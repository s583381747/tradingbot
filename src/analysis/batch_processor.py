"""Batch processing of channel videos with checkpoint/resume and concurrency."""

from __future__ import annotations

import asyncio
import logging
import signal
import threading
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from src.analysis.gemini_client import DailyQuotaExhausted
from src.analysis.video_analyzer import VideoAnalyzer, VideoAnalysisResult
from src.channel.models import ChannelInfo, VideoInfo

logger = logging.getLogger(__name__)
console = Console()


class ProcessingState(BaseModel):
    """Tracks batch processing progress so it can be resumed after interruption."""

    total_videos: int = 0
    completed: list[str] = Field(default_factory=list)
    failed: dict[str, str] = Field(default_factory=dict)
    in_progress: list[str] = Field(default_factory=list)
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class BatchProcessor:
    """Process videos concurrently with checkpoint/resume and graceful interruption."""

    def __init__(
        self,
        analyzer: VideoAnalyzer,
        output_dir: Path,
        state_file: Path,
        concurrency: int = 3,
    ) -> None:
        self._analyzer = analyzer
        self._output_dir = Path(output_dir)
        self._state_file = Path(state_file)
        self._concurrency = concurrency
        self._lock = threading.Lock()

        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._state_file.parent.mkdir(parents=True, exist_ok=True)

    async def process_channel(
        self,
        channel: ChannelInfo,
        max_videos: int | None = None,
    ) -> ProcessingState:
        state = self._load_state()

        # Filter members-only videos
        _PRIVATE_KEYWORDS = ("会员专享", "会员", "member", "专享")
        videos = [
            v for v in channel.videos
            if not any(kw in v.title for kw in _PRIVATE_KEYWORDS)
        ]
        skipped = len(channel.videos) - len(videos)
        if skipped:
            console.print(f"[dim]Skipped {skipped} members-only video(s).[/dim]")
        if max_videos is not None:
            videos = videos[:max_videos]

        state.total_videos = len(videos)
        state.in_progress = []
        self._save_state(state)

        pending = [v for v in videos if v.id not in state.completed and v.id not in state.failed]

        if not pending:
            console.print(f"[bold green]All {len(videos)} videos already processed.[/bold green]")
            return state

        console.print(
            f"[bold]Processing channel [cyan]{channel.handle}[/cyan]: "
            f"{len(pending)} remaining of {len(videos)} total "
            f"({len(state.completed)} done, {len(state.failed)} failed) "
            f"[concurrency={self._concurrency}][/bold]"
        )

        # Graceful Ctrl+C
        interrupted = False

        def _handle_interrupt(sig, frame):
            nonlocal interrupted
            if not interrupted:
                interrupted = True
                console.print(
                    f"\n[yellow]Interrupt received. Finishing current videos then stopping. "
                    f"Progress saved — run again to resume.[/yellow]"
                )

        old_handler = signal.signal(signal.SIGINT, _handle_interrupt)

        # Semaphore to limit concurrency
        sem = asyncio.Semaphore(self._concurrency)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=False,
        ) as progress:
            task_id = progress.add_task("Analyzing videos", total=len(pending))

            async def _process_one(video: VideoInfo) -> None:
                nonlocal interrupted
                if interrupted:
                    return

                async with sem:
                    if interrupted:
                        return

                    progress.update(
                        task_id,
                        description=f"[cyan]{video.title[:45]}...[/cyan]"
                        if len(video.title) > 45
                        else f"[cyan]{video.title}[/cyan]",
                    )

                    try:
                        result = await self._analyzer.analyze_video(video)
                        self._save_result(result)

                        with self._lock:
                            state.completed.append(video.id)
                            if video.id in state.in_progress:
                                state.in_progress.remove(video.id)
                            self._save_state(state)

                        console.print(
                            f"  [green]OK[/green] {video.id} — {video.title[:60]}"
                        )

                    except (KeyboardInterrupt, asyncio.CancelledError):
                        interrupted = True
                        with self._lock:
                            if video.id in state.in_progress:
                                state.in_progress.remove(video.id)
                            self._save_state(state)

                    except DailyQuotaExhausted:
                        interrupted = True
                        with self._lock:
                            if video.id in state.in_progress:
                                state.in_progress.remove(video.id)
                            self._save_state(state)
                        console.print(
                            f"\n[yellow]Daily quota exhausted after "
                            f"{len(state.completed)} videos. Run again tomorrow.[/yellow]"
                        )

                    except Exception as exc:
                        error_msg = f"{type(exc).__name__}: {exc}"
                        exc_str = str(exc)

                        if "PerDay" in exc_str or "RESOURCE_EXHAUSTED" in exc_str:
                            interrupted = True
                            with self._lock:
                                if video.id in state.in_progress:
                                    state.in_progress.remove(video.id)
                                self._save_state(state)
                            console.print(
                                f"\n[yellow]API quota exhausted after "
                                f"{len(state.completed)} videos. Run again later.[/yellow]"
                            )
                        else:
                            with self._lock:
                                state.failed[video.id] = error_msg
                                if video.id in state.in_progress:
                                    state.in_progress.remove(video.id)
                                self._save_state(state)
                            console.print(
                                f"  [red]FAILED[/red] {video.id}: {error_msg[:100]}"
                            )

                    progress.advance(task_id)

            # Launch all tasks with semaphore limiting concurrency
            tasks = [asyncio.create_task(_process_one(v)) for v in pending]
            await asyncio.gather(*tasks, return_exceptions=True)

        signal.signal(signal.SIGINT, old_handler)

        console.print(f"\n{self.get_progress_summary(state)}")
        return state

    def get_progress_summary(self, state: ProcessingState | None = None) -> str:
        if state is None:
            state = self._load_state()

        total = state.total_videos
        done = len(state.completed)
        failed = len(state.failed)
        remaining = total - done - failed

        lines = [
            f"Processing Summary:",
            f"  Total videos:  {total}",
            f"  Completed:     {done}",
            f"  Failed:        {failed}",
            f"  Remaining:     {remaining}",
        ]

        if state.failed:
            lines.append("  Failures:")
            for vid, err in list(state.failed.items())[:10]:
                lines.append(f"    - {vid}: {err[:80]}")
            if len(state.failed) > 10:
                lines.append(f"    ... and {len(state.failed) - 10} more")

        return "\n".join(lines)

    def _load_state(self) -> ProcessingState:
        if self._state_file.exists():
            try:
                raw = self._state_file.read_text(encoding="utf-8")
                return ProcessingState.model_validate_json(raw)
            except Exception as exc:
                logger.warning("Failed to load state (%s), starting fresh.", exc)
        return ProcessingState()

    def _save_state(self, state: ProcessingState) -> None:
        state.last_updated = datetime.now(timezone.utc)
        self._state_file.write_text(
            state.model_dump_json(indent=2),
            encoding="utf-8",
        )

    def _save_result(self, result: VideoAnalysisResult) -> None:
        output_path = self._output_dir / f"{result.video_id}.json"
        output_path.write_text(
            result.model_dump_json(indent=2),
            encoding="utf-8",
        )
