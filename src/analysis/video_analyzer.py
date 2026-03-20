"""Core video analysis: download → upload to Gemini → multimodal analysis."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from config.prompts.video_analysis import (
    FIRST_PASS_PROMPT,
    format_second_pass_prompt,
)
from src.analysis.gemini_client import GeminiClient
from src.channel.models import VideoInfo

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------

class KeyTimestamp(BaseModel):
    time: str = Field(description="Timestamp in MM:SS or HH:MM:SS format")
    description: str = Field(description="What happens at this moment")


class IndicatorInfo(BaseModel):
    name: str
    params: dict = Field(default_factory=dict)


class StrategyOverview(BaseModel):
    name: str
    description: str = ""
    timeframes: list[str] = Field(default_factory=list)
    markets: list[str] = Field(default_factory=list)


class FirstPassResult(BaseModel):
    topic: str = ""
    content_type: str = Field(default="other")
    strategies_overview: list[StrategyOverview] = Field(default_factory=list)
    indicators: list[IndicatorInfo] = Field(default_factory=list)
    chart_annotations: str = ""
    key_timestamps: list[KeyTimestamp] = Field(default_factory=list)


class EntryExitCondition(BaseModel):
    description: str
    indicator: str = ""
    condition: str = ""
    required: bool = True


class StopTakeProfitRule(BaseModel):
    type: str = Field(default="none_mentioned")
    value: float | None = None
    description: str = ""


class SecondPassResult(BaseModel):
    strategy_name: str
    entry_conditions: list[EntryExitCondition] = Field(default_factory=list)
    exit_conditions: list[EntryExitCondition] = Field(default_factory=list)
    indicator_params: dict = Field(default_factory=dict)
    stop_loss_rules: StopTakeProfitRule = Field(default_factory=StopTakeProfitRule)
    take_profit_rules: StopTakeProfitRule = Field(default_factory=StopTakeProfitRule)
    position_sizing: str = "not_mentioned"
    quantifiability_score: int = Field(default=1, ge=1, le=10)


class VideoAnalysisResult(BaseModel):
    video_id: str
    analysis_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    first_pass: FirstPassResult
    second_pass: list[SecondPassResult] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# JSON schemas for structured Gemini output
# ---------------------------------------------------------------------------

_FIRST_PASS_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "topic": {"type": "string"},
        "content_type": {"type": "string"},
        "strategies_overview": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "timeframes": {"type": "array", "items": {"type": "string"}},
                    "markets": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["name"],
            },
        },
        "indicators": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "params": {"type": "object"},
                },
                "required": ["name"],
            },
        },
        "chart_annotations": {"type": "string"},
        "key_timestamps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "time": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["time", "description"],
            },
        },
    },
    "required": ["topic", "content_type"],
}

_SECOND_PASS_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "strategy_name": {"type": "string"},
        "entry_conditions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "indicator": {"type": "string"},
                    "condition": {"type": "string"},
                    "required": {"type": "boolean"},
                },
                "required": ["description"],
            },
        },
        "exit_conditions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "indicator": {"type": "string"},
                    "condition": {"type": "string"},
                    "required": {"type": "boolean"},
                },
                "required": ["description"],
            },
        },
        "indicator_params": {"type": "object"},
        "stop_loss_rules": {
            "type": "object",
            "properties": {
                "type": {"type": "string"},
                "value": {"type": "number"},
                "description": {"type": "string"},
            },
        },
        "take_profit_rules": {
            "type": "object",
            "properties": {
                "type": {"type": "string"},
                "value": {"type": "number"},
                "description": {"type": "string"},
            },
        },
        "position_sizing": {"type": "string"},
        "quantifiability_score": {"type": "integer"},
    },
    "required": ["strategy_name", "quantifiability_score"],
}


# ---------------------------------------------------------------------------
# Video downloader (yt-dlp)
# ---------------------------------------------------------------------------

class VideoDownloader:
    """Download YouTube videos to local files using yt-dlp."""

    def __init__(self, download_dir: Path) -> None:
        self._download_dir = Path(download_dir)
        self._download_dir.mkdir(parents=True, exist_ok=True)

    async def download(self, video: VideoInfo) -> Path:
        """Download a video and return the local file path."""
        output_template = str(self._download_dir / f"{video.id}.%(ext)s")

        # Check if already downloaded
        for existing in self._download_dir.glob(f"{video.id}.*"):
            if existing.suffix in ('.mp4', '.webm', '.mkv'):
                logger.info("Video %s already downloaded: %s", video.id, existing)
                return existing

        import yt_dlp

        opts = {
            "format": "best[height<=720]/best",
            "outtmpl": output_template,
            "quiet": True,
            "no_warnings": True,
        }

        logger.info("Downloading video %s: %s", video.id, video.title)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._do_download, video.get_url(), opts)

        # Find the downloaded file
        for f in self._download_dir.glob(f"{video.id}.*"):
            if f.suffix in ('.mp4', '.webm', '.mkv', '.mp4.part'):
                if not f.suffix.endswith('.part'):
                    logger.info("Downloaded: %s (%.1f MB)", f.name, f.stat().st_size / 1e6)
                    return f

        raise RuntimeError(f"Download completed but file not found for {video.id}")

    @staticmethod
    def _do_download(url: str, opts: dict) -> None:
        import yt_dlp
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])

    def cleanup(self, video_id: str) -> None:
        """Delete downloaded video file to save disk space."""
        for f in self._download_dir.glob(f"{video_id}.*"):
            try:
                f.unlink()
                logger.debug("Deleted %s", f)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class VideoAnalyzer:
    """Download → Upload to Gemini File API → Multimodal analysis."""

    def __init__(
        self,
        gemini_client: GeminiClient,
        download_dir: Path | str = "data/videos/downloads",
        skip_second_pass: bool = False,
        cleanup_after: bool = True,
    ) -> None:
        self._gemini = gemini_client
        self._downloader = VideoDownloader(Path(download_dir))
        self._skip_second_pass = skip_second_pass
        self._cleanup_after = cleanup_after

    async def analyze_video(self, video: VideoInfo) -> VideoAnalysisResult:
        """Full pipeline: download → upload → first pass → second pass → cleanup."""
        import json as _json

        gemini_file = None
        try:
            # Step 1: Download video
            local_path = await self._downloader.download(video)

            # Step 2: Upload to Gemini File API
            gemini_file = await asyncio.to_thread(
                self._gemini.upload_video, local_path
            )

            # Step 3: First pass analysis
            logger.info("First pass for %s: %s", video.id, video.title)
            first_pass_prompt = self._build_first_pass_prompt(video)

            raw_first = await self._gemini.analyze_uploaded_video(
                file=gemini_file,
                prompt=first_pass_prompt,
                response_schema=_FIRST_PASS_SCHEMA,
                video_duration_seconds=video.duration,
            )

            first_pass = FirstPassResult.model_validate(raw_first)
            logger.info(
                "First pass done: %d strategies, type=%s",
                len(first_pass.strategies_overview),
                first_pass.content_type,
            )

            # Step 4: Second pass (reuse the same uploaded file)
            second_pass_results: list[SecondPassResult] = []

            if not self._skip_second_pass:
                first_pass_summary = _json.dumps(raw_first, ensure_ascii=False, indent=2)

                for strategy in first_pass.strategies_overview:
                    logger.info("Second pass for '%s' in %s", strategy.name, video.id)
                    second_pass_prompt = self._build_second_pass_prompt(
                        video, strategy, first_pass_summary
                    )

                    raw_second = await self._gemini.analyze_uploaded_video(
                        file=gemini_file,
                        prompt=second_pass_prompt,
                        response_schema=_SECOND_PASS_SCHEMA,
                        video_duration_seconds=video.duration,
                    )

                    result = SecondPassResult.model_validate(raw_second)
                    second_pass_results.append(result)
                    logger.info(
                        "Second pass done: '%s' quantifiability=%d/10",
                        result.strategy_name,
                        result.quantifiability_score,
                    )

            return VideoAnalysisResult(
                video_id=video.id,
                first_pass=first_pass,
                second_pass=second_pass_results,
            )

        finally:
            # Cleanup: delete from Gemini File API
            if gemini_file is not None:
                await asyncio.to_thread(self._gemini.delete_file, gemini_file)
            # Cleanup: delete local video file
            if self._cleanup_after:
                self._downloader.cleanup(video.id)

    # -- Prompt builders -------------------------------------------------------

    @staticmethod
    def _build_first_pass_prompt(video: VideoInfo) -> str:
        context = (
            f"Video title: {video.title}\n"
            f"Duration: {video.human_duration()}\n"
            f"Channel: {video.channel}\n"
        )
        if video.description:
            context += f"Description preview: {video.description[:500]}\n"
        return f"{context}\n{FIRST_PASS_PROMPT}"

    @staticmethod
    def _build_second_pass_prompt(
        video: VideoInfo,
        strategy_info: StrategyOverview,
        first_pass_summary: str,
    ) -> str:
        strategy_desc = strategy_info.name
        if strategy_info.description:
            strategy_desc += f" - {strategy_info.description}"
        if strategy_info.timeframes:
            strategy_desc += f" (timeframes: {', '.join(strategy_info.timeframes)})"
        if strategy_info.markets:
            strategy_desc += f" (markets: {', '.join(strategy_info.markets)})"

        prompt_text = format_second_pass_prompt(
            first_pass_summary=first_pass_summary,
            target_strategy=strategy_desc,
        )
        context = (
            f"Video title: {video.title}\n"
            f"Duration: {video.human_duration()}\n"
            f"Channel: {video.channel}\n"
        )
        return f"{context}\n{prompt_text}"
