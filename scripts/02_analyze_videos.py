#!/usr/bin/env python3
"""Pipeline Step 2: Analyze videos using Gemini to extract trading insights.

Usage:
    python scripts/02_analyze_videos.py
    python scripts/02_analyze_videos.py --input data/videos/channel_videos.json --max-videos 10
    python scripts/02_analyze_videos.py --no-resume
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from config.settings import Settings
from src.analysis.gemini_client import GeminiClient, RateLimiter
from src.analysis.video_analyzer import VideoAnalyzer
from src.analysis.batch_processor import BatchProcessor
from src.channel.models import ChannelInfo

console = Console()

DEFAULT_INPUT = "data/videos/channel_videos.json"
DEFAULT_OUTPUT_DIR = "data/analyses/"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze YouTube videos with Gemini to extract trading strategies.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_INPUT,
        help=f"Path to channel_videos.json (default: {DEFAULT_INPUT}).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for analysis output files (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Limit the number of videos to analyse (default: all).",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from where we left off (default: True).",
    )
    return parser.parse_args(argv)


def _load_channel_info(path: Path) -> ChannelInfo:
    if not path.exists():
        console.print(f"[red]Error:[/red] Input file not found: {path}")
        console.print("[dim]Run step 01 first: python scripts/01_fetch_channel.py[/dim]")
        sys.exit(1)

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return ChannelInfo.model_validate(data)
    except Exception as exc:
        console.print(f"[red]Error loading input file:[/red] {exc}")
        sys.exit(1)


async def main(args: argparse.Namespace) -> None:
    settings = Settings()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    channel_info = _load_channel_info(input_path)

    if not channel_info.videos:
        console.print("[yellow]No videos found in input file.[/yellow]")
        sys.exit(0)

    console.print(
        Panel(
            f"[bold cyan]Step 2: Analyse Videos with Gemini[/bold cyan]\n"
            f"Channel:  [green]{channel_info.handle}[/green]\n"
            f"Videos:   [green]{len(channel_info.videos)}[/green] total\n"
            f"Output:   [green]{output_dir}[/green]\n"
            f"Model:    [green]{settings.gemini.model}[/green]\n"
            f"Resume:   [green]{args.resume}[/green]",
            title="Trading Bot Pipeline",
            border_style="blue",
        )
    )

    try:
        rate_limiter = RateLimiter(
            requests_per_minute=settings.gemini.rpm,
            requests_per_day=settings.gemini.rpd,
            video_hours_per_day=settings.gemini.daily_video_hours,
        )
        gemini_client = GeminiClient(
            api_key=settings.gemini.api_key.get_secret_value(),
            model=settings.gemini.model,
            rate_limiter=rate_limiter,
        )
        analyzer = VideoAnalyzer(gemini_client, skip_second_pass=True)
        state_file = output_dir / "processing_state.json"
        processor = BatchProcessor(
            analyzer=analyzer,
            output_dir=output_dir,
            state_file=state_file,
            concurrency=3,
        )
    except Exception as exc:
        console.print(f"[red]Error initialising analysis components:[/red] {exc}")
        sys.exit(1)

    try:
        final_state = await processor.process_channel(
            channel=channel_info,
            max_videos=args.max_videos,
        )
    except Exception as exc:
        console.print(f"\n[red]Batch processing error:[/red] {exc}")
        sys.exit(1)

    console.print()
    table = Table(title="Analysis Summary", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Total Videos", str(final_state.total_videos))
    table.add_row("Completed", f"[green]{len(final_state.completed)}[/green]")
    failed_count = len(final_state.failed)
    table.add_row("Failed", f"[red]{failed_count}[/red]" if failed_count else "0")
    table.add_row("Output Directory", str(output_dir))
    console.print(table)

    if final_state.failed:
        console.print("\n[bold red]Errors:[/bold red]")
        for vid_id, err in final_state.failed.items():
            console.print(f"  [red]-[/red] {vid_id}: {err}")

    console.print(
        f"\n[bold green]Done![/bold green] Analysed {len(final_state.completed)} video(s) successfully."
    )


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
