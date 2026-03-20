#!/usr/bin/env python3
"""Pipeline Step 1: Fetch all video metadata from a YouTube channel.

Usage:
    python scripts/01_fetch_channel.py
    python scripts/01_fetch_channel.py --channel @SomeChannel
    python scripts/01_fetch_channel.py --output data/videos/my_channel.json
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from config.settings import Settings
from src.channel.scraper import ChannelScraper

console = Console()

DEFAULT_OUTPUT = "data/videos/channel_videos.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch video metadata from a YouTube channel using yt-dlp.",
    )
    parser.add_argument(
        "--channel",
        type=str,
        default=None,
        help="YouTube channel handle (e.g. @SomeChannel). "
        "Defaults to the channel configured in settings.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help=f"Output JSON file path (default: {DEFAULT_OUTPUT}).",
    )
    return parser.parse_args(argv)


async def main(args: argparse.Namespace) -> None:
    settings = Settings()

    channel = args.channel or settings.youtube.channel_handle
    if not channel:
        console.print(
            "[red]Error:[/red] No --channel provided and no default channel "
            "configured in settings."
        )
        sys.exit(1)

    output_path = Path(args.output)

    console.print(
        Panel(
            f"[bold cyan]Step 1: Fetch Channel Videos[/bold cyan]\n"
            f"Channel: [green]{channel}[/green]\n"
            f"Output:  [green]{output_path}[/green]",
            title="Trading Bot Pipeline",
            border_style="blue",
        )
    )

    try:
        with console.status(f"[bold green]Fetching videos for {channel}..."):
            scraper = ChannelScraper(channel)
            channel_info = await scraper.fetch_channel_videos()
    except Exception as exc:
        console.print(f"[red]Error fetching channel:[/red] {exc}")
        sys.exit(1)

    try:
        scraper.save_to_json(channel_info, output_path)
    except Exception as exc:
        console.print(f"[red]Error saving JSON:[/red] {exc}")
        sys.exit(1)

    table = Table(title="Fetch Summary", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Channel", channel_info.handle)
    table.add_row("Channel ID", channel_info.channel_id or "N/A")
    table.add_row("Total Videos", str(channel_info.video_count))

    if channel_info.videos:
        total_seconds = sum(v.duration for v in channel_info.videos)
        hours = total_seconds / 3600
        table.add_row("Total Duration", f"{hours:.1f} hours")
        table.add_row("Earliest Video", channel_info.videos[0].upload_date or "N/A")
        table.add_row("Latest Video", channel_info.videos[-1].upload_date or "N/A")

    table.add_row("Output File", str(output_path))

    console.print()
    console.print(table)
    console.print(
        f"\n[bold green]Done![/bold green] Saved {channel_info.video_count} videos "
        f"to [cyan]{output_path}[/cyan]"
    )


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
