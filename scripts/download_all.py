#!/usr/bin/env python3
"""Download all public videos from the channel for offline analysis."""

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import yt_dlp
from rich.console import Console
from rich.progress import track

console = Console()

DOWNLOAD_DIR = _PROJECT_ROOT / "data" / "videos" / "downloads"
CHANNEL_FILE = _PROJECT_ROOT / "data" / "videos" / "channel_videos.json"
PRIVATE_KEYWORDS = ("会员专享", "会员", "member", "专享")


def main():
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    data = json.loads(CHANNEL_FILE.read_text())
    all_videos = data["videos"]

    # Filter members-only
    videos = [v for v in all_videos if not any(kw in v["title"] for kw in PRIVATE_KEYWORDS)]
    console.print(f"Total: {len(all_videos)}, Public: {len(videos)}, Members-only: {len(all_videos) - len(videos)}")

    # Skip already downloaded
    existing = {f.stem for f in DOWNLOAD_DIR.glob("*.*") if f.suffix in ('.mp4', '.webm', '.mkv')}
    to_download = [v for v in videos if v["id"] not in existing]
    console.print(f"Already downloaded: {len(existing)}, To download: {len(to_download)}")

    if not to_download:
        console.print("[green]All videos already downloaded![/green]")
        return

    failed = []
    for v in track(to_download, description="Downloading..."):
        vid_id = v["id"]
        title = v["title"][:50]
        url = f"https://www.youtube.com/watch?v={vid_id}"
        output = str(DOWNLOAD_DIR / f"{vid_id}.%(ext)s")

        opts = {
            "format": "best[height<=720]/best",
            "outtmpl": output,
            "quiet": True,
            "no_warnings": True,
        }

        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([url])
            console.print(f"  [green]OK[/green] {vid_id} — {title}")
        except Exception as e:
            err = str(e)[:80]
            failed.append(vid_id)
            console.print(f"  [red]FAIL[/red] {vid_id} — {err}")

    console.print(f"\n[bold]Done![/bold] Downloaded: {len(to_download) - len(failed)}, Failed: {len(failed)}")
    if failed:
        console.print(f"Failed IDs: {', '.join(failed)}")


if __name__ == "__main__":
    main()
