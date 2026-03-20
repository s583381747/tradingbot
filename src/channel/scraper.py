"""YouTube channel scraper using yt-dlp (no API key required)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import yt_dlp  # type: ignore[import-untyped]

from .models import ChannelInfo, VideoInfo

logger = logging.getLogger(__name__)


class ChannelScraper:
    """Scrape video metadata from a YouTube channel via yt-dlp.

    Usage::

        scraper = ChannelScraper("@somechannel")
        info = await scraper.fetch_channel_videos()
        scraper.save_to_json(info, Path("output.json"))
    """

    _CHANNEL_URL_TEMPLATE = "https://www.youtube.com/{handle}/videos"

    def __init__(self, channel_handle: str) -> None:
        # Normalise handle — accept with or without leading '@'.
        if not channel_handle.startswith("@"):
            channel_handle = f"@{channel_handle}"
        self.channel_handle = channel_handle

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def fetch_channel_videos(self) -> ChannelInfo:
        """Fetch all video metadata from the channel.

        Uses ``flat_playlist`` so that no actual media is downloaded —
        only metadata is extracted.  The returned list is sorted by
        ``upload_date`` (oldest first).
        """
        url = self._CHANNEL_URL_TEMPLATE.format(handle=self.channel_handle)
        logger.info("Fetching videos for %s from %s", self.channel_handle, url)

        opts: dict[str, Any] = {
            "extract_flat": True,        # flat playlist — metadata only
            "quiet": True,
            "no_warnings": True,
            "ignoreerrors": True,         # skip unavailable videos
            "skip_download": True,
        }

        result = await self._run_ytdlp(url, opts)

        entries: list[dict[str, Any]] = result.get("entries") or []
        channel_id: str | None = result.get("channel_id") or result.get("id")

        videos: list[VideoInfo] = []
        for entry in entries:
            try:
                video = self._parse_entry(entry)
                videos.append(video)
            except Exception:
                vid_id = entry.get("id", "unknown")
                logger.warning("Skipping unparseable entry %s", vid_id, exc_info=True)

        # Sort by upload_date ascending (oldest first).
        videos.sort(key=lambda v: v.upload_date)

        info = ChannelInfo(
            handle=self.channel_handle,
            channel_id=channel_id,
            video_count=len(videos),
            videos=videos,
        )
        logger.info(
            "Fetched %d videos for channel %s", info.video_count, self.channel_handle
        )
        return info

    def save_to_json(self, channel_info: ChannelInfo, output_path: Path) -> None:
        """Serialise *channel_info* to a JSON file at *output_path*."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data = channel_info.model_dump(mode="json")
        output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        logger.info("Saved channel info to %s", output_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _run_ytdlp(self, url: str, opts: dict[str, Any]) -> dict[str, Any]:
        """Run yt-dlp with the given options and return the info dict.

        Runs the blocking yt-dlp extraction in a thread so it doesn't
        block the async event loop.
        """
        import asyncio

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._extract_info, url, opts)

    def _extract_info(self, url: str, opts: dict[str, Any]) -> dict[str, Any]:
        """Synchronous wrapper around ``yt_dlp.YoutubeDL.extract_info``."""
        with yt_dlp.YoutubeDL(opts) as ydl:
            try:
                info: dict[str, Any] | None = ydl.extract_info(url, download=False)
            except yt_dlp.utils.DownloadError as exc:
                logger.error("yt-dlp download error for %s: %s", url, exc)
                raise RuntimeError(f"Failed to extract info from {url}") from exc
            except Exception as exc:
                logger.error("Unexpected yt-dlp error for %s: %s", url, exc)
                raise RuntimeError(f"yt-dlp error for {url}") from exc

        if info is None:
            raise RuntimeError(f"yt-dlp returned no data for {url}")
        return info

    @staticmethod
    def _parse_entry(entry: dict[str, Any]) -> VideoInfo:
        """Convert a single yt-dlp flat-playlist entry to a ``VideoInfo``."""
        video_id: str = entry.get("id") or entry.get("url", "")
        title: str = entry.get("title") or ""
        duration: int = int(entry.get("duration") or 0)
        upload_date: str = entry.get("upload_date") or ""
        description: str = entry.get("description") or ""
        view_count: int | None = entry.get("view_count")
        channel: str = entry.get("channel") or entry.get("uploader") or ""

        return VideoInfo(
            id=video_id,
            title=title,
            duration=duration,
            upload_date=upload_date,
            description=description,
            view_count=view_count,
            channel=channel,
        )
