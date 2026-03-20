"""Pydantic models for YouTube video and channel metadata."""

from __future__ import annotations

from pydantic import BaseModel, Field, computed_field


class VideoInfo(BaseModel):
    """Metadata for a single YouTube video."""

    id: str
    title: str
    duration: int = Field(description="Duration in seconds")
    upload_date: str = Field(description="Upload date in YYYYMMDD format")
    description: str = ""
    view_count: int | None = None
    channel: str = ""

    @computed_field  # type: ignore[prop-decorator]
    @property
    def url(self) -> str:
        """Full YouTube watch URL derived from the video id."""
        return self.get_url()

    def get_url(self) -> str:
        """Return the YouTube watch URL for this video."""
        return f"https://www.youtube.com/watch?v={self.id}"

    def human_duration(self) -> str:
        """Return the duration formatted as a human-readable string.

        Examples:
            45        -> "0:45"
            125       -> "2:05"
            3661      -> "1:01:01"
        """
        total = self.duration
        if total < 0:
            return "0:00"

        hours, remainder = divmod(total, 3600)
        minutes, seconds = divmod(remainder, 60)

        if hours > 0:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        return f"{minutes}:{seconds:02d}"


class ChannelInfo(BaseModel):
    """Metadata for a YouTube channel and its videos."""

    handle: str = Field(description="Channel handle, e.g. @somechannel")
    channel_id: str | None = None
    video_count: int = 0
    videos: list[VideoInfo] = Field(default_factory=list)
