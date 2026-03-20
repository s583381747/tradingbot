"""
Settings management for the trading bot pipeline.

Loads configuration from environment variables / .env file using pydantic-settings.
Each service has its own settings class with env_prefix to avoid collisions.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root is two levels up from this file: config/settings.py -> project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"


class GeminiSettings(BaseSettings):
    """Google Gemini API configuration.

    Free-tier rate limits (as of 2025):
      - 5 requests per minute
      - 100 requests per day
      - 8 hours of video processing per day
    """

    model_config = SettingsConfigDict(
        env_prefix="GEMINI_",
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    api_key: SecretStr = Field(..., description="Gemini API key from Google AI Studio")
    model: str = Field(default="gemini-2.5-pro", description="Model name for video analysis")
    rpm: int = Field(default=5, description="Rate limit: requests per minute (free tier)")
    rpd: int = Field(default=100, description="Rate limit: requests per day (free tier)")
    daily_video_hours: float = Field(
        default=8.0, description="Rate limit: max video hours per day (free tier)"
    )
    temperature: float = Field(default=0.2, description="Sampling temperature for analysis tasks")
    max_output_tokens: int = Field(default=65536, description="Max output tokens per response")
    thinking_budget: int = Field(
        default=8192, description="Token budget for model thinking/reasoning"
    )


class OpenAISettings(BaseSettings):
    """OpenAI API configuration (optional, for Whisper fallback)."""

    model_config = SettingsConfigDict(
        env_prefix="OPENAI_",
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    api_key: Optional[SecretStr] = Field(
        default=None, description="OpenAI API key (optional, for Whisper fallback)"
    )
    whisper_model: str = Field(
        default="whisper-1", description="Whisper model for audio transcription"
    )
    base_url: Optional[str] = Field(
        default=None, description="Custom API base URL (for proxies)"
    )


class AnthropicSettings(BaseSettings):
    """Anthropic API configuration (optional, for strategy refinement)."""

    model_config = SettingsConfigDict(
        env_prefix="ANTHROPIC_",
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    api_key: Optional[SecretStr] = Field(
        default=None, description="Anthropic API key (optional)"
    )
    model: str = Field(
        default="claude-sonnet-4-20250514",
        description="Model for strategy refinement tasks",
    )


class YouTubeSettings(BaseSettings):
    """YouTube channel and download configuration."""

    model_config = SettingsConfigDict(
        env_prefix="YOUTUBE_",
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    channel_handle: str = Field(
        default="@rijindoudoujin",
        description="YouTube channel handle to scrape",
    )
    max_concurrent_downloads: int = Field(
        default=2, description="Max parallel yt-dlp downloads"
    )
    download_dir: Path = Field(
        default=PROJECT_ROOT / "data" / "videos",
        description="Directory to store downloaded videos",
    )
    cookies_file: Optional[Path] = Field(
        default=None, description="Path to cookies.txt for age-gated content"
    )
    prefer_format: str = Field(
        default="mp4", description="Preferred video container format"
    )
    max_resolution: int = Field(
        default=720,
        description="Max video resolution (lower = smaller files, faster uploads)",
    )


class PipelineSettings(BaseSettings):
    """Pipeline execution configuration."""

    model_config = SettingsConfigDict(
        env_prefix="PIPELINE_",
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_dir: Path = Field(
        default=PROJECT_ROOT / "data", description="Root data directory"
    )
    analysis_dir: Path = Field(
        default=PROJECT_ROOT / "data" / "analyses", description="Video analysis output"
    )
    strategy_dir: Path = Field(
        default=PROJECT_ROOT / "data" / "strategies", description="Generated strategy code"
    )
    checkpoint_dir: Path = Field(
        default=PROJECT_ROOT / "data" / "checkpoints", description="Pipeline state checkpoints"
    )
    max_retries: int = Field(default=3, description="Max retries for transient failures")
    retry_base_delay: float = Field(
        default=60.0,
        description="Base delay (seconds) for exponential backoff on rate limits",
    )
    batch_size: int = Field(
        default=5, description="Number of videos to process before checkpointing"
    )


class Settings(BaseSettings):
    """Root settings aggregating all service configurations.

    Usage:
        settings = get_settings()
        key = settings.gemini.api_key.get_secret_value()
    """

    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    gemini: GeminiSettings = Field(default_factory=GeminiSettings)
    openai: OpenAISettings = Field(default_factory=OpenAISettings)
    anthropic: AnthropicSettings = Field(default_factory=AnthropicSettings)
    youtube: YouTubeSettings = Field(default_factory=YouTubeSettings)
    pipeline: PipelineSettings = Field(default_factory=PipelineSettings)

    debug: bool = Field(default=False, description="Enable debug logging")
    log_level: str = Field(default="INFO", description="Logging level")

    def ensure_dirs(self) -> None:
        """Create all configured directories if they don't exist."""
        for dir_path in [
            self.youtube.download_dir,
            self.pipeline.data_dir,
            self.pipeline.analysis_dir,
            self.pipeline.strategy_dir,
            self.pipeline.checkpoint_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get cached settings singleton.

    The first call reads .env and environment variables.
    Subsequent calls return the cached instance.
    Clear with get_settings.cache_clear() if needed.
    """
    settings = Settings()
    settings.ensure_dirs()
    return settings
