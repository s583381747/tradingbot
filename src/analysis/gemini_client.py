"""Gemini API client with file upload, rate limiting and retry logic."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from google import genai
from google.genai import types
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


class DailyQuotaExhausted(Exception):
    """Raised when the Gemini daily API quota is exhausted."""
    pass


class VideoAccessDenied(Exception):
    """Raised when a video cannot be accessed."""
    pass


# ---------------------------------------------------------------------------
# Rate Limiter
# ---------------------------------------------------------------------------

@dataclass
class RateLimiter:
    """Token-bucket style rate limiter for per-minute and per-day quotas."""

    requests_per_minute: int = 10
    requests_per_day: int = 50
    video_hours_per_day: float = 8.0

    _minute_timestamps: list[float] = field(default_factory=list, repr=False)
    _day_timestamps: list[float] = field(default_factory=list, repr=False)
    _day_video_seconds: float = field(default=0.0, repr=False)
    _day_start: float = field(default_factory=time.time, repr=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    def _purge_old_entries(self) -> None:
        now = time.time()
        cutoff_minute = now - 60.0
        self._minute_timestamps = [t for t in self._minute_timestamps if t > cutoff_minute]
        if now - self._day_start >= 86400:
            self._day_timestamps.clear()
            self._day_video_seconds = 0.0
            self._day_start = now

    async def wait_if_needed(self) -> None:
        async with self._lock:
            while True:
                self._purge_old_entries()
                now = time.time()

                if len(self._minute_timestamps) >= self.requests_per_minute:
                    oldest = self._minute_timestamps[0]
                    wait_seconds = 60.0 - (now - oldest) + 0.1
                    if wait_seconds > 0:
                        logger.info("Rate limit (RPM): sleeping %.1fs", wait_seconds)
                        await asyncio.sleep(wait_seconds)
                        continue

                if len(self._day_timestamps) >= self.requests_per_day:
                    raise DailyQuotaExhausted("Daily request quota exhausted")

                if self._day_video_seconds >= self.video_hours_per_day * 3600:
                    raise DailyQuotaExhausted("Daily video hours quota exhausted")

                break

    def record_request(self, video_duration_seconds: int = 0) -> None:
        now = time.time()
        self._minute_timestamps.append(now)
        self._day_timestamps.append(now)
        if video_duration_seconds > 0:
            self._day_video_seconds += video_duration_seconds

    @property
    def remaining_rpm(self) -> int:
        self._purge_old_entries()
        return max(0, self.requests_per_minute - len(self._minute_timestamps))

    @property
    def remaining_rpd(self) -> int:
        self._purge_old_entries()
        return max(0, self.requests_per_day - len(self._day_timestamps))


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------

def _is_retriable(exc: BaseException) -> bool:
    if isinstance(exc, (DailyQuotaExhausted, VideoAccessDenied)):
        return False
    if isinstance(exc, genai.errors.ClientError):
        msg = str(exc)
        if "PerDay" in msg or "per_day" in msg.lower():
            return False
        if "PERMISSION_DENIED" in msg or "403" in msg:
            return False
        return True
    return isinstance(exc, (genai.errors.ServerError, ConnectionError, TimeoutError))


# ---------------------------------------------------------------------------
# Gemini Client
# ---------------------------------------------------------------------------

class GeminiClient:
    """Gemini API client with file upload support, rate limiting and retries."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash",
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._rate_limiter = rate_limiter or RateLimiter()

    # -- File upload ----------------------------------------------------------

    def upload_video(self, file_path: str | Path) -> types.File:
        """Upload a video file to Gemini File API.

        Returns a File object that can be used in generate_content calls.
        The file is processed asynchronously — this method polls until ready.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Video file not found: {file_path}")

        logger.info("Uploading %s to Gemini File API...", file_path.name)
        uploaded = self._client.files.upload(file=str(file_path))
        logger.info("Upload complete: %s (state=%s)", uploaded.name, uploaded.state)

        # Poll until processing is done
        import time as _time
        while uploaded.state == "PROCESSING":
            logger.debug("File %s still processing, waiting 5s...", uploaded.name)
            _time.sleep(5)
            uploaded = self._client.files.get(name=uploaded.name)

        if uploaded.state == "FAILED":
            raise RuntimeError(f"Gemini failed to process file: {file_path.name}")

        logger.info("File ready: %s", uploaded.name)
        return uploaded

    def delete_file(self, file: types.File) -> None:
        """Delete an uploaded file from Gemini File API."""
        try:
            self._client.files.delete(name=file.name)
            logger.debug("Deleted file %s", file.name)
        except Exception as exc:
            logger.warning("Failed to delete file %s: %s", file.name, exc)

    # -- Analysis with uploaded file ------------------------------------------

    async def analyze_uploaded_video(
        self,
        file: types.File,
        prompt: str,
        response_schema: dict | None = None,
        video_duration_seconds: int = 0,
    ) -> dict:
        """Analyze an already-uploaded video file with Gemini."""
        return await self._call_gemini_with_file(
            file=file,
            prompt=prompt,
            response_schema=response_schema,
            video_duration_seconds=video_duration_seconds,
        )

    # -- Text generation ------------------------------------------------------

    async def generate_text(self, prompt: str) -> str:
        """Send a text-only prompt and return the raw response."""
        await self._rate_limiter.wait_if_needed()
        logger.info("Sending text request to Gemini (%s)", self._model)
        response = await asyncio.to_thread(
            self._client.models.generate_content,
            model=self._model,
            contents=[prompt],
        )
        self._rate_limiter.record_request()
        return response.text or ""

    # -- Internal helpers -----------------------------------------------------

    @retry(
        retry=retry_if_exception(_is_retriable),
        wait=wait_exponential(multiplier=2, min=10, max=120),
        stop=stop_after_attempt(6),
        reraise=True,
        before_sleep=lambda retry_state: logger.warning(
            "Gemini request failed (%s), retrying in %.1fs (attempt %d/6)...",
            retry_state.outcome.exception().__class__.__name__,
            retry_state.next_action.sleep,
            retry_state.attempt_number,
        ),
    )
    async def _call_gemini_with_file(
        self,
        file: types.File,
        prompt: str,
        response_schema: dict | None,
        video_duration_seconds: int,
    ) -> dict:
        """Make Gemini API call with an uploaded file."""
        import json

        await self._rate_limiter.wait_if_needed()

        logger.info("Sending request to Gemini (%s) with file %s", self._model, file.name)

        config_kwargs: dict = {"response_mime_type": "application/json"}
        if response_schema is not None:
            config_kwargs["response_schema"] = response_schema

        response = await asyncio.to_thread(
            self._client.models.generate_content,
            model=self._model,
            contents=[file, prompt],
            config=types.GenerateContentConfig(**config_kwargs),
        )

        self._rate_limiter.record_request(video_duration_seconds=video_duration_seconds)

        raw_text = response.text
        if not raw_text:
            logger.error("Empty response from Gemini for file %s", file.name)
            return {}

        try:
            result: dict = json.loads(raw_text)
        except json.JSONDecodeError:
            cleaned = raw_text.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                cleaned = "\n".join(lines[1:-1]) if len(lines) > 2 else cleaned
            result = json.loads(cleaned)

        logger.info("Successfully received response for file %s", file.name)
        return result
