"""Gemini API client with rate limiting and retry logic."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

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


# ---------------------------------------------------------------------------
# Rate Limiter
# ---------------------------------------------------------------------------

@dataclass
class RateLimiter:
    """Token-bucket style rate limiter tracking per-minute, per-day, and video-hour quotas."""

    requests_per_minute: int = 10
    requests_per_day: int = 50
    video_hours_per_day: float = 8.0

    # Internal tracking (not set by caller)
    _minute_timestamps: list[float] = field(default_factory=list, repr=False)
    _day_timestamps: list[float] = field(default_factory=list, repr=False)
    _day_video_seconds: float = field(default=0.0, repr=False)
    _day_start: float = field(default_factory=time.time, repr=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    def _purge_old_entries(self) -> None:
        """Remove timestamps older than their respective windows."""
        now = time.time()

        # Purge minute window
        cutoff_minute = now - 60.0
        self._minute_timestamps = [t for t in self._minute_timestamps if t > cutoff_minute]

        # Reset day window if 24 h have passed
        if now - self._day_start >= 86400:
            self._day_timestamps.clear()
            self._day_video_seconds = 0.0
            self._day_start = now

    async def wait_if_needed(self) -> None:
        """Sleep until the next request can be made without exceeding limits."""
        async with self._lock:
            while True:
                self._purge_old_entries()
                now = time.time()

                # Check per-minute limit
                if len(self._minute_timestamps) >= self.requests_per_minute:
                    oldest = self._minute_timestamps[0]
                    wait_seconds = 60.0 - (now - oldest) + 0.1
                    if wait_seconds > 0:
                        logger.info(
                            "Rate limit (RPM): sleeping %.1fs (%d/%d used)",
                            wait_seconds,
                            len(self._minute_timestamps),
                            self.requests_per_minute,
                        )
                        await asyncio.sleep(wait_seconds)
                        continue

                # Check per-day limit
                if len(self._day_timestamps) >= self.requests_per_day:
                    seconds_left = 86400 - (now - self._day_start) + 1
                    logger.warning(
                        "Rate limit (RPD): daily quota exhausted. Sleeping %.0fs until reset.",
                        seconds_left,
                    )
                    await asyncio.sleep(seconds_left)
                    continue

                # Check daily video hours
                if self._day_video_seconds >= self.video_hours_per_day * 3600:
                    seconds_left = 86400 - (now - self._day_start) + 1
                    logger.warning(
                        "Rate limit (video hours): daily video quota exhausted. "
                        "Sleeping %.0fs until reset.",
                        seconds_left,
                    )
                    await asyncio.sleep(seconds_left)
                    continue

                # All clear
                break

    def record_request(self, video_duration_seconds: int = 0) -> None:
        """Record that a request was made, optionally with video duration consumed."""
        now = time.time()
        self._minute_timestamps.append(now)
        self._day_timestamps.append(now)
        if video_duration_seconds > 0:
            self._day_video_seconds += video_duration_seconds

    # -- Remaining quota properties -------------------------------------------

    @property
    def remaining_rpm(self) -> int:
        """Remaining requests available in the current minute window."""
        self._purge_old_entries()
        return max(0, self.requests_per_minute - len(self._minute_timestamps))

    @property
    def remaining_rpd(self) -> int:
        """Remaining requests available today."""
        self._purge_old_entries()
        return max(0, self.requests_per_day - len(self._day_timestamps))

    @property
    def remaining_video_hours(self) -> float:
        """Remaining video hours available today."""
        self._purge_old_entries()
        return max(0.0, self.video_hours_per_day - self._day_video_seconds / 3600)


# ---------------------------------------------------------------------------
# Gemini Client
# ---------------------------------------------------------------------------

class VideoAccessDenied(Exception):
    """Raised when a video cannot be accessed (private/age-restricted)."""
    pass


def _is_retriable(exc: BaseException) -> bool:
    """Return True if the exception is retriable, False if it should fail immediately."""
    if isinstance(exc, (DailyQuotaExhausted, VideoAccessDenied)):
        return False
    if isinstance(exc, genai.errors.ClientError):
        msg = str(exc)
        # Daily quota exhausted — don't retry
        if "PerDay" in msg or "per_day" in msg.lower():
            return False
        # 403 permission denied — video is private/restricted, don't retry
        if "PERMISSION_DENIED" in msg or "403" in msg:
            return False
        # Other 4xx/rate limits — retry
        return True
    return isinstance(exc, (genai.errors.ServerError, ConnectionError, TimeoutError))


class GeminiClient:
    """Wrapper around the Google GenAI SDK with rate limiting and automatic retries."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-pro",
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._rate_limiter = rate_limiter or RateLimiter()

    # -- Public API -----------------------------------------------------------

    async def analyze_video(
        self,
        youtube_url: str,
        prompt: str,
        response_schema: dict | None = None,
        video_duration_seconds: int = 0,
    ) -> dict:
        """Send a YouTube video + prompt to Gemini and return the parsed JSON response.

        Parameters
        ----------
        youtube_url:
            Full YouTube watch URL (passed as ``file_uri`` to Gemini).
        prompt:
            The analysis prompt.
        response_schema:
            Optional JSON-schema dict for structured output.
        video_duration_seconds:
            Duration of the video in seconds (used for rate-limit tracking).

        Returns
        -------
        dict
            Parsed JSON response from Gemini.
        """
        return await self._call_gemini(
            youtube_url=youtube_url,
            prompt=prompt,
            response_schema=response_schema,
            video_duration_seconds=video_duration_seconds,
        )

    async def analyze_video_segment(
        self,
        youtube_url: str,
        prompt: str,
        timestamp_start: str,
        timestamp_end: str,
    ) -> dict:
        """Analyze a specific segment of a video (second-pass).

        The timestamps are embedded into the prompt so Gemini focuses on the
        relevant portion of the video.

        Parameters
        ----------
        youtube_url:
            Full YouTube watch URL.
        prompt:
            The analysis prompt (should already reference the segment).
        timestamp_start:
            Start timestamp (e.g. ``"02:15"``).
        timestamp_end:
            End timestamp (e.g. ``"05:30"``).

        Returns
        -------
        dict
            Parsed JSON response from Gemini.
        """
        segment_instruction = (
            f"\n\nFocus specifically on the segment from {timestamp_start} to {timestamp_end}. "
            "Ignore content outside this range."
        )
        full_prompt = prompt + segment_instruction

        return await self._call_gemini(
            youtube_url=youtube_url,
            prompt=full_prompt,
            response_schema=None,
            video_duration_seconds=0,
        )

    async def generate_text(self, prompt: str) -> str:
        """Send a text-only prompt to Gemini and return the raw text response.

        Unlike :meth:`analyze_video`, this does **not** attach a video and
        does **not** force JSON output.  It is intended for code-generation
        and other free-form text tasks.

        Parameters
        ----------
        prompt:
            The text prompt to send.

        Returns
        -------
        str
            Raw text response from Gemini.
        """
        await self._rate_limiter.wait_if_needed()

        logger.info("Sending text-generation request to Gemini (%s)", self._model)

        response = await asyncio.to_thread(
            self._client.models.generate_content,
            model=self._model,
            contents=[prompt],
        )

        self._rate_limiter.record_request()

        text = response.text or ""
        logger.debug("Gemini text response length: %d", len(text))
        return text

    # -- Internal helpers -----------------------------------------------------

    @retry(
        retry=retry_if_exception(_is_retriable),
        wait=wait_exponential(multiplier=2, min=10, max=120),
        stop=stop_after_attempt(6),
        reraise=True,
        before_sleep=lambda retry_state: logger.warning(
            "Gemini request failed (%s), retrying in %.1fs (attempt %d/6)...",
            retry_state.outcome.exception().__class__.__name__,  # type: ignore[union-attr]
            retry_state.next_action.sleep,  # type: ignore[union-attr]
            retry_state.attempt_number,
        ),
    )
    async def _call_gemini(
        self,
        youtube_url: str,
        prompt: str,
        response_schema: dict | None,
        video_duration_seconds: int,
    ) -> dict:
        """Make the actual Gemini API call with rate-limiting and retries."""
        import json

        await self._rate_limiter.wait_if_needed()

        logger.info("Sending request to Gemini (%s): %s", self._model, youtube_url)

        config_kwargs: dict = {"response_mime_type": "application/json"}
        if response_schema is not None:
            config_kwargs["response_schema"] = response_schema

        # The google-genai SDK's generate_content is synchronous; run in a
        # thread so we don't block the event loop.
        response = await asyncio.to_thread(
            self._client.models.generate_content,
            model=self._model,
            contents=[
                types.Part.from_uri(file_uri=youtube_url, mime_type="video/*"),
                prompt,
            ],
            config=types.GenerateContentConfig(**config_kwargs),
        )

        # Record the request *after* success so failed attempts don't consume
        # quota from the limiter's perspective.
        self._rate_limiter.record_request(video_duration_seconds=video_duration_seconds)

        # Parse the response text as JSON
        raw_text = response.text
        if not raw_text:
            logger.error("Empty response from Gemini for %s", youtube_url)
            return {}

        try:
            result: dict = json.loads(raw_text)
        except json.JSONDecodeError:
            # Gemini sometimes wraps JSON in markdown code fences
            cleaned = raw_text.strip()
            if cleaned.startswith("```"):
                # Strip ```json ... ``` wrapper
                lines = cleaned.split("\n")
                cleaned = "\n".join(lines[1:-1]) if len(lines) > 2 else cleaned
            result = json.loads(cleaned)

        logger.info("Successfully received response for %s", youtube_url)
        return result
