"""Core two-pass video analysis using Gemini."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

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
    """A notable moment in the video."""

    time: str = Field(description="Timestamp in MM:SS or HH:MM:SS format")
    description: str = Field(description="What happens at this moment")


class IndicatorInfo(BaseModel):
    """A technical indicator with its parameters."""

    name: str
    params: dict = Field(default_factory=dict)


class StrategyOverview(BaseModel):
    """Brief overview of a strategy found during first pass."""

    name: str
    description: str = ""
    timeframes: list[str] = Field(default_factory=list)
    markets: list[str] = Field(default_factory=list)


class FirstPassResult(BaseModel):
    """Result of the first (discovery) pass over a video."""

    topic: str = ""
    content_type: str = Field(
        default="other",
        description=(
            "One of: strategy_tutorial, market_analysis, indicator_explanation, "
            "backtesting, trade_review, educational, other"
        ),
    )
    strategies_overview: list[StrategyOverview] = Field(default_factory=list)
    indicators: list[IndicatorInfo] = Field(default_factory=list)
    chart_annotations: str = ""
    key_timestamps: list[KeyTimestamp] = Field(default_factory=list)


class EntryExitCondition(BaseModel):
    """A single entry or exit condition for a strategy."""

    description: str
    indicator: str = ""
    condition: str = ""
    required: bool = True


class StopTakeProfitRule(BaseModel):
    """Stop-loss or take-profit rule."""

    type: str = Field(
        default="none_mentioned",
        description=(
            "One of: fixed_pips, atr_multiple, swing_high_low, percentage, "
            "risk_reward_ratio, indicator_based, none_mentioned"
        ),
    )
    value: float | None = None
    description: str = ""


class SecondPassResult(BaseModel):
    """Result of the second (deep-dive) pass for a single strategy."""

    strategy_name: str
    entry_conditions: list[EntryExitCondition] = Field(default_factory=list)
    exit_conditions: list[EntryExitCondition] = Field(default_factory=list)
    indicator_params: dict = Field(
        default_factory=dict,
        description="Mapping of indicator name -> parameter dict",
    )
    stop_loss_rules: StopTakeProfitRule = Field(default_factory=StopTakeProfitRule)
    take_profit_rules: StopTakeProfitRule = Field(default_factory=StopTakeProfitRule)
    position_sizing: str = "not_mentioned"
    quantifiability_score: int = Field(
        default=1,
        ge=1,
        le=10,
        description="1-10 score of how easily this strategy can be coded",
    )


class VideoAnalysisResult(BaseModel):
    """Combined result of both analysis passes for a video."""

    video_id: str
    analysis_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    first_pass: FirstPassResult
    second_pass: list[SecondPassResult] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# First-pass response schema (for Gemini structured output)
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
# Analyzer
# ---------------------------------------------------------------------------

class VideoAnalyzer:
    """Two-pass video analyzer: discovery pass then per-strategy deep dive."""

    def __init__(self, gemini_client: GeminiClient, skip_second_pass: bool = False) -> None:
        self._gemini = gemini_client
        self._skip_second_pass = skip_second_pass

    async def analyze_video(self, video: VideoInfo) -> VideoAnalysisResult:
        """Run the full two-pass analysis pipeline for a single video.

        1. First pass  -- identify strategies, indicators, key moments.
        2. Second pass -- for each strategy found, extract precise trading rules.
        """
        import json as _json

        youtube_url = video.get_url()

        # -- First pass -------------------------------------------------------
        logger.info("First pass analysis for video %s: %s", video.id, video.title)
        first_pass_prompt = self._build_first_pass_prompt(video)

        raw_first = await self._gemini.analyze_video(
            youtube_url=youtube_url,
            prompt=first_pass_prompt,
            response_schema=_FIRST_PASS_SCHEMA,
            video_duration_seconds=video.duration,
        )

        first_pass = FirstPassResult.model_validate(raw_first)
        logger.info(
            "First pass complete: %d strategies found, content_type=%s",
            len(first_pass.strategies_overview),
            first_pass.content_type,
        )

        # Serialise first-pass result for second-pass context
        first_pass_summary = _json.dumps(raw_first, ensure_ascii=False, indent=2)

        # -- Second pass (per strategy) ---------------------------------------
        second_pass_results: list[SecondPassResult] = []

        if self._skip_second_pass:
            logger.info("Skipping second pass (skip_second_pass=True)")
            return VideoAnalysisResult(
                video_id=video.id,
                first_pass=first_pass,
                second_pass=[],
            )

        for strategy in first_pass.strategies_overview:
            logger.info(
                "Second pass analysis for strategy '%s' in video %s",
                strategy.name,
                video.id,
            )
            second_pass_prompt = self._build_second_pass_prompt(
                video, strategy, first_pass_summary
            )

            raw_second = await self._gemini.analyze_video(
                youtube_url=youtube_url,
                prompt=second_pass_prompt,
                response_schema=_SECOND_PASS_SCHEMA,
                video_duration_seconds=video.duration,
            )

            result = SecondPassResult.model_validate(raw_second)
            second_pass_results.append(result)
            logger.info(
                "Second pass complete for '%s': quantifiability=%d/10",
                result.strategy_name,
                result.quantifiability_score,
            )

        return VideoAnalysisResult(
            video_id=video.id,
            first_pass=first_pass,
            second_pass=second_pass_results,
        )

    # -- Prompt builders -------------------------------------------------------

    @staticmethod
    def _build_first_pass_prompt(video: VideoInfo) -> str:
        """Construct the first-pass prompt with video metadata context."""
        context = (
            f"Video title: {video.title}\n"
            f"Duration: {video.human_duration()}\n"
            f"Channel: {video.channel}\n"
        )
        if video.description:
            # Include first 500 chars of description for additional context
            desc_preview = video.description[:500]
            context += f"Description preview: {desc_preview}\n"

        return f"{context}\n{FIRST_PASS_PROMPT}"

    @staticmethod
    def _build_second_pass_prompt(
        video: VideoInfo,
        strategy_info: StrategyOverview,
        first_pass_summary: str,
    ) -> str:
        """Construct the second-pass prompt targeting a specific strategy.

        Uses :func:`format_second_pass_prompt` from the prompts module,
        injecting the first-pass summary, strategy name, and any known
        timestamp range for the strategy segment.
        """
        # Build a strategy description string with available metadata
        strategy_desc = strategy_info.name
        if strategy_info.description:
            strategy_desc += f" - {strategy_info.description}"
        if strategy_info.timeframes:
            strategy_desc += f" (timeframes: {', '.join(strategy_info.timeframes)})"
        if strategy_info.markets:
            strategy_desc += f" (markets: {', '.join(strategy_info.markets)})"

        # Use the template formatter from the prompts module
        prompt_text = format_second_pass_prompt(
            first_pass_summary=first_pass_summary,
            target_strategy=strategy_desc,
        )

        # Prepend video context
        context = (
            f"Video title: {video.title}\n"
            f"Duration: {video.human_duration()}\n"
            f"Channel: {video.channel}\n"
        )
        return f"{context}\n{prompt_text}"
