"""Trading knowledge data models.

Pydantic models representing structured trading knowledge extracted from
video analysis — indicators, conditions, strategies, and the overall
knowledge base.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional, Union

from pydantic import BaseModel, Field


class TradingIndicator(BaseModel):
    """A technical indicator referenced in a trading strategy."""

    name: str
    type: str = Field(
        description="Category of the indicator: trend, momentum, volume, or volatility",
    )
    parameters: dict = Field(default_factory=dict)
    description: str = ""
    source_videos: list[str] = Field(
        default_factory=list,
        description="Video IDs where this indicator was discussed",
    )


class EntryCondition(BaseModel):
    """A single condition that must be met to enter a trade."""

    description: str
    indicator: str = ""
    operator: str = Field(
        default="greater_than",
        description="Comparison operator: crosses_above, crosses_below, greater_than, less_than, equals",
    )
    value: Union[str, float] = ""
    timeframe: Optional[str] = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class ExitCondition(BaseModel):
    """A single condition that triggers exiting a trade."""

    description: str
    indicator: str = ""
    operator: str = Field(
        default="greater_than",
        description="Comparison operator: crosses_above, crosses_below, greater_than, less_than, equals",
    )
    value: Union[str, float] = ""
    timeframe: Optional[str] = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    exit_type: str = Field(
        default="signal",
        description="Type of exit: stop_loss, take_profit, trailing_stop, signal",
    )


class RiskManagement(BaseModel):
    """Risk management rules for a strategy."""

    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None
    risk_reward_ratio: Optional[float] = None
    max_position_pct: Optional[float] = None
    rules: list[str] = Field(default_factory=list)


class TradingStrategy(BaseModel):
    """A complete trading strategy extracted from video analysis."""

    name: str
    description: str = ""
    market_type: str = Field(
        default="stock",
        description="Target market: stock, forex, crypto, futures",
    )
    timeframes: list[str] = Field(default_factory=list)
    indicators: list[TradingIndicator] = Field(default_factory=list)
    entry_conditions: list[EntryCondition] = Field(default_factory=list)
    exit_conditions: list[ExitCondition] = Field(default_factory=list)
    risk_management: Optional[RiskManagement] = None
    source_videos: list[str] = Field(
        default_factory=list,
        description="Video IDs that contributed to this strategy",
    )
    confidence_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Overall confidence in the strategy's completeness",
    )
    quantifiability_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="How easily the strategy can be coded into rules",
    )
    notes: list[str] = Field(default_factory=list)

    @property
    def name_snake(self) -> str:
        """Return the strategy name in snake_case for file naming."""
        import re

        s = re.sub(r"[^\w\s]", "", self.name)
        s = re.sub(r"\s+", "_", s.strip())
        return s.lower()


class TradingKnowledgeBase(BaseModel):
    """Top-level container for all extracted trading knowledge."""

    strategies: list[TradingStrategy] = Field(default_factory=list)
    indicators: list[TradingIndicator] = Field(default_factory=list)
    general_rules: list[str] = Field(default_factory=list)
    channel_info: dict = Field(default_factory=dict)
    last_updated: datetime = Field(default_factory=datetime.now)
