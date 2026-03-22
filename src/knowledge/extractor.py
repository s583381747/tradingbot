"""Extract structured trading knowledge from video analysis results.

Parses the raw analysis JSON (first_pass + second_pass) produced by the
analysis pipeline and maps it into :class:`TradingStrategy` objects that
can be stored in the knowledge base.
"""

from __future__ import annotations

import logging
from typing import Any

from src.knowledge.schema import (
    EntryCondition,
    ExitCondition,
    RiskManagement,
    TradingIndicator,
    TradingStrategy,
)

logger = logging.getLogger(__name__)

# Maps common natural-language operator phrases to canonical operator strings.
_OPERATOR_ALIASES: dict[str, str] = {
    "crosses above": "crosses_above",
    "crosses below": "crosses_below",
    "cross above": "crosses_above",
    "cross below": "crosses_below",
    "above": "greater_than",
    "below": "less_than",
    "greater than": "greater_than",
    "less than": "less_than",
    "equals": "equals",
    "equal to": "equals",
    "=": "equals",
    ">": "greater_than",
    "<": "less_than",
}

_VALID_OPERATORS = {
    "crosses_above",
    "crosses_below",
    "greater_than",
    "less_than",
    "equals",
}

_INDICATOR_TYPE_KEYWORDS: dict[str, list[str]] = {
    "trend": ["ma", "ema", "sma", "moving average", "trend", "ichimoku", "macd", "adx"],
    "momentum": ["rsi", "stochastic", "momentum", "cci", "williams", "roc"],
    "volume": ["volume", "obv", "vwap", "mfi", "accumulation", "distribution"],
    "volatility": ["bollinger", "atr", "volatility", "keltner", "std", "deviation"],
}


def _normalise_operator(raw: str) -> str:
    """Return a canonical operator string, falling back to *greater_than*."""
    cleaned = raw.strip().lower()
    if cleaned in _VALID_OPERATORS:
        return cleaned
    return _OPERATOR_ALIASES.get(cleaned, "greater_than")


def _guess_indicator_type(name: str) -> str:
    """Infer an indicator type from its name using keyword matching."""
    lower = name.lower()
    for itype, keywords in _INDICATOR_TYPE_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return itype
    return "trend"


class KnowledgeExtractor:
    """Converts raw analysis dicts into structured :class:`TradingStrategy` lists."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_from_analysis(
        self,
        analysis: dict[str, Any],
        video_id: str,
    ) -> list[TradingStrategy]:
        """Parse a ``VideoAnalysisResult`` dict and return strategies.

        The *analysis* dict is expected to have ``first_pass`` and/or
        ``second_pass`` keys, each containing strategy-relevant fields
        (indicators, entry/exit rules, risk management, etc.).

        Parameters
        ----------
        analysis:
            Raw analysis JSON as a Python dict.
        video_id:
            YouTube video ID — attached to every extracted object for
            provenance tracking.

        Returns
        -------
        list[TradingStrategy]
            Zero or more strategies found in the analysis.
        """
        strategies: list[TradingStrategy] = []

        try:
            first_pass = analysis.get("first_pass") or {}
            second_pass = analysis.get("second_pass") or {}

            # second_pass can be a list of strategy dicts (from VideoAnalysisResult)
            # or a dict with a "strategies" key.
            if isinstance(second_pass, list):
                # Each item is a strategy dict from the two-pass analysis
                for item in second_pass:
                    if isinstance(item, dict):
                        strategy = self._build_strategy(item, video_id)
                        if strategy is not None:
                            strategies.append(strategy)
            else:
                # The analysis may contain a top-level "strategies" list, or the
                # strategy information may be spread across the two passes.
                raw_strategies = (
                    second_pass.get("strategies")
                    or first_pass.get("strategies")
                    or analysis.get("strategies")
                    or []
                )

                if isinstance(raw_strategies, list) and raw_strategies:
                    for raw in raw_strategies:
                        if not isinstance(raw, dict):
                            continue
                        strategy = self._build_strategy(raw, video_id)
                        if strategy is not None:
                            strategies.append(strategy)

            # Also extract from first_pass strategies_overview if available
            if not strategies and isinstance(first_pass, dict):
                overviews = first_pass.get("strategies_overview") or []
                for ov in overviews:
                    if isinstance(ov, dict):
                        strategy = self._build_strategy(ov, video_id)
                        if strategy is not None:
                            strategies.append(strategy)

            # If still no strategies, try to build one from the
            # combined first+second pass data.
            if not strategies:
                merged_data = first_pass if isinstance(first_pass, dict) else {}
                if isinstance(second_pass, dict):
                    merged_data = {**merged_data, **second_pass}
                if merged_data:
                    strategy = self._build_strategy(merged_data, video_id)
                    if strategy is not None:
                        strategies.append(strategy)

        except Exception:
            logger.exception("Failed to extract strategies from analysis for video %s", video_id)

        return strategies

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_strategy(
        self,
        data: dict[str, Any],
        video_id: str,
    ) -> TradingStrategy | None:
        """Attempt to construct a single :class:`TradingStrategy` from *data*."""
        name = (
            data.get("strategy_name")
            or data.get("name")
            or data.get("title")
            or ""
        )
        if not name:
            # Derive a name from the video ID so we never store an unnamed
            # strategy.
            name = f"Strategy from {video_id}"

        description = (
            data.get("description")
            or data.get("summary")
            or data.get("strategy_description")
            or ""
        )

        market_type = (
            data.get("market_type")
            or data.get("market")
            or data.get("asset_class")
            or "stock"
        )
        # Normalise to one of the expected enum values.
        market_type = market_type.lower().strip()
        if market_type not in {"stock", "forex", "crypto", "futures"}:
            market_type = "stock"

        timeframes = self._as_str_list(
            data.get("timeframes") or data.get("timeframe") or []
        )

        indicators = self._extract_indicators(data)
        entry_conditions = self._extract_entry_conditions(data)
        exit_conditions = self._extract_exit_conditions(data)
        risk_management = self._extract_risk_management(data)

        confidence = self._clamp01(
            data.get("confidence_score") or data.get("confidence") or 0.5
        )
        quantifiability = self._clamp01(
            data.get("quantifiability_score") or data.get("quantifiability") or 0.5
        )

        notes = self._as_str_list(data.get("notes") or [])

        # Attach the video ID to every indicator as well.
        for ind in indicators:
            if video_id not in ind.source_videos:
                ind.source_videos.append(video_id)

        return TradingStrategy(
            name=name,
            description=description,
            market_type=market_type,
            timeframes=timeframes,
            indicators=indicators,
            entry_conditions=entry_conditions,
            exit_conditions=exit_conditions,
            risk_management=risk_management,
            source_videos=[video_id],
            confidence_score=confidence,
            quantifiability_score=quantifiability,
            notes=notes,
        )

    # ------------------------------------------------------------------
    # Extraction sub-methods
    # ------------------------------------------------------------------

    def _extract_indicators(self, data: dict[str, Any]) -> list[TradingIndicator]:
        """Extract indicator definitions from *data*."""
        raw_list = data.get("indicators") or data.get("technical_indicators") or []
        if isinstance(raw_list, str):
            # Sometimes the model returns a comma-separated string.
            raw_list = [s.strip() for s in raw_list.split(",") if s.strip()]

        indicators: list[TradingIndicator] = []
        for item in raw_list:
            try:
                if isinstance(item, str):
                    indicators.append(
                        TradingIndicator(
                            name=item,
                            type=_guess_indicator_type(item),
                        )
                    )
                elif isinstance(item, dict):
                    name = item.get("name") or item.get("indicator") or "Unknown"
                    itype = item.get("type") or _guess_indicator_type(name)
                    indicators.append(
                        TradingIndicator(
                            name=name,
                            type=itype,
                            parameters=item.get("parameters") or item.get("params") or {},
                            description=item.get("description") or "",
                        )
                    )
            except Exception:
                logger.debug("Skipping malformed indicator entry: %s", item)

        return indicators

    def _extract_entry_conditions(self, data: dict[str, Any]) -> list[EntryCondition]:
        """Extract entry conditions from *data*."""
        raw = (
            data.get("entry_conditions")
            or data.get("entry_rules")
            or data.get("entries")
            or []
        )
        if isinstance(raw, str):
            raw = [raw]

        conditions: list[EntryCondition] = []
        for item in raw:
            try:
                if isinstance(item, str):
                    conditions.append(EntryCondition(description=item))
                elif isinstance(item, dict):
                    conditions.append(
                        EntryCondition(
                            description=item.get("description") or item.get("rule") or str(item),
                            indicator=item.get("indicator") or "",
                            operator=_normalise_operator(item.get("operator") or "greater_than"),
                            value=item.get("value") if item.get("value") is not None else "",
                            timeframe=item.get("timeframe"),
                            confidence=self._clamp01(item.get("confidence") or 0.5),
                        )
                    )
            except Exception:
                logger.debug("Skipping malformed entry condition: %s", item)

        return conditions

    def _extract_exit_conditions(self, data: dict[str, Any]) -> list[ExitCondition]:
        """Extract exit conditions from *data*."""
        raw = (
            data.get("exit_conditions")
            or data.get("exit_rules")
            or data.get("exits")
            or []
        )
        if isinstance(raw, str):
            raw = [raw]

        conditions: list[ExitCondition] = []
        for item in raw:
            try:
                if isinstance(item, str):
                    # Try to infer exit type from the text.
                    exit_type = self._infer_exit_type(item)
                    conditions.append(
                        ExitCondition(description=item, exit_type=exit_type)
                    )
                elif isinstance(item, dict):
                    exit_type = item.get("exit_type") or item.get("type") or "signal"
                    if exit_type not in {
                        "stop_loss",
                        "take_profit",
                        "trailing_stop",
                        "signal",
                    }:
                        exit_type = self._infer_exit_type(
                            item.get("description") or str(item)
                        )
                    conditions.append(
                        ExitCondition(
                            description=item.get("description") or item.get("rule") or str(item),
                            indicator=item.get("indicator") or "",
                            operator=_normalise_operator(item.get("operator") or "greater_than"),
                            value=item.get("value") if item.get("value") is not None else "",
                            timeframe=item.get("timeframe"),
                            confidence=self._clamp01(item.get("confidence") or 0.5),
                            exit_type=exit_type,
                        )
                    )
            except Exception:
                logger.debug("Skipping malformed exit condition: %s", item)

        return conditions

    def _extract_risk_management(self, data: dict[str, Any]) -> RiskManagement | None:
        """Extract risk management rules from *data*.

        Returns ``None`` if there is no risk-management information at all.
        """
        raw = data.get("risk_management") or data.get("risk") or {}
        if isinstance(raw, str):
            return RiskManagement(rules=[raw])
        if isinstance(raw, list):
            return RiskManagement(rules=[str(r) for r in raw])
        if not isinstance(raw, dict):
            return None

        # Nothing useful in an empty dict.
        if not raw:
            return None

        return RiskManagement(
            stop_loss_pct=self._as_float(raw.get("stop_loss_pct") or raw.get("stop_loss")),
            take_profit_pct=self._as_float(raw.get("take_profit_pct") or raw.get("take_profit")),
            risk_reward_ratio=self._as_float(
                raw.get("risk_reward_ratio") or raw.get("risk_reward")
            ),
            max_position_pct=self._as_float(
                raw.get("max_position_pct") or raw.get("max_position") or raw.get("position_size")
            ),
            rules=self._as_str_list(raw.get("rules") or []),
        )

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_exit_type(text: str) -> str:
        lower = text.lower()
        if "stop loss" in lower or "stop-loss" in lower or "stoploss" in lower:
            return "stop_loss"
        if "take profit" in lower or "take-profit" in lower or "target" in lower:
            return "take_profit"
        if "trailing" in lower:
            return "trailing_stop"
        return "signal"

    @staticmethod
    def _clamp01(value: Any) -> float:
        try:
            v = float(value)
        except (TypeError, ValueError):
            return 0.5
        return max(0.0, min(1.0, v))

    @staticmethod
    def _as_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _as_str_list(value: Any) -> list[str]:
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            return [str(v) for v in value if v]
        return []
