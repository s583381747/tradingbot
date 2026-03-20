"""Synthesise all extracted knowledge into a coherent trading system document.

The :class:`TradingSynthesizer` can operate in two modes:

1. **AI-assisted** — uses a Gemini (or compatible) client to produce a
   polished, conflict-resolved trading system document.
2. **Rule-based** — generates the document purely from the structured data
   when no AI client is available.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from src.knowledge.schema import (
    TradingKnowledgeBase,
    TradingStrategy,
)

logger = logging.getLogger(__name__)

_OUTPUT_DIR = Path("data/knowledge")

SYNTHESIS_PROMPT = """\
You are an expert quantitative trading system designer.  Below is a
collection of trading strategies, indicators, and rules extracted from
educational trading videos.  Your task is to synthesise them into a single,
coherent, and actionable trading system document.

Requirements:
1. Identify the strongest strategies (highest confidence and most source
   evidence).
2. Resolve any conflicting rules — explain which to follow and why.
3. Organise the output into clear sections:
   - System Overview
   - Core Strategies (ranked)
   - Technical Indicators Used
   - Entry Rules
   - Exit Rules
   - Risk Management
   - Known Conflicts & Resolutions
   - Implementation Notes
4. Use precise, quantifiable language wherever possible.
5. Format the output as Markdown.

---

### Extracted Knowledge

{knowledge_context}

---

Generate the complete trading system document now.
"""


class TradingSynthesizer:
    """Build a unified trading-system document from a knowledge base.

    Parameters
    ----------
    gemini_client:
        An optional AI client instance (e.g. ``GeminiClient``) that
        exposes an async ``generate`` or ``generate_content`` method.
        If *None*, the synthesiser falls back to a deterministic,
        template-based document.
    """

    def __init__(self, gemini_client: Any = None) -> None:
        self.gemini_client = gemini_client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def synthesize(self, kb: TradingKnowledgeBase) -> str:
        """Produce a Markdown trading-system document and save it to disk.

        Returns the generated Markdown string.
        """
        if not kb.strategies:
            logger.warning("No strategies in knowledge base — generating minimal document")

        ranked = self._rank_strategies(kb.strategies)
        conflicts = self._identify_conflicts(kb.strategies)

        if self.gemini_client is not None:
            doc = await self._generate_with_ai(kb, ranked, conflicts)
        else:
            doc = self._generate_system_doc(ranked, conflicts, kb.general_rules)

        # Persist to disk.
        output_path = _OUTPUT_DIR / "trading_system.md"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(doc, encoding="utf-8")
        logger.info("Trading system document saved to %s", output_path)

        return doc

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------

    def _rank_strategies(
        self,
        strategies: list[TradingStrategy],
    ) -> list[TradingStrategy]:
        """Sort strategies by a composite score.

        Score = len(source_videos) * confidence_score * quantifiability_score
        """

        def _score(s: TradingStrategy) -> float:
            return len(s.source_videos) * s.confidence_score * s.quantifiability_score

        return sorted(strategies, key=_score, reverse=True)

    # ------------------------------------------------------------------
    # Conflict detection
    # ------------------------------------------------------------------

    def _identify_conflicts(
        self,
        strategies: list[TradingStrategy],
    ) -> list[dict[str, Any]]:
        """Find potentially contradicting rules across strategies.

        A conflict is flagged when:
        - Two strategies use the same indicator with opposing operators for
          entry (e.g. one says RSI > 70 → buy, another says RSI < 30 → buy).
        - Risk management rules disagree (e.g. different stop-loss
          percentages with high confidence on both sides).
        """
        conflicts: list[dict[str, Any]] = []

        for i, s1 in enumerate(strategies):
            for s2 in strategies[i + 1 :]:
                # --- Entry condition conflicts ---
                for ec1 in s1.entry_conditions:
                    for ec2 in s2.entry_conditions:
                        if not ec1.indicator or not ec2.indicator:
                            continue
                        if ec1.indicator.lower() != ec2.indicator.lower():
                            continue
                        if self._operators_conflict(ec1.operator, ec2.operator):
                            conflicts.append(
                                {
                                    "type": "entry_condition",
                                    "strategy_a": s1.name,
                                    "strategy_b": s2.name,
                                    "indicator": ec1.indicator,
                                    "detail": (
                                        f"{s1.name} uses {ec1.indicator} "
                                        f"{ec1.operator} {ec1.value}, while "
                                        f"{s2.name} uses {ec2.indicator} "
                                        f"{ec2.operator} {ec2.value}"
                                    ),
                                }
                            )

                # --- Risk management conflicts ---
                rm1 = s1.risk_management
                rm2 = s2.risk_management
                if rm1 and rm2:
                    if (
                        rm1.stop_loss_pct is not None
                        and rm2.stop_loss_pct is not None
                        and rm1.stop_loss_pct != rm2.stop_loss_pct
                    ):
                        conflicts.append(
                            {
                                "type": "risk_management",
                                "strategy_a": s1.name,
                                "strategy_b": s2.name,
                                "indicator": "stop_loss_pct",
                                "detail": (
                                    f"{s1.name} uses stop-loss {rm1.stop_loss_pct}%, "
                                    f"while {s2.name} uses {rm2.stop_loss_pct}%"
                                ),
                            }
                        )
                    if (
                        rm1.risk_reward_ratio is not None
                        and rm2.risk_reward_ratio is not None
                        and rm1.risk_reward_ratio != rm2.risk_reward_ratio
                    ):
                        conflicts.append(
                            {
                                "type": "risk_management",
                                "strategy_a": s1.name,
                                "strategy_b": s2.name,
                                "indicator": "risk_reward_ratio",
                                "detail": (
                                    f"{s1.name} uses R:R {rm1.risk_reward_ratio}, "
                                    f"while {s2.name} uses {rm2.risk_reward_ratio}"
                                ),
                            }
                        )

        return conflicts

    # ------------------------------------------------------------------
    # AI-assisted generation
    # ------------------------------------------------------------------

    async def _generate_with_ai(
        self,
        kb: TradingKnowledgeBase,
        ranked: list[TradingStrategy],
        conflicts: list[dict[str, Any]],
    ) -> str:
        """Use the Gemini (or compatible) client to produce the document."""
        context = self._build_knowledge_context(kb, ranked, conflicts)
        prompt = SYNTHESIS_PROMPT.format(knowledge_context=context)

        try:
            # Try the two most common async API shapes.
            if hasattr(self.gemini_client, "generate"):
                response = await self.gemini_client.generate(prompt)
            elif hasattr(self.gemini_client, "generate_content"):
                response = await self.gemini_client.generate_content(prompt)
            else:
                logger.warning(
                    "Gemini client has no recognised generation method — "
                    "falling back to rule-based document"
                )
                return self._generate_system_doc(ranked, conflicts, kb.general_rules)

            # The response may be a string or an object with a `text` attribute.
            if isinstance(response, str):
                return response
            if hasattr(response, "text"):
                return response.text  # type: ignore[union-attr]
            return str(response)

        except Exception:
            logger.exception("AI synthesis failed — falling back to rule-based document")
            return self._generate_system_doc(ranked, conflicts, kb.general_rules)

    # ------------------------------------------------------------------
    # Rule-based generation (no AI)
    # ------------------------------------------------------------------

    def _generate_system_doc(
        self,
        ranked_strategies: list[TradingStrategy],
        conflicts: list[dict[str, Any]],
        general_rules: list[str],
    ) -> str:
        """Produce a Markdown document without an AI model."""
        lines: list[str] = []

        # --- Header ---
        lines.append("# Trading System Document")
        lines.append("")
        lines.append(f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}*")
        lines.append("")

        # --- Overview ---
        lines.append("## System Overview")
        lines.append("")
        lines.append(
            f"This document synthesises **{len(ranked_strategies)}** trading "
            f"strategies extracted from educational video analysis."
        )
        if conflicts:
            lines.append(
                f"  **{len(conflicts)}** potential conflict(s) were detected "
                f"and are documented below."
            )
        lines.append("")

        # --- General rules ---
        if general_rules:
            lines.append("## General Rules")
            lines.append("")
            for rule in general_rules:
                lines.append(f"- {rule}")
            lines.append("")

        # --- Core strategies ---
        lines.append("## Core Strategies (Ranked)")
        lines.append("")
        if not ranked_strategies:
            lines.append("*No strategies extracted yet.*")
            lines.append("")
        for idx, s in enumerate(ranked_strategies, 1):
            score = len(s.source_videos) * s.confidence_score * s.quantifiability_score
            lines.append(f"### {idx}. {s.name}")
            lines.append("")
            if s.description:
                lines.append(f"{s.description}")
                lines.append("")
            lines.append(f"- **Market:** {s.market_type}")
            if s.timeframes:
                lines.append(f"- **Timeframes:** {', '.join(s.timeframes)}")
            lines.append(f"- **Confidence:** {s.confidence_score:.0%}")
            lines.append(f"- **Quantifiability:** {s.quantifiability_score:.0%}")
            lines.append(f"- **Composite score:** {score:.2f}")
            lines.append(f"- **Sources:** {len(s.source_videos)} video(s)")
            lines.append("")

        # --- Indicators ---
        lines.append("## Technical Indicators")
        lines.append("")
        seen_indicators: set[str] = set()
        for s in ranked_strategies:
            for ind in s.indicators:
                key = ind.name.lower()
                if key in seen_indicators:
                    continue
                seen_indicators.add(key)
                params = (
                    ", ".join(f"{k}={v}" for k, v in ind.parameters.items())
                    if ind.parameters
                    else "default"
                )
                lines.append(f"- **{ind.name}** ({ind.type}) — params: {params}")
                if ind.description:
                    lines.append(f"  {ind.description}")
        if not seen_indicators:
            lines.append("*No indicators extracted yet.*")
        lines.append("")

        # --- Entry rules ---
        lines.append("## Entry Rules")
        lines.append("")
        entry_count = 0
        for s in ranked_strategies:
            if s.entry_conditions:
                lines.append(f"**{s.name}:**")
                for ec in s.entry_conditions:
                    indicator_part = f" [{ec.indicator} {ec.operator} {ec.value}]" if ec.indicator else ""
                    lines.append(f"- {ec.description}{indicator_part}")
                lines.append("")
                entry_count += len(s.entry_conditions)
        if entry_count == 0:
            lines.append("*No explicit entry rules extracted yet.*")
            lines.append("")

        # --- Exit rules ---
        lines.append("## Exit Rules")
        lines.append("")
        exit_count = 0
        for s in ranked_strategies:
            if s.exit_conditions:
                lines.append(f"**{s.name}:**")
                for ec in s.exit_conditions:
                    indicator_part = f" [{ec.indicator} {ec.operator} {ec.value}]" if ec.indicator else ""
                    lines.append(f"- ({ec.exit_type}) {ec.description}{indicator_part}")
                lines.append("")
                exit_count += len(s.exit_conditions)
        if exit_count == 0:
            lines.append("*No explicit exit rules extracted yet.*")
            lines.append("")

        # --- Risk management ---
        lines.append("## Risk Management")
        lines.append("")
        rm_found = False
        for s in ranked_strategies:
            rm = s.risk_management
            if rm is None:
                continue
            rm_found = True
            lines.append(f"**{s.name}:**")
            if rm.stop_loss_pct is not None:
                lines.append(f"- Stop-loss: {rm.stop_loss_pct}%")
            if rm.take_profit_pct is not None:
                lines.append(f"- Take-profit: {rm.take_profit_pct}%")
            if rm.risk_reward_ratio is not None:
                lines.append(f"- Risk/Reward ratio: {rm.risk_reward_ratio}")
            if rm.max_position_pct is not None:
                lines.append(f"- Max position size: {rm.max_position_pct}%")
            for rule in rm.rules:
                lines.append(f"- {rule}")
            lines.append("")
        if not rm_found:
            lines.append("*No risk management rules extracted yet.*")
            lines.append("")

        # --- Conflicts ---
        if conflicts:
            lines.append("## Known Conflicts")
            lines.append("")
            for c in conflicts:
                lines.append(
                    f"- **{c['type']}** between *{c['strategy_a']}* and "
                    f"*{c['strategy_b']}*: {c['detail']}"
                )
            lines.append("")

        # --- Footer ---
        lines.append("---")
        lines.append("")
        lines.append(
            "*This document was generated automatically from extracted "
            "video knowledge. Review and validate before live trading.*"
        )
        lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _operators_conflict(op1: str, op2: str) -> bool:
        """Return *True* if two operators point in opposing directions."""
        opposing_pairs = {
            frozenset({"crosses_above", "crosses_below"}),
            frozenset({"greater_than", "less_than"}),
        }
        return frozenset({op1, op2}) in opposing_pairs

    @staticmethod
    def _build_knowledge_context(
        kb: TradingKnowledgeBase,
        ranked: list[TradingStrategy],
        conflicts: list[dict[str, Any]],
    ) -> str:
        """Serialise the knowledge base into a text block for the AI prompt."""
        parts: list[str] = []

        parts.append(f"Total strategies: {len(ranked)}")
        parts.append(f"Total conflicts: {len(conflicts)}")
        parts.append("")

        for idx, s in enumerate(ranked, 1):
            parts.append(f"### Strategy {idx}: {s.name}")
            parts.append(f"Description: {s.description}")
            parts.append(f"Market: {s.market_type}")
            parts.append(f"Timeframes: {', '.join(s.timeframes) if s.timeframes else 'N/A'}")
            parts.append(f"Confidence: {s.confidence_score}")
            parts.append(f"Quantifiability: {s.quantifiability_score}")
            parts.append(f"Sources: {len(s.source_videos)} video(s)")

            if s.indicators:
                parts.append("Indicators:")
                for ind in s.indicators:
                    parts.append(f"  - {ind.name} ({ind.type}): {ind.description}")

            if s.entry_conditions:
                parts.append("Entry conditions:")
                for ec in s.entry_conditions:
                    parts.append(
                        f"  - {ec.description} "
                        f"[{ec.indicator} {ec.operator} {ec.value}] "
                        f"(confidence: {ec.confidence})"
                    )

            if s.exit_conditions:
                parts.append("Exit conditions:")
                for ec in s.exit_conditions:
                    parts.append(
                        f"  - ({ec.exit_type}) {ec.description} "
                        f"[{ec.indicator} {ec.operator} {ec.value}]"
                    )

            if s.risk_management:
                rm = s.risk_management
                parts.append("Risk management:")
                if rm.stop_loss_pct is not None:
                    parts.append(f"  - Stop-loss: {rm.stop_loss_pct}%")
                if rm.take_profit_pct is not None:
                    parts.append(f"  - Take-profit: {rm.take_profit_pct}%")
                if rm.risk_reward_ratio is not None:
                    parts.append(f"  - R:R ratio: {rm.risk_reward_ratio}")
                for rule in rm.rules:
                    parts.append(f"  - {rule}")

            if s.notes:
                parts.append("Notes:")
                for note in s.notes:
                    parts.append(f"  - {note}")

            parts.append("")

        if conflicts:
            parts.append("### Detected Conflicts")
            for c in conflicts:
                parts.append(f"- {c['detail']}")
            parts.append("")

        if kb.general_rules:
            parts.append("### General Rules")
            for rule in kb.general_rules:
                parts.append(f"- {rule}")
            parts.append("")

        return "\n".join(parts)
