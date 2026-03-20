"""JSON persistence layer for the trading knowledge base.

Provides load / save / merge semantics so that strategies extracted from
multiple videos can be accumulated into a single knowledge base file
without duplicating entries.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

from src.knowledge.schema import (
    TradingIndicator,
    TradingKnowledgeBase,
    TradingStrategy,
)

logger = logging.getLogger(__name__)

# Two strategy names are considered duplicates when their similarity
# ratio exceeds this threshold.
_NAME_SIMILARITY_THRESHOLD = 0.85


class KnowledgeStore:
    """Read and write a :class:`TradingKnowledgeBase` as a JSON file.

    Parameters
    ----------
    store_path:
        Path to the JSON file used for persistence.  Parent directories
        are created automatically on first write.
    """

    def __init__(self, store_path: Path) -> None:
        self.store_path = Path(store_path)

    # ------------------------------------------------------------------
    # Core I/O
    # ------------------------------------------------------------------

    def load(self) -> TradingKnowledgeBase:
        """Load the knowledge base from disk.

        Returns an empty :class:`TradingKnowledgeBase` if the file does
        not exist or cannot be parsed.
        """
        if not self.store_path.exists():
            logger.info("Knowledge store not found at %s — returning empty KB", self.store_path)
            return TradingKnowledgeBase()

        try:
            raw = self.store_path.read_text(encoding="utf-8")
            data = json.loads(raw)
            return TradingKnowledgeBase.model_validate(data)
        except json.JSONDecodeError:
            logger.exception("Corrupt JSON in %s — returning empty KB", self.store_path)
            return TradingKnowledgeBase()
        except Exception:
            logger.exception("Failed to load knowledge base from %s", self.store_path)
            return TradingKnowledgeBase()

    def save(self, kb: TradingKnowledgeBase) -> None:
        """Persist the knowledge base to disk as pretty-printed JSON.

        The ``last_updated`` timestamp is set to *now* before writing.
        """
        kb.last_updated = datetime.now()
        self.store_path.parent.mkdir(parents=True, exist_ok=True)

        payload = kb.model_dump(mode="json")
        # datetime objects are serialised as ISO strings by Pydantic's
        # ``mode="json"`` — no custom encoder needed.
        self.store_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        logger.info("Knowledge base saved to %s", self.store_path)

    # ------------------------------------------------------------------
    # Convenience mutators
    # ------------------------------------------------------------------

    def add_strategies(self, strategies: list[TradingStrategy]) -> None:
        """Merge *strategies* into the persisted knowledge base.

        Loads the existing file, deduplicates by strategy-name similarity,
        and writes the result back.
        """
        if not strategies:
            return

        kb = self.load()
        kb.strategies = self._merge_strategies(kb.strategies, strategies)

        # Also update the top-level indicator list.
        all_indicators = self._collect_indicators(kb.strategies)
        kb.indicators = self._dedupe_indicators(kb.indicators, all_indicators)

        self.save(kb)

    # ------------------------------------------------------------------
    # Merge logic
    # ------------------------------------------------------------------

    def _merge_strategies(
        self,
        existing: list[TradingStrategy],
        new: list[TradingStrategy],
    ) -> list[TradingStrategy]:
        """Merge *new* strategies into *existing*, deduplicating by name.

        When a new strategy name is similar enough to an existing one
        (≥ 85 % similarity), the two are merged:
        * ``source_videos`` lists are combined.
        * The version with the higher ``confidence_score`` is kept as the
          base, but it inherits the merged video list.

        Completely novel strategies are appended.
        """
        merged: list[TradingStrategy] = list(existing)

        for new_strategy in new:
            match_idx = self._find_similar(new_strategy.name, merged)

            if match_idx is not None:
                old = merged[match_idx]
                # Combine source videos (deduplicated, order-preserving).
                combined_videos = list(dict.fromkeys(old.source_videos + new_strategy.source_videos))
                # Keep the higher-confidence version.
                if new_strategy.confidence_score >= old.confidence_score:
                    winner = new_strategy.model_copy(
                        update={"source_videos": combined_videos}
                    )
                else:
                    winner = old.model_copy(
                        update={"source_videos": combined_videos}
                    )
                merged[match_idx] = winner
                logger.debug(
                    "Merged strategy '%s' with existing '%s'",
                    new_strategy.name,
                    old.name,
                )
            else:
                merged.append(new_strategy)
                logger.debug("Added new strategy '%s'", new_strategy.name)

        return merged

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_strategy_by_name(self, name: str) -> Optional[TradingStrategy]:
        """Return the first strategy whose name matches *name* (case-insensitive).

        Also checks for fuzzy similarity (≥ threshold) so that minor
        wording differences don't prevent a hit.
        """
        kb = self.load()
        name_lower = name.strip().lower()

        for strategy in kb.strategies:
            if strategy.name.strip().lower() == name_lower:
                return strategy

        # Fall back to fuzzy match.
        for strategy in kb.strategies:
            ratio = SequenceMatcher(
                None, name_lower, strategy.name.strip().lower()
            ).ratio()
            if ratio >= _NAME_SIMILARITY_THRESHOLD:
                return strategy

        return None

    def get_all_indicators(self) -> list[TradingIndicator]:
        """Return a deduplicated list of all indicators across strategies."""
        kb = self.load()
        # Start with top-level indicators, then add any that only appear
        # inside individual strategies.
        all_from_strategies = self._collect_indicators(kb.strategies)
        return self._dedupe_indicators(kb.indicators, all_from_strategies)

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _find_similar(
        name: str,
        strategies: list[TradingStrategy],
    ) -> int | None:
        """Return the index of the first strategy whose name is similar."""
        name_lower = name.strip().lower()
        for idx, s in enumerate(strategies):
            existing_lower = s.name.strip().lower()
            if existing_lower == name_lower:
                return idx
            ratio = SequenceMatcher(None, name_lower, existing_lower).ratio()
            if ratio >= _NAME_SIMILARITY_THRESHOLD:
                return idx
        return None

    @staticmethod
    def _collect_indicators(
        strategies: list[TradingStrategy],
    ) -> list[TradingIndicator]:
        indicators: list[TradingIndicator] = []
        for s in strategies:
            indicators.extend(s.indicators)
        return indicators

    @staticmethod
    def _dedupe_indicators(
        existing: list[TradingIndicator],
        new: list[TradingIndicator],
    ) -> list[TradingIndicator]:
        """Combine two indicator lists, deduplicating by name (case-insensitive)."""
        seen: dict[str, TradingIndicator] = {}
        for ind in existing:
            key = ind.name.strip().lower()
            if key not in seen:
                seen[key] = ind
            else:
                # Merge source_videos.
                combined = list(dict.fromkeys(seen[key].source_videos + ind.source_videos))
                seen[key] = seen[key].model_copy(update={"source_videos": combined})

        for ind in new:
            key = ind.name.strip().lower()
            if key not in seen:
                seen[key] = ind
            else:
                combined = list(dict.fromkeys(seen[key].source_videos + ind.source_videos))
                seen[key] = seen[key].model_copy(update={"source_videos": combined})

        return list(seen.values())
