"""AI-assisted strategy code generation.

Uses Gemini (when available) or a deterministic template to turn a
``TradingStrategy`` schema object into a runnable backtrader strategy
Python file.
"""

from __future__ import annotations

import logging
import re
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING

from src.knowledge.schema import TradingStrategy

if TYPE_CHECKING:
    from src.analysis.gemini_client import GeminiClient

logger = logging.getLogger(__name__)

# Directory where generated strategy files are stored.
_GENERATED_DIR = Path(__file__).resolve().parent / "generated"

# ---------------------------------------------------------------------------
# Indicator-name -> bt.indicators mapping used by the template generator
# ---------------------------------------------------------------------------
_INDICATOR_MAP: dict[str, str] = {
    "sma": "bt.indicators.SMA",
    "simple moving average": "bt.indicators.SMA",
    "ema": "bt.indicators.EMA",
    "exponential moving average": "bt.indicators.EMA",
    "rsi": "bt.indicators.RSI",
    "relative strength index": "bt.indicators.RSI",
    "macd": "bt.indicators.MACD",
    "bollinger": "bt.indicators.BollingerBands",
    "bollinger bands": "bt.indicators.BollingerBands",
    "bb": "bt.indicators.BollingerBands",
    "atr": "bt.indicators.ATR",
    "average true range": "bt.indicators.ATR",
    "stochastic": "bt.indicators.Stochastic",
    "adx": "bt.indicators.ADX",
    "cci": "bt.indicators.CCI",
    "williams r": "bt.indicators.WilliamsR",
    "williamsr": "bt.indicators.WilliamsR",
    "obv": "bt.indicators.OBV",
    "vwap": "bt.indicators.VWAP",
}


class StrategyCodeGenerator:
    """Generate backtrader strategy source code from a strategy spec.

    Parameters
    ----------
    gemini_client:
        Optional async Gemini client.  When provided the generator asks
        Gemini to produce the code; otherwise it falls back to a
        deterministic template.
    """

    def __init__(self, gemini_client: GeminiClient | None = None) -> None:
        self._ai = gemini_client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate(self, strategy: TradingStrategy) -> str:
        """Generate a backtrader strategy file and return the code string.

        1. Build a prompt from the strategy specification.
        2. If an AI client is available, ask Gemini; otherwise use the
           deterministic template.
        3. Validate the generated code (syntax + basic structural checks).
        4. Save to ``src/strategy/generated/<name_snake>.py``.

        Returns
        -------
        str
            The generated Python source code.

        Raises
        ------
        SyntaxError
            If the generated code fails validation.
        """
        if self._ai is not None:
            code = await self._ai_generate(strategy)
        else:
            code = self._template_generate(strategy)

        # Validate -------------------------------------------------------
        ok, err_msg = self._validate_code(code)
        if not ok:
            # When AI code is broken fall back to template
            if self._ai is not None:
                logger.warning(
                    "AI-generated code failed validation (%s); falling back to template.",
                    err_msg,
                )
                code = self._template_generate(strategy)
                ok2, err2 = self._validate_code(code)
                if not ok2:
                    raise SyntaxError(f"Template code also invalid: {err2}")
            else:
                raise SyntaxError(f"Generated code invalid: {err_msg}")

        # Persist ---------------------------------------------------------
        _GENERATED_DIR.mkdir(parents=True, exist_ok=True)
        out_path = _GENERATED_DIR / f"{strategy.name_snake}.py"
        out_path.write_text(code, encoding="utf-8")
        logger.info("Strategy code saved to %s", out_path)
        return code

    # ------------------------------------------------------------------
    # AI generation
    # ------------------------------------------------------------------

    async def _ai_generate(self, strategy: TradingStrategy) -> str:
        """Use Gemini to produce a backtrader strategy."""
        import json

        from config.prompts.strategy_synthesis import format_codegen_prompt

        class_name = _to_class_name(strategy.name)
        strategy_json = json.dumps(
            strategy.model_dump(mode="json"), indent=2, ensure_ascii=False,
        )

        prompt = format_codegen_prompt(
            strategy_json=strategy_json,
            strategy_class_name=class_name,
            strategy_description=strategy.description or strategy.name,
            quantifiability_score=int(strategy.quantifiability_score * 10),
        )

        assert self._ai is not None
        raw = await self._ai.generate_text(prompt)
        return self._extract_python_code(raw)

    @staticmethod
    def _extract_python_code(response: str) -> str:
        """Pull the Python code block out of a Gemini response."""
        # Try fenced code block first: ```python ... ```
        match = re.search(r"```python\s*\n(.*?)```", response, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Try generic fenced block: ``` ... ```
        match = re.search(r"```\s*\n(.*?)```", response, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Last resort: assume the whole response is code
        return response.strip()

    # ------------------------------------------------------------------
    # Template generation (no AI)
    # ------------------------------------------------------------------

    def _template_generate(self, strategy: TradingStrategy) -> str:
        """Generate a backtrader strategy from a deterministic template.

        Maps indicators to ``bt.indicators`` calls and entry/exit
        conditions to ``next()`` logic.
        """
        class_name = _to_class_name(strategy.name)

        # Build params tuple entries
        param_lines: list[str] = [
            '        ("printlog", False),',
        ]

        # Build indicator init lines
        indicator_lines: list[str] = []
        indicator_vars: list[str] = []

        for idx, ind in enumerate(strategy.indicators):
            bt_cls = _INDICATOR_MAP.get(ind.name.lower().strip())
            var_name = f"self.ind_{_sanitize_var(ind.name)}_{idx}"
            indicator_vars.append(var_name)

            if bt_cls:
                params_kwargs = _build_indicator_kwargs(ind.parameters)
                indicator_lines.append(
                    f"        {var_name} = {bt_cls}(self.data.close{params_kwargs})"
                )
                for pname, pval in ind.parameters.items():
                    safe = _sanitize_var(f"{ind.name}_{pname}")
                    param_lines.append(f'        ("{safe}", {pval!r}),')
            else:
                indicator_lines.append(
                    f"        # TODO: unknown indicator '{ind.name}' — implement manually"
                )
                indicator_lines.append(
                    f"        {var_name} = None  # placeholder"
                )

        # Build entry/exit logic
        entry_logic = _build_condition_logic(strategy.entry_conditions, "entry")
        exit_logic = _build_condition_logic(strategy.exit_conditions, "exit")

        # Build stop-loss / take-profit params
        sl_tp_next: list[str] = []
        if strategy.risk_management:
            rm = strategy.risk_management
            if rm.stop_loss_pct:
                param_lines.append(f'        ("stop_loss_pct", {rm.stop_loss_pct}),')
                sl_tp_next.extend([
                    "        # Stop-loss check",
                    "        if self.position:",
                    "            current_pnl_pct = (self.data.close[0] - self.buy_price) / self.buy_price * 100 if self.buy_price else 0",
                    "            if current_pnl_pct <= -self.p.stop_loss_pct:",
                    '                self.log(f"STOP LOSS triggered at {current_pnl_pct:.2f}%")',
                    "                self.order = self.sell()",
                    "                return",
                    "",
                ])
            if rm.take_profit_pct:
                param_lines.append(f'        ("take_profit_pct", {rm.take_profit_pct}),')
                sl_tp_next.extend([
                    "        # Take-profit check",
                    "        if self.position:",
                    "            current_pnl_pct = (self.data.close[0] - self.buy_price) / self.buy_price * 100 if self.buy_price else 0",
                    "            if current_pnl_pct >= self.p.take_profit_pct:",
                    '                self.log(f"TAKE PROFIT triggered at {current_pnl_pct:.2f}%")',
                    "                self.order = self.sell()",
                    "                return",
                    "",
                ])

        # Assemble the code line by line (no textwrap.dedent issues)
        lines: list[str] = []
        desc_escaped = (strategy.description or "").replace('"""', "'''")
        name_escaped = strategy.name.replace('"""', "'''")
        lines.append(f'"""Auto-generated backtrader strategy: {name_escaped}."""')
        lines.append("")
        lines.append("from __future__ import annotations")
        lines.append("")
        lines.append("import backtrader as bt")
        lines.append("")
        lines.append("from src.strategy.templates.backtrader_base import BaseGeneratedStrategy")
        lines.append("")
        lines.append("")
        lines.append(f"class {class_name}(BaseGeneratedStrategy):")
        lines.append(f'    """Generated strategy: {name_escaped}.')
        lines.append("")
        lines.append(f"    {desc_escaped}")
        lines.append('    """')
        lines.append("")
        lines.append("    params = (")
        lines.extend(param_lines)
        lines.append("    )")
        lines.append("")
        lines.append("    def __init__(self) -> None:")
        lines.append("        super().__init__()")
        if indicator_lines:
            lines.extend(indicator_lines)
        else:
            lines.append("        pass  # no indicators")
        lines.append("")
        lines.append("    def next(self) -> None:")
        lines.append("        # Skip if an order is pending")
        lines.append("        if self.has_pending_order():")
        lines.append("            return")
        lines.append("")
        if sl_tp_next:
            lines.extend(sl_tp_next)
        lines.append("        # Entry logic")
        lines.append("        if not self.position:")
        if entry_logic:
            lines.append(f"            if {entry_logic}:")
        else:
            lines.append("            if True:  # TODO: define entry conditions")
        lines.append("                self.log('BUY CREATE %.2f' % self.data.close[0])")
        lines.append("                self.order = self.buy()")
        lines.append("")
        lines.append("        # Exit logic")
        lines.append("        else:")
        if exit_logic:
            lines.append(f"            if {exit_logic}:")
        else:
            lines.append("            if True:  # TODO: define exit conditions")
        lines.append("                self.log('SELL CREATE %.2f' % self.data.close[0])")
        lines.append("                self.order = self.sell()")
        lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_code(code: str) -> tuple[bool, str]:
        """Validate generated Python code.

        Checks:
        1. Syntax via ``compile()``.
        2. Required imports are present (``backtrader``).
        3. A class inheriting from ``bt.Strategy`` (or our base) exists.

        Returns ``(True, "")`` on success or ``(False, reason)`` on failure.
        """
        # 1. Syntax check
        try:
            compile(code, "<generated>", "exec")
        except SyntaxError as exc:
            return False, f"SyntaxError at line {exc.lineno}: {exc.msg}"

        # 2. Required imports
        if "backtrader" not in code and "bt" not in code:
            return False, "Missing backtrader import"

        # 3. Strategy class inheritance
        has_strategy_class = bool(
            re.search(
                r"class\s+\w+\s*\(\s*(?:bt\.Strategy|BaseGeneratedStrategy)\s*\)",
                code,
            )
        )
        if not has_strategy_class:
            return False, "No class inheriting bt.Strategy or BaseGeneratedStrategy found"

        return True, ""


# ======================================================================
# Module-level helpers
# ======================================================================


def _to_class_name(name: str) -> str:
    """Convert a human strategy name to a PascalCase class name."""
    cleaned = re.sub(r"[^\w\s]", "", name)
    return "".join(word.capitalize() for word in cleaned.split()) or "GeneratedStrategy"


def _sanitize_var(name: str) -> str:
    """Turn an arbitrary string into a safe Python identifier fragment."""
    s = re.sub(r"[^\w]", "_", name.lower().strip())
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "var"


def _build_indicator_kwargs(params: dict) -> str:
    """Turn indicator parameter dict into a kwargs string like ``, period=14``."""
    if not params:
        return ""
    parts: list[str] = []
    for k, v in params.items():
        parts.append(f"{k}={v!r}")
    return ", " + ", ".join(parts)


def _build_condition_logic(conditions: list, label: str) -> str:
    """Best-effort conversion of human-readable conditions to Python expressions.

    Because conditions are natural-language strings we can only do
    keyword-based heuristic matching.  The result is a boolean expression
    string or empty string when nothing can be inferred.
    """
    expressions: list[str] = []
    for cond in conditions:
        desc = cond.description.lower() if hasattr(cond, "description") else str(cond).lower()

        # Detect crossover patterns
        if "cross" in desc and "above" in desc:
            expressions.append("True  # TODO: crossover-above condition")
        elif "cross" in desc and "below" in desc:
            expressions.append("True  # TODO: crossover-below condition")
        elif "greater" in desc or "above" in desc or ">" in desc:
            expressions.append("True  # TODO: greater-than condition")
        elif "less" in desc or "below" in desc or "<" in desc:
            expressions.append("True  # TODO: less-than condition")
        elif "overbought" in desc:
            expressions.append("True  # TODO: overbought condition")
        elif "oversold" in desc:
            expressions.append("True  # TODO: oversold condition")

    if not expressions:
        return ""

    return " and ".join(expressions)
