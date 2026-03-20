"""Backtest execution engine.

Dynamically loads a generated strategy module, configures a backtrader
``Cerebro`` instance with data and standard analysers, runs the backtest,
and collects results into a typed ``BacktestResult`` model.
"""

from __future__ import annotations

import importlib.util
import inspect
import logging
import sys
from pathlib import Path
from typing import Any

import backtrader as bt
from pydantic import BaseModel, Field

from src.backtest.data_feed import DataFeedProvider

logger = logging.getLogger(__name__)


# ======================================================================
# Result model
# ======================================================================


class BacktestResult(BaseModel):
    """Structured output of a single backtest run."""

    strategy_name: str = Field(description="Name of the strategy class")
    symbol: str = Field(description="Ticker symbol tested")
    period: str = Field(description="Date range, e.g. '2020-01-01 to 2023-12-31'")
    initial_cash: float = Field(description="Starting capital")
    final_value: float = Field(description="Portfolio value at the end")
    total_return_pct: float = Field(description="Total return in percent")
    sharpe_ratio: float | None = Field(default=None, description="Annualised Sharpe ratio")
    max_drawdown_pct: float | None = Field(default=None, description="Maximum drawdown percent")
    win_rate: float | None = Field(default=None, description="Winning trades / total trades")
    total_trades: int = Field(default=0, description="Number of completed round-trip trades")
    profit_factor: float | None = Field(
        default=None,
        description="Gross profit / gross loss (None if no losing trades)",
    )


# ======================================================================
# Runner
# ======================================================================


class BacktestRunner:
    """Execute backtests for generated strategy files.

    Usage::

        runner = BacktestRunner(initial_cash=100_000)
        result = runner.run(
            strategy_code_path="src/strategy/generated/my_strat.py",
            symbol="AAPL",
            start="2020-01-01",
            end="2023-12-31",
        )
        print(result.total_return_pct)

    Parameters
    ----------
    initial_cash:
        Starting portfolio value.  Defaults to ``100_000``.
    """

    def __init__(self, initial_cash: float = 100_000.0) -> None:
        self.initial_cash = initial_cash
        self._data_provider = DataFeedProvider()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        strategy_code_path: str,
        symbol: str,
        start: str,
        end: str,
        **kwargs: Any,
    ) -> BacktestResult:
        """Run a full backtest.

        Parameters
        ----------
        strategy_code_path:
            Path to a ``.py`` file containing a backtrader strategy class.
        symbol:
            Ticker symbol to test against (e.g. ``"AAPL"``, ``"BTC-USD"``).
        start:
            Start date ``YYYY-MM-DD``.
        end:
            End date ``YYYY-MM-DD``.
        **kwargs:
            Extra keyword arguments forwarded as strategy params
            (overrides the class defaults).

        Returns
        -------
        BacktestResult
            Structured result with return, Sharpe, drawdown, etc.
        """
        # 1. Import strategy class from file
        strategy_cls = self._import_strategy(strategy_code_path)
        logger.info(
            "Running backtest: strategy=%s symbol=%s period=%s–%s cash=%.0f",
            strategy_cls.__name__,
            symbol,
            start,
            end,
            self.initial_cash,
        )

        # 2. Fetch market data
        is_crypto = "-" in symbol or symbol.upper() in (
            "BTC", "ETH", "SOL", "DOGE", "ADA", "XRP", "DOT", "AVAX", "MATIC", "LINK",
        )
        if is_crypto:
            df = self._data_provider.get_crypto_data(symbol, start, end)
        else:
            df = self._data_provider.get_stock_data(symbol, start, end)
        data_feed = self._data_provider.to_backtrader_feed(df)

        # 3. Configure Cerebro
        cerebro = bt.Cerebro()
        cerebro.addstrategy(strategy_cls, **kwargs)
        cerebro.adddata(data_feed)
        cerebro.broker.setcash(self.initial_cash)

        # Commission — 0.1 % per trade (a reasonable default)
        cerebro.broker.setcommission(commission=0.001)

        # 4. Add analysers
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", riskfreerate=0.0)
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
        cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")

        # 5. Run
        strategies = cerebro.run()

        # 6. Collect results
        result = self._extract_results(
            cerebro=cerebro,
            strategies=strategies,
            strategy_name=strategy_cls.__name__,
            symbol=symbol,
            start=start,
            end=end,
        )
        logger.info(
            "Backtest complete: %s  return=%.2f%%  sharpe=%s  trades=%d",
            result.strategy_name,
            result.total_return_pct,
            f"{result.sharpe_ratio:.2f}" if result.sharpe_ratio is not None else "N/A",
            result.total_trades,
        )
        return result

    # ------------------------------------------------------------------
    # Strategy import
    # ------------------------------------------------------------------

    @staticmethod
    def _import_strategy(code_path: str) -> type:
        """Dynamically import a strategy class from a Python file.

        The file must contain exactly one class that inherits from
        ``bt.Strategy`` (directly or via ``BaseGeneratedStrategy``).

        Parameters
        ----------
        code_path:
            Filesystem path to the ``.py`` file.

        Returns
        -------
        type
            The strategy class object.

        Raises
        ------
        FileNotFoundError
            If *code_path* does not exist.
        ImportError
            If no valid strategy class is found in the file.
        """
        path = Path(code_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Strategy file not found: {path}")

        module_name = f"_generated_strategy_{path.stem}"

        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot create module spec for {path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module

        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            sys.modules.pop(module_name, None)
            raise ImportError(f"Failed to execute strategy module {path}: {exc}") from exc

        # Find the strategy class
        strategy_cls: type | None = None
        for _name, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__ != module_name:
                continue  # skip imported base classes
            if issubclass(obj, bt.Strategy):
                strategy_cls = obj
                break

        if strategy_cls is None:
            sys.modules.pop(module_name, None)
            raise ImportError(
                f"No bt.Strategy subclass found in {path}. "
                "The file must define a class inheriting from bt.Strategy."
            )

        return strategy_cls

    # ------------------------------------------------------------------
    # Result extraction
    # ------------------------------------------------------------------

    def _extract_results(
        self,
        cerebro: bt.Cerebro,
        strategies: list,
        strategy_name: str,
        symbol: str,
        start: str,
        end: str,
    ) -> BacktestResult:
        """Pull metrics from backtrader analysers into a BacktestResult."""
        strat = strategies[0]
        final_value = cerebro.broker.getvalue()
        total_return_pct = ((final_value - self.initial_cash) / self.initial_cash) * 100.0

        # --- Sharpe Ratio ---
        sharpe_ratio: float | None = None
        try:
            sharpe_analysis = strat.analyzers.sharpe.get_analysis()
            sr = sharpe_analysis.get("sharperatio")
            if sr is not None:
                sharpe_ratio = float(sr)
        except Exception:
            logger.debug("Could not extract Sharpe ratio", exc_info=True)

        # --- Max Drawdown ---
        max_drawdown_pct: float | None = None
        try:
            dd_analysis = strat.analyzers.drawdown.get_analysis()
            max_dd = dd_analysis.get("max", {})
            if isinstance(max_dd, dict):
                max_drawdown_pct = max_dd.get("drawdown")
            else:
                max_drawdown_pct = float(max_dd)
        except Exception:
            logger.debug("Could not extract drawdown", exc_info=True)

        # --- Trade Analyzer ---
        total_trades = 0
        win_rate: float | None = None
        profit_factor: float | None = None
        try:
            ta = strat.analyzers.trades.get_analysis()

            total_obj = ta.get("total", {})
            total_closed = total_obj.get("closed", 0) if isinstance(total_obj, dict) else 0
            total_trades = int(total_closed)

            won_obj = ta.get("won", {})
            won_total = won_obj.get("total", 0) if isinstance(won_obj, dict) else 0

            if total_trades > 0:
                win_rate = float(won_total) / float(total_trades)

            # Profit factor = gross_profit / gross_loss
            gross_profit = 0.0
            gross_loss = 0.0

            pnl_won = won_obj.get("pnl", {}) if isinstance(won_obj, dict) else {}
            if isinstance(pnl_won, dict):
                gross_profit = abs(float(pnl_won.get("total", 0.0)))
            else:
                gross_profit = abs(float(pnl_won))

            lost_obj = ta.get("lost", {})
            pnl_lost = lost_obj.get("pnl", {}) if isinstance(lost_obj, dict) else {}
            if isinstance(pnl_lost, dict):
                gross_loss = abs(float(pnl_lost.get("total", 0.0)))
            else:
                gross_loss = abs(float(pnl_lost))

            if gross_loss > 0:
                profit_factor = gross_profit / gross_loss

        except Exception:
            logger.debug("Could not extract trade analysis", exc_info=True)

        return BacktestResult(
            strategy_name=strategy_name,
            symbol=symbol,
            period=f"{start} to {end}",
            initial_cash=self.initial_cash,
            final_value=round(final_value, 2),
            total_return_pct=round(total_return_pct, 2),
            sharpe_ratio=round(sharpe_ratio, 4) if sharpe_ratio is not None else None,
            max_drawdown_pct=round(max_drawdown_pct, 2) if max_drawdown_pct is not None else None,
            win_rate=round(win_rate, 4) if win_rate is not None else None,
            total_trades=total_trades,
            profit_factor=round(profit_factor, 4) if profit_factor is not None else None,
        )
