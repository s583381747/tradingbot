"""
Fixed evaluation harness for strategy optimization.
DO NOT MODIFY — the agent only modifies strategy.py.

Loads cached data, runs backtest, computes composite score, prints metrics.
"""

from __future__ import annotations

import importlib
import sys
import traceback

import backtrader as bt
import pandas as pd

# ── Constants ──────────────────────────────────────────────────
DATA_PATH = "data/QQQ_1Min_2025-09-21_2026-03-21.csv"
CASH = 100_000
COMMISSION = 0.001

# ── Score weights ──────────────────────────────────────────────
W_SHARPE = 0.30
W_PROFIT_FACTOR = 0.25
W_RETURN = 0.20
W_WIN_RATE = 0.10
W_DRAWDOWN = 0.15


def load_data() -> pd.DataFrame:
    """Load cached CSV into a DataFrame suitable for backtrader."""
    df = pd.read_csv(DATA_PATH, index_col="timestamp", parse_dates=True)
    # Ensure expected columns exist
    for col in ("Open", "High", "Low", "Close", "Volume"):
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
    return df


def run_backtest(df: pd.DataFrame) -> dict:
    """Run backtest with Strategy from strategy.py, return raw metrics."""
    # Import strategy module fresh each time
    if "strategy" in sys.modules:
        del sys.modules["strategy"]
    mod = importlib.import_module("strategy")
    StrategyClass = mod.Strategy

    cerebro = bt.Cerebro()

    data = bt.feeds.PandasData(
        dataname=df,
        datetime=None,
        open="Open",
        high="High",
        low="Low",
        close="Close",
        volume="Volume",
        openinterest=-1,
    )
    cerebro.adddata(data)
    cerebro.addstrategy(StrategyClass)

    cerebro.broker.setcash(CASH)
    cerebro.broker.setcommission(commission=COMMISSION)

    # Analyzers
    cerebro.addanalyzer(
        bt.analyzers.SharpeRatio,
        _name="sharpe",
        timeframe=bt.TimeFrame.Days,
        riskfreerate=0.05,
    )
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")

    results = cerebro.run()
    strat = results[0]

    # Extract metrics
    final_value = cerebro.broker.getvalue()
    total_return = (final_value - CASH) / CASH * 100

    sharpe_analysis = strat.analyzers.sharpe.get_analysis()
    sharpe = sharpe_analysis.get("sharperatio", 0.0) or 0.0

    dd = strat.analyzers.drawdown.get_analysis()
    max_dd = dd.get("max", {}).get("drawdown", 0.0) or 0.0

    ta = strat.analyzers.trades.get_analysis()
    total_trades = ta.get("total", {}).get("total", 0)
    won = ta.get("won", {}).get("total", 0)
    lost = ta.get("lost", {}).get("total", 0)
    win_rate = won / total_trades * 100 if total_trades > 0 else 0.0

    gross_won = ta.get("won", {}).get("pnl", {}).get("total", 0.0) or 0.0
    gross_lost = abs(ta.get("lost", {}).get("pnl", {}).get("total", 0.0) or 0.0)
    profit_factor = gross_won / gross_lost if gross_lost > 0 else 0.0

    return {
        "final_value": final_value,
        "total_return": round(total_return, 4),
        "sharpe": round(sharpe, 4),
        "max_drawdown": round(max_dd, 4),
        "total_trades": total_trades,
        "win_rate": round(win_rate, 2),
        "profit_factor": round(profit_factor, 4),
        "won": won,
        "lost": lost,
    }


def compute_score(m: dict) -> float:
    """
    Composite score (higher = better).

    Components (each normalized to roughly 0–1 range):
      - Sharpe (30%): clamp to [-2, 3], rescale to [0, 1]
      - Profit Factor (25%): clamp to [0, 3], rescale to [0, 1]
      - Return (20%): clamp to [-20, 50], rescale to [0, 1]
      - Win Rate (10%): clamp to [0, 100], rescale to [0, 1]
      - Drawdown penalty (15%): clamp to [0, 30], rescale 1→0

    Trade count gate:
      < 5  trades → -10 (strategy is broken)
      < 20 trades → heavy penalty (0.2x multiplier)
      20-50       → mild penalty (0.6x)
      50-500      → full credit (1.0x)
      > 500       → mild penalty (0.8x, overtrading)
    """
    trades = m["total_trades"]

    # Trade count gate
    if trades < 5:
        return -10.0

    # Normalize components
    sharpe_norm = max(0.0, min(1.0, (m["sharpe"] + 2) / 5.0))
    pf_norm = max(0.0, min(1.0, m["profit_factor"] / 3.0))
    ret_norm = max(0.0, min(1.0, (m["total_return"] + 20) / 70.0))
    wr_norm = max(0.0, min(1.0, m["win_rate"] / 100.0))
    dd_norm = max(0.0, min(1.0, 1.0 - m["max_drawdown"] / 30.0))

    raw = (
        W_SHARPE * sharpe_norm
        + W_PROFIT_FACTOR * pf_norm
        + W_RETURN * ret_norm
        + W_WIN_RATE * wr_norm
        + W_DRAWDOWN * dd_norm
    )

    # Trade count multiplier
    if trades < 20:
        mult = 0.2
    elif trades < 50:
        mult = 0.6
    elif trades <= 500:
        mult = 1.0
    else:
        mult = 0.8

    return round(raw * mult, 6)


def main():
    df = load_data()
    print(f"bars: {len(df)}")

    metrics = run_backtest(df)
    score = compute_score(metrics)

    # Print extractable output
    print(f"score: {score:.6f}")
    print(f"total_return: {metrics['total_return']:.4f}")
    print(f"sharpe: {metrics['sharpe']:.4f}")
    print(f"max_drawdown: {metrics['max_drawdown']:.4f}")
    print(f"win_rate: {metrics['win_rate']:.2f}")
    print(f"profit_factor: {metrics['profit_factor']:.4f}")
    print(f"total_trades: {metrics['total_trades']}")
    print(f"won: {metrics['won']}")
    print(f"lost: {metrics['lost']}")
    print(f"final_value: {metrics['final_value']:.2f}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        print("score: -10.000000")
        sys.exit(1)
