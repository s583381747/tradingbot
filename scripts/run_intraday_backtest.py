#!/usr/bin/env python3
"""
Intraday backtest runner with parameter optimization.

Usage:
    # Quick test with defaults
    python scripts/run_intraday_backtest.py

    # Full optimization
    python scripts/run_intraday_backtest.py --optimize

    # Use CSV data
    python scripts/run_intraday_backtest.py --data-file data/qqq_1m.csv

    # Custom symbol / period
    python scripts/run_intraday_backtest.py --symbol QQQ --interval 5m --period 60d
"""

from __future__ import annotations

import argparse
import datetime
import importlib
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(_PROJECT_ROOT / ".env")

import backtrader as bt
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

# ────────────────────────────────────────────────────────────────
# Named parameter sets — each tests a different "unclear" aspect
# ────────────────────────────────────────────────────────────────

PARAM_SETS: dict[str, dict] = {
    # ── Baseline ──
    "baseline": {},

    # ── Trend sensitivity ──
    "trend_loose":   {"ema_slope_threshold": 0.005},
    "trend_strict":  {"ema_slope_threshold": 0.02},
    "no_vwap":       {"require_above_vwap": False},

    # ── Pullback distance ──
    "pb_tight":      {"pullback_touch_mult": 0.3},
    "pb_loose":      {"pullback_touch_mult": 1.0},

    # ── Volume filter ──
    "no_vol_filter": {"use_volume_filter": False},
    "vol_strict":    {"volume_breakout_mult": 2.0},

    # ── Stop loss type ──
    "sl_atr":        {"sl_type": "atr", "sl_atr_mult": 1.5},
    "sl_tight":      {"sl_offset": 0.03},
    "sl_wide":       {"sl_offset": 0.10},

    # ── Take profit ──
    "tp_50":         {"partial_tp_pct": 0.5},
    "tp_80":         {"partial_tp_pct": 0.8},
    "no_partial_tp": {"use_partial_tp": False},

    # ── Trailing stop ──
    "trail_atr":     {"trail_type": "atr"},
    "trail_none":    {"trail_type": "none"},
    "trail_tight":   {"trail_atr_mult": 0.5},

    # ── Opening range ──
    "or_strict":     {"opening_range_atr_mult": 1.0},
    "or_ignore":     {"opening_range_atr_mult": 999},

    # ── Bundle: aggressive ──
    "aggressive": {
        "ema_slope_threshold": 0.005,
        "pullback_touch_mult": 0.8,
        "sl_offset": 0.03,
        "partial_tp_pct": 0.5,
        "trail_atr_mult": 0.5,
    },

    # ── Bundle: conservative ──
    "conservative": {
        "ema_slope_threshold": 0.02,
        "pullback_touch_mult": 0.3,
        "sl_offset": 0.08,
        "partial_tp_pct": 0.8,
        "trail_atr_mult": 1.5,
        "volume_breakout_mult": 2.0,
    },

    # ── 1-Minute calibrated sets ──────────────────────────────
    # 1m bars have more noise: need looser slope, wider pullback,
    # lower volume threshold, and ATR-based stops

    "1m_base": {
        "ema_slope_threshold": 0.003,
        "pullback_touch_mult": 1.0,
        "use_volume_filter": False,
        "sl_type": "atr",
        "sl_atr_mult": 1.5,
        "tp_atr_mult": 2.5,
        "trail_type": "ema",
        "trail_atr_mult": 0.8,
    },

    "1m_trend_follow": {
        "ema_slope_threshold": 0.005,
        "pullback_touch_mult": 1.2,
        "use_volume_filter": False,
        "sl_type": "atr",
        "sl_atr_mult": 2.0,
        "tp_atr_mult": 3.0,
        "trail_type": "ema",
        "trail_atr_mult": 1.0,
        "partial_tp_pct": 0.5,
    },

    "1m_scalp": {
        "ema_slope_threshold": 0.002,
        "pullback_touch_mult": 1.5,
        "use_volume_filter": False,
        "sl_type": "atr",
        "sl_atr_mult": 1.0,
        "tp_atr_mult": 1.5,
        "trail_type": "none",
        "use_partial_tp": False,
        "max_daily_trades": 10,
    },

    "1m_wide_stop": {
        "ema_slope_threshold": 0.003,
        "pullback_touch_mult": 1.0,
        "use_volume_filter": False,
        "sl_type": "atr",
        "sl_atr_mult": 2.5,
        "tp_atr_mult": 3.0,
        "trail_type": "ema",
        "trail_atr_mult": 1.5,
        "partial_tp_pct": 0.8,
    },

    "1m_ema_only": {
        "ema_slope_threshold": 0.003,
        "pullback_touch_mult": 0.8,
        "use_volume_filter": False,
        "require_above_vwap": False,
        "sl_type": "atr",
        "sl_atr_mult": 1.5,
        "tp_atr_mult": 2.0,
        "trail_type": "ema",
        "trail_atr_mult": 0.5,
    },

    "1m_vwap_bounce": {
        "ema_slope_threshold": 0.002,
        "pullback_touch_mult": 1.5,
        "require_above_vwap": True,
        "use_volume_filter": False,
        "sl_type": "atr",
        "sl_atr_mult": 1.5,
        "tp_atr_mult": 2.0,
        "trail_type": "atr",
        "trail_atr_mult": 1.0,
    },

    # ── Pseudo-5m on 1m data (scale indicators by 5x) ────────
    # The video strategy was taught on ~5m charts.
    # EMA20@5m ≈ EMA100@1m, ATR14@5m ≈ ATR70@1m, etc.

    "1m_5x_A": {
        "ema_period": 100,
        "atr_period": 70,
        "volume_avg_period": 100,
        "ema_slope_period": 25,
        "ema_slope_threshold": 0.01,
        "pullback_touch_mult": 0.5,
        "use_volume_filter": True,
        "volume_breakout_mult": 1.5,
        "sl_type": "swing",
        "sl_offset": 0.10,
        "tp_atr_mult": 2.0,
        "trail_type": "ema",
        "trail_atr_mult": 1.0,
        "opening_range_bars": 30,
    },

    "1m_5x_B": {
        "ema_period": 100,
        "atr_period": 70,
        "volume_avg_period": 100,
        "ema_slope_period": 25,
        "ema_slope_threshold": 0.005,
        "pullback_touch_mult": 0.8,
        "use_volume_filter": True,
        "volume_breakout_mult": 1.3,
        "sl_type": "atr",
        "sl_atr_mult": 1.5,
        "tp_atr_mult": 2.5,
        "trail_type": "ema",
        "trail_atr_mult": 0.8,
        "opening_range_bars": 30,
    },

    "1m_5x_C": {
        "ema_period": 100,
        "atr_period": 70,
        "volume_avg_period": 100,
        "ema_slope_period": 25,
        "ema_slope_threshold": 0.008,
        "pullback_touch_mult": 0.5,
        "use_volume_filter": False,
        "sl_type": "swing",
        "sl_offset": 0.15,
        "tp_atr_mult": 2.0,
        "trail_type": "ema",
        "trail_atr_mult": 1.0,
        "partial_tp_pct": 0.6,
        "opening_range_bars": 30,
    },

    "1m_5x_loose": {
        "ema_period": 100,
        "atr_period": 70,
        "volume_avg_period": 100,
        "ema_slope_period": 25,
        "ema_slope_threshold": 0.003,
        "pullback_touch_mult": 1.0,
        "use_volume_filter": False,
        "require_above_vwap": False,
        "sl_type": "atr",
        "sl_atr_mult": 2.0,
        "tp_atr_mult": 3.0,
        "trail_type": "ema",
        "trail_atr_mult": 1.5,
        "partial_tp_pct": 0.5,
        "opening_range_bars": 30,
    },

    "1m_5x_tight": {
        "ema_period": 100,
        "atr_period": 70,
        "volume_avg_period": 100,
        "ema_slope_period": 25,
        "ema_slope_threshold": 0.01,
        "pullback_touch_mult": 0.3,
        "use_volume_filter": True,
        "volume_breakout_mult": 1.5,
        "sl_type": "swing",
        "sl_offset": 0.05,
        "tp_atr_mult": 1.5,
        "trail_type": "none",
        "use_partial_tp": False,
        "opening_range_bars": 30,
    },
}


# ────────────────────────────────────────────────────────────────
# Data loading
# ────────────────────────────────────────────────────────────────

def load_yfinance(symbol: str, period: str, interval: str) -> pd.DataFrame:
    """Download data from yfinance."""
    import yfinance as yf

    console.print(f"[dim]Downloading {symbol} {interval} data for {period} via yfinance...[/dim]")
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval)

    if df.empty:
        console.print("[red]No data returned from yfinance.[/red]")
        console.print("[dim]Note: yfinance limits — 1m: ~7d, 5m: ~60d, 15m: ~60d, 1h: ~730d[/dim]")
        sys.exit(1)

    # Normalize columns
    df.index.name = "datetime"
    for col in ("Open", "High", "Low", "Close", "Volume"):
        if col not in df.columns:
            alt = col.lower()
            if alt in df.columns:
                df.rename(columns={alt: col}, inplace=True)

    # Drop timezone info (backtrader doesn't handle it)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # Filter to regular trading hours (9:30-16:00 ET)
    df = df.between_time("09:30", "16:00")

    console.print(f"[green]Loaded {len(df)} bars[/green]  "
                  f"from {df.index[0]} to {df.index[-1]}")
    return df


def load_csv(path: str) -> pd.DataFrame:
    """Load data from a CSV file.

    Expected columns: datetime (or Date/Datetime), Open, High, Low, Close, Volume.
    """
    console.print(f"[dim]Loading CSV: {path}[/dim]")
    df = pd.read_csv(path, parse_dates=True, index_col=0)

    # Normalize column names
    col_map = {}
    for col in df.columns:
        lower = col.lower().strip()
        if lower in ("open",):
            col_map[col] = "Open"
        elif lower in ("high",):
            col_map[col] = "High"
        elif lower in ("low",):
            col_map[col] = "Low"
        elif lower in ("close", "adj close"):
            col_map[col] = "Close"
        elif lower in ("volume",):
            col_map[col] = "Volume"
    df.rename(columns=col_map, inplace=True)

    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    df = df.between_time("09:30", "16:00")

    console.print(f"[green]Loaded {len(df)} bars[/green]  "
                  f"from {df.index[0]} to {df.index[-1]}")
    return df


def load_alpaca(symbol: str, start: str, end: str, interval: str = "1Min") -> pd.DataFrame:
    """Load data from Alpaca API. Requires ALPACA_KEY and ALPACA_SECRET env vars."""
    import os

    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    except ImportError:
        console.print("[red]alpaca-py not installed. Run: pip install alpaca-py[/red]")
        sys.exit(1)

    key = os.environ.get("ALPACA_KEY", "")
    secret = os.environ.get("ALPACA_SECRET", "")
    if not key or not secret:
        console.print("[red]Set ALPACA_KEY and ALPACA_SECRET env vars (or in .env).[/red]")
        sys.exit(1)

    tf_map = {
        "1Min": TimeFrame.Minute,
        "5Min": TimeFrame(5, TimeFrameUnit.Minute),
        "15Min": TimeFrame(15, TimeFrameUnit.Minute),
        "1Hour": TimeFrame.Hour,
    }
    tf = tf_map.get(interval, TimeFrame.Minute)

    console.print(f"[dim]Fetching {symbol} {interval} bars from Alpaca "
                  f"({start} to {end})...[/dim]")

    client = StockHistoricalDataClient(key, secret)

    # Alpaca limits to ~10k bars per request — paginate by month for 1Min
    start_dt = datetime.datetime.fromisoformat(start)
    end_dt = datetime.datetime.fromisoformat(end)
    all_frames: list[pd.DataFrame] = []

    chunk_start = start_dt
    while chunk_start < end_dt:
        # ~1 month chunks (22 trading days * 390 min = 8580 bars)
        chunk_end = min(chunk_start + datetime.timedelta(days=30), end_dt)

        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=tf,
            start=chunk_start,
            end=chunk_end,
        )
        try:
            bars = client.get_stock_bars(request)
            chunk_df = bars.df
            if not chunk_df.empty:
                if isinstance(chunk_df.index, pd.MultiIndex):
                    chunk_df = chunk_df.droplevel(0)
                all_frames.append(chunk_df)
                console.print(f"  [dim]{chunk_start.date()} → {chunk_end.date()}: "
                              f"{len(chunk_df)} bars[/dim]")
        except Exception as exc:
            console.print(f"  [yellow]Chunk {chunk_start.date()}-{chunk_end.date()} "
                          f"failed: {exc}[/yellow]")

        chunk_start = chunk_end

    if not all_frames:
        console.print("[red]No data returned from Alpaca.[/red]")
        sys.exit(1)

    df = pd.concat(all_frames)
    df = df[~df.index.duplicated(keep="first")]
    df.sort_index(inplace=True)

    df.rename(columns={"open": "Open", "high": "High", "low": "Low",
                        "close": "Close", "volume": "Volume"}, inplace=True)

    if df.index.tz is not None:
        df.index = df.index.tz_convert("America/New_York").tz_localize(None)

    df = df.between_time("09:30", "16:00")

    # Cache to CSV for faster re-runs
    cache_path = _PROJECT_ROOT / "data" / f"{symbol}_{interval}_{start}_{end}.csv"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path)
    console.print(f"[dim]Cached to {cache_path}[/dim]")

    console.print(f"[green]Loaded {len(df)} bars from Alpaca[/green]  "
                  f"from {df.index[0]} to {df.index[-1]}")
    return df


# ────────────────────────────────────────────────────────────────
# Run single backtest
# ────────────────────────────────────────────────────────────────

def run_single(df: pd.DataFrame, params: dict, cash: float = 100_000,
               commission: float = 0.001) -> dict:
    """Run a single backtest and return metrics."""
    from src.strategy.trend_following_system import TrendFollowingSystem

    cerebro = bt.Cerebro()

    data = bt.feeds.PandasData(
        dataname=df,
        datetime=None,
        open="Open", high="High", low="Low", close="Close", volume="Volume",
        openinterest=-1,
    )
    cerebro.adddata(data)
    cerebro.addstrategy(TrendFollowingSystem, **params)

    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission=commission)

    # Analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe",
                        timeframe=bt.TimeFrame.Days, riskfreerate=0.05)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")

    results = cerebro.run()
    strat = results[0]

    # Extract metrics
    final_value = cerebro.broker.getvalue()
    total_return = (final_value - cash) / cash * 100

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
    profit_factor = gross_won / gross_lost if gross_lost > 0 else float("inf")

    return {
        "final_value": final_value,
        "total_return_pct": round(total_return, 2),
        "sharpe": round(sharpe, 3),
        "max_drawdown_pct": round(max_dd, 2),
        "total_trades": total_trades,
        "win_rate_pct": round(win_rate, 1),
        "profit_factor": round(profit_factor, 2),
        "won": won,
        "lost": lost,
    }


# ────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Intraday backtest with parameter optimization")
    parser.add_argument("--symbol", default="QQQ", help="Ticker symbol (default: QQQ)")
    parser.add_argument("--interval", default="1m", help="Bar interval (default: 1m)")
    parser.add_argument("--period", default="60d", help="yfinance period (default: 60d)")
    parser.add_argument("--data-file", type=str, default=None, help="Path to CSV data file")
    parser.add_argument("--data-source", choices=["yfinance", "alpaca", "csv"], default="alpaca")
    parser.add_argument("--alpaca-start", type=str, default=None, help="Start date for Alpaca (YYYY-MM-DD)")
    parser.add_argument("--alpaca-end", type=str, default=None, help="End date for Alpaca (YYYY-MM-DD)")
    parser.add_argument("--cash", type=float, default=100_000, help="Starting cash (default: 100000)")
    parser.add_argument("--commission", type=float, default=0.001, help="Commission rate (default: 0.001)")
    parser.add_argument("--optimize", action="store_true", help="Run all parameter sets")
    parser.add_argument("--sets", nargs="*", default=None, help="Specific param sets to run")
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON")
    return parser.parse_args()


def main():
    args = parse_args()

    console.print(Panel(
        f"[bold cyan]Intraday Backtest Runner[/bold cyan]\n"
        f"Symbol:     [green]{args.symbol}[/green]\n"
        f"Interval:   [green]{args.interval}[/green]\n"
        f"Cash:       [green]${args.cash:,.0f}[/green]\n"
        f"Optimize:   [green]{args.optimize}[/green]",
        title="Trading Bot", border_style="blue",
    ))

    # ── Load data ──
    if args.data_file:
        df = load_csv(args.data_file)
    elif args.data_source == "alpaca":
        start = args.alpaca_start or (datetime.date.today() - datetime.timedelta(days=180)).isoformat()
        end = args.alpaca_end or datetime.date.today().isoformat()
        interval_map = {"1m": "1Min", "5m": "5Min", "15m": "15Min", "1h": "1Hour"}
        df = load_alpaca(args.symbol, start, end, interval_map.get(args.interval, "1Min"))
    else:
        df = load_yfinance(args.symbol, args.period, args.interval)

    if len(df) < 100:
        console.print(f"[red]Not enough data ({len(df)} bars). Need at least 100.[/red]")
        sys.exit(1)

    # ── Determine which sets to run ──
    if args.optimize:
        sets_to_run = PARAM_SETS
    elif args.sets:
        sets_to_run = {k: PARAM_SETS[k] for k in args.sets if k in PARAM_SETS}
    else:
        sets_to_run = {"baseline": PARAM_SETS["baseline"]}

    # ── Run backtests ──
    results: dict[str, dict] = {}

    for name, params in sets_to_run.items():
        console.print(f"\n[bold]Running: {name}[/bold]  params={params or 'defaults'}")
        try:
            metrics = run_single(df, params, cash=args.cash, commission=args.commission)
            results[name] = metrics
            console.print(
                f"  Return: [{'green' if metrics['total_return_pct'] > 0 else 'red'}]"
                f"{metrics['total_return_pct']:+.2f}%[/]  "
                f"Sharpe: {metrics['sharpe']:.3f}  "
                f"MaxDD: {metrics['max_drawdown_pct']:.1f}%  "
                f"Trades: {metrics['total_trades']}  "
                f"WinRate: {metrics['win_rate_pct']:.0f}%  "
                f"PF: {metrics['profit_factor']:.2f}"
            )
        except Exception as exc:
            console.print(f"  [red]FAILED: {exc}[/red]")
            results[name] = {"error": str(exc)}

    # ── Results table ──
    console.print()
    table = Table(title="Optimization Results", show_header=True, header_style="bold magenta")
    table.add_column("Param Set", style="cyan", min_width=15)
    table.add_column("Return %", justify="right")
    table.add_column("Sharpe", justify="right")
    table.add_column("MaxDD %", justify="right")
    table.add_column("Trades", justify="right")
    table.add_column("Win %", justify="right")
    table.add_column("PF", justify="right")

    sorted_results = sorted(
        [(k, v) for k, v in results.items() if "error" not in v],
        key=lambda x: x[1].get("total_return_pct", -999),
        reverse=True,
    )

    for name, m in sorted_results:
        ret_str = f"{m['total_return_pct']:+.2f}%"
        ret_style = "green" if m["total_return_pct"] > 0 else "red"
        table.add_row(
            name,
            f"[{ret_style}]{ret_str}[/]",
            f"{m['sharpe']:.3f}",
            f"{m['max_drawdown_pct']:.1f}%",
            str(m["total_trades"]),
            f"{m['win_rate_pct']:.0f}%",
            f"{m['profit_factor']:.2f}",
        )

    console.print(table)

    # ── Best result ──
    if sorted_results:
        best_name, best = sorted_results[0]
        console.print(f"\n[bold green]Best: {best_name}[/bold green]  "
                      f"Return={best['total_return_pct']:+.2f}%  "
                      f"Sharpe={best['sharpe']:.3f}  "
                      f"PF={best['profit_factor']:.2f}")
        console.print(f"[dim]Params: {PARAM_SETS.get(best_name, {})}[/dim]")

    # ── Save results ──
    output_path = args.output or "data/backtest_results/optimization_results.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    console.print(f"\n[dim]Results saved to {output_path}[/dim]")


if __name__ == "__main__":
    main()
