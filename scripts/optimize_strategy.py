#!/usr/bin/env python3
"""Optimize the EMA Pullback Strategy on QQQ with multiple parameter sets."""

from __future__ import annotations

import sys
from pathlib import Path
from itertools import product

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import backtrader as bt
import yfinance as yf
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from src.strategy.generated.ema_pullback_system import EmaPullbackSystem

console = Console()


def fetch_data(symbol: str, period: str = "60d", interval: str = "5m") -> pd.DataFrame:
    """Fetch data from yfinance."""
    console.print(f"[cyan]Downloading {symbol} {interval} data ({period})...[/cyan]")
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval)

    if df.empty:
        raise ValueError(f"No data for {symbol}")

    # Keep OHLCV only
    keep = [c for c in df.columns if c.lower() in ("open", "high", "low", "close", "volume")]
    df = df[keep]

    # Strip timezone
    if hasattr(df.index, "tz") and df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    console.print(f"[green]Got {len(df)} bars from {df.index[0]} to {df.index[-1]}[/green]")
    return df


def run_backtest(df: pd.DataFrame, params: dict, cash: float = 100000) -> dict:
    """Run a single backtest with given params, return metrics."""
    cerebro = bt.Cerebro()
    cerebro.addstrategy(EmaPullbackSystem, **params)

    data = bt.feeds.PandasData(dataname=df)
    cerebro.adddata(data)
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission=0.001)

    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", riskfreerate=0.0)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="dd")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")

    results = cerebro.run()
    strat = results[0]

    final_val = cerebro.broker.getvalue()
    ret_pct = (final_val - cash) / cash * 100

    # Sharpe
    sharpe = None
    try:
        sa = strat.analyzers.sharpe.get_analysis()
        sr = sa.get("sharperatio")
        if sr is not None:
            sharpe = float(sr)
    except Exception:
        pass

    # Drawdown
    max_dd = None
    try:
        da = strat.analyzers.dd.get_analysis()
        md = da.get("max", {})
        max_dd = md.get("drawdown") if isinstance(md, dict) else float(md)
    except Exception:
        pass

    # Trades
    total_trades = 0
    win_rate = None
    profit_factor = None
    try:
        ta = strat.analyzers.trades.get_analysis()
        total_obj = ta.get("total", {})
        total_trades = int(total_obj.get("closed", 0)) if isinstance(total_obj, dict) else 0

        won_obj = ta.get("won", {})
        won_total = won_obj.get("total", 0) if isinstance(won_obj, dict) else 0

        if total_trades > 0:
            win_rate = float(won_total) / float(total_trades) * 100

        pnl_won = won_obj.get("pnl", {}) if isinstance(won_obj, dict) else {}
        gross_profit = abs(float(pnl_won.get("total", 0))) if isinstance(pnl_won, dict) else abs(float(pnl_won))

        lost_obj = ta.get("lost", {})
        pnl_lost = lost_obj.get("pnl", {}) if isinstance(lost_obj, dict) else {}
        gross_loss = abs(float(pnl_lost.get("total", 0))) if isinstance(pnl_lost, dict) else abs(float(pnl_lost))

        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
    except Exception:
        pass

    return {
        "return_pct": round(ret_pct, 2),
        "sharpe": round(sharpe, 3) if sharpe else None,
        "max_dd": round(max_dd, 2) if max_dd else None,
        "trades": total_trades,
        "win_rate": round(win_rate, 1) if win_rate else None,
        "profit_factor": round(profit_factor, 3) if profit_factor else None,
        "final_value": round(final_val, 2),
    }


def main():
    # ================================================================
    # Fetch QQQ data - use 5m for 60 days (max for intraday yfinance)
    # ================================================================
    df = fetch_data("QQQ", period="60d", interval="5m")

    # ================================================================
    # Define parameter grid
    # ================================================================
    param_sets = [
        # --- Set 1: Baseline (video defaults) ---
        {
            "name": "基准-20EMA标准",
            "params": {
                "ema_fast": 20, "ema_slow": 40, "ema_filter": 60,
                "trend_bars": 5, "trend_atr_mult": 1.0,
                "pullback_atr_tol": 0.3, "max_pullback_bars": 8,
                "use_vol_filter": True, "vol_thresh": 0.8,
                "sl_atr_mult": 1.5, "use_trail": True, "trail_atr_offset": 0.3,
                "tp_atr_mult": 3.0, "use_partial_tp": True,
                "risk_per_trade": 0.02, "max_daily_trades": 3,
            },
        },
        # --- Set 2: Faster EMA ---
        {
            "name": "快速-9EMA激进",
            "params": {
                "ema_fast": 9, "ema_slow": 21, "ema_filter": 50,
                "trend_bars": 3, "trend_atr_mult": 0.8,
                "pullback_atr_tol": 0.4, "max_pullback_bars": 5,
                "use_vol_filter": True, "vol_thresh": 0.7,
                "sl_atr_mult": 1.2, "use_trail": True, "trail_atr_offset": 0.2,
                "tp_atr_mult": 2.5, "use_partial_tp": True,
                "risk_per_trade": 0.015, "max_daily_trades": 5,
            },
        },
        # --- Set 3: Conservative / slower ---
        {
            "name": "保守-慢速趋势",
            "params": {
                "ema_fast": 20, "ema_slow": 50, "ema_filter": 100,
                "trend_bars": 8, "trend_atr_mult": 1.5,
                "pullback_atr_tol": 0.2, "max_pullback_bars": 10,
                "use_vol_filter": True, "vol_thresh": 1.0,
                "sl_atr_mult": 2.0, "use_trail": True, "trail_atr_offset": 0.5,
                "tp_atr_mult": 4.0, "use_partial_tp": True,
                "risk_per_trade": 0.01, "max_daily_trades": 2,
            },
        },
        # --- Set 4: Tight stops, quick profits ---
        {
            "name": "短线-紧止损快止盈",
            "params": {
                "ema_fast": 12, "ema_slow": 26, "ema_filter": 50,
                "trend_bars": 4, "trend_atr_mult": 0.8,
                "pullback_atr_tol": 0.25, "max_pullback_bars": 6,
                "use_vol_filter": True, "vol_thresh": 0.8,
                "sl_atr_mult": 1.0, "use_trail": True, "trail_atr_offset": 0.15,
                "tp_atr_mult": 2.0, "use_partial_tp": True,
                "partial_tp_rr": 1.0,
                "risk_per_trade": 0.02, "max_daily_trades": 4,
            },
        },
        # --- Set 5: Long only (bull bias) ---
        {
            "name": "仅做多-牛市偏向",
            "params": {
                "ema_fast": 20, "ema_slow": 40, "ema_filter": 60,
                "trend_bars": 5, "trend_atr_mult": 1.0,
                "pullback_atr_tol": 0.3, "max_pullback_bars": 8,
                "use_vol_filter": True, "vol_thresh": 0.8,
                "sl_atr_mult": 1.5, "use_trail": True, "trail_atr_offset": 0.3,
                "tp_atr_mult": 3.0, "use_partial_tp": True,
                "risk_per_trade": 0.025, "max_daily_trades": 3,
                "long_only": True,
            },
        },
        # --- Set 6: Wide pullback tolerance ---
        {
            "name": "宽容-回撤容差大",
            "params": {
                "ema_fast": 20, "ema_slow": 40, "ema_filter": 60,
                "trend_bars": 5, "trend_atr_mult": 0.8,
                "pullback_atr_tol": 0.5, "max_pullback_bars": 12,
                "use_vol_filter": False,
                "sl_atr_mult": 1.5, "use_trail": True, "trail_atr_offset": 0.3,
                "tp_atr_mult": 3.0, "use_partial_tp": True,
                "risk_per_trade": 0.02, "max_daily_trades": 5,
            },
        },
        # --- Set 7: No volume filter, no partial TP ---
        {
            "name": "简化-无量价无分批",
            "params": {
                "ema_fast": 20, "ema_slow": 40, "ema_filter": 60,
                "trend_bars": 5, "trend_atr_mult": 1.0,
                "pullback_atr_tol": 0.3, "max_pullback_bars": 8,
                "use_vol_filter": False,
                "sl_atr_mult": 1.5, "use_trail": True, "trail_atr_offset": 0.3,
                "tp_atr_mult": 3.0, "use_partial_tp": False,
                "risk_per_trade": 0.02, "max_daily_trades": 3,
            },
        },
        # --- Set 8: High RR, wide TP ---
        {
            "name": "高RR-大止盈小止损",
            "params": {
                "ema_fast": 20, "ema_slow": 40, "ema_filter": 60,
                "trend_bars": 6, "trend_atr_mult": 1.2,
                "pullback_atr_tol": 0.25, "max_pullback_bars": 8,
                "use_vol_filter": True, "vol_thresh": 0.9,
                "sl_atr_mult": 1.0, "use_trail": True, "trail_atr_offset": 0.2,
                "tp_atr_mult": 5.0, "use_partial_tp": True,
                "partial_tp_rr": 2.0,
                "risk_per_trade": 0.02, "max_daily_trades": 3,
            },
        },
        # --- Set 9: No trailing stop ---
        {
            "name": "固定止损-无追踪",
            "params": {
                "ema_fast": 20, "ema_slow": 40, "ema_filter": 60,
                "trend_bars": 5, "trend_atr_mult": 1.0,
                "pullback_atr_tol": 0.3, "max_pullback_bars": 8,
                "use_vol_filter": True, "vol_thresh": 0.8,
                "sl_atr_mult": 1.5, "use_trail": False,
                "tp_atr_mult": 3.0, "use_partial_tp": True,
                "risk_per_trade": 0.02, "max_daily_trades": 3,
            },
        },
        # --- Set 10: Aggressive scalp ---
        {
            "name": "超短线-高频剥头皮",
            "params": {
                "ema_fast": 8, "ema_slow": 20, "ema_filter": 40,
                "trend_bars": 3, "trend_atr_mult": 0.5,
                "pullback_atr_tol": 0.5, "max_pullback_bars": 4,
                "use_vol_filter": False,
                "sl_atr_mult": 0.8, "use_trail": True, "trail_atr_offset": 0.1,
                "tp_atr_mult": 1.5, "use_partial_tp": False,
                "risk_per_trade": 0.01, "max_daily_trades": 8,
            },
        },
    ]

    # ================================================================
    # Run all backtests
    # ================================================================
    console.print(Panel(
        f"[bold cyan]EMA Pullback Strategy Optimizer[/bold cyan]\n"
        f"Symbol: [green]QQQ[/green]  |  Interval: [green]5m[/green]  |  Data: [green]{len(df)} bars[/green]\n"
        f"Testing [green]{len(param_sets)}[/green] parameter configurations",
        title="Strategy Optimization",
        border_style="blue",
    ))

    results = []
    for i, ps in enumerate(param_sets, 1):
        name = ps["name"]
        params = ps["params"]
        console.print(f"  [{i:2d}/{len(param_sets)}] {name}...", end=" ")
        try:
            metrics = run_backtest(df, params)
            metrics["name"] = name
            results.append(metrics)
            ret_color = "green" if metrics["return_pct"] > 0 else "red"
            console.print(
                f"[{ret_color}]{metrics['return_pct']:+.2f}%[/{ret_color}] | "
                f"Trades: {metrics['trades']} | "
                f"WR: {metrics['win_rate'] or 'N/A'}% | "
                f"PF: {metrics['profit_factor'] or 'N/A'}"
            )
        except Exception as e:
            console.print(f"[red]FAILED: {e}[/red]")

    # ================================================================
    # Results table
    # ================================================================
    console.print()

    # Sort by composite score: return * profit_factor (if available)
    def score(r):
        ret = r["return_pct"]
        pf = r.get("profit_factor") or 0
        wr = (r.get("win_rate") or 0) / 100
        dd = abs(r.get("max_dd") or 100)
        trades = r.get("trades") or 0
        if trades < 3:
            return -999
        # Composite: weighted combination
        return ret * 0.3 + (pf * 10 if pf else 0) * 0.3 + wr * 20 * 0.2 - dd * 0.2

    results.sort(key=score, reverse=True)

    table = Table(title="QQQ 5m 回测结果排名", show_header=True, header_style="bold magenta")
    table.add_column("#", style="dim", width=3)
    table.add_column("策略名称", min_width=20)
    table.add_column("收益%", justify="right")
    table.add_column("Sharpe", justify="right")
    table.add_column("最大回撤%", justify="right")
    table.add_column("交易次数", justify="right")
    table.add_column("胜率%", justify="right")
    table.add_column("盈亏比", justify="right")
    table.add_column("最终资金", justify="right")

    for i, r in enumerate(results, 1):
        ret_str = f"[green]{r['return_pct']:+.2f}[/green]" if r["return_pct"] > 0 else f"[red]{r['return_pct']:+.2f}[/red]"
        sharpe_str = f"{r['sharpe']:.3f}" if r['sharpe'] else "N/A"
        dd_str = f"{r['max_dd']:.2f}" if r['max_dd'] else "N/A"
        wr_str = f"{r['win_rate']:.1f}" if r['win_rate'] else "N/A"
        pf_str = f"{r['profit_factor']:.3f}" if r['profit_factor'] else "N/A"
        marker = " [bold yellow]★[/bold yellow]" if i == 1 else ""

        table.add_row(
            str(i),
            r["name"] + marker,
            ret_str,
            sharpe_str,
            dd_str,
            str(r["trades"]),
            wr_str,
            pf_str,
            f"${r['final_value']:,.0f}",
        )

    console.print(table)

    # ================================================================
    # Best strategy details
    # ================================================================
    if results:
        best = results[0]
        best_params = next(ps for ps in param_sets if ps["name"] == best["name"])

        console.print(Panel(
            f"[bold green]最佳策略: {best['name']}[/bold green]\n\n"
            f"  收益率:     [green]{best['return_pct']:+.2f}%[/green]\n"
            f"  Sharpe:     {best['sharpe'] or 'N/A'}\n"
            f"  最大回撤:   {best['max_dd'] or 'N/A'}%\n"
            f"  交易次数:   {best['trades']}\n"
            f"  胜率:       {best['win_rate'] or 'N/A'}%\n"
            f"  盈亏比:     {best['profit_factor'] or 'N/A'}\n"
            f"  最终资金:   ${best['final_value']:,.2f}\n\n"
            f"[bold]最优参数:[/bold]\n" +
            "\n".join(f"  {k}: {v}" for k, v in best_params["params"].items()),
            title="Winner",
            border_style="green",
        ))


if __name__ == "__main__":
    main()
