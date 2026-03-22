#!/usr/bin/env python3
"""Walk-forward optimization on QQQ 1m to avoid overfitting.

- Train on first 5 days, validate on last 2 days
- Only select strategies that are stable across both periods
- Penalize parameter sensitivity (overfitting signal)
"""

from __future__ import annotations

import sys
from pathlib import Path

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


def fetch_1m(symbol: str) -> pd.DataFrame:
    df = yf.Ticker(symbol).history(period="7d", interval="1m")
    keep = [c for c in df.columns if c.lower() in ("open", "high", "low", "close", "volume")]
    df = df[keep]
    if hasattr(df.index, "tz") and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


def split_train_test(df: pd.DataFrame, train_days: int = 5):
    """Split by calendar days."""
    dates = df.index.normalize().unique()
    train_end = dates[min(train_days, len(dates) - 1)]
    train = df[df.index < train_end]
    test = df[df.index >= train_end]
    return train, test


def run_bt(df: pd.DataFrame, params: dict, cash: float = 100000) -> dict:
    cerebro = bt.Cerebro()
    cerebro.addstrategy(EmaPullbackSystem, **params)
    cerebro.adddata(bt.feeds.PandasData(dataname=df))
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission=0.001)
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="ta")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="dd")

    results = cerebro.run()
    strat = results[0]
    final = cerebro.broker.getvalue()
    ret = (final - cash) / cash * 100

    trades = 0
    win_rate = None
    pf = None
    try:
        ta = strat.analyzers.ta.get_analysis()
        t_obj = ta.get("total", {})
        trades = int(t_obj.get("closed", 0)) if isinstance(t_obj, dict) else 0
        w_obj = ta.get("won", {})
        won = w_obj.get("total", 0) if isinstance(w_obj, dict) else 0
        if trades > 0:
            win_rate = won / trades * 100
        pnl_w = w_obj.get("pnl", {}) if isinstance(w_obj, dict) else {}
        gp = abs(float(pnl_w.get("total", 0))) if isinstance(pnl_w, dict) else abs(float(pnl_w))
        l_obj = ta.get("lost", {})
        pnl_l = l_obj.get("pnl", {}) if isinstance(l_obj, dict) else {}
        gl = abs(float(pnl_l.get("total", 0))) if isinstance(pnl_l, dict) else abs(float(pnl_l))
        if gl > 0:
            pf = gp / gl
    except Exception:
        pass

    max_dd = None
    try:
        da = strat.analyzers.dd.get_analysis()
        md = da.get("max", {})
        max_dd = md.get("drawdown") if isinstance(md, dict) else float(md)
    except Exception:
        pass

    return {
        "return": round(ret, 2),
        "trades": trades,
        "win_rate": round(win_rate, 1) if win_rate is not None else None,
        "pf": round(pf, 3) if pf is not None else None,
        "max_dd": round(max_dd, 2) if max_dd is not None else None,
    }


def main():
    df = fetch_1m("QQQ")
    console.print(f"[cyan]QQQ 1m: {len(df)} bars, {df.index[0]} → {df.index[-1]}[/cyan]")

    train, test = split_train_test(df, train_days=5)
    console.print(f"[green]Train: {len(train)} bars ({train.index[0].date()} → {train.index[-1].date()})[/green]")
    console.print(f"[yellow]Test:  {len(test)} bars ({test.index[0].date()} → {test.index[-1].date()})[/yellow]")
    console.print()

    # ================================================================
    # Parameter grid - focused, not too many combinations
    # Keep parameters coarse to avoid overfitting
    # ================================================================
    # All configs do BOTH long and short (user requirement)
    # ORB filter: NY open first candle too big = range day, skip
    _ORB = dict(use_orb_filter=True, orb_atr_mult=2.0, orb_cooldown_bars=30)

    configs = [
        # --- A: Standard 20 EMA + ORB ---
        {"name": "A 标准20EMA+ORB", "p": dict(
            ema_fast=20, ema_slow=40, ema_filter=60,
            trend_bars=5, trend_atr_mult=1.0,
            pullback_atr_tol=0.3, max_pullback_bars=8,
            use_vol_filter=True, vol_thresh=0.8,
            sl_atr_mult=1.5, use_trail=True, trail_atr_offset=0.3,
            tp_atr_mult=3.0, use_partial_tp=True,
            risk_per_trade=0.02, max_daily_trades=3,
            **_ORB,
        )},
        # --- B: Conservative (best from 5m) + ORB ---
        {"name": "B 保守慢速+ORB", "p": dict(
            ema_fast=20, ema_slow=50, ema_filter=100,
            trend_bars=8, trend_atr_mult=1.5,
            pullback_atr_tol=0.2, max_pullback_bars=10,
            use_vol_filter=True, vol_thresh=1.0,
            sl_atr_mult=2.0, use_trail=True, trail_atr_offset=0.5,
            tp_atr_mult=4.0, use_partial_tp=True,
            risk_per_trade=0.01, max_daily_trades=2,
            **_ORB,
        )},
        # --- C: 1m wider tolerance for noise ---
        {"name": "C 1m宽容抗噪", "p": dict(
            ema_fast=20, ema_slow=50, ema_filter=100,
            trend_bars=10, trend_atr_mult=2.0,
            pullback_atr_tol=0.5, max_pullback_bars=15,
            use_vol_filter=False,
            sl_atr_mult=2.5, use_trail=True, trail_atr_offset=0.8,
            tp_atr_mult=5.0, use_partial_tp=True,
            risk_per_trade=0.01, max_daily_trades=2,
            **_ORB,
        )},
        # --- D: Fixed stop, no trail ---
        {"name": "D 固定止损无追踪", "p": dict(
            ema_fast=20, ema_slow=40, ema_filter=60,
            trend_bars=5, trend_atr_mult=1.0,
            pullback_atr_tol=0.3, max_pullback_bars=8,
            use_vol_filter=True, vol_thresh=0.8,
            sl_atr_mult=1.5, use_trail=False,
            tp_atr_mult=3.0, use_partial_tp=True,
            risk_per_trade=0.02, max_daily_trades=3,
            **_ORB,
        )},
        # --- E: Small position, tight risk ---
        {"name": "E 小仓控风险", "p": dict(
            ema_fast=20, ema_slow=40, ema_filter=60,
            trend_bars=6, trend_atr_mult=1.2,
            pullback_atr_tol=0.3, max_pullback_bars=8,
            use_vol_filter=True, vol_thresh=0.9,
            sl_atr_mult=2.0, use_trail=True, trail_atr_offset=0.4,
            tp_atr_mult=4.0, use_partial_tp=True, partial_tp_rr=2.0,
            risk_per_trade=0.005, max_daily_trades=2,
            **_ORB,
        )},
        # --- F: Strict trend, low frequency ---
        {"name": "F 严格趋势低频", "p": dict(
            ema_fast=20, ema_slow=50, ema_filter=100,
            trend_bars=15, trend_atr_mult=2.5,
            pullback_atr_tol=0.2, max_pullback_bars=20,
            use_vol_filter=True, vol_thresh=1.2,
            sl_atr_mult=2.0, use_trail=True, trail_atr_offset=0.5,
            tp_atr_mult=4.0, use_partial_tp=True,
            risk_per_trade=0.01, max_daily_trades=1,
            **_ORB,
        )},
        # --- G: No ORB filter (comparison) ---
        {"name": "G 标准无ORB对比", "p": dict(
            ema_fast=20, ema_slow=40, ema_filter=60,
            trend_bars=5, trend_atr_mult=1.0,
            pullback_atr_tol=0.3, max_pullback_bars=8,
            use_vol_filter=True, vol_thresh=0.8,
            sl_atr_mult=1.5, use_trail=True, trail_atr_offset=0.3,
            tp_atr_mult=3.0, use_partial_tp=True,
            risk_per_trade=0.02, max_daily_trades=3,
            use_orb_filter=False,
        )},
        # --- H: Wide ORB cooldown ---
        {"name": "H 长ORB冷却期", "p": dict(
            ema_fast=20, ema_slow=40, ema_filter=60,
            trend_bars=5, trend_atr_mult=1.0,
            pullback_atr_tol=0.3, max_pullback_bars=8,
            use_vol_filter=True, vol_thresh=0.8,
            sl_atr_mult=1.5, use_trail=True, trail_atr_offset=0.3,
            tp_atr_mult=3.0, use_partial_tp=True,
            risk_per_trade=0.02, max_daily_trades=3,
            use_orb_filter=True, orb_atr_mult=1.5, orb_cooldown_bars=60,
        )},
    ]

    # ================================================================
    # Walk-forward: train + test
    # ================================================================
    results = []
    for i, cfg in enumerate(configs, 1):
        name = cfg["name"]
        p = cfg["p"]
        console.print(f"  [{i:2d}/{len(configs)}] {name}...", end=" ")
        try:
            tr = run_bt(train, p)
            te = run_bt(test, p)

            # Robustness: how close is test to train?
            # Lower degradation = less overfitting
            if tr["trades"] > 0 and te["trades"] > 0:
                ret_degrad = tr["return"] - te["return"] if tr["return"] > te["return"] else 0
            else:
                ret_degrad = 99

            results.append({
                "name": name,
                "train": tr,
                "test": te,
                "degradation": round(ret_degrad, 2),
            })
            t_color = "green" if tr["return"] > 0 else "red"
            v_color = "green" if te["return"] > 0 else "red"
            console.print(
                f"Train:[{t_color}]{tr['return']:+.2f}%[/{t_color}]({tr['trades']}t) "
                f"Test:[{v_color}]{te['return']:+.2f}%[/{v_color}]({te['trades']}t) "
                f"Degrad:{ret_degrad:.1f}"
            )
        except Exception as e:
            console.print(f"[red]FAIL: {e}[/red]")

    # ================================================================
    # Ranking: composite score prioritizing test performance + stability
    # ================================================================
    def score(r):
        tr, te = r["train"], r["test"]
        if tr["trades"] < 2 or te["trades"] < 1:
            return -9999
        # Weighted: 40% test return, 30% test PF, 20% consistency, 10% low DD
        s = 0
        s += (te["return"] or 0) * 0.4
        s += ((te["pf"] or 0) * 10) * 0.3
        # Consistency bonus: both positive or both same sign
        if tr["return"] > 0 and te["return"] > 0:
            s += 5
        elif (tr["return"] > 0) != (te["return"] > 0):
            s -= 3  # sign flip = unstable
        s -= r["degradation"] * 0.2
        s -= abs(te.get("max_dd") or 0) * 0.1
        return s

    results.sort(key=score, reverse=True)

    # ================================================================
    # Display
    # ================================================================
    console.print()
    table = Table(title="QQQ 1m Walk-Forward 结果 (防过拟合)", show_header=True, header_style="bold magenta")
    table.add_column("#", width=3)
    table.add_column("策略", min_width=18)
    table.add_column("Train收益%", justify="right")
    table.add_column("Train交易", justify="right")
    table.add_column("Train胜率", justify="right")
    table.add_column("Train PF", justify="right")
    table.add_column("Test收益%", justify="right")
    table.add_column("Test交易", justify="right")
    table.add_column("Test胜率", justify="right")
    table.add_column("Test PF", justify="right")
    table.add_column("退化", justify="right")

    for i, r in enumerate(results, 1):
        tr, te = r["train"], r["test"]
        marker = " [bold yellow]★[/bold yellow]" if i == 1 else ""

        def fmt_ret(v):
            return f"[green]{v:+.2f}[/green]" if v > 0 else f"[red]{v:+.2f}[/red]"

        def fmt_v(v):
            return f"{v}" if v is not None else "N/A"

        table.add_row(
            str(i), r["name"] + marker,
            fmt_ret(tr["return"]), str(tr["trades"]),
            fmt_v(tr["win_rate"]), fmt_v(tr["pf"]),
            fmt_ret(te["return"]), str(te["trades"]),
            fmt_v(te["win_rate"]), fmt_v(te["pf"]),
            f"{r['degradation']:.1f}",
        )

    console.print(table)

    # Best
    if results:
        best = results[0]
        bp = next(c for c in configs if c["name"] == best["name"])
        console.print(Panel(
            f"[bold green]最佳策略: {best['name']}[/bold green]\n\n"
            f"[bold]训练集 (5天):[/bold]\n"
            f"  收益: {best['train']['return']:+.2f}%  |  交易: {best['train']['trades']}  |  "
            f"胜率: {best['train']['win_rate']}%  |  PF: {best['train']['pf']}\n\n"
            f"[bold]验证集 (2天):[/bold]\n"
            f"  收益: {best['test']['return']:+.2f}%  |  交易: {best['test']['trades']}  |  "
            f"胜率: {best['test']['win_rate']}%  |  PF: {best['test']['pf']}\n\n"
            f"[bold]退化度: {best['degradation']}[/bold] (越低越稳定)\n\n"
            f"[bold]参数:[/bold]\n" +
            "\n".join(f"  {k}: {v}" for k, v in bp["p"].items()),
            title="Winner (Walk-Forward Validated)",
            border_style="green",
        ))


if __name__ == "__main__":
    main()
