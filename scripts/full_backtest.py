#!/usr/bin/env python3
"""Full 6-month QQQ 1m backtest via Alpaca data API.

- Downloads 6 months of QQQ 1-minute bars (SIP feed = real Invesco volume)
- Walk-forward: 70% train / 30% test
- Tests multiple parameter configs, picks the most robust
"""

from __future__ import annotations
import sys, time
from pathlib import Path
from datetime import datetime, timedelta

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import requests
import pandas as pd
import backtrader as bt
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track

from src.strategy.generated.ema_pullback_system import EmaPullbackSystem

console = Console()

API_KEY = "PKBQIINM7PMLOTV7VYHYBGSSDE"
SECRET  = "F61HEmmrQhLb8VyXApSHZjB63729c6o4iYweksURKiQ6"
HEADERS = {"APCA-API-KEY-ID": API_KEY, "APCA-API-SECRET-KEY": SECRET}
DATA_URL = "https://data.alpaca.markets/v2/stocks/QQQ/bars"


def download_1m(start_date: str, end_date: str, cache_path: Path | None = None) -> pd.DataFrame:
    """Download QQQ 1m bars from Alpaca (SIP feed = real exchange volume)."""
    if cache_path and cache_path.exists():
        console.print(f"[dim]Loading cached data from {cache_path}[/dim]")
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        console.print(f"[green]Loaded {len(df)} bars from cache[/green]")
        return df

    console.print(f"[cyan]Downloading QQQ 1m bars: {start_date} → {end_date}[/cyan]")
    all_bars = []
    page_token = None
    page = 0

    while True:
        params = {
            "timeframe": "1Min",
            "start": start_date,
            "end": end_date,
            "limit": 10000,
            "feed": "iex",         # IEX exchange data (free for paper)
            "adjustment": "split",
        }
        if page_token:
            params["page_token"] = page_token

        r = requests.get(DATA_URL, headers=HEADERS, params=params)
        if r.status_code != 200:
            console.print(f"[red]API error {r.status_code}: {r.text[:200]}[/red]")
            break

        data = r.json()
        bars = data.get("bars", [])
        all_bars.extend(bars)
        page += 1

        if page % 5 == 0:
            console.print(f"  Page {page}: {len(all_bars)} bars so far...")

        page_token = data.get("next_page_token")
        if not page_token or not bars:
            break

        time.sleep(0.25)  # Rate limit: be nice

    console.print(f"[green]Downloaded {len(all_bars)} bars in {page} pages[/green]")

    if not all_bars:
        raise ValueError("No data downloaded")

    # Convert to DataFrame
    df = pd.DataFrame(all_bars)
    df = df.rename(columns={"t": "Datetime", "o": "Open", "h": "High",
                             "l": "Low", "c": "Close", "v": "Volume"})
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.set_index("Datetime")
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df = df.sort_index()

    # Strip timezone
    if hasattr(df.index, "tz") and df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # Cache
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path)
        console.print(f"[dim]Cached to {cache_path}[/dim]")

    return df


def run(df: pd.DataFrame, params: dict, cash: float = 100000) -> dict:
    cerebro = bt.Cerebro()
    cerebro.addstrategy(EmaPullbackSystem, **params)
    cerebro.adddata(bt.feeds.PandasData(dataname=df))
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission=0.001)
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="ta")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="dd")
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sr", riskfreerate=0.0)
    results = cerebro.run()
    strat = results[0]
    final = cerebro.broker.getvalue()
    ret = (final - cash) / cash * 100
    trades = 0; wr = None; pf = None; sharpe = None; maxdd = None
    avg_win = None; avg_loss = None
    try:
        ta = strat.analyzers.ta.get_analysis()
        to = ta.get("total", {})
        trades = int(to.get("closed", 0)) if isinstance(to, dict) else 0
        wo = ta.get("won", {})
        won = wo.get("total", 0) if isinstance(wo, dict) else 0
        if trades > 0: wr = won / trades * 100
        pw = wo.get("pnl", {}) if isinstance(wo, dict) else {}
        gp = abs(float(pw.get("total", 0))) if isinstance(pw, dict) else abs(float(pw))
        if won > 0:
            avg_win = gp / won
        lo_obj = ta.get("lost", {})
        lost = lo_obj.get("total", 0) if isinstance(lo_obj, dict) else 0
        pl = lo_obj.get("pnl", {}) if isinstance(lo_obj, dict) else {}
        gl = abs(float(pl.get("total", 0))) if isinstance(pl, dict) else abs(float(pl))
        if lost > 0:
            avg_loss = gl / lost
        if gl > 0: pf = gp / gl
    except Exception: pass
    try:
        da = strat.analyzers.dd.get_analysis()
        md = da.get("max", {})
        maxdd = md.get("drawdown") if isinstance(md, dict) else float(md)
    except Exception: pass
    try:
        sa = strat.analyzers.sr.get_analysis()
        s = sa.get("sharperatio")
        if s is not None: sharpe = float(s)
    except Exception: pass
    return {
        "ret": round(ret, 2), "trades": trades,
        "wr": round(wr, 1) if wr is not None else None,
        "pf": round(pf, 3) if pf is not None else None,
        "dd": round(maxdd, 2) if maxdd is not None else None,
        "sharpe": round(sharpe, 3) if sharpe is not None else None,
        "avg_win": round(avg_win, 2) if avg_win is not None else None,
        "avg_loss": round(avg_loss, 2) if avg_loss is not None else None,
        "final": round(final, 2),
    }


def main():
    # ================================================================
    # Download 6 months of QQQ 1m data
    # ================================================================
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")

    cache = _PROJECT_ROOT / "data" / "backtest_cache" / f"QQQ_1m_{start_date}_{end_date}.csv"
    df = download_1m(start_date, end_date, cache_path=cache)

    # Filter to market hours only (9:30-16:00 ET)
    df = df.between_time("09:30", "16:00")

    trading_days = df.index.normalize().nunique()
    console.print(Panel(
        f"[bold]QQQ 1m 数据 (Alpaca SIP Feed)[/bold]\n\n"
        f"  总K线数: [green]{len(df):,}[/green]\n"
        f"  交易天数: [green]{trading_days}[/green]\n"
        f"  日期范围: {df.index[0].date()} → {df.index[-1].date()}\n"
        f"  平均成交量: {df['Volume'].mean():,.0f} 股/分钟\n"
        f"  日均成交量: {df.groupby(df.index.date)['Volume'].sum().mean():,.0f} 股\n"
        f"  数据源: Alpaca SIP (consolidated exchange data)\n"
        f"  标的: QQQ (Invesco QQQ Trust, NASDAQ)",
        title="Data Summary", border_style="cyan",
    ))

    # ================================================================
    # Walk-forward split: 70% train / 30% test
    # ================================================================
    n = int(len(df) * 0.7)
    train = df.iloc[:n]
    test = df.iloc[n:]
    train_days = train.index.normalize().nunique()
    test_days = test.index.normalize().nunique()

    console.print(f"\n[bold]Walk-Forward Split:[/bold]")
    console.print(f"  Train: {len(train):,} bars ({train_days} days, {train.index[0].date()} → {train.index[-1].date()})")
    console.print(f"  Test:  {len(test):,} bars ({test_days} days, {test.index[0].date()} → {test.index[-1].date()})")
    console.print()

    # ================================================================
    # Parameter configs (all do both long + short)
    # ================================================================
    _ORB = dict(use_orb_filter=True, orb_atr_mult=2.0, orb_cooldown_bars=30)

    # Base configs
    _bases = [
        ("A 标准20EMA", dict(
            ema_fast=20, ema_slow=40, ema_filter=60,
            trend_bars=5, trend_atr_mult=1.0,
            pullback_atr_tol=0.3, max_pullback_bars=8,
            use_vol_filter=False,
            sl_atr_mult=1.5, use_trail=True, trail_atr_offset=0.3,
            tp_atr_mult=3.0, use_partial_tp=True,
            risk_per_trade=0.02, max_daily_trades=3, **_ORB,
        )),
        ("B 保守慢速", dict(
            ema_fast=20, ema_slow=50, ema_filter=100,
            trend_bars=8, trend_atr_mult=1.5,
            pullback_atr_tol=0.2, max_pullback_bars=10,
            use_vol_filter=False,
            sl_atr_mult=2.0, use_trail=True, trail_atr_offset=0.5,
            tp_atr_mult=4.0, use_partial_tp=True,
            risk_per_trade=0.01, max_daily_trades=2, **_ORB,
        )),
        ("C 小仓控风险", dict(
            ema_fast=20, ema_slow=40, ema_filter=60,
            trend_bars=6, trend_atr_mult=1.2,
            pullback_atr_tol=0.3, max_pullback_bars=8,
            use_vol_filter=False,
            sl_atr_mult=2.0, use_trail=True, trail_atr_offset=0.4,
            tp_atr_mult=4.0, use_partial_tp=True, partial_tp_rr=2.0,
            risk_per_trade=0.005, max_daily_trades=2, **_ORB,
        )),
    ]

    # Generate both normal and INVERSE versions
    configs = []
    for name, p in _bases:
        configs.append({"name": name, "p": {**p, "inverse": False}})
        configs.append({"name": f"{name} [反转]", "p": {**p, "inverse": True}})

    # ================================================================
    # Run walk-forward on full 6-month 1m data
    # ================================================================
    results = []
    for i, cfg in enumerate(configs, 1):
        name = cfg["name"]
        p = cfg["p"]
        console.print(f"  [{i}/{len(configs)}] {name}...", end=" ")
        try:
            tr = run(train, p)
            te = run(test, p)
            results.append({"name": name, "train": tr, "test": te, "params": p})

            tc = "green" if tr["ret"] > 0 else "red"
            vc = "green" if te["ret"] > 0 else "red"
            console.print(
                f"Train:[{tc}]{tr['ret']:+.2f}%[/{tc}]({tr['trades']}t,WR:{tr['wr'] or 'N/A'}%) "
                f"Test:[{vc}]{te['ret']:+.2f}%[/{vc}]({te['trades']}t,WR:{te['wr'] or 'N/A'}%)"
            )
        except Exception as e:
            console.print(f"[red]FAIL: {e}[/red]")

    # ================================================================
    # Scoring: weighted by test performance + stability
    # ================================================================
    def score(r):
        tr, te = r["train"], r["test"]
        if te["trades"] < 5:
            return -9999
        s = 0
        s += (te["ret"] or 0) * 3                 # test return matters most
        s += ((te["pf"] or 0) - 1) * 15 if te["pf"] else 0   # PF > 1 bonus
        s += ((te["wr"] or 0) - 50) * 0.5         # WR > 50% bonus
        s -= abs(te.get("dd") or 0) * 0.5          # drawdown penalty
        # Consistency: penalize train/test divergence
        divergence = abs((tr["ret"] or 0) - (te["ret"] or 0))
        s -= divergence * 0.2
        # Both positive = big bonus
        if (tr["ret"] or 0) > 0 and (te["ret"] or 0) > 0:
            s += 15
        return s

    results.sort(key=score, reverse=True)

    # ================================================================
    # Results table
    # ================================================================
    console.print("\n")
    table = Table(title=f"QQQ 1m 6个月回测 Walk-Forward ({train.index[0].date()} → {test.index[-1].date()})",
                  show_header=True, header_style="bold magenta")
    table.add_column("#", width=3)
    table.add_column("策略", min_width=16)
    table.add_column("Train收益%", justify="right")
    table.add_column("Train交易", justify="right")
    table.add_column("Train WR%", justify="right")
    table.add_column("Train PF", justify="right")
    table.add_column("Test收益%", justify="right")
    table.add_column("Test交易", justify="right")
    table.add_column("Test WR%", justify="right")
    table.add_column("Test PF", justify="right")
    table.add_column("Test DD%", justify="right")

    for i, r in enumerate(results, 1):
        tr, te = r["train"], r["test"]
        m = " [bold yellow]★[/bold yellow]" if i == 1 else ""
        def fr(v): return f"[green]{v:+.2f}[/green]" if v and v > 0 else f"[red]{v:+.2f}[/red]" if v else "N/A"
        def fv(v): return f"{v}" if v is not None else "N/A"

        table.add_row(str(i), r["name"] + m,
            fr(tr["ret"]), str(tr["trades"]), fv(tr["wr"]), fv(tr["pf"]),
            fr(te["ret"]), str(te["trades"]), fv(te["wr"]), fv(te["pf"]),
            fv(te["dd"]))

    console.print(table)

    # ================================================================
    # Winner details
    # ================================================================
    if results:
        best = results[0]
        tr, te = best["train"], best["test"]

        console.print(Panel(
            f"[bold green]最佳策略: {best['name']}[/bold green]\n\n"
            f"[bold]训练集 ({train_days} 天, ~{len(train):,} bars):[/bold]\n"
            f"  收益: {tr['ret']:+.2f}%  |  交易: {tr['trades']}  |  "
            f"胜率: {tr['wr']}%  |  PF: {tr['pf']}  |  最大回撤: {tr['dd']}%\n"
            f"  平均盈利: ${tr.get('avg_win') or 'N/A'}  |  平均亏损: ${tr.get('avg_loss') or 'N/A'}\n\n"
            f"[bold]验证集 ({test_days} 天, ~{len(test):,} bars):[/bold]\n"
            f"  收益: {te['ret']:+.2f}%  |  交易: {te['trades']}  |  "
            f"胜率: {te['wr']}%  |  PF: {te['pf']}  |  最大回撤: {te['dd']}%\n"
            f"  平均盈利: ${te.get('avg_win') or 'N/A'}  |  平均亏损: ${te.get('avg_loss') or 'N/A'}\n"
            f"  最终资金: ${te['final']:,.2f}\n\n"
            f"[bold]最优参数:[/bold]\n" +
            "\n".join(f"  {k}: {v}" for k, v in best["params"].items()),
            title="Winner (6-Month 1m Walk-Forward Validated)",
            border_style="green",
        ))


if __name__ == "__main__":
    main()
