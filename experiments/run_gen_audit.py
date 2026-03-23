"""
Generation audit runner: backtest + walk-forward + quarterly + bootstrap.
Runs all validation tests for a given strategy file on Polygon SIP data.
Uses all CPUs.
"""
from __future__ import annotations
import importlib, sys, os, time, shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from pathlib import Path

import pandas as pd
import numpy as np
import backtrader as bt

POLYGON_CLEAN = "data/QQQ_1Min_Polygon_2y_clean.csv"
IEX_CLEAN = "data/QQQ_1Min_2y_clean.csv"
CASH = 100_000
COMMISSION = 0.001
NCPU = cpu_count()


def _bt_run(data_path, strategy_file="strategy", overrides=None, cash=CASH):
    """Run backtest, return metrics dict."""
    df = pd.read_csv(data_path, index_col="timestamp", parse_dates=True)

    if "strategy" in sys.modules:
        del sys.modules["strategy"]
    # Copy strategy file to strategy.py if needed
    mod = importlib.import_module(strategy_file)
    if overrides:
        for k, v in overrides.items():
            setattr(mod, k, v)
    StrategyClass = mod.Strategy

    pmap = {
        "EMA_PERIOD": "ema_period", "EMA_SLOW_PERIOD": "ema_slow_period",
        "ATR_PERIOD": "atr_period", "EMA_SLOPE_PERIOD": "ema_slope_period",
        "CHOP_SLOPE_AVG_PERIOD": "chop_slope_avg_period",
        "CHOP_SLOPE_THRESHOLD": "chop_slope_threshold",
        "CHOP_BOX_MIN_BARS": "chop_box_min_bars",
        "PULLBACK_TOUCH_MULT": "pullback_touch_mult",
        "MIN_PULLBACK_BARS": "min_pullback_bars",
        "INITIAL_STOP_ATR_MULT": "initial_stop_atr_mult",
        "TP_ACTIVATE_ATR": "tp_activate_atr",
        "TP1_PCT": "tp1_pct", "TP2_PCT": "tp2_pct", "TP3_PCT": "tp3_pct",
        "TP1_CANDLE_OFFSET": "tp1_candle_offset",
        "TP2_EMA_ATR_MULT": "tp2_ema_atr_mult",
        "TP3_EMA_ATR_MULT": "tp3_ema_atr_mult",
        "ENABLE_ADDON": "enable_addon", "MAX_ADDONS": "max_addons",
        "ADDON_PULLBACK_MULT": "addon_pullback_mult",
        "ADDON_MIN_BARS": "addon_min_bars",
        "RISK_PCT": "risk_pct", "MAX_POSITION_PCT": "max_position_pct",
        "MAX_DAILY_TRADES": "max_daily_trades",
        "MAX_DAILY_LOSS_PCT": "max_daily_loss_pct",
        "LOSERS_MAX_BARS": "losers_max_bars",
        "TREND_FLICKER_BARS": "trend_flicker_bars",
    }
    params = {}
    if overrides:
        params = {pmap[k]: v for k, v in overrides.items() if k in pmap}

    cerebro = bt.Cerebro()
    data = bt.feeds.PandasData(dataname=df, datetime=None, open="Open", high="High",
                                low="Low", close="Close", volume="Volume", openinterest=-1)
    cerebro.adddata(data)
    cerebro.addstrategy(StrategyClass, **params)
    cerebro.broker.setcash(cash)
    # IBKR-style per-share commission: $0.005/share
    cerebro.broker.setcommission(commission=0.005, commtype=bt.CommInfoBase.COMM_FIXED)
    # Allow pyramiding (adding to positions)
    cerebro.addsizer(bt.sizers.FixedSize, stake=1)  # size controlled by strategy
    cerebro.broker.set_checksubmit(False)  # allow orders even with pending
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe",
                        timeframe=bt.TimeFrame.Days, riskfreerate=0.05)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="dd")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="ta")

    results = cerebro.run()
    strat = results[0]
    fv = cerebro.broker.getvalue()
    ret = (fv - cash) / cash * 100
    sa = strat.analyzers.sharpe.get_analysis()
    sharpe = sa.get("sharperatio", 0.0) or 0.0
    dd = strat.analyzers.dd.get_analysis()
    mdd = dd.get("max", {}).get("drawdown", 0.0) or 0.0
    ta = strat.analyzers.ta.get_analysis()
    tt = ta.get("total", {}).get("total", 0)
    w = ta.get("won", {}).get("total", 0)
    l = ta.get("lost", {}).get("total", 0)
    wr = w / tt * 100 if tt > 0 else 0
    gw = ta.get("won", {}).get("pnl", {}).get("total", 0.0) or 0.0
    gl = abs(ta.get("lost", {}).get("pnl", {}).get("total", 0.0) or 0.0)
    pf = gw / gl if gl > 0 else 0
    days = df.index.normalize().nunique()

    return {
        "ret": round(ret, 2), "sharpe": round(sharpe, 3), "mdd": round(mdd, 2),
        "trades": tt, "won": w, "lost": l, "wr": round(wr, 1),
        "pf": round(pf, 3), "gw": round(gw, 2), "gl": round(gl, 2),
        "tpd": round(tt / max(days, 1), 2),
        "avg_win": round(gw / w, 2) if w > 0 else 0,
        "avg_loss": round(gl / l, 2) if l > 0 else 0,
    }


def _wf_half(args):
    """Walk-forward half."""
    name, strategy_file, data_path, start, end = args
    df = pd.read_csv(data_path, index_col="timestamp", parse_dates=True)
    df = df[(df.index >= start) & (df.index < end)]
    if len(df) < 500:
        return {"name": name, "trades": 0, "pf": 0, "ret": 0, "wr": 0, "won": 0, "lost": 0}
    tmp = f"/tmp/wf_{name}_{os.getpid()}.csv"
    df.to_csv(tmp)
    try:
        r = _bt_run(tmp, strategy_file)
        r["name"] = name
        return r
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def run_full_audit(gen_name: str, strategy_file: str = "strategy"):
    """Run complete audit suite. Returns report string."""
    data_path = os.path.abspath(POLYGON_CLEAN)
    lines = []
    p = lambda s: lines.append(s)

    p(f"{'='*80}")
    p(f"AUDIT REPORT: {gen_name}")
    p(f"{'='*80}")
    p(f"Strategy file: {strategy_file}.py")
    p(f"Data: {POLYGON_CLEAN}")
    p("")

    # 1. Full 2Y backtest
    p("1. FULL 2-YEAR BACKTEST")
    p("-" * 40)
    r = _bt_run(data_path, strategy_file)
    p(f"  PF={r['pf']:.3f}  Return={r['ret']:.2f}%  Trades={r['trades']} ({r['tpd']:.2f}/day)")
    p(f"  WR={r['wr']:.1f}%  Sharpe={r['sharpe']:.3f}  MaxDD={r['mdd']:.2f}%")
    p(f"  Won={r['won']}  Lost={r['lost']}  GrossWon=${r['gw']:.0f}  GrossLost=${r['gl']:.0f}")
    if r['won'] > 0 and r['lost'] > 0:
        p(f"  AvgWin=${r['avg_win']:.0f}  AvgLoss=${r['avg_loss']:.0f}  R:R={r['avg_win']/r['avg_loss']:.1f}")
    full_result = r

    # 2. Buy & Hold comparison
    p("")
    p("2. BUY & HOLD COMPARISON")
    p("-" * 40)
    df = pd.read_csv(data_path, index_col="timestamp", parse_dates=True)
    bnh = (df["Close"].iloc[-1] - df["Close"].iloc[0]) / df["Close"].iloc[0] * 100
    p(f"  QQQ Buy&Hold: {bnh:.2f}%")
    p(f"  Strategy: {r['ret']:.2f}%")
    p(f"  {'BEATS' if r['ret'] > bnh else 'UNDERPERFORMS'} buy-and-hold")

    # 3. Walk-forward Y1 vs Y2
    p("")
    p("3. WALK-FORWARD (Year 1 vs Year 2)")
    p("-" * 40)
    mid = "2025-03-22"
    wf_tasks = [
        ("Y1", strategy_file, data_path, "2024-01-01", mid),
        ("Y2", strategy_file, data_path, mid, "2027-01-01"),
    ]
    with ProcessPoolExecutor(max_workers=NCPU) as pool:
        futures = {pool.submit(_wf_half, t): t[0] for t in wf_tasks}
        wf = {}
        for f in as_completed(futures):
            wr = f.result()
            wf[wr["name"]] = wr
    for name in ["Y1", "Y2"]:
        if name in wf:
            wr = wf[name]
            status = "OK" if wr["pf"] >= 1.0 else "FAIL"
            p(f"  {name}: PF={wr['pf']:.3f} Ret={wr['ret']:.2f}% Trades={wr['trades']} "
              f"({wr['won']}W/{wr['lost']}L) [{status}]")
    both_ok = all(wf.get(n, {}).get("pf", 0) >= 1.0 for n in ["Y1", "Y2"])
    p(f"  Walk-forward: {'PASS' if both_ok else 'FAIL'}")

    # 4. Quarterly breakdown
    p("")
    p("4. QUARTERLY BREAKDOWN")
    p("-" * 40)
    quarters = []
    for year in sorted(df.index.year.unique()):
        for q in range(1, 5):
            ms = (q-1)*3+1
            me = q*3+1 if q < 4 else 1
            ye = year if q < 4 else year+1
            qs = f"{year}-{ms:02d}-01"
            qe = f"{ye}-{me:02d}-01"
            qdf = df[(df.index >= qs) & (df.index < qe)]
            if len(qdf) >= 500:
                quarters.append((f"{year}Q{q}", strategy_file, data_path, qs, qe))

    pos_q = 0
    total_q = 0
    with ProcessPoolExecutor(max_workers=NCPU) as pool:
        futures = {pool.submit(_wf_half, t): t[0] for t in quarters}
        qr = []
        for f in as_completed(futures):
            qr.append(f.result())

    for r in sorted(qr, key=lambda x: x["name"]):
        if r["trades"] < 1:
            continue
        total_q += 1
        ok = r["pf"] >= 1.0
        if ok:
            pos_q += 1
        mark = "+" if ok else "-"
        p(f"  {mark} {r['name']}: PF={r['pf']:.3f} Ret={r['ret']:.2f}% "
          f"Trades={r['trades']} ({r['won']}W/{r['lost']}L)")
    p(f"  Profitable quarters: {pos_q}/{total_q} ({pos_q/total_q*100:.0f}%)" if total_q > 0 else "  No quarters")

    # 5. Bootstrap significance
    p("")
    p("5. STATISTICAL SIGNIFICANCE")
    p("-" * 40)
    if full_result["won"] > 0 and full_result["lost"] > 0:
        avg_w = full_result["avg_win"]
        avg_l = full_result["avg_loss"]
        n = full_result["trades"]
        be_wr = 1 / (1 + avg_w / avg_l)
        p(f"  Breakeven WR: {be_wr:.1%}  Actual WR: {full_result['wr']:.1f}%")

        wins = [avg_w] * full_result["won"]
        losses = [-avg_l] * full_result["lost"]
        all_pnls = np.array(wins + losses)
        np.random.seed(42)
        mc_pfs = []
        for _ in range(10000):
            shuf = np.random.choice(all_pnls, size=n, replace=True)
            w = shuf[shuf > 0].sum()
            l = abs(shuf[shuf < 0].sum())
            mc_pfs.append(w / l if l > 0 else 0)
        obs_pf = full_result["pf"]
        pval = sum(1 for x in mc_pfs if x >= obs_pf) / len(mc_pfs)
        p(f"  Observed PF: {obs_pf:.3f}")
        p(f"  MC mean PF: {np.mean(mc_pfs):.3f}  MC 95th: {np.percentile(mc_pfs, 95):.3f}")
        p(f"  p-value: {pval:.4f}")
        if pval < 0.05:
            p(f"  → SIGNIFICANT (p<0.05)")
        elif pval < 0.10:
            p(f"  → MARGINAL (p<0.10)")
        else:
            p(f"  → NOT SIGNIFICANT")
    else:
        p("  Insufficient trades for significance test")

    # 6. IEX comparison (data quality impact)
    if os.path.exists(IEX_CLEAN):
        p("")
        p("6. DATA QUALITY COMPARISON (IEX vs Polygon)")
        p("-" * 40)
        iex_r = _bt_run(os.path.abspath(IEX_CLEAN), strategy_file)
        p(f"  IEX:     PF={iex_r['pf']:.3f} Ret={iex_r['ret']:.2f}% Trades={iex_r['trades']}")
        p(f"  Polygon: PF={full_result['pf']:.3f} Ret={full_result['ret']:.2f}% Trades={full_result['trades']}")

    report = "\n".join(lines)
    # Save report
    Path("reports").mkdir(exist_ok=True)
    report_path = f"reports/{gen_name}_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(report)
    print(f"\nReport saved to {report_path}")
    return full_result


if __name__ == "__main__":
    gen = sys.argv[1] if len(sys.argv) > 1 else "gen0"
    strat = sys.argv[2] if len(sys.argv) > 2 else "strategy"
    run_full_audit(gen, strat)
