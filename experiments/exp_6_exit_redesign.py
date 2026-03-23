"""
Experiment 6: Fix trade count + redesign exits.

Problem: 64 trades / 2 years = 0.13/day. Need ~1/day = 500 trades.
Approach: Loosen entry (chop threshold), then sweep exit params to find PF>1.0.

Uses ALL CPUs.
"""
from __future__ import annotations
import importlib, sys, time, os
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

import pandas as pd
import numpy as np
import backtrader as bt

CLEAN_PATH = "data/QQQ_1Min_2y_clean.csv"
CASH = 100_000
COMMISSION = 0.001
NCPU = cpu_count()


def _run_single(args):
    name, overrides, data_path = args
    df = pd.read_csv(data_path, index_col="timestamp", parse_dates=True)
    if "strategy" in sys.modules:
        del sys.modules["strategy"]
    mod = importlib.import_module("strategy")
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
    }
    params = {pmap[k]: v for k, v in overrides.items() if k in pmap}
    cerebro = bt.Cerebro()
    data = bt.feeds.PandasData(dataname=df, datetime=None, open="Open", high="High",
                                low="Low", close="Close", volume="Volume", openinterest=-1)
    cerebro.adddata(data)
    cerebro.addstrategy(StrategyClass, **params)
    cerebro.broker.setcash(CASH)
    cerebro.broker.setcommission(commission=COMMISSION)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe",
                        timeframe=bt.TimeFrame.Days, riskfreerate=0.05)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    results = cerebro.run()
    strat = results[0]
    fv = cerebro.broker.getvalue()
    ret = (fv - CASH) / CASH * 100
    sa = strat.analyzers.sharpe.get_analysis()
    sharpe = sa.get("sharperatio", 0.0) or 0.0
    dd = strat.analyzers.drawdown.get_analysis()
    max_dd = dd.get("max", {}).get("drawdown", 0.0) or 0.0
    ta = strat.analyzers.trades.get_analysis()
    tt = ta.get("total", {}).get("total", 0)
    w = ta.get("won", {}).get("total", 0)
    l = ta.get("lost", {}).get("total", 0)
    wr = w / tt * 100 if tt > 0 else 0
    gw = ta.get("won", {}).get("pnl", {}).get("total", 0.0) or 0.0
    gl = abs(ta.get("lost", {}).get("pnl", {}).get("total", 0.0) or 0.0)
    pf = gw / gl if gl > 0 else 0
    return {"name": name, "ret": round(ret, 2), "sharpe": round(sharpe, 3),
            "dd": round(max_dd, 2), "trades": tt, "won": w, "lost": l,
            "wr": round(wr, 1), "pf": round(pf, 3), "overrides": overrides,
            "tpd": round(tt / 499, 2)}  # trades per day


def main():
    data_path = os.path.abspath(CLEAN_PATH)
    t0 = time.time()

    exps = {}

    # ═══════════════════════════════════════════════════════════
    # PHASE 1: Map entry looseness → trade count
    # Goal: find chop thresholds that give 200-600 trades
    # ═══════════════════════════════════════════════════════════
    for ct in [0.003, 0.005, 0.006, 0.007, 0.008, 0.009, 0.010]:
        for pb in [1.0, 1.2, 1.5]:
            exps[f"entry_c{ct}_pb{pb}"] = {
                "CHOP_SLOPE_THRESHOLD": ct, "PULLBACK_TOUCH_MULT": pb,
                "INITIAL_STOP_ATR_MULT": 5.0,
                "TP_ACTIVATE_ATR": 100.0, "ENABLE_ADDON": False,
                "LOSERS_MAX_BARS": 60,
            }

    # ═══════════════════════════════════════════════════════════
    # PHASE 2: Fix exit with ~200-trade entry configs
    # Use chop=0.005-0.008 as base, sweep exit params
    # ═══════════════════════════════════════════════════════════
    for ct in [0.005, 0.007, 0.008]:
        BASE = {"CHOP_SLOPE_THRESHOLD": ct, "PULLBACK_TOUCH_MULT": 1.2,
                "TP_ACTIVATE_ATR": 100.0, "ENABLE_ADDON": False}

        # A: Initial stop sweep (the #1 lever from 2y data)
        for stop in [3.0, 5.0, 7.0, 10.0, 15.0]:
            exps[f"c{ct}_stop{stop}"] = {**BASE, "INITIAL_STOP_ATR_MULT": stop}

        # B: Losers max bars sweep (kill losers earlier/later)
        for lb in [15, 20, 30, 45, 60, 90, 120]:
            exps[f"c{ct}_los{lb}"] = {**BASE, "LOSERS_MAX_BARS": lb, "INITIAL_STOP_ATR_MULT": 7.0}

        # C: EMA trail width (TP3 = the main exit in no-TP mode)
        # Currently TP3_EMA_ATR_MULT=8.0 is the trail when TP is off
        # Actually when TP_ACTIVATE=100, the only exit is initial_stop + losers_max + force_close
        # We need the EMA trail ALWAYS active, not just after TP activation
        # For now, test the stop + losers combos

        # D: Combined stop + losers
        for stop in [5.0, 7.0, 10.0]:
            for lb in [20, 30, 45, 60]:
                exps[f"c{ct}_s{stop}_l{lb}"] = {
                    **BASE, "INITIAL_STOP_ATR_MULT": stop, "LOSERS_MAX_BARS": lb,
                }

    # ═══════════════════════════════════════════════════════════
    # PHASE 3: TP system with loose entry
    # Test if TP helps when we have more trades
    # ═══════════════════════════════════════════════════════════
    for ct in [0.005, 0.007]:
        for act in [3.0, 5.0, 8.0]:
            for tp2, tp3 in [(3.0, 6.0), (5.0, 8.0), (5.0, 10.0)]:
                exps[f"c{ct}_tp_a{act}_t{tp2}_{tp3}"] = {
                    "CHOP_SLOPE_THRESHOLD": ct, "PULLBACK_TOUCH_MULT": 1.2,
                    "INITIAL_STOP_ATR_MULT": 7.0,
                    "TP_ACTIVATE_ATR": act, "TP1_CANDLE_OFFSET": 1.0,
                    "TP2_EMA_ATR_MULT": tp2, "TP3_EMA_ATR_MULT": tp3,
                    "ENABLE_ADDON": False, "LOSERS_MAX_BARS": 45,
                }

    # ═══════════════════════════════════════════════════════════
    # PHASE 4: TP ratio experiments with loose entry
    # ═══════════════════════════════════════════════════════════
    for ct in [0.005, 0.007]:
        for t1, t2, t3 in [(0.10, 0.20, 0.70), (0.0, 0.0, 1.0), (0.30, 0.40, 0.30)]:
            exps[f"c{ct}_r{int(t1*100)}_{int(t2*100)}_{int(t3*100)}"] = {
                "CHOP_SLOPE_THRESHOLD": ct, "PULLBACK_TOUCH_MULT": 1.2,
                "INITIAL_STOP_ATR_MULT": 7.0,
                "TP_ACTIVATE_ATR": 5.0, "TP1_PCT": t1, "TP2_PCT": t2, "TP3_PCT": t3,
                "TP1_CANDLE_OFFSET": 1.0, "TP2_EMA_ATR_MULT": 5.0, "TP3_EMA_ATR_MULT": 8.0,
                "ENABLE_ADDON": False, "LOSERS_MAX_BARS": 45,
            }

    # ═══════════════════════════════════════════════════════════
    # PHASE 5: Addon with loose entry
    # ═══════════════════════════════════════════════════════════
    for ct in [0.005, 0.007]:
        exps[f"c{ct}_addon1"] = {
            "CHOP_SLOPE_THRESHOLD": ct, "PULLBACK_TOUCH_MULT": 1.2,
            "INITIAL_STOP_ATR_MULT": 7.0,
            "TP_ACTIVATE_ATR": 100.0, "ENABLE_ADDON": True, "MAX_ADDONS": 1,
            "LOSERS_MAX_BARS": 45,
        }
        exps[f"c{ct}_full"] = {
            "CHOP_SLOPE_THRESHOLD": ct, "PULLBACK_TOUCH_MULT": 1.2,
            "INITIAL_STOP_ATR_MULT": 7.0,
            "TP_ACTIVATE_ATR": 5.0, "TP1_CANDLE_OFFSET": 1.0,
            "TP1_PCT": 0.10, "TP2_PCT": 0.20, "TP3_PCT": 0.70,
            "TP2_EMA_ATR_MULT": 5.0, "TP3_EMA_ATR_MULT": 8.0,
            "ENABLE_ADDON": True, "MAX_ADDONS": 1,
            "LOSERS_MAX_BARS": 45,
        }

    # ═══════════════════════════════════════════════════════════
    # PHASE 6: Daily trade limit relaxation
    # ═══════════════════════════════════════════════════════════
    for ct in [0.005, 0.007]:
        for dt in [3, 6, 10, 20]:
            exps[f"c{ct}_daily{dt}"] = {
                "CHOP_SLOPE_THRESHOLD": ct, "PULLBACK_TOUCH_MULT": 1.2,
                "INITIAL_STOP_ATR_MULT": 7.0,
                "TP_ACTIVATE_ATR": 100.0, "ENABLE_ADDON": False,
                "MAX_DAILY_TRADES": dt, "LOSERS_MAX_BARS": 45,
            }

    # ═══════════════════════════════════════════════════════════
    # PHASE 7: Slope period with loose chop
    # ═══════════════════════════════════════════════════════════
    for sp in [3, 5, 8]:
        for ct in [0.005, 0.007]:
            exps[f"c{ct}_sp{sp}"] = {
                "CHOP_SLOPE_THRESHOLD": ct, "EMA_SLOPE_PERIOD": sp,
                "PULLBACK_TOUCH_MULT": 1.2,
                "INITIAL_STOP_ATR_MULT": 7.0,
                "TP_ACTIVATE_ATR": 100.0, "ENABLE_ADDON": False,
                "LOSERS_MAX_BARS": 45,
            }

    # RUN
    print(f"{len(exps)} experiments × {NCPU} CPUs")
    tasks = [(n, o, data_path) for n, o in exps.items()]
    results = []
    done = 0

    with ProcessPoolExecutor(max_workers=NCPU) as pool:
        futures = {pool.submit(_run_single, t): t[0] for t in tasks}
        for f in as_completed(futures):
            done += 1
            try:
                r = f.result()
                results.append(r)
                m = " ***" if r["pf"] >= 1.0 else ""
                if done % 20 == 0 or r["pf"] >= 1.0:
                    print(f"[{done}/{len(tasks)}] {r['name']:<35} PF={r['pf']:>6.3f} "
                          f"Ret={r['ret']:>6.2f}% T={r['trades']:>4} "
                          f"({r['tpd']:.1f}/d) WR={r['wr']:.0f}% DD={r['dd']:.1f}%{m}")
            except Exception as e:
                print(f"[{done}/{len(tasks)}] {futures[f]:<35} ERR: {str(e)[:50]}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")

    # ═══ ANALYSIS ═══
    print(f"\n{'='*120}")

    # Group by trade count ranges
    for label, lo, hi in [("< 100 trades", 0, 100), ("100-300", 100, 300),
                           ("300-600", 300, 600), ("> 600", 600, 9999)]:
        group = [r for r in results if lo <= r["trades"] < hi]
        profitable = [r for r in group if r["pf"] >= 1.0]
        print(f"\n{label}: {len(group)} configs, {len(profitable)} profitable")
        if profitable:
            for r in sorted(profitable, key=lambda x: x["pf"], reverse=True)[:5]:
                print(f"  {r['name']:<35} PF={r['pf']:.3f} Ret={r['ret']:.2f}% "
                      f"T={r['trades']} ({r['tpd']:.1f}/d) WR={r['wr']:.0f}% DD={r['dd']:.1f}%")

    # Best configs with 200+ trades
    print(f"\n{'='*120}")
    print("TOP 20 BY PF (with 100+ trades for statistical significance)")
    enough = sorted([r for r in results if r["trades"] >= 100], key=lambda x: x["pf"], reverse=True)
    for i, r in enumerate(enough[:20]):
        m = " ***" if r["pf"] >= 1.0 else ""
        print(f"  #{i+1}: {r['name']:<35} PF={r['pf']:.3f} Ret={r['ret']:.2f}% "
              f"T={r['trades']} ({r['tpd']:.1f}/d) WR={r['wr']:.0f}% DD={r['dd']:.1f}%{m}")
        if r["pf"] >= 1.0:
            kp = {k: v for k, v in r["overrides"].items()
                  if k not in ("EMA_PERIOD", "EMA_SLOW_PERIOD", "ATR_PERIOD")}
            print(f"       {kp}")

    # Trade count mapping
    print(f"\n{'='*120}")
    print("ENTRY CONFIG → TRADE COUNT (for reference)")
    entry_map = sorted([r for r in results if r["name"].startswith("entry_")],
                       key=lambda x: x["trades"])
    for r in entry_map:
        print(f"  {r['name']:<35} T={r['trades']:>4} ({r['tpd']:.1f}/d) PF={r['pf']:.3f}")


if __name__ == "__main__":
    main()
