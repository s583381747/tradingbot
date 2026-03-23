"""
Clean 2-year data to NY market hours, run v3 backtest + parallel parameter sweep.
Uses ALL CPUs via multiprocessing.
"""
from __future__ import annotations
import importlib, sys, time, os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

import pandas as pd
import backtrader as bt

RAW_PATH = "data/QQQ_1Min_2024-03-22_2026-03-22.csv"
CLEAN_PATH = "data/QQQ_1Min_2y_clean.csv"
CASH = 100_000
COMMISSION = 0.001
NCPU = cpu_count()


# ═══════════════════════════════════════════════════════════
# DATA CLEANING
# ═══════════════════════════════════════════════════════════

def clean_data():
    if Path(CLEAN_PATH).exists():
        df = pd.read_csv(CLEAN_PATH, index_col="timestamp", parse_dates=True)
        print(f"Loaded clean data: {len(df)} bars")
        return df

    print("Cleaning raw data → NY market hours only...")
    df = pd.read_csv(RAW_PATH, index_col=0, parse_dates=True)
    df.index.name = "timestamp"

    # Raw data is UTC. Convert to US/Eastern.
    df.index = df.index.tz_localize("UTC").tz_convert("US/Eastern")

    # Filter to regular market hours: 9:30 - 15:59 ET
    t = df.index.time
    import datetime as dt
    mask = (t >= dt.time(9, 30)) & (t < dt.time(16, 0))
    df = df[mask].copy()

    # Strip timezone for backtrader compatibility
    df.index = df.index.tz_localize(None)

    df.to_csv(CLEAN_PATH)
    print(f"Clean data: {len(df)} bars, {df.index[0]} → {df.index[-1]}")
    print(f"Trading days: {df.index.normalize().nunique()}")
    return df


# ═══════════════════════════════════════════════════════════
# BACKTEST RUNNER (pickle-safe for multiprocessing)
# ═══════════════════════════════════════════════════════════

def _run_single(args):
    """Run one backtest config. Must be top-level for pickling."""
    name, overrides, data_path = args

    df = pd.read_csv(data_path, index_col="timestamp", parse_dates=True)

    if "strategy" in sys.modules:
        del sys.modules["strategy"]
    mod = importlib.import_module("strategy")

    for k, v in overrides.items():
        setattr(mod, k, v)

    StrategyClass = mod.Strategy
    param_name_map = {
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
    params = {param_name_map[k]: v for k, v in overrides.items() if k in param_name_map}

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

    final_value = cerebro.broker.getvalue()
    total_return = (final_value - CASH) / CASH * 100
    sharpe_a = strat.analyzers.sharpe.get_analysis()
    sharpe = sharpe_a.get("sharperatio", 0.0) or 0.0
    dd = strat.analyzers.drawdown.get_analysis()
    max_dd = dd.get("max", {}).get("drawdown", 0.0) or 0.0
    ta = strat.analyzers.trades.get_analysis()
    total_trades = ta.get("total", {}).get("total", 0)
    won = ta.get("won", {}).get("total", 0)
    lost = ta.get("lost", {}).get("total", 0)
    win_rate = won / total_trades * 100 if total_trades > 0 else 0.0
    gross_won = ta.get("won", {}).get("pnl", {}).get("total", 0.0) or 0.0
    gross_lost = abs(ta.get("lost", {}).get("pnl", {}).get("total", 0.0) or 0.0)
    pf = gross_won / gross_lost if gross_lost > 0 else 0.0

    return {
        "name": name,
        "total_return": round(total_return, 2),
        "sharpe": round(sharpe, 3),
        "max_drawdown": round(max_dd, 2),
        "total_trades": total_trades,
        "won": won, "lost": lost,
        "win_rate": round(win_rate, 1),
        "profit_factor": round(pf, 3),
        "overrides": overrides,
    }


def score(m):
    trades = m["total_trades"]
    if trades < 5: return -10.0
    sharpe_norm = max(0, min(1, (m["sharpe"] + 2) / 5))
    pf_norm = max(0, min(1, m["profit_factor"] / 3))
    ret_norm = max(0, min(1, (m["total_return"] + 20) / 70))
    wr_norm = max(0, min(1, m["win_rate"] / 100))
    dd_norm = max(0, min(1, 1 - m["max_drawdown"] / 30))
    raw = 0.30*sharpe_norm + 0.25*pf_norm + 0.20*ret_norm + 0.10*wr_norm + 0.15*dd_norm
    if trades < 20: mult = 0.2
    elif trades < 50: mult = 0.6
    elif trades <= 500: mult = 1.0
    else: mult = 0.8
    return round(raw * mult, 4)


# ═══════════════════════════════════════════════════════════
# EXPERIMENTS
# ═══════════════════════════════════════════════════════════

def build_experiments():
    BASE = {"EMA_PERIOD": 20, "EMA_SLOPE_PERIOD": 3}
    exps = {}

    # v3 defaults (current strategy.py)
    exps["v3_default"] = {}

    # No TP/no addon (pure base)
    exps["v3_noTP"] = {"TP_ACTIVATE_ATR": 100.0, "ENABLE_ADDON": False}

    # Chop threshold sweep
    for ct in [0.008, 0.010, 0.012, 0.013, 0.014, 0.015]:
        exps[f"chop_{ct}"] = {
            **BASE, "CHOP_SLOPE_THRESHOLD": ct,
            "TP_ACTIVATE_ATR": 100.0, "ENABLE_ADDON": False,
        }
        # Also with TP
        exps[f"chop_{ct}_tp"] = {
            **BASE, "CHOP_SLOPE_THRESHOLD": ct,
            "TP_ACTIVATE_ATR": 5.0, "TP1_CANDLE_OFFSET": 1.00,
            "TP2_EMA_ATR_MULT": 5.0, "TP3_EMA_ATR_MULT": 8.0,
            "ENABLE_ADDON": False,
        }

    # Stop sweep
    for st in [1.5, 2.0, 2.5, 3.0, 5.0, 7.0]:
        exps[f"stop_{st}"] = {
            **BASE, "INITIAL_STOP_ATR_MULT": st,
            "TP_ACTIVATE_ATR": 100.0, "ENABLE_ADDON": False,
        }

    # Losers sweep
    for lb in [20, 30, 45, 60, 90]:
        exps[f"losers_{lb}"] = {
            **BASE, "LOSERS_MAX_BARS": lb,
            "TP_ACTIVATE_ATR": 100.0, "ENABLE_ADDON": False,
        }

    # TP activation sweep
    for act in [3.0, 5.0, 8.0]:
        for tp1 in [0.50, 1.00]:
            for tp2, tp3 in [(4.0, 8.0), (5.0, 8.0), (5.0, 10.0)]:
                exps[f"tp_a{act}_o{tp1}_t{tp2}_{tp3}"] = {
                    **BASE, "TP_ACTIVATE_ATR": act, "TP1_CANDLE_OFFSET": tp1,
                    "TP2_EMA_ATR_MULT": tp2, "TP3_EMA_ATR_MULT": tp3,
                    "ENABLE_ADDON": False,
                }

    # Full system (TP + addon)
    for act in [5.0, 8.0]:
        exps[f"full_act{act}"] = {
            **BASE, "TP_ACTIVATE_ATR": act, "TP1_CANDLE_OFFSET": 1.00,
            "TP2_EMA_ATR_MULT": 5.0, "TP3_EMA_ATR_MULT": 8.0,
            "ENABLE_ADDON": True, "MAX_ADDONS": 1,
        }

    # TP ratio sweep
    for t1, t2, t3 in [(0.10, 0.20, 0.70), (0.30, 0.40, 0.30), (0.20, 0.30, 0.50)]:
        exps[f"ratio_{int(t1*100)}_{int(t2*100)}_{int(t3*100)}"] = {
            **BASE, "TP1_PCT": t1, "TP2_PCT": t2, "TP3_PCT": t3,
            "TP_ACTIVATE_ATR": 5.0, "TP1_CANDLE_OFFSET": 1.00,
            "TP2_EMA_ATR_MULT": 5.0, "TP3_EMA_ATR_MULT": 8.0,
            "ENABLE_ADDON": False,
        }

    # Pullback sweep
    for pb in [0.8, 1.0, 1.2, 1.5, 2.0]:
        exps[f"pb_{pb}"] = {
            **BASE, "PULLBACK_TOUCH_MULT": pb,
            "TP_ACTIVATE_ATR": 100.0, "ENABLE_ADDON": False,
        }

    return exps


def main():
    t0 = time.time()

    # Step 1: Clean data
    df = clean_data()
    data_path = os.path.abspath(CLEAN_PATH)

    # Step 2: Build experiments
    experiments = build_experiments()
    print(f"\n{len(experiments)} experiments × {NCPU} CPUs")
    print("=" * 130)

    # Step 3: Run ALL in parallel
    tasks = [(name, overrides, data_path) for name, overrides in experiments.items()]

    results = []
    done = 0
    with ProcessPoolExecutor(max_workers=NCPU) as pool:
        futures = {pool.submit(_run_single, t): t[0] for t in tasks}
        for future in as_completed(futures):
            name = futures[future]
            done += 1
            try:
                r = future.result()
                r["score"] = score(r)
                results.append(r)
                marker = " ***" if r["profit_factor"] >= 1.0 else ""
                print(f"[{done}/{len(tasks)}] {r['name']:<35} PF={r['profit_factor']:>7.3f} "
                      f"Ret={r['total_return']:>6.2f}% WR={r['win_rate']:>5.1f}% "
                      f"Trades={r['total_trades']:>4} DD={r['max_drawdown']:>5.2f}%{marker}")
            except Exception as e:
                print(f"[{done}/{len(tasks)}] {name:<35} ERROR: {e}")

    elapsed = time.time() - t0

    # Step 4: Summary
    print(f"\n{'='*130}")
    print(f"Completed in {elapsed:.1f}s ({NCPU} CPUs)")
    print(f"{'='*130}")

    valid = [r for r in results if r["score"] > -10]
    profitable = [r for r in valid if r.get("profit_factor", 0) >= 1.0]

    print(f"\nPROFITABLE (PF >= 1.0): {len(profitable)} / {len(valid)}")
    for r in sorted(profitable, key=lambda x: x["profit_factor"], reverse=True)[:20]:
        key_p = {k: v for k, v in r.get("overrides", {}).items()
                 if k not in ("EMA_PERIOD", "EMA_SLOW_PERIOD", "ATR_PERIOD")}
        print(f"  {r['name']:<35} PF={r['profit_factor']:.3f} Ret={r['total_return']:.2f}% "
              f"WR={r['win_rate']:.1f}% Trades={r['total_trades']} DD={r['max_drawdown']:.2f}%")
        print(f"    {key_p}")

    print(f"\nTOP 15 BY SCORE:")
    for i, r in enumerate(sorted(valid, key=lambda x: x["score"], reverse=True)[:15]):
        pf_mark = " ***" if r["profit_factor"] >= 1.0 else ""
        print(f"  #{i+1}: {r['name']:<33} Score={r['score']:.4f} PF={r['profit_factor']:.3f} "
              f"Trades={r['total_trades']} Ret={r['total_return']:.2f}%{pf_mark}")

    # v3 vs noTP comparison
    v3 = next((r for r in results if r["name"] == "v3_default"), None)
    notp = next((r for r in results if r["name"] == "v3_noTP"), None)
    if v3 and notp:
        print(f"\n{'='*130}")
        print(f"v3 DEFAULT:  PF={v3['profit_factor']:.3f} Ret={v3['total_return']:.2f}% "
              f"Trades={v3['total_trades']} WR={v3['win_rate']:.1f}% DD={v3['max_drawdown']:.2f}%")
        print(f"v3 NO TP:    PF={notp['profit_factor']:.3f} Ret={notp['total_return']:.2f}% "
              f"Trades={notp['total_trades']} WR={notp['win_rate']:.1f}% DD={notp['max_drawdown']:.2f}%")


if __name__ == "__main__":
    main()
