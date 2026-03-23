"""
Experiment 3: Bold parameter sweep for the Chop-Box v2 framework.

Tests many configurations to find the optimal parameter set.
Key hypotheses:
1. Initial stop too tight (1.5x → try 3-7x ATR)
2. TP candle trail too aggressive → widen or disable TP1
3. TP activation too early or too late
4. Addon may help or hurt
5. Chop threshold sensitivity
"""

from __future__ import annotations

import importlib
import sys
import itertools
import time
from copy import deepcopy

import backtrader as bt
import pandas as pd

DATA_PATH = "data/QQQ_1Min_2025-09-21_2026-03-21.csv"
CASH = 100_000
COMMISSION = 0.001


def load_data():
    df = pd.read_csv(DATA_PATH, index_col="timestamp", parse_dates=True)
    return df


def run_one(df, overrides: dict) -> dict:
    """Run backtest with parameter overrides, return metrics."""
    # Fresh import
    if "strategy" in sys.modules:
        del sys.modules["strategy"]
    mod = importlib.import_module("strategy")

    # Monkey-patch module-level constants
    for k, v in overrides.items():
        setattr(mod, k, v)

    # Also need to update the Strategy class params defaults
    StrategyClass = mod.Strategy

    cerebro = bt.Cerebro()
    data = bt.feeds.PandasData(
        dataname=df, datetime=None,
        open="Open", high="High", low="Low", close="Close",
        volume="Volume", openinterest=-1,
    )
    cerebro.adddata(data)

    # Build params dict for strategy
    param_overrides = {}
    param_name_map = {
        "EMA_PERIOD": "ema_period",
        "EMA_SLOW_PERIOD": "ema_slow_period",
        "ATR_PERIOD": "atr_period",
        "EMA_SLOPE_PERIOD": "ema_slope_period",
        "CHOP_SLOPE_AVG_PERIOD": "chop_slope_avg_period",
        "CHOP_SLOPE_THRESHOLD": "chop_slope_threshold",
        "CHOP_BOX_MIN_BARS": "chop_box_min_bars",
        "PULLBACK_TOUCH_MULT": "pullback_touch_mult",
        "MIN_PULLBACK_BARS": "min_pullback_bars",
        "INITIAL_STOP_ATR_MULT": "initial_stop_atr_mult",
        "TP_ACTIVATE_ATR": "tp_activate_atr",
        "TP1_PCT": "tp1_pct",
        "TP2_PCT": "tp2_pct",
        "TP3_PCT": "tp3_pct",
        "TP1_CANDLE_OFFSET": "tp1_candle_offset",
        "TP2_EMA_ATR_MULT": "tp2_ema_atr_mult",
        "TP3_EMA_ATR_MULT": "tp3_ema_atr_mult",
        "ENABLE_ADDON": "enable_addon",
        "MAX_ADDONS": "max_addons",
        "ADDON_PULLBACK_MULT": "addon_pullback_mult",
        "ADDON_MIN_BARS": "addon_min_bars",
        "RISK_PCT": "risk_pct",
        "MAX_POSITION_PCT": "max_position_pct",
        "MAX_DAILY_TRADES": "max_daily_trades",
        "MAX_DAILY_LOSS_PCT": "max_daily_loss_pct",
        "LOSERS_MAX_BARS": "losers_max_bars",
    }
    for k, v in overrides.items():
        if k in param_name_map:
            param_overrides[param_name_map[k]] = v

    cerebro.addstrategy(StrategyClass, **param_overrides)
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
        "total_return": round(total_return, 2),
        "sharpe": round(sharpe, 3),
        "max_drawdown": round(max_dd, 2),
        "total_trades": total_trades,
        "won": won,
        "lost": lost,
        "win_rate": round(win_rate, 1),
        "profit_factor": round(pf, 3),
    }


def score(m):
    trades = m["total_trades"]
    if trades < 5:
        return -10.0
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


def main():
    df = load_data()
    print(f"Data: {len(df)} bars")
    print()

    # ═══════════════════════════════════════════════════════════
    # EXPERIMENT CONFIGURATIONS
    # ═══════════════════════════════════════════════════════════

    experiments = {}

    # --- Phase 1: Diagnose the problem ---

    # A: Baseline (current defaults)
    experiments["A0_baseline"] = {}

    # B: Just widen initial stop (proven fix from exp 2.1)
    for mult in [3.0, 5.0, 7.0]:
        experiments[f"B_stop_{mult}"] = {"INITIAL_STOP_ATR_MULT": mult}

    # C: Disable TP system entirely (use only EMA trail like old strategy)
    # Simulate by making TP activate very late (100 ATR = never)
    experiments["C_no_tp"] = {"TP_ACTIVATE_ATR": 100.0, "INITIAL_STOP_ATR_MULT": 5.0}

    # D: Disable addon
    experiments["D_no_addon"] = {"ENABLE_ADDON": False, "INITIAL_STOP_ATR_MULT": 5.0}

    # E: Both no TP + no addon + wide stop
    experiments["E_pure_trail"] = {
        "TP_ACTIVATE_ATR": 100.0,
        "ENABLE_ADDON": False,
        "INITIAL_STOP_ATR_MULT": 5.0,
    }

    # --- Phase 2: Fix TP system ---

    # F: TP with wider candle offset (give more room)
    for offset in [0.20, 0.50, 1.00]:
        experiments[f"F_tp1off_{offset}"] = {
            "TP1_CANDLE_OFFSET": offset,
            "INITIAL_STOP_ATR_MULT": 5.0,
        }

    # G: TP with higher activation threshold
    for act in [3.0, 4.0, 5.0]:
        experiments[f"G_tpact_{act}"] = {
            "TP_ACTIVATE_ATR": act,
            "INITIAL_STOP_ATR_MULT": 5.0,
        }

    # H: TP2/TP3 tighter (let winners run less)
    experiments["H_tp_tight"] = {
        "TP2_EMA_ATR_MULT": 2.0,
        "TP3_EMA_ATR_MULT": 4.0,
        "INITIAL_STOP_ATR_MULT": 5.0,
    }

    # I: TP2/TP3 much wider (let winners run more)
    experiments["I_tp_wide"] = {
        "TP2_EMA_ATR_MULT": 5.0,
        "TP3_EMA_ATR_MULT": 10.0,
        "INITIAL_STOP_ATR_MULT": 5.0,
    }

    # J: Only use EMA trail for all 3 portions (no candle trail)
    # TP1 also uses EMA - 2x ATR instead of candle trail
    # (Can't easily change logic, but make TP1 offset huge so it never triggers before TP2)
    experiments["J_all_ema_trail"] = {
        "TP1_CANDLE_OFFSET": 5.0,  # huge offset = effectively EMA trail
        "TP2_EMA_ATR_MULT": 2.0,
        "TP3_EMA_ATR_MULT": 5.0,
        "INITIAL_STOP_ATR_MULT": 5.0,
    }

    # --- Phase 3: Chop Box sensitivity ---

    # K: Looser chop detection (more bars in trend mode → more trades)
    for thresh in [0.003, 0.008, 0.010]:
        experiments[f"K_chop_{thresh}"] = {
            "CHOP_SLOPE_THRESHOLD": thresh,
            "INITIAL_STOP_ATR_MULT": 5.0,
            "TP_ACTIVATE_ATR": 100.0,  # disable TP for clean test
            "ENABLE_ADDON": False,
        }

    # L: Pullback sensitivity
    for pb in [0.8, 1.0, 1.5, 2.0]:
        experiments[f"L_pb_{pb}"] = {
            "PULLBACK_TOUCH_MULT": pb,
            "INITIAL_STOP_ATR_MULT": 5.0,
            "TP_ACTIVATE_ATR": 100.0,
            "ENABLE_ADDON": False,
        }

    # --- Phase 4: Bold combinations ---

    # M: Wide stop + looser chop + wider pullback + no TP + no addon
    experiments["M_loose_pure"] = {
        "INITIAL_STOP_ATR_MULT": 5.0,
        "CHOP_SLOPE_THRESHOLD": 0.008,
        "PULLBACK_TOUCH_MULT": 1.5,
        "TP_ACTIVATE_ATR": 100.0,
        "ENABLE_ADDON": False,
        "MIN_PULLBACK_BARS": 1,
    }

    # N: Best chop + wide stop + careful TP (high activation, wide offset)
    experiments["N_careful_tp"] = {
        "INITIAL_STOP_ATR_MULT": 5.0,
        "CHOP_SLOPE_THRESHOLD": 0.008,
        "PULLBACK_TOUCH_MULT": 1.5,
        "TP_ACTIVATE_ATR": 4.0,
        "TP1_CANDLE_OFFSET": 0.50,
        "TP2_EMA_ATR_MULT": 4.0,
        "TP3_EMA_ATR_MULT": 8.0,
        "ENABLE_ADDON": False,
    }

    # O: Same as N but with addon
    experiments["O_careful_tp_addon"] = {
        "INITIAL_STOP_ATR_MULT": 5.0,
        "CHOP_SLOPE_THRESHOLD": 0.008,
        "PULLBACK_TOUCH_MULT": 1.5,
        "TP_ACTIVATE_ATR": 4.0,
        "TP1_CANDLE_OFFSET": 0.50,
        "TP2_EMA_ATR_MULT": 4.0,
        "TP3_EMA_ATR_MULT": 8.0,
        "ENABLE_ADDON": True,
        "MAX_ADDONS": 1,
    }

    # P: Ultra aggressive: very wide stop, no TP, loose everything
    experiments["P_ultra_loose"] = {
        "INITIAL_STOP_ATR_MULT": 10.0,
        "CHOP_SLOPE_THRESHOLD": 0.010,
        "PULLBACK_TOUCH_MULT": 2.0,
        "TP_ACTIVATE_ATR": 100.0,
        "ENABLE_ADDON": False,
        "LOSERS_MAX_BARS": 120,
    }

    # Q: Tighter chop (more selective), wide stop, no TP
    experiments["Q_strict_chop"] = {
        "INITIAL_STOP_ATR_MULT": 5.0,
        "CHOP_SLOPE_THRESHOLD": 0.003,
        "CHOP_BOX_MIN_BARS": 10,
        "TP_ACTIVATE_ATR": 100.0,
        "ENABLE_ADDON": False,
    }

    # R: Losers max bars sensitivity
    for bars in [30, 45, 90, 120]:
        experiments[f"R_losers_{bars}"] = {
            "INITIAL_STOP_ATR_MULT": 5.0,
            "LOSERS_MAX_BARS": bars,
            "TP_ACTIVATE_ATR": 100.0,
            "ENABLE_ADDON": False,
        }

    # ═══════════════════════════════════════════════════════════
    # RUN ALL EXPERIMENTS
    # ═══════════════════════════════════════════════════════════

    print(f"Running {len(experiments)} experiments...")
    print("=" * 120)
    print(f"{'Name':<25} {'Score':>8} {'PF':>8} {'Return%':>8} {'Sharpe':>8} {'WR%':>6} {'Trades':>7} {'W/L':>8} {'MaxDD%':>8}")
    print("-" * 120)

    results = []
    for name, overrides in experiments.items():
        t0 = time.time()
        try:
            m = run_one(df, overrides)
            s = score(m)
            elapsed = time.time() - t0
            results.append({"name": name, "score": s, **m, "overrides": overrides})
            print(f"{name:<25} {s:>8.4f} {m['profit_factor']:>8.3f} {m['total_return']:>7.2f}% "
                  f"{m['sharpe']:>8.3f} {m['win_rate']:>5.1f}% {m['total_trades']:>6} "
                  f"{m['won']}/{m['lost']:>5} {m['max_drawdown']:>7.2f}%  ({elapsed:.1f}s)")
        except Exception as e:
            print(f"{name:<25} {'ERROR':>8}  {str(e)[:60]}")
            results.append({"name": name, "score": -10, "error": str(e)})

    # ═══════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════

    print()
    print("=" * 120)
    print("TOP 10 CONFIGURATIONS")
    print("=" * 120)

    valid = [r for r in results if r["score"] > -10]
    valid.sort(key=lambda x: x["score"], reverse=True)

    for i, r in enumerate(valid[:10]):
        print(f"\n  #{i+1}: {r['name']}")
        print(f"      Score={r['score']:.4f} | PF={r['profit_factor']:.3f} | Return={r['total_return']:.2f}% | "
              f"Sharpe={r['sharpe']:.3f} | WR={r['win_rate']:.1f}% | Trades={r['total_trades']} | MaxDD={r['max_drawdown']:.2f}%")
        if "overrides" in r:
            params = ", ".join(f"{k}={v}" for k, v in r["overrides"].items())
            print(f"      Params: {params}")

    # Key findings
    print()
    print("=" * 120)
    print("KEY FINDINGS")
    print("=" * 120)

    baseline = next((r for r in results if r["name"] == "A0_baseline"), None)
    if baseline:
        print(f"\n  Baseline: Score={baseline['score']:.4f}, PF={baseline.get('profit_factor', 'N/A')}")

    # Compare TP on vs off
    no_tp = next((r for r in results if r["name"] == "E_pure_trail"), None)
    if no_tp:
        print(f"  Pure trail (no TP, no addon): Score={no_tp['score']:.4f}, PF={no_tp.get('profit_factor', 'N/A')}")

    # Best overall
    if valid:
        best = valid[0]
        print(f"\n  BEST: {best['name']} → Score={best['score']:.4f}, PF={best.get('profit_factor', 'N/A')}")


if __name__ == "__main__":
    main()
