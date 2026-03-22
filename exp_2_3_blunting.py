"""
Experiment 2.3: Parameter Blunting Test
========================================
Tests whether the strategy edge survives when overfit entry parameters
are progressively relaxed ("blunted") to more generic values.

If the edge is real, moderate blunting should retain profitability.
If it's noise, even mild blunting will collapse performance.
"""

from __future__ import annotations

import sys
import traceback

import backtrader as bt
import pandas as pd

# ── Config ─────────────────────────────────────────────────────
DATA_PATH = "data/QQQ_1Min_2025-09-21_2026-03-21.csv"
CASH = 100_000
COMMISSION = 0.001

# ── Score function (matches prepare.py) ────────────────────────
def compute_score(m: dict) -> float:
    t = m["total_trades"]
    if t < 5:
        return -10.0
    sharpe_norm = max(0.0, min(1.0, (m["sharpe"] + 2) / 5.0))
    pf_norm = max(0.0, min(1.0, m["profit_factor"] / 3.0))
    ret_norm = max(0.0, min(1.0, (m["total_return"] + 20) / 70.0))
    wr_norm = max(0.0, min(1.0, m["win_rate"] / 100.0))
    dd_norm = max(0.0, min(1.0, 1.0 - m["max_drawdown"] / 30.0))
    raw = (0.30 * sharpe_norm + 0.25 * pf_norm + 0.20 * ret_norm
           + 0.10 * wr_norm + 0.15 * dd_norm)
    if t < 20:
        return round(raw * 0.2, 6)
    elif t < 50:
        return round(raw * 0.6, 6)
    elif t <= 500:
        return round(raw, 6)
    else:
        return round(raw * 0.8, 6)

# ── Data loader ────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, index_col="timestamp", parse_dates=True)
    for col in ("Open", "High", "Low", "Close", "Volume"):
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
    return df

# ── Run one backtest with param overrides ──────────────────────
def run_one(df: pd.DataFrame, overrides: dict) -> dict:
    """Run backtest with given parameter overrides, return metrics dict."""
    # Fresh import each time
    if "strategy" in sys.modules:
        del sys.modules["strategy"]
    import strategy as mod
    StrategyClass = mod.Strategy

    cerebro = bt.Cerebro()
    data_feed = bt.feeds.PandasData(
        dataname=df,
        datetime=None,
        open="Open",
        high="High",
        low="Low",
        close="Close",
        volume="Volume",
        openinterest=-1,
    )
    cerebro.adddata(data_feed)
    cerebro.addstrategy(StrategyClass, **overrides)

    cerebro.broker.setcash(CASH)
    cerebro.broker.setcommission(commission=COMMISSION)

    cerebro.addanalyzer(
        bt.analyzers.SharpeRatio, _name="sharpe",
        timeframe=bt.TimeFrame.Days, riskfreerate=0.05,
    )
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")

    results = cerebro.run()
    strat = results[0]

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

    m = {
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
    m["score"] = compute_score(m)
    return m


# ── Blunting levels ────────────────────────────────────────────
BLUNTING_LEVELS = [
    {
        "name": "Level 0 (Current Overfit)",
        "params": {
            "ema_slope_threshold": 0.012,
            "pullback_touch_mult": 1.2,
            "rsi_overbought": 63,
            "rsi_oversold": 32,
        },
    },
    {
        "name": "Level 1 (Mild Blunting)",
        "params": {
            "ema_slope_threshold": 0.01,
            "pullback_touch_mult": 1.3,
            "rsi_overbought": 65,
            "rsi_oversold": 35,
        },
    },
    {
        "name": "Level 2 (Moderate Blunting)",
        "params": {
            "ema_slope_threshold": 0.01,
            "pullback_touch_mult": 1.5,
            "rsi_overbought": 68,
            "rsi_oversold": 28,
        },
    },
    {
        "name": "Level 3 (Heavy Blunting)",
        "params": {
            "ema_slope_threshold": 0.008,
            "pullback_touch_mult": 1.5,
            "rsi_overbought": 70,
            "rsi_oversold": 25,
        },
    },
    {
        "name": "Level 4 (Near-Disabled)",
        "params": {
            "ema_slope_threshold": 0.005,
            "pullback_touch_mult": 2.0,
            "rsi_overbought": 80,
            "rsi_oversold": 20,
        },
    },
]

STOP_CONFIGS = [
    {"label": "stop=2.5 trail=6.0",  "initial_stop_atr_mult": 2.5, "ema_trail_offset": 6.0},
    {"label": "stop=5.0 trail=6.0",  "initial_stop_atr_mult": 5.0, "ema_trail_offset": 6.0},
    {"label": "stop=5.0 trail=10.0", "initial_stop_atr_mult": 5.0, "ema_trail_offset": 10.0},
]

# Extra tests: blunted + different min_pullback_bars
PULLBACK_TESTS = [
    {"label": "L1 + pb_bars=1", "level_idx": 1, "min_pullback_bars": 1},
    {"label": "L1 + pb_bars=2", "level_idx": 1, "min_pullback_bars": 2},
    {"label": "L2 + pb_bars=1", "level_idx": 2, "min_pullback_bars": 1},
    {"label": "L2 + pb_bars=2", "level_idx": 2, "min_pullback_bars": 2},
]


def fmt_result(m: dict, tag: str = "") -> str:
    sign = "+" if m["total_return"] >= 0 else ""
    tag_str = f" ({tag})" if tag else ""
    return (
        f"sc={m['score']:.4f} t={m['total_trades']:>3d} "
        f"wr={m['win_rate']:.0f}% pf={m['profit_factor']:.3f} "
        f"ret={sign}{m['total_return']:.2f}% sh={m['sharpe']:.2f} "
        f"dd={m['max_drawdown']:.1f}%{tag_str}"
    )


def main():
    print("=" * 70)
    print("EXPERIMENT 2.3: Parameter Blunting Test")
    print("=" * 70)
    print()

    df = load_data()
    print(f"Data loaded: {len(df)} bars")
    print()

    # Store all results for summary
    # results[level_idx][stop_idx] = metrics
    all_results = {}

    # ── Core tests: 5 levels x 3 stop configs = 15 tests ──────
    test_num = 0
    for lvl_idx, level in enumerate(BLUNTING_LEVELS):
        print(f"=== Blunting {level['name']} ===")
        print(f"    slope={level['params']['ema_slope_threshold']}, "
              f"pb_mult={level['params']['pullback_touch_mult']}, "
              f"rsi_ob={level['params']['rsi_overbought']}, "
              f"rsi_os={level['params']['rsi_oversold']}")
        all_results[lvl_idx] = {}

        for stop_idx, stop_cfg in enumerate(STOP_CONFIGS):
            test_num += 1
            overrides = {}
            overrides.update(level["params"])
            overrides["initial_stop_atr_mult"] = stop_cfg["initial_stop_atr_mult"]
            overrides["ema_trail_offset"] = stop_cfg["ema_trail_offset"]

            m = run_one(df, overrides)
            all_results[lvl_idx][stop_idx] = m

            tag = "BASELINE" if (lvl_idx == 0 and stop_idx == 0) else ""
            print(f"  {stop_cfg['label']:>22s}:  {fmt_result(m, tag)}")
            sys.stdout.flush()

        print()

    # ── Pullback bars tests ────────────────────────────────────
    print("=== Blunted + min_pullback_bars Variants ===")
    pb_results = []
    for pb_test in PULLBACK_TESTS:
        test_num += 1
        lvl = BLUNTING_LEVELS[pb_test["level_idx"]]
        overrides = {}
        overrides.update(lvl["params"])
        # Use current stop config (2.5, 6.0)
        overrides["initial_stop_atr_mult"] = 2.5
        overrides["ema_trail_offset"] = 6.0
        overrides["min_pullback_bars"] = pb_test["min_pullback_bars"]

        m = run_one(df, overrides)
        pb_results.append((pb_test["label"], m))
        print(f"  {pb_test['label']:>22s}:  {fmt_result(m)}")
        sys.stdout.flush()

    print()
    print(f"Total tests run: {test_num}")
    print()

    # ── SUMMARY ────────────────────────────────────────────────
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()

    # Baseline = Level 0, stop=2.5, trail=6.0
    baseline = all_results[0][0]
    baseline_sc = baseline["score"]
    baseline_pf = baseline["profit_factor"]

    print(f"Baseline (Level 0, stop=2.5, trail=6.0): sc={baseline_sc:.4f} pf={baseline_pf:.3f}")
    print()

    # Degradation per level (using stop=2.5 trail=6.0 for apples-to-apples)
    print("--- Degradation by Level (stop=2.5, trail=6.0) ---")
    for lvl_idx in range(1, len(BLUNTING_LEVELS)):
        m = all_results[lvl_idx][0]
        sc = m["score"]
        pf = m["profit_factor"]
        if baseline_sc > 0:
            sc_drop = (1 - sc / baseline_sc) * 100
        elif baseline_sc == 0:
            sc_drop = 0 if sc == 0 else -100
        else:
            sc_drop = float("nan")
        print(f"  Level 0 -> Level {lvl_idx}: "
              f"sc {baseline_sc:.4f} -> {sc:.4f} ({sc_drop:+.1f}% drop), "
              f"PF {baseline_pf:.3f} -> {pf:.3f}, "
              f"trades {baseline['total_trades']} -> {m['total_trades']}")

    print()

    # Best combo at each level
    print("--- Best Stop Config per Blunting Level ---")
    for lvl_idx in range(len(BLUNTING_LEVELS)):
        best_stop_idx = max(all_results[lvl_idx], key=lambda si: all_results[lvl_idx][si]["score"])
        best_m = all_results[lvl_idx][best_stop_idx]
        print(f"  Level {lvl_idx}: {STOP_CONFIGS[best_stop_idx]['label']} "
              f"-> sc={best_m['score']:.4f} pf={best_m['profit_factor']:.3f}")

    print()

    # Most robust blunting level with PF > 0.9
    print("--- Robustness Check ---")
    most_robust_pf09 = -1
    for lvl_idx in range(len(BLUNTING_LEVELS)):
        # Check across all stop configs
        for stop_idx in range(len(STOP_CONFIGS)):
            m = all_results[lvl_idx][stop_idx]
            if m["profit_factor"] > 0.9 and m["total_trades"] >= 5:
                most_robust_pf09 = lvl_idx

    if most_robust_pf09 >= 0:
        print(f"  Most aggressive blunting with PF > 0.9 (any stop): Level {most_robust_pf09}")
    else:
        print("  No blunting level achieves PF > 0.9")

    # Best overall blunted combo
    best_combo_sc = -999
    best_combo_desc = ""
    best_combo_m = None
    for lvl_idx in range(len(BLUNTING_LEVELS)):
        for stop_idx in range(len(STOP_CONFIGS)):
            m = all_results[lvl_idx][stop_idx]
            if m["score"] > best_combo_sc:
                best_combo_sc = m["score"]
                best_combo_desc = (
                    f"Level {lvl_idx} + {STOP_CONFIGS[stop_idx]['label']}"
                )
                best_combo_m = m

    print(f"  Best overall combo: {best_combo_desc} "
          f"-> sc={best_combo_m['score']:.4f} pf={best_combo_m['profit_factor']:.3f}")

    # Best blunted (Level >= 1) + wide stop combo
    best_blunted_sc = -999
    best_blunted_desc = ""
    best_blunted_m = None
    for lvl_idx in range(1, len(BLUNTING_LEVELS)):
        for stop_idx in range(len(STOP_CONFIGS)):
            m = all_results[lvl_idx][stop_idx]
            if m["score"] > best_blunted_sc:
                best_blunted_sc = m["score"]
                best_blunted_desc = (
                    f"Level {lvl_idx} + {STOP_CONFIGS[stop_idx]['label']}"
                )
                best_blunted_m = m

    if best_blunted_m:
        print(f"  Best blunted (L1+) combo: {best_blunted_desc} "
              f"-> sc={best_blunted_m['score']:.4f} pf={best_blunted_m['profit_factor']:.3f}")

    print()

    # Wide stop effect
    print("--- Wide Stop Effect ---")
    for lvl_idx in range(len(BLUNTING_LEVELS)):
        m_narrow = all_results[lvl_idx][0]
        m_wide = all_results[lvl_idx][1]
        m_widest = all_results[lvl_idx][2]
        print(f"  Level {lvl_idx}: "
              f"narrow PF={m_narrow['profit_factor']:.3f} -> "
              f"wide PF={m_wide['profit_factor']:.3f} -> "
              f"widest PF={m_widest['profit_factor']:.3f}")

    print()

    # Pullback bars effect
    print("--- min_pullback_bars Effect ---")
    for label, m in pb_results:
        print(f"  {label}: sc={m['score']:.4f} pf={m['profit_factor']:.3f} t={m['total_trades']}")

    print()

    # ── VERDICT ────────────────────────────────────────────────
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)

    # Determine verdict based on PF survival
    # Check Level 2 with best stop config
    level2_best_pf = max(all_results[2][si]["profit_factor"] for si in range(len(STOP_CONFIGS)))
    level1_best_pf = max(all_results[1][si]["profit_factor"] for si in range(len(STOP_CONFIGS)))
    level3_best_pf = max(all_results[3][si]["profit_factor"] for si in range(len(STOP_CONFIGS)))

    # Also check with standard stop (index 0) for strict comparison
    level1_std_pf = all_results[1][0]["profit_factor"]
    level2_std_pf = all_results[2][0]["profit_factor"]

    if level2_best_pf > 0.9:
        verdict = "REAL"
        reason = (f"Level 2 (moderate blunting) still achieves PF={level2_best_pf:.3f} > 0.9. "
                  f"Level 3 PF={level3_best_pf:.3f}.")
    elif level1_best_pf > 0.9:
        verdict = "FRAGILE"
        reason = (f"Only Level 1 (mild) survives with PF={level1_best_pf:.3f} > 0.9. "
                  f"Level 2 collapses to PF={level2_best_pf:.3f}.")
    else:
        verdict = "NOISE"
        reason = (f"Even Level 1 collapses: best PF={level1_best_pf:.3f}. "
                  f"The edge was pure curve-fitting.")

    print()
    print(f"  Edge is: {verdict}")
    print(f"    REAL    = Level 2+ still PF > 0.9")
    print(f"    FRAGILE = Only Level 1 survives")
    print(f"    NOISE   = Even Level 1 collapses")
    print()
    print(f"  Result: {reason}")
    print()

    # Additional context
    print("  Key observations:")
    if baseline_pf > 0:
        l1_retention = level1_std_pf / baseline_pf * 100
        l2_retention = level2_std_pf / baseline_pf * 100
        print(f"    - Level 1 retains {l1_retention:.0f}% of baseline PF")
        print(f"    - Level 2 retains {l2_retention:.0f}% of baseline PF")

    # Check if wide stops help blunted entries
    wide_helps = 0
    for lvl_idx in range(1, len(BLUNTING_LEVELS)):
        if all_results[lvl_idx][1]["score"] > all_results[lvl_idx][0]["score"]:
            wide_helps += 1
    if wide_helps > 0:
        print(f"    - Wide stops improved {wide_helps}/{len(BLUNTING_LEVELS)-1} blunted levels")
    else:
        print(f"    - Wide stops did NOT help blunted entries")

    print()
    print("=" * 70)
    print("EXPERIMENT 2.3 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
