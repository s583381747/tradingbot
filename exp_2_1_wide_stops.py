"""
Experiment 2.1: Wide stops + fixed time exits.

Hypothesis: Entry signal is predictive at 30-60 min (p<0.05) but NOT at 5-10 min.
Current stop (2.5x ATR ~ $0.50) kills trades in 5-15 min before signal materializes.
Test: widen stops and vary exit mechanisms to match signal timescale.
"""

from __future__ import annotations

import importlib
import sys
import time
import traceback

import backtrader as bt
import pandas as pd

# ── Constants ──────────────────────────────────────────────────
DATA_PATH = "data/QQQ_1Min_2025-09-21_2026-03-21.csv"
CASH = 100_000
COMMISSION = 0.001


# ── Score function (matches prepare.py exactly) ───────────────
def sc(m):
    t = m["t"]
    if t < 5:
        return -10.0
    raw = (
        0.30 * max(0, min(1, (m["sh"] + 2) / 5))
        + 0.25 * max(0, min(1, m["pf"] / 3))
        + 0.20 * max(0, min(1, (m["ret"] + 20) / 70))
        + 0.10 * max(0, min(1, m["wr"] / 100))
        + 0.15 * max(0, min(1, 1 - m["dd"] / 30))
    )
    if t < 20:
        return round(raw * 0.2, 6)
    elif t < 50:
        return round(raw * 0.6, 6)
    elif t <= 500:
        return round(raw, 6)
    else:
        return round(raw * 0.8, 6)


# ── Data loading ──────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, index_col="timestamp", parse_dates=True)
    for col in ("Open", "High", "Low", "Close", "Volume"):
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
    return df


# ── Single backtest run ───────────────────────────────────────
def run_backtest(df: pd.DataFrame, param_overrides: dict) -> dict:
    """Run backtest with given parameter overrides, return metrics dict."""
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
    cerebro.addstrategy(StrategyClass, **param_overrides)

    cerebro.broker.setcash(CASH)
    cerebro.broker.setcommission(commission=COMMISSION)

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

    final_value = cerebro.broker.getvalue()
    total_return = (final_value - CASH) / CASH * 100

    sharpe_analysis = strat.analyzers.sharpe.get_analysis()
    sharpe_val = sharpe_analysis.get("sharperatio", 0.0) or 0.0

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
        "t": total_trades,
        "sh": round(sharpe_val, 4),
        "pf": round(profit_factor, 4),
        "ret": round(total_return, 4),
        "wr": round(win_rate, 2),
        "dd": round(max_dd, 4),
        "won": won,
        "lost": lost,
    }


# ── Test matrix ───────────────────────────────────────────────
# Each entry: (label, {param_overrides})

GROUP_A = []
for stop in [2.5, 5.0, 7.0, 10.0, 999.0]:
    GROUP_A.append((
        f"stop={stop:<5.1f} trail=6.0  losers=45",
        {"initial_stop_atr_mult": stop, "ema_trail_offset": 6.0, "losers_max_bars": 45},
    ))

GROUP_B = []
for stop, trail in [
    (5.0, 10.0), (7.0, 10.0), (10.0, 10.0),
    (5.0, 15.0), (7.0, 15.0),
]:
    GROUP_B.append((
        f"stop={stop:<5.1f} trail={trail:<5.1f} losers=45",
        {"initial_stop_atr_mult": stop, "ema_trail_offset": trail, "losers_max_bars": 45},
    ))

GROUP_C = []
for stop in [5.0, 7.0]:
    for losers in [30, 60, 90]:
        GROUP_C.append((
            f"stop={stop:<5.1f} trail=6.0  losers={losers}",
            {"initial_stop_atr_mult": stop, "ema_trail_offset": 6.0, "losers_max_bars": losers},
        ))

GROUP_D = []
for losers in [30, 60, 90, 120, 180, 999]:
    GROUP_D.append((
        f"stop=999   trail=999   losers={losers}",
        {"initial_stop_atr_mult": 999.0, "ema_trail_offset": 999.0, "losers_max_bars": losers},
    ))

ALL_GROUPS = [
    ("Group A: Wide Stop + Current Trail (trail=6.0, losers=45)", GROUP_A),
    ("Group B: Wide Stop + Wider Trail (losers=45)", GROUP_B),
    ("Group C: Wide Stop + Time Exit Variations (trail=6.0)", GROUP_C),
    ("Group D: No Stop, No Trail, Time + Force Close Only", GROUP_D),
]


# ── Run all ───────────────────────────────────────────────────
def main():
    print("=" * 78)
    print("EXPERIMENT 2.1: Wide Stops + Fixed Time Exits")
    print("=" * 78)
    print()
    print("Hypothesis: Signal is predictive at 30-60 min, not 5-10 min.")
    print("Current 2.5x ATR stop kills trades before signal materializes.")
    print("Test: widen stops to give signal time to work.")
    print()

    df = load_data()
    print(f"Data loaded: {len(df)} bars")
    print()

    all_results = []  # (label, metrics, score)
    total_runs = sum(len(g) for _, g in ALL_GROUPS)
    run_count = 0
    t_start = time.time()

    for group_name, group_tests in ALL_GROUPS:
        print(f"=== {group_name} ===")
        for label, overrides in group_tests:
            run_count += 1
            try:
                metrics = run_backtest(df, overrides)
                score_val = sc(metrics)
            except Exception as e:
                print(f"  {label} => ERROR: {e}")
                metrics = {"t": 0, "sh": 0, "pf": 0, "ret": 0, "wr": 0, "dd": 0, "won": 0, "lost": 0}
                score_val = -10.0

            all_results.append((label, metrics, score_val))

            # Format output
            ret_str = f"{metrics['ret']:+.2f}%" if metrics['ret'] != 0 else "+0.00%"
            elapsed = time.time() - t_start
            eta = (elapsed / run_count) * (total_runs - run_count) if run_count > 0 else 0

            print(
                f"  {label} => sc={score_val:.4f}"
                f"  t={metrics['t']:<3d}"
                f"  wr={metrics['wr']:.1f}%"
                f"  pf={metrics['pf']:.3f}"
                f"  ret={ret_str:<8s}"
                f"  dd={metrics['dd']:.2f}%"
                f"  sh={metrics['sh']:.3f}"
                f"  [{run_count}/{total_runs}, ETA {eta:.0f}s]"
            )
            sys.stdout.flush()

        print()

    # ── Summary ───────────────────────────────────────────────
    elapsed_total = time.time() - t_start
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)

    # Find best overall score
    valid = [(l, m, s) for l, m, s in all_results if s > -10.0]
    if valid:
        best_score = max(valid, key=lambda x: x[2])
        print(f"  Best overall:  {best_score[0]}  score={best_score[2]:.4f}")
    else:
        print("  Best overall:  No valid results!")

    # Find best PF (with at least 5 trades)
    valid_pf = [(l, m, s) for l, m, s in all_results if m["t"] >= 5]
    if valid_pf:
        best_pf = max(valid_pf, key=lambda x: x[1]["pf"])
        print(f"  Best PF:       {best_pf[0]}  PF={best_pf[1]['pf']:.3f}")

    # Find best return
    if valid_pf:
        best_ret = max(valid_pf, key=lambda x: x[1]["ret"])
        print(f"  Best return:   {best_ret[0]}  ret={best_ret[1]['ret']:+.2f}%")

    # Find best sharpe
    if valid_pf:
        best_sh = max(valid_pf, key=lambda x: x[1]["sh"])
        print(f"  Best Sharpe:   {best_sh[0]}  sh={best_sh[1]['sh']:.3f}")

    # Find lowest DD (with positive return)
    valid_pos = [(l, m, s) for l, m, s in all_results if m["t"] >= 5 and m["ret"] > 0]
    if valid_pos:
        best_dd = min(valid_pos, key=lambda x: x[1]["dd"])
        print(f"  Lowest DD:     {best_dd[0]}  dd={best_dd[1]['dd']:.2f}%")

    print()

    # Baseline comparison
    baseline_label, baseline_m, baseline_s = all_results[0]  # first entry is baseline
    print(f"  Baseline (stop=2.5, trail=6.0): score={baseline_s:.4f}, PF={baseline_m['pf']:.3f}, ret={baseline_m['ret']:+.2f}%")
    print()

    # Per-group bests
    offset = 0
    for group_name, group_tests in ALL_GROUPS:
        group_results = all_results[offset : offset + len(group_tests)]
        offset += len(group_tests)

        group_valid = [(l, m, s) for l, m, s in group_results if s > -10.0]
        if group_valid:
            best = max(group_valid, key=lambda x: x[2])
            delta = best[2] - baseline_s
            print(f"  {group_name}")
            print(f"    Best: {best[0]}  score={best[2]:.4f}  (delta={delta:+.4f} vs baseline)")
        else:
            print(f"  {group_name}")
            print(f"    No valid results.")
        print()

    # Key insight
    print("  Key insight:")
    if valid:
        # Compare Group A scores: does wider stop help?
        group_a_results = all_results[:len(GROUP_A)]
        baseline_score = group_a_results[0][2]
        wider_better = sum(1 for _, _, s in group_a_results[1:] if s > baseline_score)
        wider_total = len(group_a_results) - 1

        if wider_better > wider_total / 2:
            print(f"    WIDER STOPS HELP: {wider_better}/{wider_total} wider-stop configs beat baseline.")
            print("    Signal needs time to materialize -- tight stops are harmful.")
        elif wider_better == 0:
            print(f"    WIDER STOPS DON'T HELP: 0/{wider_total} wider-stop configs beat baseline.")
            print("    Current stop width may already be appropriate, or entries need improvement.")
        else:
            print(f"    MIXED: {wider_better}/{wider_total} wider-stop configs beat baseline.")
            print("    Some benefit to wider stops but not universal.")

        # Compare Group D: does holding longer help?
        group_d_start = len(GROUP_A) + len(GROUP_B) + len(GROUP_C)
        group_d_results = all_results[group_d_start : group_d_start + len(GROUP_D)]
        d_valid = [(l, m, s) for l, m, s in group_d_results if s > -10.0 and m["t"] >= 5]
        if d_valid:
            d_best = max(d_valid, key=lambda x: x[2])
            print(f"    Pure hold test: best={d_best[0]} score={d_best[2]:.4f}")
            if d_best[2] > baseline_score:
                print("    Removing stops/trails IMPROVES results -- stops are net negative!")
            else:
                print("    Removing stops/trails HURTS results -- some stop mechanism is needed.")

    print()
    print(f"  Total runtime: {elapsed_total:.1f}s ({total_runs} backtests)")
    print()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
