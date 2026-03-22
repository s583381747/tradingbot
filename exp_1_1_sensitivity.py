"""
Parameter sensitivity analysis for strategy.py.

For each key parameter, vary it while holding all others at baseline,
run the backtest, and record the score. Determines if the strategy is
robust or overfit.
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

# ── Baseline parameter values ─────────────────────────────────
BASELINE = {
    "ema_slope_threshold": 0.012,
    "pullback_touch_mult": 1.2,
    "min_pullback_bars": 1,
    "rsi_overbought": 63,
    "rsi_oversold": 32,
    "initial_stop_atr_mult": 2.5,
    "ema_trail_offset": 6.0,
    "losers_max_bars": 45,
}

# ── Test values for each parameter ────────────────────────────
def pct_variations(base, pcts=(-0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30)):
    """Generate test values as percentage variations from baseline."""
    return [round(base * (1 + p), 6) for p in pcts]

PARAM_TEST_VALUES = {
    "ema_slope_threshold": pct_variations(0.012),
    "pullback_touch_mult": pct_variations(1.2),
    "min_pullback_bars": [1, 2, 3],
    "rsi_overbought": [58, 60, 62, 63, 64, 66, 68],
    "rsi_oversold": [26, 28, 30, 32, 34, 36, 38],
    "initial_stop_atr_mult": pct_variations(2.5),
    "ema_trail_offset": pct_variations(6.0),
    "losers_max_bars": [30, 35, 40, 45, 50, 55, 60],
}


# ── Score function (matches prepare.py exactly) ──────────────
def score(m):
    t = m["trades"]
    if t < 5:
        return -10.0
    sn = max(0, min(1, (m["sharpe"] + 2) / 5))
    pn = max(0, min(1, m["pf"] / 3))
    rn = max(0, min(1, (m["ret"] + 20) / 70))
    wn = max(0, min(1, m["wr"] / 100))
    dn = max(0, min(1, 1 - m["dd"] / 30))
    raw = 0.30 * sn + 0.25 * pn + 0.20 * rn + 0.10 * wn + 0.15 * dn
    if t < 20:
        mult = 0.2
    elif t < 50:
        mult = 0.6
    elif t <= 500:
        mult = 1.0
    else:
        mult = 0.8
    return round(raw * mult, 6)


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
    # Fresh import of strategy
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
        "trades": total_trades,
        "sharpe": round(sharpe_val, 4),
        "pf": round(profit_factor, 4),
        "ret": round(total_return, 4),
        "wr": round(win_rate, 2),
        "dd": round(max_dd, 4),
        "won": won,
        "lost": lost,
    }


# ── Verdict logic ─────────────────────────────────────────────
def classify_parameter(results: list[dict], baseline_score: float) -> str:
    """
    Classify parameter sensitivity.
    results: list of {value, score, ...} dicts in order of test values.
    """
    if baseline_score <= 0:
        return "FRAGILE"

    scores = [r["score"] for r in results]

    # Find baseline index (the entry where value matches baseline)
    baseline_idx = None
    for i, r in enumerate(results):
        if r.get("is_baseline", False):
            baseline_idx = i
            break

    if baseline_idx is None:
        # Fallback: middle element
        baseline_idx = len(results) // 2

    # Check for CLIFF: sharp discontinuity between adjacent values
    is_cliff = False
    for i in range(1, len(scores)):
        if baseline_score > 0:
            change_a = abs(scores[i] - scores[i - 1]) / baseline_score
            if change_a > 0.40:
                is_cliff = True
                break

    # Check for FRAGILE: +/-10% causes >30% score drop
    is_fragile = False
    for r in results:
        if r.get("is_baseline", False):
            continue
        if baseline_score > 0 and r["score"] >= 0:
            drop = (baseline_score - r["score"]) / baseline_score
            # Check if this is a "nearby" variant (within ~15% of baseline value or 1-step)
            pct_diff = abs(r["value"] - results[baseline_idx]["value"])
            base_val = results[baseline_idx]["value"]
            if base_val != 0:
                rel_diff = pct_diff / abs(base_val)
            else:
                rel_diff = 0
            if rel_diff <= 0.15 and drop > 0.30:
                is_fragile = True
                break

    # Check ROBUST: all variants within +/-20% of baseline value score >= 80% of baseline
    is_robust = True
    for r in results:
        if r.get("is_baseline", False):
            continue
        base_val = results[baseline_idx]["value"]
        if base_val != 0:
            rel_diff = abs(r["value"] - base_val) / abs(base_val)
        else:
            rel_diff = 0
        if rel_diff <= 0.25:  # within ~20% range
            if baseline_score > 0:
                if r["score"] < 0 or r["score"] < baseline_score * 0.80:
                    is_robust = False
                    break

    if is_cliff:
        return "CLIFF"
    if is_fragile:
        return "FRAGILE"
    if is_robust:
        return "ROBUST"
    return "FRAGILE"


# ── Main ──────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("PARAMETER SENSITIVITY ANALYSIS")
    print("=" * 70)
    print()

    df = load_data()
    print(f"Data loaded: {len(df)} bars")
    print()

    param_names = list(PARAM_TEST_VALUES.keys())
    all_verdicts = {}
    total_runs = sum(len(v) for v in PARAM_TEST_VALUES.values())
    run_count = 0
    t_start = time.time()

    for param_name in param_names:
        test_values = PARAM_TEST_VALUES[param_name]
        baseline_val = BASELINE[param_name]

        print(f"=== {param_name} (baseline={baseline_val}) ===")
        print(f"  {'value':>10s}  {'score':>8s}  {'trades':>6s}  {'PF':>7s}  {'return':>8s}  {'change_from_baseline':>20s}")

        results = []
        baseline_score = None

        for val in test_values:
            run_count += 1
            # Build overrides: only override the one parameter
            overrides = {param_name: val}

            try:
                metrics = run_backtest(df, overrides)
                sc = score(metrics)
            except Exception as e:
                print(f"  {val:>10}  ERROR: {e}")
                metrics = {"trades": 0, "sharpe": 0, "pf": 0, "ret": 0, "wr": 0, "dd": 0}
                sc = -10.0

            is_baseline = (val == baseline_val)
            if is_baseline:
                baseline_score = sc

            results.append({
                "value": val,
                "score": sc,
                "trades": metrics["trades"],
                "pf": metrics["pf"],
                "ret": metrics["ret"],
                "is_baseline": is_baseline,
            })

            elapsed = time.time() - t_start
            eta = (elapsed / run_count) * (total_runs - run_count) if run_count > 0 else 0
            sys.stdout.flush()

        # Now print with change_from_baseline
        for r in results:
            if baseline_score is not None and baseline_score > 0 and r["score"] >= 0:
                change = (r["score"] - baseline_score) / baseline_score * 100
                change_str = f"{change:+.1f}%"
            elif r["is_baseline"]:
                change_str = "BASELINE"
            else:
                change_str = "N/A"

            marker = " <-- baseline" if r["is_baseline"] else ""
            print(f"  {r['value']:>10}  {r['score']:>8.6f}  {r['trades']:>6d}  {r['pf']:>7.4f}  {r['ret']:>7.4f}%  {change_str:>20s}{marker}")

        # Classify
        if baseline_score is None:
            baseline_score = 0.0

        verdict = classify_parameter(results, baseline_score)
        all_verdicts[param_name] = verdict
        print(f"  Verdict: {verdict}")
        print()

    # ── Summary ───────────────────────────────────────────────
    elapsed_total = time.time() - t_start
    robust_count = sum(1 for v in all_verdicts.values() if v == "ROBUST")
    total_params = len(all_verdicts)
    fragile_cliff = [k for k, v in all_verdicts.items() if v in ("FRAGILE", "CLIFF")]

    print("=" * 70)
    print("SENSITIVITY SUMMARY:")
    print("=" * 70)
    print(f"  Robust: {robust_count} / {total_params} parameters")
    print()

    if fragile_cliff:
        print("  FRAGILE/CLIFF parameters:")
        for p in fragile_cliff:
            print(f"    - {p}: {all_verdicts[p]}")
    else:
        print("  No FRAGILE/CLIFF parameters found.")
    print()

    # Overall verdict
    if robust_count == total_params:
        overall = "LIKELY ROBUST"
    elif len(fragile_cliff) <= 1 and robust_count >= total_params - 2:
        overall = "LIKELY ROBUST"
    elif len(fragile_cliff) >= total_params // 2:
        overall = "DEFINITELY OVERFIT"
    else:
        overall = "LIKELY OVERFIT"

    print(f"  Overall verdict: {overall}")
    print(f"  Total runtime: {elapsed_total:.1f}s ({total_runs} backtests)")
    print()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
