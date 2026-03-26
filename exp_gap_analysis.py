"""
Gap Analysis: How much does Open overshoot signal price on trigger bars?

Investigates whether $0.104 average gap is real or inflated.
Breaks down gap distribution, checks for data artifacts,
and compares Open vs High on trigger bars.
"""
from __future__ import annotations
import functools, datetime as dt
import numpy as np, pandas as pd
from entry_signal import add_indicators, detect_trend, check_touch, check_bounce, calc_signal_line

print = functools.partial(print, flush=True)

DATA_PATH = "data/QQQ_1Min_Polygon_2y_clean.csv"

PARAMS = {
    "ema_fast": 20, "ema_slow": 50, "atr_period": 14,
    "touch_tol": 0.15, "touch_below_max": 0.5,
    "signal_offset": 0.05, "stop_buffer": 0.3,
    "signal_valid_bars": 3,
    "no_entry_after": dt.time(15, 30),
    "daily_loss_r": 2.5,
}


def collect_gaps(df):
    """Collect gap data for every trigger bar."""
    p = PARAMS
    df = add_indicators(df, p)
    high = df["High"].values
    low = df["Low"].values
    close = df["Close"].values
    open_p = df["Open"].values
    ema = df["ema20"].values
    ema_s = df["ema50"].values
    atr_v = df["atr"].values
    times = df.index.time
    dates = df.index.date
    n = len(df)

    gaps = []
    bar = max(p["ema_slow"], p["atr_period"]) + 5
    daily_r_loss = 0.0
    current_date = None

    while bar < n - 200:
        a = atr_v[bar]
        if np.isnan(a) or a <= 0 or np.isnan(ema[bar]) or np.isnan(ema_s[bar]):
            bar += 1; continue
        if times[bar] >= p["no_entry_after"]:
            bar += 1; continue

        d_date = dates[bar]
        if current_date != d_date:
            current_date = d_date; daily_r_loss = 0.0
        if daily_r_loss >= p["daily_loss_r"]:
            bar += 1; continue

        trend = detect_trend(close[bar], ema[bar], ema_s[bar])
        if trend == 0: bar += 1; continue

        if not check_touch(trend, low[bar], high[bar], ema[bar], a,
                           p["touch_tol"], p["touch_below_max"]):
            bar += 1; continue

        bb = bar + 1
        if bb >= n: bar += 1; continue
        if not check_bounce(trend, close[bb], high[bar], low[bar]):
            bar += 1; continue

        sl = calc_signal_line(trend, high[bar], low[bar], a,
                              p["signal_offset"], p["stop_buffer"])
        if sl is None: bar += 1; continue
        sig, stop, risk = sl

        # Signal trigger
        for j in range(1, p["signal_valid_bars"] + 1):
            cb = bb + j
            if cb >= n: break
            if trend == 1 and high[cb] >= sig:
                # Gap = how much Open overshoots sig
                gap = open_p[cb] - sig  # positive = open above sig (bad for long)
                prev_close = close[cb - 1]
                bar_range = high[cb] - low[cb]
                gaps.append({
                    "trend": trend,
                    "sig": sig,
                    "open": open_p[cb],
                    "high": high[cb],
                    "low": low[cb],
                    "close_prev": prev_close,
                    "gap_raw": gap,
                    "gap_abs": abs(gap) if gap > 0 else 0,  # only adverse gap
                    "gap_pct_of_risk": gap / risk * 100 if risk > 0 else 0,
                    "bar_range": bar_range,
                    "atr": a,
                    "risk": risk,
                    "open_above_sig": open_p[cb] > sig,
                    "open_eq_sig": abs(open_p[cb] - sig) < 0.005,
                    "time": times[cb],
                    "date": dates[cb],
                    "trigger_delay": j,  # which bar triggered (1st, 2nd, 3rd)
                })
                bar = cb + 1
                break
            if trend == -1 and low[cb] <= sig:
                gap = sig - open_p[cb]  # positive = open below sig (bad for short)
                prev_close = close[cb - 1]
                bar_range = high[cb] - low[cb]
                gaps.append({
                    "trend": trend,
                    "sig": sig,
                    "open": open_p[cb],
                    "high": high[cb],
                    "low": low[cb],
                    "close_prev": prev_close,
                    "gap_raw": gap,
                    "gap_abs": abs(gap) if gap > 0 else 0,
                    "gap_pct_of_risk": gap / risk * 100 if risk > 0 else 0,
                    "bar_range": bar_range,
                    "atr": a,
                    "risk": risk,
                    "open_above_sig": open_p[cb] < sig,  # for short, open below sig = gap
                    "open_eq_sig": abs(open_p[cb] - sig) < 0.005,
                    "time": times[cb],
                    "date": dates[cb],
                    "trigger_delay": j,
                })
                bar = cb + 1
                break
        else:
            bar += 1
            continue

    return pd.DataFrame(gaps)


def main():
    df = pd.read_csv(DATA_PATH, index_col="timestamp", parse_dates=True)
    print(f"Data: {len(df):,} bars\n")

    gdf = collect_gaps(df)
    print(f"Total triggers analyzed: {len(gdf)}\n")

    # ═══ 1. Overall gap distribution ═══
    print(f"{'='*80}")
    print(f"  GAP DISTRIBUTION (Open vs Signal price)")
    print(f"{'='*80}")

    gap_raw = gdf["gap_raw"].values
    print(f"\n  All gaps (positive = adverse, Open past signal):")
    print(f"    Mean:   ${np.mean(gap_raw):.4f}")
    print(f"    Median: ${np.median(gap_raw):.4f}")
    print(f"    Std:    ${np.std(gap_raw):.4f}")
    print(f"    Min:    ${np.min(gap_raw):.4f}")
    print(f"    Max:    ${np.max(gap_raw):.4f}")

    # ═══ 2. How many have adverse gap? ═══
    print(f"\n  {'Category':<40} {'Count':>6} {'%':>7}")
    print(f"  {'-'*55}")

    n = len(gdf)
    open_past = (gdf["gap_raw"] > 0.005).sum()   # Open past sig by > $0.005
    open_at = (gdf["gap_raw"].abs() <= 0.005).sum()  # Open ≈ sig
    open_before = (gdf["gap_raw"] < -0.005).sum()  # Open hasn't reached sig yet

    print(f"  {'Open past signal (adverse gap)':>40} {open_past:>6} {open_past/n*100:>6.1f}%")
    print(f"  {'Open ≈ signal (±$0.005)':>40} {open_at:>6} {open_at/n*100:>6.1f}%")
    print(f"  {'Open before signal (intrabar trigger)':>40} {open_before:>6} {open_before/n*100:>6.1f}%")

    # ═══ 3. Adverse gap size buckets ═══
    print(f"\n{'='*80}")
    print(f"  ADVERSE GAP SIZE DISTRIBUTION (only Open-past-signal cases)")
    print(f"{'='*80}")

    adverse = gdf[gdf["gap_raw"] > 0.005]
    if len(adverse) > 0:
        ag = adverse["gap_raw"].values
        print(f"\n  Count: {len(adverse)}  ({len(adverse)/n*100:.1f}% of all triggers)")
        print(f"  Mean:   ${np.mean(ag):.4f}")
        print(f"  Median: ${np.median(ag):.4f}")
        print(f"  P75:    ${np.percentile(ag, 75):.4f}")
        print(f"  P90:    ${np.percentile(ag, 90):.4f}")
        print(f"  P95:    ${np.percentile(ag, 95):.4f}")
        print(f"  P99:    ${np.percentile(ag, 99):.4f}")
        print(f"  Max:    ${np.max(ag):.4f}")

        print(f"\n  {'Gap range':<25} {'Count':>6} {'%':>7} {'Cum%':>7}")
        print(f"  {'-'*48}")
        buckets = [(0, 0.01), (0.01, 0.02), (0.02, 0.05), (0.05, 0.10),
                   (0.10, 0.20), (0.20, 0.50), (0.50, 1.00), (1.00, 99)]
        cum = 0
        for lo, hi in buckets:
            cnt = ((ag >= lo) & (ag < hi)).sum()
            cum += cnt
            label = f"${lo:.2f}-${hi:.2f}" if hi < 99 else f"${lo:.2f}+"
            print(f"  {label:<25} {cnt:>6} {cnt/len(adverse)*100:>6.1f}% {cum/len(adverse)*100:>6.1f}%")

    # ═══ 4. Gap as % of risk (1R) ═══
    print(f"\n{'='*80}")
    print(f"  GAP AS PERCENTAGE OF RISK (1R)")
    print(f"{'='*80}")

    gap_pct = gdf["gap_pct_of_risk"].values
    print(f"\n  Mean gap as % of risk: {np.mean(gap_pct):.1f}%")
    print(f"  Median:                {np.median(gap_pct):.1f}%")

    adverse_pct = gdf.loc[gdf["gap_raw"] > 0.005, "gap_pct_of_risk"].values
    if len(adverse_pct) > 0:
        print(f"\n  Adverse gaps only:")
        print(f"    Mean:   {np.mean(adverse_pct):.1f}% of 1R")
        print(f"    Median: {np.median(adverse_pct):.1f}% of 1R")
        print(f"    P95:    {np.percentile(adverse_pct, 95):.1f}% of 1R")

    # ═══ 5. Intrabar triggers: Open is BEFORE sig ═══
    print(f"\n{'='*80}")
    print(f"  INTRABAR TRIGGERS (Open < Signal, price reaches sig during bar)")
    print(f"{'='*80}")

    intrabar = gdf[gdf["gap_raw"] < -0.005]
    if len(intrabar) > 0:
        print(f"\n  Count: {len(intrabar)} ({len(intrabar)/n*100:.1f}% of triggers)")
        print(f"  These would fill at sig (correct), stop-buy works as intended")
        print(f"  → NO gap issue for these trades")

    # ═══ 6. Trigger delay analysis ═══
    print(f"\n{'='*80}")
    print(f"  TRIGGER DELAY (which bar triggers after bounce)")
    print(f"{'='*80}")

    for delay in [1, 2, 3]:
        sub = gdf[gdf["trigger_delay"] == delay]
        if len(sub) == 0: continue
        adv = sub[sub["gap_raw"] > 0.005]
        print(f"\n  Bar +{delay}: {len(sub)} triggers ({len(sub)/n*100:.1f}%)")
        print(f"    Adverse gaps: {len(adv)} ({len(adv)/len(sub)*100:.1f}%)")
        if len(adv) > 0:
            print(f"    Mean adverse gap: ${adv['gap_raw'].mean():.4f}")

    # ═══ 7. Time-of-day analysis ═══
    print(f"\n{'='*80}")
    print(f"  GAP BY TIME OF DAY")
    print(f"{'='*80}")

    gdf["hour"] = [t.hour for t in gdf["time"]]
    for hour in sorted(gdf["hour"].unique()):
        sub = gdf[gdf["hour"] == hour]
        adv = sub[sub["gap_raw"] > 0.005]
        mean_gap = sub["gap_raw"].mean()
        print(f"  {hour:02d}:xx  triggers={len(sub):>5}  adverse={len(adv):>4} ({len(adv)/len(sub)*100:>5.1f}%)"
              f"  mean_gap=${mean_gap:.4f}")

    # ═══ 8. Check: is this a 1-min bar boundary artifact? ═══
    print(f"\n{'='*80}")
    print(f"  DATA CHECK: Open vs prev Close (bar boundary gaps)")
    print(f"{'='*80}")

    # On 1-min bars, Open should be very close to previous Close
    # Large Open-prevClose gaps suggest data issues
    raw_df = pd.read_csv(DATA_PATH, index_col="timestamp", parse_dates=True)
    open_vs_prev = (raw_df["Open"] - raw_df["Close"].shift(1)).abs().dropna()
    print(f"\n  |Open - prevClose| across all bars:")
    print(f"    Mean:   ${open_vs_prev.mean():.4f}")
    print(f"    Median: ${open_vs_prev.median():.4f}")
    print(f"    P95:    ${open_vs_prev.quantile(0.95):.4f}")
    print(f"    P99:    ${open_vs_prev.quantile(0.99):.4f}")
    print(f"    Max:    ${open_vs_prev.max():.4f}")

    # Check how many are > $0.10
    big_jumps = (open_vs_prev > 0.10).sum()
    print(f"    Bars with |gap| > $0.10: {big_jumps} ({big_jumps/len(open_vs_prev)*100:.2f}%)")

    # Intraday only (exclude 9:30 bars)
    times_all = raw_df.index.time
    intraday_mask = [t > dt.time(9, 31) for t in times_all[1:]]
    intraday_gaps = open_vs_prev.iloc[:len(intraday_mask)][intraday_mask]
    print(f"\n  Intraday only (exclude 9:30-9:31):")
    print(f"    Mean:   ${intraday_gaps.mean():.4f}")
    print(f"    Median: ${intraday_gaps.median():.4f}")
    print(f"    P95:    ${intraday_gaps.quantile(0.95):.4f}")
    print(f"    P99:    ${intraday_gaps.quantile(0.99):.4f}")
    print(f"    Bars with |gap| > $0.10: {(intraday_gaps > 0.10).sum()}")

    # ═══ Verdict ═══
    print(f"\n{'='*80}")
    print(f"  VERDICT")
    print(f"{'='*80}")

    intrabar_pct = len(gdf[gdf["gap_raw"] < -0.005]) / n * 100
    adverse_pct_all = len(gdf[gdf["gap_raw"] > 0.005]) / n * 100
    print(f"\n  {intrabar_pct:.1f}% of triggers: Open < sig → stop-buy fills correctly at sig")
    print(f"  {adverse_pct_all:.1f}% of triggers: Open > sig → stop-buy should fill at Open, not sig")
    if len(adverse) > 0:
        print(f"  Average adverse gap: ${np.mean(ag):.4f} ({np.mean(adverse_pct):.1f}% of 1R)")
        print(f"  Median adverse gap: ${np.median(ag):.4f}")


if __name__ == "__main__":
    main()
