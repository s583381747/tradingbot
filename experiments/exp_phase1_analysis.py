"""
Phase 1: Component-level statistical analysis.
No strategy code — pure pandas/numpy to understand each component's edge.

Experiments:
A. Chop Box quantification
B. EMA20 pullback quality + entry method comparison
C. Exit design (R:R optimization)
D. Component combination

All vectorized. Uses multiprocessing where applicable.
"""
from __future__ import annotations
import functools
import numpy as np
import pandas as pd

print = functools.partial(print, flush=True)

DATA_PATH = "data/QQQ_1Min_Polygon_2y_clean.csv"


def load_and_prepare():
    df = pd.read_csv(DATA_PATH, index_col="timestamp", parse_dates=True)

    # Core indicators
    df["ema20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["Close"].ewm(span=50, adjust=False).mean()

    tr = np.maximum(df["High"] - df["Low"],
                    np.maximum((df["High"] - df["Close"].shift(1)).abs(),
                               (df["Low"] - df["Close"].shift(1)).abs()))
    df["atr"] = tr.rolling(14).mean()

    # Volume MA
    df["vol_ma20"] = df["Volume"].rolling(20).mean()

    # Range compression (various periods)
    for n in [15, 20, 30]:
        rh = df["High"].rolling(n).max()
        rl = df["Low"].rolling(n).min()
        df[f"range_{n}"] = rh - rl
        df[f"range_atr_{n}"] = df[f"range_{n}"] / df["atr"]
        df[f"range_high_{n}"] = rh
        df[f"range_low_{n}"] = rl

    # Date and time
    df["date"] = df.index.date
    df["hour"] = df.index.hour
    df["minute"] = df.index.minute
    df["time_min"] = df["hour"] * 60 + df["minute"]

    # EMA alignment
    df["bull_align"] = (df["ema20"] > df["ema50"]).astype(int)
    df["bear_align"] = (df["ema20"] < df["ema50"]).astype(int)
    df["ema_gap"] = (df["ema20"] - df["ema50"]).abs() / df["atr"]

    # Forward returns (for measuring entry quality)
    for n in [10, 20, 30, 60, 120]:
        df[f"fwd_high_{n}"] = df["High"].shift(-1).rolling(n).max().shift(-(n - 1))
        df[f"fwd_low_{n}"] = df["Low"].shift(-1).rolling(n).min().shift(-(n - 1))
        df[f"fwd_close_{n}"] = df["Close"].shift(-n)

    return df.dropna(subset=["atr", "ema50", "fwd_close_30"])


def exp_a_chopbox(df):
    """Experiment A: Chop Box quantification."""
    print("=" * 80)
    print("EXPERIMENT A: CHOP BOX QUANTIFICATION")
    print("=" * 80)

    # Find EMA20 pullback bounces (baseline set for comparison)
    # Long bounce: low touches EMA20 zone, next bar close > this bar high
    touch_long = (df["Low"] <= df["ema20"] + df["atr"] * 1.2) & \
                 (df["Low"] >= df["ema20"] - df["atr"] * 1.2) & \
                 (df["bull_align"] == 1)
    bounce_long = touch_long & (df["Close"].shift(-1) > df["High"])

    # Short bounce: high touches EMA20 zone
    touch_short = (df["High"] >= df["ema20"] - df["atr"] * 1.2) & \
                  (df["High"] <= df["ema20"] + df["atr"] * 1.2) & \
                  (df["bear_align"] == 1)
    bounce_short = touch_short & (df["Close"].shift(-1) < df["Low"])

    bounces = df[bounce_long | bounce_short].copy()
    bounces["direction"] = np.where(bounce_long[bounces.index], 1, -1)

    # Win = 30-bar forward move > 2 ATR in direction
    bounces["fwd_move"] = np.where(
        bounces["direction"] == 1,
        (bounces["fwd_high_30"] - bounces["Close"]) / bounces["atr"],
        (bounces["Close"] - bounces["fwd_low_30"]) / bounces["atr"],
    )
    bounces["win"] = bounces["fwd_move"] >= 2.0

    baseline_wr = bounces["win"].mean()
    print(f"\n  Baseline (all bounces): {len(bounces)} entries, WR={baseline_wr:.1%}")

    # Test chop box filtering at various thresholds
    print(f"\n  Chop Box filter (range_atr_20 threshold):")
    print(f"  {'Threshold':<12} {'Chop%':>8} {'Entries':>8} {'WR':>8} {'WR Δ':>8} {'Avg FwdMove':>12}")

    for thresh in [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]:
        # "In chop" = range/ATR < threshold. Filter OUT entries in chop.
        in_chop = df[f"range_atr_20"] < thresh
        chop_pct = in_chop.mean() * 100

        # Only keep bounces NOT in chop
        filtered = bounces[~in_chop.reindex(bounces.index, fill_value=True)]
        if len(filtered) < 20:
            continue
        wr = filtered["win"].mean()
        delta = wr - baseline_wr
        avg_move = filtered["fwd_move"].mean()
        print(f"  <{thresh:<10} {chop_pct:>7.1f}% {len(filtered):>7} {wr:>7.1%} {delta:>+7.1%} {avg_move:>11.2f}")

    # Also test: entries IN chop vs NOT in chop
    print(f"\n  In-chop vs Out-of-chop comparison (threshold=4.0):")
    in_chop = df["range_atr_20"] < 4.0
    in_b = bounces[in_chop.reindex(bounces.index, fill_value=True)]
    out_b = bounces[~in_chop.reindex(bounces.index, fill_value=True)]
    if len(in_b) > 20 and len(out_b) > 20:
        print(f"    IN  chop: {len(in_b)} entries, WR={in_b['win'].mean():.1%}, AvgMove={in_b['fwd_move'].mean():.2f}")
        print(f"    OUT chop: {len(out_b)} entries, WR={out_b['win'].mean():.1%}, AvgMove={out_b['fwd_move'].mean():.2f}")

    return bounces


def exp_b_entry_quality(df, bounces):
    """Experiment B: Entry quality factors + entry method comparison."""
    print(f"\n{'=' * 80}")
    print("EXPERIMENT B: ENTRY QUALITY + ENTRY METHOD")
    print("=" * 80)

    # Factor 1: EMA gap (trend maturity)
    print("\n  Factor: EMA20-EMA50 gap (in ATR units)")
    for lo, hi, label in [(0, 0.5, "<0.5"), (0.5, 1.0, "0.5-1.0"), (1.0, 2.0, "1.0-2.0"), (2.0, 99, ">2.0")]:
        mask = (bounces["ema_gap"] >= lo) & (bounces["ema_gap"] < hi)
        sub = bounces[mask]
        if len(sub) < 20:
            continue
        print(f"    Gap {label:<8}: N={len(sub):>5} WR={sub['win'].mean():.1%} AvgMove={sub['fwd_move'].mean():.2f}")

    # Factor 2: Range/ATR at entry (just broke out vs established trend)
    print("\n  Factor: Range/ATR at entry")
    for lo, hi, label in [(0, 3, "<3"), (3, 4, "3-4"), (4, 5, "4-5"), (5, 7, "5-7"), (7, 99, ">7")]:
        mask = (bounces["range_atr_20"] >= lo) & (bounces["range_atr_20"] < hi)
        sub = bounces[mask]
        if len(sub) < 20:
            continue
        print(f"    R/ATR {label:<8}: N={len(sub):>5} WR={sub['win'].mean():.1%} AvgMove={sub['fwd_move'].mean():.2f}")

    # Factor 3: Time of day
    print("\n  Factor: Time of day")
    for start_h, start_m, end_h, end_m, label in [
        (9, 30, 10, 30, "09:30-10:30"), (10, 30, 12, 0, "10:30-12:00"),
        (12, 0, 14, 0, "12:00-14:00"), (14, 0, 15, 30, "14:00-15:30"),
    ]:
        start = start_h * 60 + start_m
        end = end_h * 60 + end_m
        mask = (bounces["time_min"] >= start) & (bounces["time_min"] < end)
        sub = bounces[mask]
        if len(sub) < 20:
            continue
        print(f"    {label}: N={len(sub):>5} WR={sub['win'].mean():.1%} AvgMove={sub['fwd_move'].mean():.2f}")

    # ═══ Entry method comparison ═══
    print(f"\n  --- ENTRY METHOD COMPARISON ---")

    # Method 1 (current): close > prev high (bar must complete)
    # This is what "bounces" already captures

    # Method 2: price touches prev high + offset → stop order entry
    # Simulate: entry at prev_high + 0.05. Win if high of next bar reaches that level.
    # For long: after wick touch EMA20, entry = current bar high + 0.05
    touch_long = (df["Low"] <= df["ema20"] + df["atr"] * 1.2) & \
                 (df["Low"] >= df["ema20"] - df["atr"] * 1.2) & \
                 (df["bull_align"] == 1)

    stop_entry_price = df["High"] + 0.05  # stop buy order at prev high + offset
    # Triggered if next bar's high >= stop_entry_price
    triggered = touch_long & (df["High"].shift(-1) >= stop_entry_price)

    stop_entries = df[triggered].copy()
    stop_entries["entry_price"] = stop_entry_price[triggered]
    stop_entries["direction"] = 1

    # Win: 30-bar fwd high from entry_price > entry_price + 2*ATR
    stop_entries["fwd_move"] = (stop_entries["fwd_high_30"] - stop_entries["entry_price"]) / stop_entries["atr"]
    stop_entries["win"] = stop_entries["fwd_move"] >= 2.0

    # Also compute for short side
    touch_short = (df["High"] >= df["ema20"] - df["atr"] * 1.2) & \
                  (df["High"] <= df["ema20"] + df["atr"] * 1.2) & \
                  (df["bear_align"] == 1)

    stop_sell_price = df["Low"] - 0.05
    triggered_s = touch_short & (df["Low"].shift(-1) <= stop_sell_price)
    stop_entries_s = df[triggered_s].copy()
    stop_entries_s["entry_price"] = stop_sell_price[triggered_s]
    stop_entries_s["direction"] = -1
    stop_entries_s["fwd_move"] = (stop_entries_s["entry_price"] - stop_entries_s["fwd_low_30"]) / stop_entries_s["atr"]
    stop_entries_s["win"] = stop_entries_s["fwd_move"] >= 2.0

    all_stop = pd.concat([stop_entries, stop_entries_s])

    print(f"\n  Method 1 (close > prev high): N={len(bounces)} WR={bounces['win'].mean():.1%}")
    print(f"  Method 2 (stop order at high+0.05): N={len(all_stop)} WR={all_stop['win'].mean():.1%}")
    print(f"  Method 2 entries/day: {len(all_stop)/500:.1f}")

    return all_stop


def exp_c_exit_design(df, bounces):
    """Experiment C: Exit design for R:R >= 3.0."""
    print(f"\n{'=' * 80}")
    print("EXPERIMENT C: EXIT DESIGN")
    print("=" * 80)

    # For each bounce entry, compute various exit scenarios
    # Use direction-aware forward prices

    long_b = bounces[bounces["direction"] == 1].copy()
    short_b = bounces[bounces["direction"] == -1].copy()

    # Stop-loss at various ATR levels below entry
    print("\n  Stop-Loss Sweep (fixed target = 3 ATR):")
    print(f"  {'Stop(ATR)':<12} {'WR':>8} {'AvgWin':>10} {'AvgLoss':>10} {'R:R':>8} {'PF':>8}")

    for stop_mult in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        # Long: stop at close - stop_mult * ATR, target at close + 3*ATR
        stop_dist = bounces["atr"] * stop_mult
        target_dist = bounces["atr"] * 3.0

        # Check if stop hit first or target hit first (simplified: use fwd_low/fwd_high)
        for n in [30, 60]:
            if bounces["direction"].iloc[0] == 1:  # just check structure
                pass

        # Simplified: use 30-bar forward data
        hit_stop = np.where(
            bounces["direction"] == 1,
            bounces["fwd_low_30"] <= bounces["Close"] - stop_dist,
            bounces["fwd_high_30"] >= bounces["Close"] + stop_dist,
        )
        hit_target = np.where(
            bounces["direction"] == 1,
            bounces["fwd_high_30"] >= bounces["Close"] + target_dist,
            bounces["fwd_low_30"] <= bounces["Close"] - target_dist,
        )

        # Outcome: win if target hit and stop not hit, or target hit before stop
        # Simplified: if both hit, assume stop hit first (conservative)
        win = hit_target & ~hit_stop
        lose = hit_stop.astype(bool)
        neutral = ~win & ~lose  # neither hit in 30 bars

        wins = win.sum()
        losses = lose.sum()
        neutrals = neutral.sum()
        total = len(bounces)
        wr = wins / total if total > 0 else 0

        avg_win = (target_dist[win]).mean() if wins > 0 else 0
        avg_loss = (stop_dist[lose]).mean() if losses > 0 else 0
        rr = avg_win / avg_loss if avg_loss > 0 else 0
        pf = (wins * avg_win) / (losses * avg_loss) if losses > 0 and avg_loss > 0 else 0

        print(f"  {stop_mult:<12} {wr:>7.1%} ${avg_win:>9.2f} ${avg_loss:>9.2f} {rr:>7.1f} {pf:>7.2f}")

    # Fixed stop, variable target
    print(f"\n  Target Sweep (fixed stop = 1.5 ATR):")
    print(f"  {'Target(ATR)':<12} {'WR':>8} {'R:R':>8} {'PF':>8} {'Expectancy':>12}")

    for target_mult in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
        stop_dist = bounces["atr"] * 1.5
        target_dist = bounces["atr"] * target_mult

        hit_stop = np.where(
            bounces["direction"] == 1,
            bounces["fwd_low_30"] <= bounces["Close"] - stop_dist,
            bounces["fwd_high_30"] >= bounces["Close"] + stop_dist,
        )
        hit_target = np.where(
            bounces["direction"] == 1,
            bounces["fwd_high_30"] >= bounces["Close"] + target_dist,
            bounces["fwd_low_30"] <= bounces["Close"] - target_dist,
        )

        win = hit_target & ~hit_stop
        lose = hit_stop.astype(bool)
        total = len(bounces)
        wins = win.sum()
        losses = lose.sum()
        wr = wins / total
        rr = target_mult / 1.5
        pf = (wins * target_mult) / (losses * 1.5) if losses > 0 else 0
        expectancy = wr * target_mult - (1 - wr) * 1.5  # in ATR per trade

        print(f"  {target_mult:<12} {wr:>7.1%} {rr:>7.1f} {pf:>7.2f} {expectancy:>11.3f} ATR")

    # Hold N bars then exit
    print(f"\n  Time Exit (hold N bars):")
    print(f"  {'Bars':<8} {'AvgPnL(ATR)':>12} {'WR':>8} {'Median':>10} {'Std':>10}")

    for n in [10, 20, 30, 60, 120]:
        col = f"fwd_close_{n}"
        if col not in bounces.columns:
            continue
        pnl = np.where(
            bounces["direction"] == 1,
            (bounces[col] - bounces["Close"]) / bounces["atr"],
            (bounces["Close"] - bounces[col]) / bounces["atr"],
        )
        pnl_s = pd.Series(pnl)
        wr = (pnl_s > 0).mean()
        print(f"  {n:<8} {pnl_s.mean():>11.3f} {wr:>7.1%} {pnl_s.median():>9.3f} {pnl_s.std():>9.3f}")


def exp_d_combination(df, bounces):
    """Experiment D: Best combination analysis."""
    print(f"\n{'=' * 80}")
    print("EXPERIMENT D: COMPONENT COMBINATION")
    print("=" * 80)

    # Best filters from A and B applied together
    # Chop filter: range_atr_20 >= best_threshold
    # Time filter: if any hour was significantly better

    for chop_thresh in [3.5, 4.0, 4.5]:
        not_chop = bounces["range_atr_20"] >= chop_thresh
        filtered = bounces[not_chop]
        if len(filtered) < 30:
            continue

        wr = filtered["win"].mean()
        avg_move = filtered["fwd_move"].mean()

        # Compute theoretical PF at various R:R targets
        for target in [2.0, 3.0]:
            stop = 1.5
            # Simplified PF estimate
            theoretical_pf = (wr * target) / ((1 - wr) * stop)

            print(f"  ChopFilter>={chop_thresh} | Target={target}ATR Stop={stop}ATR | "
                  f"N={len(filtered)} WR={wr:.1%} PF≈{theoretical_pf:.2f}")

    # Interaction: chop filter + EMA gap
    print(f"\n  Combined: ChopFilter + EMA Gap:")
    for chop_t in [4.0, 4.5]:
        for gap_lo in [0.5, 1.0]:
            mask = (bounces["range_atr_20"] >= chop_t) & (bounces["ema_gap"] >= gap_lo)
            sub = bounces[mask]
            if len(sub) < 20:
                continue
            wr = sub["win"].mean()
            n = len(sub)
            per_day = n / 500
            print(f"    Chop>={chop_t} + Gap>={gap_lo}: N={n} ({per_day:.1f}/day) WR={wr:.1%} AvgFwd={sub['fwd_move'].mean():.2f}")


def main():
    print("Loading data...")
    df = load_and_prepare()
    print(f"Ready: {len(df)} bars\n")

    bounces = exp_a_chopbox(df)
    stop_entries = exp_b_entry_quality(df, bounces)
    exp_c_exit_design(df, bounces)
    exp_d_combination(df, bounces)

    print(f"\n{'=' * 80}")
    print("PHASE 1 COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
