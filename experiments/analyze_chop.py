"""
Chop Box Analysis: Quantify consolidation zones in QQQ 1-min data.

Goal: Find the best way to detect "chop" (range-bound price action)
so the trend-following strategy can avoid it.

Methods tested:
1. Price range compression (N-bar range vs ATR)
2. EMA20 cross frequency (many crosses = chop)
3. Choppiness Index (classic indicator)
4. ADX (directional movement)
5. EMA slope persistence (flat slope = chop)
"""

from __future__ import annotations
import pandas as pd
import numpy as np

DATA_PATH = "data/QQQ_1Min_2025-09-21_2026-03-21.csv"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, index_col="timestamp", parse_dates=True)
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all indicators needed for chop analysis."""
    # EMA20
    df["ema20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["Close"].ewm(span=50, adjust=False).mean()

    # ATR14
    df["tr"] = np.maximum(
        df["High"] - df["Low"],
        np.maximum(
            abs(df["High"] - df["Close"].shift(1)),
            abs(df["Low"] - df["Close"].shift(1)),
        ),
    )
    df["atr14"] = df["tr"].rolling(14).mean()

    # EMA20 slope (% per bar, 5-bar)
    df["ema20_slope"] = (df["ema20"] - df["ema20"].shift(5)) / 5 / df["Close"] * 100

    # EMA20 crosses: price crosses EMA20
    above = df["Close"] > df["ema20"]
    df["ema20_cross"] = above != above.shift(1)

    # Rolling range (high-low over N bars) / ATR = range compression ratio
    for n in [20, 30, 50]:
        roll_high = df["High"].rolling(n).max()
        roll_low = df["Low"].rolling(n).min()
        df[f"range_{n}"] = roll_high - roll_low
        df[f"range_atr_{n}"] = df[f"range_{n}"] / df["atr14"]

    # EMA20 cross count in last N bars
    for n in [20, 30, 50]:
        df[f"cross_count_{n}"] = df["ema20_cross"].astype(int).rolling(n).sum()

    # Choppiness Index: 100 * log10(sum(ATR, n) / (HH - LL)) / log10(n)
    for n in [14, 20, 30]:
        atr_sum = df["tr"].rolling(n).sum()
        hh = df["High"].rolling(n).max()
        ll = df["Low"].rolling(n).min()
        range_val = hh - ll
        range_val = range_val.replace(0, np.nan)
        df[f"chop_{n}"] = 100 * np.log10(atr_sum / range_val) / np.log10(n)

    # ADX (simplified)
    plus_dm = df["High"].diff()
    minus_dm = -df["Low"].diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

    atr_smooth = df["tr"].ewm(span=14, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(span=14, adjust=False).mean() / atr_smooth
    minus_di = 100 * minus_dm.ewm(span=14, adjust=False).mean() / atr_smooth
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
    df["adx"] = dx.ewm(span=14, adjust=False).mean()

    # Slope persistence: abs(slope) averaged over N bars
    for n in [10, 20, 30]:
        df[f"slope_abs_avg_{n}"] = df["ema20_slope"].abs().rolling(n).mean()

    return df


def mark_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """Mark intraday sessions (9:30-16:00)."""
    df["time"] = df.index.time
    df["date"] = df.index.date
    # Only keep market hours
    import datetime
    mask = (df["time"] >= datetime.time(9, 30)) & (df["time"] < datetime.time(16, 0))
    return df[mask].copy()


def analyze_future_moves(df: pd.DataFrame) -> pd.DataFrame:
    """For each bar, compute the max favorable/adverse move in next N bars."""
    for n in [15, 30, 60]:
        # Future max high and min low
        df[f"future_high_{n}"] = df["High"].shift(-1).rolling(n).max().shift(-(n - 1))
        df[f"future_low_{n}"] = df["Low"].shift(-1).rolling(n).min().shift(-(n - 1))
        # Max move up from close
        df[f"future_up_{n}"] = (df[f"future_high_{n}"] - df["Close"]) / df["atr14"]
        # Max move down from close
        df[f"future_down_{n}"] = (df["Close"] - df[f"future_low_{n}"]) / df["atr14"]
        # Directional move (absolute max move / ATR)
        df[f"future_dir_{n}"] = df[f"future_up_{n}"] - df[f"future_down_{n}"]
        # Range of future move (how trendy the next N bars are)
        df[f"future_range_{n}"] = (df[f"future_high_{n}"] - df[f"future_low_{n}"]) / df["atr14"]

    return df


def evaluate_chop_detector(df: pd.DataFrame, col: str, threshold: float,
                           chop_when: str = "above", horizon: int = 30):
    """Evaluate a chop detector: compare future trendiness when chop vs non-chop."""
    valid = df.dropna(subset=[col, f"future_range_{horizon}"])

    if chop_when == "above":
        is_chop = valid[col] >= threshold
    else:
        is_chop = valid[col] <= threshold

    chop_bars = valid[is_chop]
    trend_bars = valid[~is_chop]

    if len(chop_bars) < 100 or len(trend_bars) < 100:
        return None

    # Compare future range (trendiness) in chop vs non-chop
    chop_range = chop_bars[f"future_range_{horizon}"].mean()
    trend_range = trend_bars[f"future_range_{horizon}"].mean()

    # Compare absolute directional move
    chop_dir = chop_bars[f"future_dir_{horizon}"].abs().mean()
    trend_dir = trend_bars[f"future_dir_{horizon}"].abs().mean()

    # What % of bars are marked as chop
    chop_pct = len(chop_bars) / len(valid) * 100

    return {
        "col": col,
        "threshold": threshold,
        "chop_when": chop_when,
        "horizon": horizon,
        "chop_pct": round(chop_pct, 1),
        "chop_bars": len(chop_bars),
        "trend_bars": len(trend_bars),
        "chop_future_range": round(chop_range, 3),
        "trend_future_range": round(trend_range, 3),
        "separation": round(trend_range - chop_range, 3),
        "ratio": round(trend_range / chop_range, 3) if chop_range > 0 else 0,
        "chop_future_dir": round(chop_dir, 3),
        "trend_future_dir": round(trend_dir, 3),
    }


def main():
    print("=" * 80)
    print("CHOP BOX ANALYSIS: Quantifying consolidation zones")
    print("=" * 80)
    print()

    df = load_data()
    print(f"Raw bars: {len(df)}")

    df = add_indicators(df)
    df = mark_sessions(df)
    print(f"Market-hours bars: {len(df)}")

    df = analyze_future_moves(df)
    valid = df.dropna(subset=["future_range_30"])
    print(f"Valid bars (with 30-bar future): {len(valid)}")
    print()

    # ── Test each chop detection method ──────────────────────────
    tests = []

    # Method 1: Range compression (range/ATR over N bars)
    print("METHOD 1: Range Compression (N-bar range / ATR)")
    print("-" * 60)
    for n in [20, 30, 50]:
        for thresh in [2.0, 3.0, 4.0, 5.0, 6.0]:
            r = evaluate_chop_detector(df, f"range_atr_{n}", thresh, "below", 30)
            if r:
                tests.append(r)
                print(f"  range_atr_{n} < {thresh}: chop={r['chop_pct']:.0f}% | "
                      f"chop_range={r['chop_future_range']:.2f} trend_range={r['trend_future_range']:.2f} "
                      f"sep={r['separation']:.2f} ratio={r['ratio']:.2f}")
    print()

    # Method 2: EMA20 cross frequency
    print("METHOD 2: EMA20 Cross Frequency")
    print("-" * 60)
    for n in [20, 30, 50]:
        for thresh in [2, 3, 4, 5, 6]:
            r = evaluate_chop_detector(df, f"cross_count_{n}", thresh, "above", 30)
            if r:
                tests.append(r)
                print(f"  cross_count_{n} >= {thresh}: chop={r['chop_pct']:.0f}% | "
                      f"chop_range={r['chop_future_range']:.2f} trend_range={r['trend_future_range']:.2f} "
                      f"sep={r['separation']:.2f} ratio={r['ratio']:.2f}")
    print()

    # Method 3: Choppiness Index
    print("METHOD 3: Choppiness Index")
    print("-" * 60)
    for n in [14, 20, 30]:
        for thresh in [40, 50, 55, 60, 65]:
            r = evaluate_chop_detector(df, f"chop_{n}", thresh, "above", 30)
            if r:
                tests.append(r)
                print(f"  chop_{n} >= {thresh}: chop={r['chop_pct']:.0f}% | "
                      f"chop_range={r['chop_future_range']:.2f} trend_range={r['trend_future_range']:.2f} "
                      f"sep={r['separation']:.2f} ratio={r['ratio']:.2f}")
    print()

    # Method 4: ADX
    print("METHOD 4: ADX (low = chop)")
    print("-" * 60)
    for thresh in [15, 20, 25, 30, 35]:
        r = evaluate_chop_detector(df, "adx", thresh, "below", 30)
        if r:
            tests.append(r)
            print(f"  adx < {thresh}: chop={r['chop_pct']:.0f}% | "
                  f"chop_range={r['chop_future_range']:.2f} trend_range={r['trend_future_range']:.2f} "
                  f"sep={r['separation']:.2f} ratio={r['ratio']:.2f}")
    print()

    # Method 5: EMA slope persistence (low avg abs slope = chop)
    print("METHOD 5: EMA Slope Persistence")
    print("-" * 60)
    for n in [10, 20, 30]:
        for thresh in [0.005, 0.008, 0.010, 0.012, 0.015]:
            r = evaluate_chop_detector(df, f"slope_abs_avg_{n}", thresh, "below", 30)
            if r:
                tests.append(r)
                print(f"  slope_abs_{n} < {thresh}: chop={r['chop_pct']:.0f}% | "
                      f"chop_range={r['chop_future_range']:.2f} trend_range={r['trend_future_range']:.2f} "
                      f"sep={r['separation']:.2f} ratio={r['ratio']:.2f}")
    print()

    # ── Best detectors ──────────────────────────────────────────
    print("=" * 80)
    print("TOP 10 CHOP DETECTORS (by separation)")
    print("=" * 80)

    # Filter: chop_pct between 20-80% (useful filter, not too extreme)
    useful = [t for t in tests if 20 <= t["chop_pct"] <= 80]
    useful.sort(key=lambda x: x["separation"], reverse=True)

    for i, t in enumerate(useful[:10]):
        print(f"  #{i+1}: {t['col']} {t['chop_when']} {t['threshold']}")
        print(f"      chop={t['chop_pct']:.0f}% | separation={t['separation']:.3f} | ratio={t['ratio']:.2f}")
        print(f"      chop_range={t['chop_future_range']:.3f} vs trend_range={t['trend_future_range']:.3f}")
        print()

    # ── Also test at 60-bar horizon ─────────────────────────────
    print("=" * 80)
    print("TOP 5 at 60-bar horizon (matches signal timescale)")
    print("=" * 80)
    tests_60 = []
    for t in useful[:10]:
        r = evaluate_chop_detector(df, t["col"], t["threshold"], t["chop_when"], 60)
        if r:
            tests_60.append(r)

    tests_60.sort(key=lambda x: x["separation"], reverse=True)
    for i, t in enumerate(tests_60[:5]):
        print(f"  #{i+1}: {t['col']} {t['chop_when']} {t['threshold']}")
        print(f"      chop={t['chop_pct']:.0f}% | separation={t['separation']:.3f} | ratio={t['ratio']:.2f}")
        print()

    # ── Combined detector test ──────────────────────────────────
    print("=" * 80)
    print("COMBINED DETECTORS")
    print("=" * 80)

    # Test combining top 2 methods
    if len(useful) >= 2:
        # Try: range compression + cross count
        for range_n, range_t in [(30, 4.0), (30, 3.0), (20, 3.0)]:
            for cross_n, cross_t in [(30, 3), (30, 4), (20, 3)]:
                col_r = f"range_atr_{range_n}"
                col_c = f"cross_count_{cross_n}"
                if col_r not in df.columns or col_c not in df.columns:
                    continue

                is_chop = (df[col_r] <= range_t) | (df[col_c] >= cross_t)
                v = df.dropna(subset=[col_r, col_c, "future_range_30"])
                is_chop_v = is_chop.loc[v.index]

                chop_bars = v[is_chop_v]
                trend_bars = v[~is_chop_v]

                if len(chop_bars) < 100 and len(trend_bars) < 100:
                    continue

                chop_pct = len(chop_bars) / len(v) * 100
                chop_range = chop_bars["future_range_30"].mean()
                trend_range = trend_bars["future_range_30"].mean()
                sep = trend_range - chop_range

                print(f"  range_atr_{range_n}<{range_t} OR cross_{cross_n}>={cross_t}: "
                      f"chop={chop_pct:.0f}% sep={sep:.3f} "
                      f"(chop={chop_range:.2f} trend={trend_range:.2f})")

    print()
    print("DONE")


if __name__ == "__main__":
    main()
