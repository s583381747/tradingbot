"""
Debug entry logic: dump bar-by-bar state to find why trade count is so low.
Check for implementation bugs in EMA, chop box, pullback state machine.
"""
from __future__ import annotations
import datetime, sys, importlib
from collections import deque

import pandas as pd
import numpy as np

DATA_PATH = "data/QQQ_1Min_2y_clean.csv"


def main():
    df = pd.read_csv(DATA_PATH, index_col="timestamp", parse_dates=True)
    print(f"Data: {len(df)} bars, {df.index[0]} → {df.index[-1]}\n")

    # Replicate the indicators manually to check
    ema20 = df["Close"].ewm(span=20, adjust=False).mean()
    ema50 = df["Close"].ewm(span=50, adjust=False).mean()
    atr_tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift(1)).abs(),
        (df["Low"] - df["Close"].shift(1)).abs()
    ], axis=1).max(axis=1)
    atr14 = atr_tr.rolling(14).mean()

    # EMA slope (3-bar, normalized)
    slope_period = 3
    ema_slope = (ema20 - ema20.shift(slope_period)) / slope_period / df["Close"] * 100

    # Avg abs slope (20 bar)
    avg_abs_slope = ema_slope.abs().rolling(20).mean()

    df["ema20"] = ema20
    df["ema50"] = ema50
    df["atr"] = atr14
    df["slope"] = ema_slope
    df["avg_abs_slope"] = avg_abs_slope

    # ═══ CHECK 1: How often is the market in "trend" mode? ═══
    print("=" * 80)
    print("CHECK 1: TREND DETECTION FREQUENCY")
    print("=" * 80)

    threshold = 0.012
    slope_ok = ema_slope.abs() >= threshold
    align_bull = (ema20 > ema50) & (ema_slope > 0)
    align_bear = (ema20 < ema50) & (ema_slope < 0)
    trend_bars = slope_ok & (align_bull | align_bear)

    total_bars = len(df.dropna())
    trend_count = trend_bars.sum()
    print(f"  Total bars: {total_bars}")
    print(f"  |slope| >= {threshold}: {slope_ok.sum()} ({slope_ok.sum()/total_bars*100:.1f}%)")
    print(f"  Trend (slope + alignment): {trend_count} ({trend_count/total_bars*100:.1f}%)")

    for t in [0.005, 0.008, 0.010, 0.012, 0.015]:
        s_ok = ema_slope.abs() >= t
        tr = s_ok & (align_bull | align_bear)
        print(f"    threshold={t}: {tr.sum()} trend bars ({tr.sum()/total_bars*100:.1f}%)")

    # ═══ CHECK 2: How often does chop box allow entry? ═══
    print(f"\n{'='*80}")
    print("CHECK 2: CHOP BOX STATE")
    print("=" * 80)

    # Simulate chop box
    in_chop = True
    box_high = 0.0
    box_low = 999999.0
    chop_bar_count = 0
    chop_state = []
    slope_hist = deque(maxlen=20)
    prev_date = None

    for i in range(len(df)):
        dt = df.index[i]
        cur_date = dt.date()

        # Daily reset
        if cur_date != prev_date:
            prev_date = cur_date
            in_chop = True
            box_high = df["High"].iloc[i]
            box_low = df["Low"].iloc[i]
            chop_bar_count = 1
            slope_hist.clear()

        s = df["slope"].iloc[i]
        if pd.isna(s):
            chop_state.append(True)
            continue

        slope_hist.append(abs(s))

        if len(slope_hist) < 5:
            box_high = max(box_high, df["High"].iloc[i])
            box_low = min(box_low, df["Low"].iloc[i])
            chop_bar_count += 1
            chop_state.append(True)
            continue

        avg_s = sum(slope_hist) / len(slope_hist)

        if in_chop:
            if avg_s >= threshold and chop_bar_count >= 5:
                if df["Close"].iloc[i] > box_high or df["Close"].iloc[i] < box_low:
                    in_chop = False
            if in_chop and avg_s < threshold:
                box_high = max(box_high, df["High"].iloc[i])
                box_low = min(box_low, df["Low"].iloc[i])
                chop_bar_count += 1
        else:
            if avg_s < threshold:
                in_chop = True
                box_high = df["High"].iloc[i]
                box_low = df["Low"].iloc[i]
                chop_bar_count = 1

        chop_state.append(in_chop)

    df["in_chop"] = chop_state[:len(df)]
    not_chop = (~df["in_chop"]).sum()
    print(f"  Bars NOT in chop: {not_chop} ({not_chop/total_bars*100:.1f}%)")
    print(f"  Bars in chop: {df['in_chop'].sum()} ({df['in_chop'].sum()/total_bars*100:.1f}%)")

    # Per day: how many minutes in trend mode?
    daily_trend_mins = df.groupby(df.index.date).apply(
        lambda g: (~g["in_chop"]).sum()
    )
    print(f"  Avg trend minutes/day: {daily_trend_mins.mean():.1f}")
    print(f"  Days with 0 trend minutes: {(daily_trend_mins == 0).sum()} / {len(daily_trend_mins)}")
    print(f"  Days with <30 trend minutes: {(daily_trend_mins < 30).sum()} / {len(daily_trend_mins)}")

    # ═══ CHECK 3: Pullback opportunities (price near EMA in trend) ═══
    print(f"\n{'='*80}")
    print("CHECK 3: PULLBACK OPPORTUNITIES")
    print("=" * 80)

    pb_mult = 1.2
    dist_to_ema = (df["Close"] - df["ema20"]).abs()
    in_pb_zone = dist_to_ema <= df["atr"] * pb_mult

    # Pullback in trend mode
    pb_in_trend = in_pb_zone & ~df["in_chop"] & trend_bars
    print(f"  Total pullback zone bars (trend mode): {pb_in_trend.sum()}")

    # Alternative: use Low/High for touch detection
    touch_long = (df["Low"] <= df["ema20"] + df["atr"] * pb_mult) & (df["Low"] >= df["ema20"] - df["atr"] * pb_mult)
    touch_short = (df["High"] >= df["ema20"] - df["atr"] * pb_mult) & (df["High"] <= df["ema20"] + df["atr"] * pb_mult)

    print(f"  Using Close distance: {in_pb_zone.sum()} bars in zone")
    print(f"  Using Low touch (long): {touch_long.sum()} bars")
    print(f"  Using High touch (short): {touch_short.sum()} bars")

    # ═══ CHECK 4: The state machine reset bug ═══
    print(f"\n{'='*80}")
    print("CHECK 4: TREND CONTINUITY (flicker analysis)")
    print("=" * 80)

    # How often does trend flicker 0→1→0 in short succession?
    trend_dir = pd.Series(0, index=df.index)
    trend_dir[align_bull & slope_ok] = 1
    trend_dir[align_bear & slope_ok] = -1

    # Count trend "gaps" (trend=0 for 1-3 bars then back)
    flickers = 0
    gap_len = 0
    in_gap = False
    last_trend = 0
    for i in range(len(trend_dir)):
        t = trend_dir.iloc[i]
        if t != 0:
            if in_gap and gap_len <= 3 and t == last_trend:
                flickers += 1
            in_gap = False
            gap_len = 0
            last_trend = t
        else:
            if last_trend != 0:
                in_gap = True
                gap_len += 1

    print(f"  Short trend gaps (1-3 bars, same direction resumes): {flickers}")
    print(f"  → Each one resets the pullback state machine!")
    print(f"  → This kills {flickers} potential entry setups")

    # ═══ CHECK 5: What if we use simpler trend detection? ═══
    print(f"\n{'='*80}")
    print("CHECK 5: SIMPLE TREND DETECTION (price vs EMA alignment)")
    print("=" * 80)

    # Simple: price above EMA20 AND EMA20 above EMA50 = bull
    simple_bull = (df["Close"] > df["ema20"]) & (df["ema20"] > df["ema50"])
    simple_bear = (df["Close"] < df["ema20"]) & (df["ema20"] < df["ema50"])
    simple_trend = simple_bull | simple_bear

    print(f"  Simple trend bars: {simple_trend.sum()} ({simple_trend.sum()/total_bars*100:.1f}%)")
    print(f"  Current complex trend bars: {trend_count} ({trend_count/total_bars*100:.1f}%)")
    print(f"  Ratio: simple gives {simple_trend.sum()/max(trend_count,1):.1f}x more trend bars")

    # ═══ CHECK 6: Data quality check ═══
    print(f"\n{'='*80}")
    print("CHECK 6: DATA QUALITY")
    print("=" * 80)

    # Check for gaps, zero volume, etc.
    zero_vol = (df["Volume"] == 0).sum()
    zero_range = (df["High"] == df["Low"]).sum()
    big_gaps = ((df["Open"] - df["Close"].shift(1)).abs() / df["Close"].shift(1) * 100 > 1).sum()

    print(f"  Zero volume bars: {zero_vol}")
    print(f"  Zero range bars (H=L): {zero_range}")
    print(f"  Big gaps (>1%): {big_gaps}")

    # IEX volume vs typical QQQ volume
    avg_vol = df["Volume"].mean()
    print(f"  Avg bar volume: {avg_vol:.0f}")
    print(f"  NOTE: IEX is ~2-3% of total QQQ volume. Prices may differ from SIP.")

    # ═══ SUMMARY ═══
    print(f"\n{'='*80}")
    print("DIAGNOSIS SUMMARY")
    print("=" * 80)
    print(f"""
  PROBLEM: Only 65 trades in 2 years (need 500+)

  ROOT CAUSES:
  1. Chop box daily reset: {(daily_trend_mins == 0).sum()}/{len(daily_trend_mins)} days have ZERO trend minutes
     → avg only {daily_trend_mins.mean():.0f} min/day in trend mode (out of 390)
  2. Trend flicker resets: {flickers} pullback setups killed by 1-3 bar trend gaps
  3. Trend detection too strict: only {trend_count/total_bars*100:.1f}% of bars in trend
     → Simple alignment gives {simple_trend.sum()/total_bars*100:.1f}% (much more)
  4. Pullback uses Close distance — misses candle wick touches
""")


if __name__ == "__main__":
    main()
