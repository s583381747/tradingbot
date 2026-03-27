"""
Blue Strategy 3: Multi-Timeframe Pullback (V3)
===============================================
Reverted to simpler V1 logic that worked (PF=1.33) with targeted improvements.

Top-down trend alignment:
1. Daily trend: close > EMA50 (bull) or close < EMA50 (bear) -- simple, no dual-EMA
2. 30-min pullback: Low touches 30-min EMA20 (bull) or High touches (bear)
   Close must be on the right side of EMA (not a breakdown)
3. 10-min entry: close crosses above EMA8 (bull) or below (bear)

Key difference from EMA20 touch baseline:
- Uses 3 timeframes instead of 1 (more selective)
- 30-min structure for stops (wider = lower cost/risk)
- Daily trend alignment reduces counter-trend trades

V3 changes from V1:
- Removed broken ffill-based touch detection
- Proper per-day iteration with 30-min bar scanning
- Simplified daily trend (just close vs EMA50, proven to work)
- Stop at 30-min swing low (5-bar) with buffer
- Allow up to 2 trades per day (V1 had 1)
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np, pandas as pd
from src.backtest_engine import load_nq, resample, COSTS, compute_stats, gap_through_fill

INSTRUMENT = "NQ"
NC = 1
DAILY_EMA = 50
TF_30M_EMA = 20
TF_10M_EMA = 8
ATR_30M_PERIOD = 14
TRAIL_ATR_MULT = 1.5
TRAIL_TRIGGER_R = 2.0
SWING_LOOKBACK = 5
PULLBACK_TOUCH_PCT = 0.3  # within 0.3% of 30m EMA
ENTRY_CUTOFF_HOUR = 14
EOD_HOUR = 15
EOD_MINUTE = 50
MAX_TRADES_DAY = 2


def run_backtest(start="2022-01-01", end="2026-12-31"):
    cost = COSTS[INSTRUMENT]
    pt_val = cost["pt_val"]

    df1 = load_nq(start="2021-06-01")
    df1 = df1[df1.index <= end]

    # Daily
    df_daily = df1.resample("D").agg(
        {"Open": "first", "High": "max", "Low": "min",
         "Close": "last", "Volume": "sum"}).dropna()
    df_daily["ema50"] = df_daily["Close"].ewm(span=DAILY_EMA, adjust=False).mean()

    # 30-min
    df30 = resample(df1, 30)
    df30["ema20"] = df30["Close"].ewm(span=TF_30M_EMA, adjust=False).mean()
    tr30 = np.maximum(df30["High"] - df30["Low"],
        np.maximum((df30["High"] - df30["Close"].shift(1)).abs(),
                   (df30["Low"] - df30["Close"].shift(1)).abs()))
    df30["atr"] = tr30.rolling(ATR_30M_PERIOD).mean()
    df30["swing_low"] = df30["Low"].rolling(SWING_LOOKBACK).min()
    df30["swing_high"] = df30["High"].rolling(SWING_LOOKBACK).max()

    # Pullback touch
    tol = PULLBACK_TOUCH_PCT / 100
    df30["touch_bull"] = (df30["Low"] <= df30["ema20"] * (1 + tol)) & \
                         (df30["Close"] > df30["ema20"])
    df30["touch_bear"] = (df30["High"] >= df30["ema20"] * (1 - tol)) & \
                         (df30["Close"] < df30["ema20"])
    df30["date"] = df30.index.date

    # 10-min
    df10 = resample(df1, 10)
    df10["ema8"] = df10["Close"].ewm(span=TF_10M_EMA, adjust=False).mean()
    df10["date"] = df10.index.date

    # Filter to period
    df10 = df10[df10.index >= start]
    dates = sorted(df10["date"].unique())

    trades = []

    for day in dates:
        # Daily trend (previous day)
        day_ts = pd.Timestamp(day)
        prev_daily = df_daily[df_daily.index < day_ts]
        if len(prev_daily) == 0:
            continue
        prev_row = prev_daily.iloc[-1]
        prev_close = prev_row["Close"]
        ema50 = prev_row["ema50"]
        if pd.isna(ema50):
            continue

        is_bull = prev_close > ema50
        is_bear = prev_close < ema50

        if not is_bull and not is_bear:
            continue

        # Get today's 30-min bars
        day30 = df30[df30["date"] == day]
        day10 = df10[df10["date"] == day]

        if len(day10) < 5 or len(day30) < 2:
            continue

        # Pre-compute 30-min touch signals with their metadata
        touch_events = []
        for j in range(len(day30)):
            bar30 = day30.iloc[j]
            ts30 = day30.index[j]
            if is_bull and bar30["touch_bull"]:
                touch_events.append({
                    "ts": ts30,
                    "direction": 1,
                    "swing_low": bar30["swing_low"],
                    "swing_high": bar30["swing_high"],
                    "atr": bar30["atr"],
                })
            elif is_bear and bar30["touch_bear"]:
                touch_events.append({
                    "ts": ts30,
                    "direction": -1,
                    "swing_low": bar30["swing_low"],
                    "swing_high": bar30["swing_high"],
                    "atr": bar30["atr"],
                })

        if len(touch_events) == 0:
            continue

        # Scan 10-min bars for entry
        in_trade = False
        entry_price = 0
        stop_price = 0
        direction = 0
        trail_active = False
        trail_stop = 0
        risk_pts = 0
        entry_ts = ""
        trade_count = 0
        active_touch = None  # current 30-min touch event we're looking to enter on
        touch_idx = 0

        for i in range(1, len(day10)):
            bar = day10.iloc[i]
            prev_bar = day10.iloc[i - 1]
            ts = day10.index[i]
            h, l, o, c = bar["High"], bar["Low"], bar["Open"], bar["Close"]

            # EOD exit
            if ts.hour >= EOD_HOUR and ts.minute >= EOD_MINUTE:
                if in_trade:
                    exit_price = c
                    pnl_pts = (exit_price - entry_price) * direction
                    cost_total = cost["comm_rt"] * NC + cost["spread"] * NC
                    pnl_dollar = pnl_pts * pt_val * NC - cost_total
                    r = pnl_pts / risk_pts if risk_pts > 0 else 0
                    trades.append({
                        "date": str(day), "entry_time": entry_ts, "exit_time": str(ts),
                        "direction": direction, "entry": entry_price, "exit": exit_price,
                        "pnl": pnl_dollar, "r": r, "exit_type": "eod",
                        "cost": cost_total, "risk": risk_pts * pt_val * NC,
                    })
                    in_trade = False
                break

            # Update active touch event
            while touch_idx < len(touch_events) and touch_events[touch_idx]["ts"] <= ts:
                active_touch = touch_events[touch_idx]
                touch_idx += 1

            if in_trade:
                atr_val = active_touch["atr"] if active_touch else 30
                if direction == 1:
                    eff_stop = trail_stop if trail_active else stop_price
                    if l <= eff_stop:
                        fill = gap_through_fill(eff_stop, o, 1)
                        pnl_pts = fill - entry_price
                        exit_type = "trail" if trail_active else "stop"
                        slip = cost["slip_stop"] * NC
                        cost_total = cost["comm_rt"] * NC + cost["spread"] * NC + slip
                        pnl_dollar = pnl_pts * pt_val * NC - cost_total
                        r = pnl_pts / risk_pts if risk_pts > 0 else 0
                        trades.append({
                            "date": str(day), "entry_time": entry_ts, "exit_time": str(ts),
                            "direction": direction, "entry": entry_price, "exit": fill,
                            "pnl": pnl_dollar, "r": r, "exit_type": exit_type,
                            "cost": cost_total, "risk": risk_pts * pt_val * NC,
                        })
                        in_trade = False
                        continue
                    if not trail_active and (h - entry_price) >= TRAIL_TRIGGER_R * risk_pts:
                        trail_active = True
                        trail_stop = h - TRAIL_ATR_MULT * atr_val
                    elif trail_active:
                        new_trail = h - TRAIL_ATR_MULT * atr_val
                        if new_trail > trail_stop:
                            trail_stop = new_trail

                elif direction == -1:
                    eff_stop = trail_stop if trail_active else stop_price
                    if h >= eff_stop:
                        fill = gap_through_fill(eff_stop, o, -1)
                        pnl_pts = entry_price - fill
                        exit_type = "trail" if trail_active else "stop"
                        slip = cost["slip_stop"] * NC
                        cost_total = cost["comm_rt"] * NC + cost["spread"] * NC + slip
                        pnl_dollar = pnl_pts * pt_val * NC - cost_total
                        r = pnl_pts / risk_pts if risk_pts > 0 else 0
                        trades.append({
                            "date": str(day), "entry_time": entry_ts, "exit_time": str(ts),
                            "direction": direction, "entry": entry_price, "exit": fill,
                            "pnl": pnl_dollar, "r": r, "exit_type": exit_type,
                            "cost": cost_total, "risk": risk_pts * pt_val * NC,
                        })
                        in_trade = False
                        continue
                    if not trail_active and (entry_price - l) >= TRAIL_TRIGGER_R * risk_pts:
                        trail_active = True
                        trail_stop = l + TRAIL_ATR_MULT * atr_val
                    elif trail_active:
                        new_trail = l + TRAIL_ATR_MULT * atr_val
                        if new_trail < trail_stop:
                            trail_stop = new_trail

            else:
                if trade_count >= MAX_TRADES_DAY:
                    continue
                if ts.hour >= ENTRY_CUTOFF_HOUR:
                    continue
                if active_touch is None:
                    continue

                atr_val = active_touch["atr"]
                if pd.isna(atr_val) or atr_val <= 0:
                    continue

                ema8 = bar["ema8"]
                prev_ema8 = prev_bar["ema8"]
                prev_c = prev_bar["Close"]
                if pd.isna(ema8) or pd.isna(prev_ema8):
                    continue

                # Long: 30m bullish touch + 10m close crosses above EMA8
                if active_touch["direction"] == 1 and prev_c <= prev_ema8 and c > ema8:
                    entry_price = c
                    sw_low = active_touch["swing_low"]
                    if pd.isna(sw_low):
                        continue
                    stop_price = sw_low - 2.0
                    risk_pts = entry_price - stop_price
                    if 5 < risk_pts < 100:
                        direction = 1
                        in_trade = True
                        trail_active = False
                        trail_stop = 0
                        entry_ts = str(ts)
                        trade_count += 1
                        active_touch = None  # consumed this touch
                        continue

                # Short: 30m bearish touch + 10m close crosses below EMA8
                elif active_touch["direction"] == -1 and prev_c >= prev_ema8 and c < ema8:
                    entry_price = c
                    sw_high = active_touch["swing_high"]
                    if pd.isna(sw_high):
                        continue
                    stop_price = sw_high + 2.0
                    risk_pts = stop_price - entry_price
                    if 5 < risk_pts < 100:
                        direction = -1
                        in_trade = True
                        trail_active = False
                        trail_stop = 0
                        entry_ts = str(ts)
                        trade_count += 1
                        active_touch = None
                        continue

    return pd.DataFrame(trades)


def main():
    print("=" * 70)
    print("BLUE STRATEGY 3: Multi-Timeframe Pullback (V3)")
    print("=" * 70)

    trades = run_backtest()
    if len(trades) == 0:
        print("No trades generated!")
        return

    trades["year"] = pd.to_datetime(trades["date"]).dt.year

    is_trades = trades[(trades["year"] >= 2022) & (trades["year"] <= 2023)]
    oos_trades = trades[(trades["year"] >= 2024)]
    full_trades = trades

    print(f"\nTotal trades: {len(trades)}")
    print(f"IS trades (2022-2023): {len(is_trades)}")
    print(f"OOS trades (2024-2026): {len(oos_trades)}")

    for label, tdf in [("IS 2022-2023", is_trades),
                        ("OOS 2024-2026", oos_trades),
                        ("FULL 4Y", full_trades)]:
        if len(tdf) == 0:
            continue
        s = compute_stats(tdf)
        print(f"\n--- {label} ---")
        print(f"  PF={s['pf']:.3f}  WR={s['wr']}%  N={s['n']}  DD=${s['dd']:.0f}")
        print(f"  $/day={s['dpnl']:.1f}  Sharpe={s['sharpe']:.2f}  Sortino={s['sortino']:.2f}")
        print(f"  APR={s['apr']:.1f}%  Calmar={s['calmar']:.2f}  5R+={s['b5']}")
        print(f"  AvgWinR={s['avg_win_r']:.3f}  AvgLossR={s['avg_loss_r']:.3f}")
        print(f"  Cost/Risk={s['cost_pct']:.1f}%  MCL={s['mcl']}")

    print("\n--- Yearly Breakdown ---")
    for yr in sorted(trades["year"].unique()):
        yt = trades[trades["year"] == yr]
        s = compute_stats(yt)
        print(f"  {yr}: PF={s['pf']:.3f} N={s['n']} PnL=${s['pnl']:.0f} DD=${s['dd']:.0f} "
              f"$/day={s['dpnl']:.1f} Sharpe={s['sharpe']:.2f} Cost/Risk={s['cost_pct']:.1f}%")

    print("\n--- Exit Types ---")
    for et in sorted(trades["exit_type"].unique()):
        sub = trades[trades["exit_type"] == et]
        print(f"  {et}: N={len(sub)} AvgPnL=${sub['pnl'].mean():.1f} AvgR={sub['r'].mean():.3f}")

    print("\n--- Direction ---")
    for d, lbl in [(1, "LONG"), (-1, "SHORT")]:
        sub = trades[trades["direction"] == d]
        if len(sub) > 0:
            s = compute_stats(sub)
            print(f"  {lbl}: N={s['n']} PF={s['pf']:.3f} AvgR={sub['r'].mean():.3f}")

    return trades


if __name__ == "__main__":
    main()
