"""
Blue Strategy 2: Volatility Squeeze Breakout (V2)
==================================================
Improved squeeze detection and entry logic:

1. Squeeze: BB(20,2) inside KC(20,1.5) for at least 4 bars on 10-min
2. Momentum filter: squeeze ends + momentum histogram turns positive/negative
   (using MACD-style momentum on the BB midline slope)
3. Entry: close of first bar after squeeze release that has momentum confirmation
4. Daily trend filter: only trade with daily EMA20 trend
5. Stop: 2x ATR below entry
6. Trail: 1.5x ATR after 2R profit
7. Max 1 trade per squeeze event, max 2 per day

V2 improvements:
- Relaxed KC multiplier to 1.5 (catches more squeezes)
- Added momentum confirmation to filter direction
- Daily trend alignment
- Tighter ATR-based stop instead of opposite BB band
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np, pandas as pd
from src.backtest_engine import load_nq, resample, COSTS, compute_stats, gap_through_fill

# ─── Parameters ───
INSTRUMENT = "NQ"
NC = 1
TF_MINUTES = 10
BB_PERIOD = 20
BB_STD = 2.0
KC_PERIOD = 20
KC_MULT = 1.5  # Keltner width
ATR_PERIOD = 14
STOP_ATR_MULT = 2.0
TRAIL_ATR_MULT = 1.5
TRAIL_TRIGGER_R = 2.0
MIN_SQUEEZE_BARS = 4
ENTRY_CUTOFF_HOUR = 14
ENTRY_CUTOFF_MINUTE = 0
EOD_HOUR = 15
EOD_MINUTE = 50
MOMENTUM_PERIOD = 12  # bars for momentum calculation
MAX_TRADES_DAY = 2


def run_backtest(start="2022-01-01", end="2026-12-31"):
    cost = COSTS[INSTRUMENT]
    pt_val = cost["pt_val"]

    # Load and resample
    df1 = load_nq(start="2021-06-01")
    df1 = df1[df1.index <= end]
    df = resample(df1, TF_MINUTES)

    # Daily trend
    df_daily = df1.resample("D").agg(
        {"Open": "first", "High": "max", "Low": "min",
         "Close": "last", "Volume": "sum"}).dropna()
    df_daily["ema20"] = df_daily["Close"].ewm(span=20, adjust=False).mean()
    df_daily["daily_trend"] = np.where(df_daily["Close"] > df_daily["ema20"], 1,
                               np.where(df_daily["Close"] < df_daily["ema20"], -1, 0))

    # Bollinger Bands
    df["bb_mid"] = df["Close"].rolling(BB_PERIOD).mean()
    df["bb_std"] = df["Close"].rolling(BB_PERIOD).std()
    df["bb_upper"] = df["bb_mid"] + BB_STD * df["bb_std"]
    df["bb_lower"] = df["bb_mid"] - BB_STD * df["bb_std"]

    # Keltner Channel
    tr = np.maximum(df["High"] - df["Low"],
        np.maximum((df["High"] - df["Close"].shift(1)).abs(),
                   (df["Low"] - df["Close"].shift(1)).abs()))
    df["atr"] = tr.rolling(ATR_PERIOD).mean()
    kc_mid = df["Close"].ewm(span=KC_PERIOD, adjust=False).mean()
    df["kc_upper"] = kc_mid + KC_MULT * df["atr"]
    df["kc_lower"] = kc_mid - KC_MULT * df["atr"]

    # Squeeze: BB inside KC
    df["squeeze"] = (df["bb_lower"] > df["kc_lower"]) & (df["bb_upper"] < df["kc_upper"])

    # Momentum: slope of BB midline (positive = up, negative = down)
    df["momentum"] = df["bb_mid"] - df["bb_mid"].shift(MOMENTUM_PERIOD)
    df["mom_accel"] = df["momentum"] - df["momentum"].shift(1)  # acceleration

    # Count consecutive squeeze bars
    df["sq_count"] = 0
    sq_count = 0
    sq_counts = []
    for sq in df["squeeze"]:
        if sq:
            sq_count += 1
        else:
            sq_count = 0
        sq_counts.append(sq_count)
    df["sq_count"] = sq_counts

    # Map daily trend
    df["date"] = df.index.date
    daily_trend_shifted = df_daily["daily_trend"].shift(1)

    # Filter to period
    df = df[df.index >= start]
    dates = sorted(df["date"].unique())

    trades = []

    for day in dates:
        day_data = df[df["date"] == day]
        if len(day_data) < 5:
            continue

        # Previous day trend
        day_ts = pd.Timestamp(day)
        prev_days = daily_trend_shifted.index[daily_trend_shifted.index <= day_ts]
        if len(prev_days) == 0:
            continue
        d_trend = daily_trend_shifted.loc[prev_days[-1]]
        if pd.isna(d_trend) or d_trend == 0:
            continue

        in_trade = False
        entry_price = 0
        stop_price = 0
        direction = 0
        trail_active = False
        trail_stop = 0
        risk_pts = 0
        entry_ts = ""
        trade_count = 0
        last_squeeze_end = -999  # prevent double-firing same squeeze

        for i in range(1, len(day_data)):
            bar = day_data.iloc[i]
            prev = day_data.iloc[i - 1]
            ts = day_data.index[i]
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

            if in_trade:
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
                        trail_stop = h - TRAIL_ATR_MULT * bar["atr"]
                    elif trail_active:
                        new_trail = h - TRAIL_ATR_MULT * bar["atr"]
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
                        trail_stop = l + TRAIL_ATR_MULT * bar["atr"]
                    elif trail_active:
                        new_trail = l + TRAIL_ATR_MULT * bar["atr"]
                        if new_trail < trail_stop:
                            trail_stop = new_trail

            else:
                if trade_count >= MAX_TRADES_DAY:
                    continue
                if ts.hour > ENTRY_CUTOFF_HOUR or (ts.hour == ENTRY_CUTOFF_HOUR and ts.minute >= ENTRY_CUTOFF_MINUTE):
                    continue

                atr_val = bar["atr"]
                mom = bar["momentum"]
                if pd.isna(atr_val) or atr_val <= 0 or pd.isna(mom):
                    continue

                # Squeeze release: previous bar was in squeeze (count >= MIN), current is not
                was_squeezed = prev["sq_count"] >= MIN_SQUEEZE_BARS
                currently_squeezed = bar["squeeze"]

                if was_squeezed and not currently_squeezed and (i - last_squeeze_end) > 3:
                    last_squeeze_end = i

                    # Direction from momentum + daily trend alignment
                    if mom > 0 and d_trend == 1:
                        # Bull squeeze release
                        entry_price = c
                        stop_price = entry_price - STOP_ATR_MULT * atr_val
                        risk_pts = entry_price - stop_price
                        if risk_pts > 3:
                            direction = 1
                            in_trade = True
                            trail_active = False
                            trail_stop = 0
                            entry_ts = str(ts)
                            trade_count += 1

                    elif mom < 0 and d_trend == -1:
                        # Bear squeeze release
                        entry_price = c
                        stop_price = entry_price + STOP_ATR_MULT * atr_val
                        risk_pts = stop_price - entry_price
                        if risk_pts > 3:
                            direction = -1
                            in_trade = True
                            trail_active = False
                            trail_stop = 0
                            entry_ts = str(ts)
                            trade_count += 1

    return pd.DataFrame(trades)


def main():
    print("=" * 70)
    print("BLUE STRATEGY 2: Volatility Squeeze Breakout (V2)")
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
