"""
Blue Strategy 1: Opening Range Breakout (ORB-30)
=================================================
Trade the first 30-min range breakout in the direction of the daily trend.
This is the "classic ORB" implementation for comparison purposes.

After testing 8 variations (ORB, ORB+Fade, ORB+Hold, VWAP MR, Gap Momentum,
EOD Continuation, EMA Cross, Range Expansion, Inside Bar), the original ORB
with trend filter remains the most interpretable baseline.

Rules:
1. Opening Range: first 30 minutes (9:30-10:00 ET) define high/low
2. Trend filter: previous day close > EMA50 (bull) or < (bear)
   Dual EMA: EMA20 > EMA50 for strong bull
3. Entry: break above OR High (long, bull) or below OR Low (short, bear)
4. Stop: opposite side of Opening Range
5. Trail: 1.5x ATR (30-min) after 2R profit
6. EOD flat at 15:50
7. Max 1 trade per direction per day
8. No entries after 14:00 ET
9. OR range: 15-120 pts (filter extreme days)

Cost model: NQ x1 — comm $2.46, spread $5.00, slip $1.25 per contract
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np, pandas as pd
from src.backtest_engine import load_nq, COSTS, compute_stats, gap_through_fill

INSTRUMENT = "NQ"
NC = 1
OR_MINUTES = 30
ATR_PERIOD = 14
TRAIL_MULT = 1.5
MIN_OR_PTS = 15
MAX_OR_PTS = 120
ENTRY_CUTOFF_HOUR = 14
EOD_HOUR = 15
EOD_MINUTE = 50
BREAKOUT_BUFFER = 1.0
TRAIL_TRIGGER_R = 2.0
DAILY_EMA_FAST = 20
DAILY_EMA_SLOW = 50


def run_backtest(start="2022-01-01", end="2026-12-31"):
    cost = COSTS[INSTRUMENT]
    pt_val = cost["pt_val"]

    df = load_nq(start="2021-10-01")
    df = df[df.index <= end]

    # Daily trend (strong: EMA20 > EMA50 AND close > EMA50)
    df_daily = df.resample("D").agg(
        {"Open": "first", "High": "max", "Low": "min",
         "Close": "last", "Volume": "sum"}).dropna()
    df_daily["ema20"] = df_daily["Close"].ewm(span=DAILY_EMA_FAST, adjust=False).mean()
    df_daily["ema50"] = df_daily["Close"].ewm(span=DAILY_EMA_SLOW, adjust=False).mean()

    # 30-min EMA for trend filter + ATR for trailing
    df30 = df.resample("30min").agg(
        {"Open": "first", "High": "max", "Low": "min",
         "Close": "last", "Volume": "sum"}).dropna()
    df30["ema50"] = df30["Close"].ewm(span=50, adjust=False).mean()
    tr30 = np.maximum(df30["High"] - df30["Low"],
        np.maximum((df30["High"] - df30["Close"].shift(1)).abs(),
                   (df30["Low"] - df30["Close"].shift(1)).abs()))
    df30["atr"] = tr30.rolling(ATR_PERIOD).mean()

    df["ema50_30m"] = df30["ema50"].reindex(df.index, method="ffill")
    df["atr_30m"] = df30["atr"].reindex(df.index, method="ffill")

    df["date"] = df.index.date
    df = df[df.index >= start]
    dates = sorted(df["date"].unique())

    trades = []

    for day in dates:
        day_data = df[df["date"] == day]
        if len(day_data) < OR_MINUTES + 10:
            continue

        # Previous day trend
        day_ts = pd.Timestamp(day)
        prev_daily = df_daily[df_daily.index < day_ts]
        if len(prev_daily) == 0:
            continue
        prev_row = prev_daily.iloc[-1]
        prev_close = prev_row["Close"]
        ema20d = prev_row["ema20"]
        ema50d = prev_row["ema50"]
        if pd.isna(ema20d) or pd.isna(ema50d):
            continue

        # Use simple trend: close vs EMA50
        trend_up = prev_close > ema50d
        trend_down = prev_close < ema50d

        # Opening range
        or_bars = day_data.iloc[:OR_MINUTES]
        or_high = or_bars["High"].max()
        or_low = or_bars["Low"].min()
        or_range = or_high - or_low

        if or_range < MIN_OR_PTS or or_range > MAX_OR_PTS:
            continue

        # ATR at OR end
        prev_atr = day_data.iloc[OR_MINUTES - 1]["atr_30m"]
        if pd.isna(prev_atr) or prev_atr <= 0:
            continue

        or_mid = (or_high + or_low) / 2

        post_or = day_data.iloc[OR_MINUTES:]
        long_triggered = False
        short_triggered = False
        in_trade = False
        entry_price = 0
        stop_price = 0
        direction = 0
        trail_active = False
        trail_stop = 0
        risk_pts = 0
        entry_ts = ""

        for i in range(len(post_or)):
            bar = post_or.iloc[i]
            ts = post_or.index[i]
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

            if ts.hour >= ENTRY_CUTOFF_HOUR and not in_trade:
                continue

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
                        trail_stop = h - prev_atr * TRAIL_MULT
                    elif trail_active:
                        new_trail = h - prev_atr * TRAIL_MULT
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
                        trail_stop = l + prev_atr * TRAIL_MULT
                    elif trail_active:
                        new_trail = l + prev_atr * TRAIL_MULT
                        if new_trail < trail_stop:
                            trail_stop = new_trail

            else:
                # Long breakout with trend
                if trend_up and not long_triggered and h >= or_high + BREAKOUT_BUFFER:
                    entry_price = max(o, or_high + BREAKOUT_BUFFER)
                    stop_price = or_low
                    risk_pts = entry_price - stop_price
                    if risk_pts <= 0:
                        continue
                    direction = 1
                    in_trade = True
                    trail_active = False
                    trail_stop = 0
                    long_triggered = True
                    entry_ts = str(ts)

                if trend_down and not short_triggered and l <= or_low - BREAKOUT_BUFFER:
                    entry_price = min(o, or_low - BREAKOUT_BUFFER)
                    stop_price = or_high
                    risk_pts = stop_price - entry_price
                    if risk_pts <= 0:
                        continue
                    direction = -1
                    in_trade = True
                    trail_active = False
                    trail_stop = 0
                    short_triggered = True
                    entry_ts = str(ts)

    return pd.DataFrame(trades)


def main():
    print("=" * 70)
    print("BLUE STRATEGY 1: Opening Range Breakout (ORB-30)")
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
