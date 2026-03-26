"""
Edge Analysis — Where does the profit actually come from?

Deep diagnostic of touch close + BE-lock champion:
  1. R-distribution: exactly which trades make money
  2. Time in trade: how long do winners vs losers last
  3. Win streaks / loss streaks: what does the P&L path look like
  4. BE exit analysis: how many trades move to BE then get stopped
  5. Chandelier contribution: profit from chandelier vs initial stop
  6. Long vs short deep dive
  7. Time of day edge
  8. ATR regime analysis (high vol vs low vol)
  9. Trend strength (EMA gap) vs performance
  10. Sequential dependency: does previous trade outcome affect next
"""
from __future__ import annotations
import functools, datetime as dt
import numpy as np, pandas as pd
from entry_signal import add_indicators, detect_trend, check_touch

print = functools.partial(print, flush=True)
DATA_PATH = "data/QQQ_1Min_Polygon_2y_clean.csv"

BASE = {
    "ema_fast": 20, "ema_slow": 50, "atr_period": 14,
    "touch_tol": 0.15, "touch_below_max": 0.5,
    "stop_buffer": 0.3,
    "lock_rr": 0.1, "lock_pct": 0.05,
    "chand_bars": 40, "chand_mult": 0.5,
    "max_hold_bars": 180,
    "risk_pct": 0.01, "max_pos_pct": 0.25,
    "no_entry_after": dt.time(15, 30), "force_close_at": dt.time(15, 58),
    "commission_per_share": 0.005, "daily_loss_r": 2.5,
}


def run_detailed(df, capital=100_000):
    """Run backtest with detailed per-trade logging."""
    p = BASE.copy()
    df = add_indicators(df, p)
    high = df["High"].values; low = df["Low"].values
    close = df["Close"].values; open_p = df["Open"].values
    ema = df["ema20"].values; ema_s = df["ema50"].values; atr_v = df["atr"].values
    times = df.index.time; dates = df.index.date; hours = df.index.hour; n = len(df)
    comm = p["commission_per_share"]

    equity = capital; trades = []
    bar = max(p["ema_slow"], p["atr_period"]) + 5
    daily_r_loss = 0.0; current_date = None

    while bar < n - p["max_hold_bars"] - 5:
        a = atr_v[bar]
        if np.isnan(a) or a <= 0 or np.isnan(ema[bar]) or np.isnan(ema_s[bar]):
            bar += 1; continue
        if times[bar] >= p["no_entry_after"]: bar += 1; continue
        d = dates[bar]
        if current_date != d: current_date = d; daily_r_loss = 0.0
        if daily_r_loss >= p["daily_loss_r"]: bar += 1; continue

        trend = detect_trend(close[bar], ema[bar], ema_s[bar])
        if trend == 0: bar += 1; continue
        if not check_touch(trend, low[bar], high[bar], ema[bar], a,
                           p["touch_tol"], p["touch_below_max"]):
            bar += 1; continue

        actual_entry = close[bar]
        stop = low[bar] - p["stop_buffer"] * a if trend == 1 else high[bar] + p["stop_buffer"] * a
        risk = abs(actual_entry - stop)
        if risk <= 0: bar += 1; continue
        entry_bar = bar

        shares = max(1, int(equity * p["risk_pct"] / risk))
        if shares * abs(actual_entry) > equity * p["max_pos_pct"]:
            shares = max(1, int(equity * p["max_pos_pct"] / abs(actual_entry)))
        if equity < shares * risk or shares < 1: bar += 1; continue

        lock_sh = max(1, int(shares * p["lock_pct"]))
        remaining = shares; runner_stop = stop; lock_done = False
        trade_pnl = -shares * comm; end_bar = entry_bar
        exit_reason = "timeout"
        max_favorable = 0.0  # max favorable excursion in R
        max_adverse = 0.0    # max adverse excursion in R
        be_moved = False
        bars_to_lock = -1
        bars_held = 0

        # EMA gap at entry (trend strength)
        ema_gap = abs(ema[bar] - ema_s[bar]) / a if a > 0 else 0

        for k in range(1, p["max_hold_bars"] + 1):
            bi = entry_bar + k
            if bi >= n: break
            h = high[bi]; l = low[bi]
            ca = atr_v[bi] if not np.isnan(atr_v[bi]) else a

            # Track excursions
            if trend == 1:
                fav = (h - actual_entry) / risk
                adv = (actual_entry - l) / risk
            else:
                fav = (actual_entry - l) / risk
                adv = (h - actual_entry) / risk
            max_favorable = max(max_favorable, fav)
            max_adverse = max(max_adverse, adv)

            if times[bi] >= p["force_close_at"]:
                trade_pnl += remaining * (close[bi] - actual_entry) * trend - remaining * comm
                end_bar = bi; exit_reason = "session_close"; bars_held = k; break

            stopped = (trend == 1 and l <= runner_stop) or (trend == -1 and h >= runner_stop)
            if stopped:
                trade_pnl += remaining * (runner_stop - actual_entry) * trend - remaining * comm
                end_bar = bi; bars_held = k
                if lock_done and abs(runner_stop - actual_entry) < 0.02:
                    exit_reason = "be_stop"
                elif lock_done:
                    exit_reason = "trail_stop"
                else:
                    exit_reason = "initial_stop"
                break

            if not lock_done and remaining > lock_sh:
                target = actual_entry + p["lock_rr"] * risk * trend
                hit = (trend == 1 and h >= target) or (trend == -1 and l <= target)
                if hit:
                    trade_pnl += lock_sh * p["lock_rr"] * risk - lock_sh * comm
                    remaining -= lock_sh; lock_done = True; be_moved = True
                    bars_to_lock = k
                    if trend == 1: runner_stop = max(runner_stop, actual_entry)
                    else: runner_stop = min(runner_stop, actual_entry)

            if lock_done and k >= p["chand_bars"]:
                sk = max(1, k - p["chand_bars"] + 1)
                if trend == 1:
                    hh = max(high[entry_bar + kk] for kk in range(sk, k) if entry_bar + kk < n)
                    runner_stop = max(runner_stop, hh - p["chand_mult"] * ca)
                else:
                    ll = min(low[entry_bar + kk] for kk in range(sk, k) if entry_bar + kk < n)
                    runner_stop = min(runner_stop, ll + p["chand_mult"] * ca)
        else:
            ep = close[min(entry_bar + p["max_hold_bars"], n - 1)]
            trade_pnl += remaining * (ep - actual_entry) * trend - remaining * comm
            end_bar = min(entry_bar + p["max_hold_bars"], n - 1)
            bars_held = p["max_hold_bars"]

        equity += trade_pnl
        r_mult = trade_pnl / (shares * risk) if shares * risk > 0 else 0
        if trade_pnl < 0 and shares * risk > 0:
            daily_r_loss += abs(r_mult)

        trades.append({
            "pnl": trade_pnl, "r": r_mult, "dir": trend,
            "shares": shares, "risk": risk,
            "exit": exit_reason, "lock": lock_done,
            "bars_held": bars_held, "bars_to_lock": bars_to_lock,
            "mfe": max_favorable, "mae": max_adverse,
            "hour": hours[entry_bar], "date": dates[entry_bar],
            "atr": a, "ema_gap": ema_gap,
            "entry_price": actual_entry,
        })
        bar = end_bar + 1

    return pd.DataFrame(trades)


def main():
    df = pd.read_csv(DATA_PATH, index_col="timestamp", parse_dates=True)
    print(f"Data: {len(df):,} bars\n")

    tdf = run_detailed(df)
    total = len(tdf)
    print(f"Total trades: {total}\n")

    # ═══ 1. R-Distribution ═══
    print(f"{'='*90}")
    print(f"  1. R-DISTRIBUTION — Where does the money come from?")
    print(f"{'='*90}")

    r = tdf["r"].values
    bins = [(-999,-1), (-1,-0.5), (-0.5,-0.1), (-0.1,0), (0,0.1), (0.1,0.5),
            (0.5,1), (1,2), (2,5), (5,10), (10,20), (20,999)]
    labels = ["<-1R","-.5 to -1R","-.1 to -.5R","0 to -.1R","0 to .1R",".1 to .5R",
              ".5 to 1R","1 to 2R","2 to 5R","5 to 10R","10 to 20R","20R+"]

    print(f"\n  {'Bucket':<15} {'Count':>6} {'%':>7} {'Total R':>9} {'% of Profit':>12} {'Cum R':>9}")
    print(f"  {'-'*60}")
    total_profit = r[r > 0].sum()
    total_loss = abs(r[r < 0].sum())
    cum_r = 0
    for (lo, hi), label in zip(bins, labels):
        mask = (r >= lo) & (r < hi)
        cnt = mask.sum()
        tr = r[mask].sum()
        cum_r += tr
        pct_profit = tr / total_profit * 100 if total_profit > 0 and tr > 0 else 0
        print(f"  {label:<15} {cnt:>6} {cnt/total*100:>6.1f}% {tr:>+8.1f}R {pct_profit:>10.1f}% {cum_r:>+8.1f}R")

    print(f"\n  Total profit R: {total_profit:+.1f}R")
    print(f"  Total loss R:   {-total_loss:+.1f}R")
    print(f"  Net R:          {total_profit - total_loss:+.1f}R")
    print(f"  PF (R-based):   {total_profit/total_loss:.3f}")

    # ═══ 2. Exit Reason Breakdown ═══
    print(f"\n{'='*90}")
    print(f"  2. EXIT REASON BREAKDOWN")
    print(f"{'='*90}")

    print(f"\n  {'Exit Reason':<20} {'Count':>6} {'%':>7} {'Avg R':>8} {'Total R':>9} {'Avg Bars':>9}")
    print(f"  {'-'*62}")
    for reason in sorted(tdf["exit"].unique()):
        sub = tdf[tdf["exit"] == reason]
        avg_r = sub["r"].mean()
        total_r = sub["r"].sum()
        avg_bars = sub["bars_held"].mean()
        print(f"  {reason:<20} {len(sub):>6} {len(sub)/total*100:>6.1f}% {avg_r:>+7.3f} {total_r:>+8.1f}R {avg_bars:>8.1f}")

    # ═══ 3. MFE/MAE Analysis ═══
    print(f"\n{'='*90}")
    print(f"  3. MFE/MAE — How far do trades go before exit?")
    print(f"{'='*90}")

    winners = tdf[tdf["r"] > 0]
    losers = tdf[tdf["r"] <= 0]

    print(f"\n  Winners ({len(winners)}):")
    print(f"    MFE: mean={winners['mfe'].mean():.2f}R  median={winners['mfe'].median():.2f}R  P95={winners['mfe'].quantile(0.95):.2f}R")
    print(f"    MAE: mean={winners['mae'].mean():.2f}R  median={winners['mae'].median():.2f}R")

    print(f"\n  Losers ({len(losers)}):")
    print(f"    MFE: mean={losers['mfe'].mean():.2f}R  median={losers['mfe'].median():.2f}R  P95={losers['mfe'].quantile(0.95):.2f}R")
    print(f"    MAE: mean={losers['mae'].mean():.2f}R  median={losers['mae'].median():.2f}R")

    # How many losers had MFE > 1R? (went profitable but came back)
    losers_profitable = losers[losers["mfe"] > 1.0]
    print(f"\n  Losers that reached 1R+ before losing: {len(losers_profitable)} ({len(losers_profitable)/len(losers)*100:.1f}%)")
    losers_be = losers[losers["mfe"] > 0.1]
    print(f"  Losers that reached 0.1R+ (lock trigger): {len(losers_be)} ({len(losers_be)/len(losers)*100:.1f}%)")

    # ═══ 4. BE Stop Deep Dive ═══
    print(f"\n{'='*90}")
    print(f"  4. BREAKEVEN STOP ANALYSIS")
    print(f"{'='*90}")

    be_trades = tdf[tdf["exit"] == "be_stop"]
    print(f"\n  BE exits: {len(be_trades)} ({len(be_trades)/total*100:.1f}% of all trades)")
    if len(be_trades) > 0:
        print(f"  Avg R of BE exits: {be_trades['r'].mean():+.4f}R (should be ≈ 0)")
        print(f"  Total R from BE exits: {be_trades['r'].sum():+.2f}R")
        print(f"  Avg bars to lock: {be_trades['bars_to_lock'].mean():.1f}")
        print(f"  Avg bars held: {be_trades['bars_held'].mean():.1f}")
        print(f"  Avg MFE before BE exit: {be_trades['mfe'].mean():.2f}R")
        print(f"  These trades reached avg {be_trades['mfe'].mean():.2f}R profit before coming back to BE")

    # ═══ 5. Time Analysis ═══
    print(f"\n{'='*90}")
    print(f"  5. BARS HELD — Time in Trade")
    print(f"{'='*90}")

    for label, sub in [("All", tdf), ("Winners (R>0)", winners), ("5R+ winners", tdf[tdf["r"]>=5]),
                        ("Initial stop", tdf[tdf["exit"]=="initial_stop"]),
                        ("BE stop", tdf[tdf["exit"]=="be_stop"]),
                        ("Trail stop", tdf[tdf["exit"]=="trail_stop"])]:
        if len(sub) == 0: continue
        print(f"\n  {label} ({len(sub)}):")
        print(f"    Bars: mean={sub['bars_held'].mean():.1f}  median={sub['bars_held'].median():.0f}"
              f"  P25={sub['bars_held'].quantile(0.25):.0f}  P75={sub['bars_held'].quantile(0.75):.0f}"
              f"  P95={sub['bars_held'].quantile(0.95):.0f}")

    # ═══ 6. Time of Day ═══
    print(f"\n{'='*90}")
    print(f"  6. TIME OF DAY")
    print(f"{'='*90}")

    print(f"\n  {'Hour':>6} {'Trades':>7} {'WR%':>6} {'Avg R':>8} {'Total R':>9} {'PF':>7} {'5R+':>5}")
    print(f"  {'-'*55}")
    for hour in sorted(tdf["hour"].unique()):
        sub = tdf[tdf["hour"] == hour]
        wr = (sub["r"] > 0).mean() * 100
        gw = sub.loc[sub["r"]>0, "r"].sum()
        gl = abs(sub.loc[sub["r"]<=0, "r"].sum())
        pf = gw/gl if gl > 0 else 0
        big5 = (sub["r"] >= 5).sum()
        print(f"  {hour:>5}:xx {len(sub):>6} {wr:>5.1f}% {sub['r'].mean():>+7.3f} {sub['r'].sum():>+8.1f}R {pf:>6.3f} {big5:>4}")

    # ═══ 7. Long vs Short Deep Dive ═══
    print(f"\n{'='*90}")
    print(f"  7. LONG vs SHORT")
    print(f"{'='*90}")

    for label, sub in [("LONG", tdf[tdf["dir"]==1]), ("SHORT", tdf[tdf["dir"]==-1])]:
        gw = sub.loc[sub["r"]>0, "r"].sum()
        gl = abs(sub.loc[sub["r"]<=0, "r"].sum())
        pf = gw/gl if gl > 0 else 0
        big5 = (sub["r"] >= 5).sum()
        be = sub[sub["exit"]=="be_stop"]
        init = sub[sub["exit"]=="initial_stop"]
        trail = sub[sub["exit"]=="trail_stop"]
        print(f"\n  {label} ({len(sub)} trades, PF={pf:.3f}):")
        print(f"    WR={( sub['r']>0).mean()*100:.1f}%  Avg R={sub['r'].mean():+.4f}  5R+={big5}")
        print(f"    Exit: initial_stop={len(init)} ({len(init)/len(sub)*100:.1f}%)"
              f"  be_stop={len(be)} ({len(be)/len(sub)*100:.1f}%)"
              f"  trail_stop={len(trail)} ({len(trail)/len(sub)*100:.1f}%)")
        print(f"    Avg bars: winners={sub[sub['r']>0]['bars_held'].mean():.1f}"
              f"  losers={sub[sub['r']<=0]['bars_held'].mean():.1f}")

    # ═══ 8. ATR Regime ═══
    print(f"\n{'='*90}")
    print(f"  8. ATR REGIME — Does volatility affect edge?")
    print(f"{'='*90}")

    atr_q = tdf["atr"].quantile([0, 0.2, 0.4, 0.6, 0.8, 1.0]).values
    print(f"\n  ATR quintiles: {['$'+f'{q:.3f}' for q in atr_q]}")
    print(f"\n  {'ATR Range':<20} {'Trades':>7} {'PF':>7} {'Avg R':>8} {'5R+':>5} {'Total R':>9}")
    print(f"  {'-'*60}")
    for i in range(5):
        lo, hi = atr_q[i], atr_q[i+1]
        sub = tdf[(tdf["atr"] >= lo) & (tdf["atr"] < hi + 0.0001)]
        gw = sub.loc[sub["r"]>0, "r"].sum()
        gl = abs(sub.loc[sub["r"]<=0, "r"].sum())
        pf = gw/gl if gl > 0 else 0
        print(f"  ${lo:.3f}-${hi:.3f}  {len(sub):>7} {pf:>6.3f} {sub['r'].mean():>+7.3f}"
              f" {(sub['r']>=5).sum():>4} {sub['r'].sum():>+8.1f}R")

    # ═══ 9. Trend Strength (EMA gap) ═══
    print(f"\n{'='*90}")
    print(f"  9. TREND STRENGTH — EMA20/EMA50 gap as ATR multiple")
    print(f"{'='*90}")

    eg_q = tdf["ema_gap"].quantile([0, 0.2, 0.4, 0.6, 0.8, 1.0]).values
    print(f"\n  {'EMA Gap':<20} {'Trades':>7} {'PF':>7} {'Avg R':>8} {'5R+':>5} {'Total R':>9}")
    print(f"  {'-'*60}")
    for i in range(5):
        lo, hi = eg_q[i], eg_q[i+1]
        sub = tdf[(tdf["ema_gap"] >= lo) & (tdf["ema_gap"] < hi + 0.0001)]
        gw = sub.loc[sub["r"]>0, "r"].sum()
        gl = abs(sub.loc[sub["r"]<=0, "r"].sum())
        pf = gw/gl if gl > 0 else 0
        print(f"  {lo:.2f}-{hi:.2f} ATR   {len(sub):>7} {pf:>6.3f} {sub['r'].mean():>+7.3f}"
              f" {(sub['r']>=5).sum():>4} {sub['r'].sum():>+8.1f}R")

    # ═══ 10. Sequential Dependency ═══
    print(f"\n{'='*90}")
    print(f"  10. SEQUENTIAL DEPENDENCY — Does losing streak affect next trade?")
    print(f"{'='*90}")

    # Count consecutive losses before each trade
    wins_losses = (tdf["r"] > 0).values
    streak = np.zeros(len(wins_losses), dtype=int)
    for i in range(1, len(wins_losses)):
        if not wins_losses[i-1]:  # previous was loss
            streak[i] = streak[i-1] + 1
        else:
            streak[i] = 0
    tdf["loss_streak"] = streak

    print(f"\n  {'Prev Losses':>12} {'Trades':>7} {'WR%':>6} {'Avg R':>8} {'PF':>7}")
    print(f"  {'-'*45}")
    for s in [0, 1, 2, 3, 5, 10, 15, 20]:
        if s < 20:
            sub = tdf[tdf["loss_streak"] == s]
            label = f"{s}"
        else:
            sub = tdf[tdf["loss_streak"] >= s]
            label = f"{s}+"
        if len(sub) < 10: continue
        gw = sub.loc[sub["r"]>0, "r"].sum()
        gl = abs(sub.loc[sub["r"]<=0, "r"].sum())
        pf = gw/gl if gl > 0 else 0
        wr = (sub["r"] > 0).mean() * 100
        print(f"  {label:>12} {len(sub):>6} {wr:>5.1f}% {sub['r'].mean():>+7.3f} {pf:>6.3f}")

    # ═══ 11. Loss streak distribution ═══
    print(f"\n  Loss streak distribution:")
    streaks = []
    current = 0
    for w in wins_losses:
        if not w:
            current += 1
        else:
            if current > 0:
                streaks.append(current)
            current = 0
    if current > 0:
        streaks.append(current)
    streaks = np.array(streaks)
    print(f"    Mean: {streaks.mean():.1f}  Median: {np.median(streaks):.0f}"
          f"  P95: {np.percentile(streaks, 95):.0f}  Max: {streaks.max()}")

    # ═══ 12. Profit concentration ═══
    print(f"\n{'='*90}")
    print(f"  12. PROFIT CONCENTRATION")
    print(f"{'='*90}")

    sorted_r = np.sort(r)[::-1]  # descending
    total_r = sorted_r.sum()
    for pct in [1, 2, 3, 5, 10, 20]:
        n_top = max(1, int(total * pct / 100))
        top_r = sorted_r[:n_top].sum()
        print(f"  Top {pct}% of trades ({n_top}): {top_r:+.1f}R = {top_r/total_r*100:.1f}% of total profit")


if __name__ == "__main__":
    main()
