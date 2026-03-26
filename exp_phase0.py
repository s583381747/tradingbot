"""
Phase 0: Pre-Analysis — Answer 3 critical questions before experimenting.

0A: BE exit aftermath — what happens to price after BE stop-out?
0B: Trail stop aftermath — what happens after chandelier exit?
0C: Daily chop regime — does day-level trend/chop classification work?
0D: Time filters — quick wins (14:xx filter, win-after-win cooldown)
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


def run_detailed(df, capital=100_000, no_entry_after=None, skip_after_win=0):
    """Run with detailed per-trade + post-exit tracking."""
    p = BASE.copy()
    if no_entry_after is not None:
        p["no_entry_after"] = no_entry_after

    df = add_indicators(df, p)
    high = df["High"].values; low = df["Low"].values
    close = df["Close"].values
    ema = df["ema20"].values; ema_s = df["ema50"].values; atr_v = df["atr"].values
    times = df.index.time; dates = df.index.date; hours = df.index.hour; n = len(df)
    comm = p["commission_per_share"]

    equity = capital; trades = []
    bar = max(p["ema_slow"], p["atr_period"]) + 5
    daily_r_loss = 0.0; current_date = None
    skip_count = 0

    while bar < n - p["max_hold_bars"] - 70:  # extra room for post-exit analysis
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

        # Skip after win
        if skip_count > 0:
            skip_count -= 1
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
        exit_reason = "timeout"; bars_held = 0
        max_dd_r = 0.0

        for k in range(1, p["max_hold_bars"] + 1):
            bi = entry_bar + k
            if bi >= n: break
            h = high[bi]; l = low[bi]
            ca = atr_v[bi] if not np.isnan(atr_v[bi]) else a

            # Track max adverse for DD
            if trend == 1:
                adv = (actual_entry - l) / risk
            else:
                adv = (h - actual_entry) / risk
            max_dd_r = max(max_dd_r, adv)

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
                if (trend == 1 and h >= target) or (trend == -1 and l <= target):
                    trade_pnl += lock_sh * p["lock_rr"] * risk - lock_sh * comm
                    remaining -= lock_sh; lock_done = True
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

        # Post-exit price tracking
        post_r = {}
        for lookahead in [5, 10, 20, 40, 60]:
            fb = end_bar + lookahead
            if fb < n:
                if trend == 1:
                    post_move = (close[fb] - close[end_bar]) / risk
                else:
                    post_move = (close[end_bar] - close[fb]) / risk
                post_r[f"post_{lookahead}"] = post_move
            else:
                post_r[f"post_{lookahead}"] = np.nan

        # Post-exit MFE (max favorable after exit)
        post_mfe = 0.0
        for fb in range(end_bar + 1, min(end_bar + 61, n)):
            if trend == 1:
                fav = (high[fb] - close[end_bar]) / risk
            else:
                fav = (close[end_bar] - low[fb]) / risk
            post_mfe = max(post_mfe, fav)
        post_r["post_mfe_60"] = post_mfe

        if r_mult > 0:
            skip_count = skip_after_win

        # Daily ADX (approximate: use ADX at entry bar)
        # We need ADX — compute simply
        trades.append({
            "pnl": trade_pnl, "r": r_mult, "dir": trend,
            "exit": exit_reason, "bars_held": bars_held,
            "shares": shares, "risk": risk,
            "entry_bar": entry_bar, "end_bar": end_bar,
            "hour": hours[entry_bar], "date": dates[entry_bar],
            "atr": a, "max_dd_r": max_dd_r,
            **post_r,
        })
        bar = end_bar + 1

    return pd.DataFrame(trades), equity


def compute_daily_stats(df):
    """Compute daily regime indicators."""
    df = add_indicators(df, BASE)
    daily = df.groupby(df.index.date).agg(
        day_high=("High", "max"),
        day_low=("Low", "min"),
        day_open=("Open", "first"),
        day_close=("Close", "last"),
        mean_atr=("atr", "mean"),
        bars=("Close", "count"),
    )
    daily["day_range"] = daily["day_high"] - daily["day_low"]
    daily["range_atr"] = daily["day_range"] / daily["mean_atr"]

    # EMA cross count per day
    above = df["Close"] > df["ema20"]
    crosses = (above != above.shift(1)).astype(int)
    daily["ema_crosses"] = crosses.groupby(df.index.date).sum()

    return daily


def main():
    df = pd.read_csv(DATA_PATH, index_col="timestamp", parse_dates=True)
    print(f"Data: {len(df):,} bars\n")

    print("Running detailed backtest...")
    tdf, final_equity = run_detailed(df)
    total = len(tdf)

    gw = tdf.loc[tdf["r"] > 0, "r"].sum()
    gl = abs(tdf.loc[tdf["r"] <= 0, "r"].sum())
    base_pf = gw / gl if gl > 0 else 0
    base_ret = (final_equity - 100_000) / 100_000 * 100
    base_5r = (tdf["r"] >= 5).sum()
    base_max_dd = tdf["max_dd_r"].max()
    print(f"Baseline: PF={base_pf:.3f}, ret={base_ret:+.2f}%, trades={total}, 5R+={base_5r}, maxDD={base_max_dd:.2f}R\n")

    # ══════════════════════════════════════════════════════════════
    # 0A: BE EXIT AFTERMATH
    # ══════════════════════════════════════════════════════════════
    print(f"{'='*90}")
    print(f"  0A. BE EXIT AFTERMATH — What happens after BE stop-out?")
    print(f"{'='*90}")

    be = tdf[tdf["exit"] == "be_stop"].copy()
    print(f"\n  BE exits: {len(be)}")

    print(f"\n  Price move after BE exit (in R, favorable direction):")
    print(f"  {'Lookahead':<12} {'Mean':>8} {'Median':>8} {'P25':>8} {'P75':>8} {'% > +1R':>8} {'% > +2R':>8} {'% < -1R':>8}")
    print(f"  {'-'*75}")
    for la in [5, 10, 20, 40, 60]:
        col = f"post_{la}"
        vals = be[col].dropna().values
        if len(vals) == 0: continue
        print(f"  {la:>3} bars     {vals.mean():>+7.3f} {np.median(vals):>+7.3f}"
              f" {np.percentile(vals,25):>+7.3f} {np.percentile(vals,75):>+7.3f}"
              f" {(vals > 1).mean()*100:>7.1f}% {(vals > 2).mean()*100:>7.1f}% {(vals < -1).mean()*100:>7.1f}%")

    # Post-exit MFE
    post_mfe = be["post_mfe_60"].dropna().values
    print(f"\n  Post-exit MFE (max favorable in 60 bars after BE exit):")
    print(f"    Mean: {post_mfe.mean():.3f}R  Median: {np.median(post_mfe):.3f}R"
          f"  P75: {np.percentile(post_mfe, 75):.3f}R  P90: {np.percentile(post_mfe, 90):.3f}R")
    print(f"    Reached +1R after exit: {(post_mfe >= 1).mean()*100:.1f}%")
    print(f"    Reached +2R after exit: {(post_mfe >= 2).mean()*100:.1f}%")
    print(f"    Reached +5R after exit: {(post_mfe >= 5).mean()*100:.1f}%")

    # Verdict
    resumed_1r = (post_mfe >= 1).mean() * 100
    print(f"\n  ★ VERDICT 0A: {resumed_1r:.1f}% of BE exits resumed to +1R within 60 bars")
    if resumed_1r > 40:
        print(f"    → GO: Phase 1A full (放宽 BE 有意义)")
    elif resumed_1r > 20:
        print(f"    → PARTIAL: Phase 1A reduced (仅测 BE+buffer 和 time-based)")
    else:
        print(f"    → NO-GO: Phase 1A minimal (BE 是正确的)")

    # ══════════════════════════════════════════════════════════════
    # 0B: TRAIL STOP AFTERMATH
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*90}")
    print(f"  0B. TRAIL STOP AFTERMATH — Is chandelier exiting too early?")
    print(f"{'='*90}")

    ts = tdf[tdf["exit"] == "trail_stop"].copy()
    print(f"\n  Trail stop exits: {len(ts)}")

    print(f"\n  Price move after trail exit (in R, favorable direction):")
    print(f"  {'Lookahead':<12} {'Mean':>8} {'Median':>8} {'P25':>8} {'P75':>8} {'% > +1R':>8} {'% > +2R':>8}")
    print(f"  {'-'*65}")
    for la in [5, 10, 20, 40, 60]:
        col = f"post_{la}"
        vals = ts[col].dropna().values
        if len(vals) == 0: continue
        print(f"  {la:>3} bars     {vals.mean():>+7.3f} {np.median(vals):>+7.3f}"
              f" {np.percentile(vals,25):>+7.3f} {np.percentile(vals,75):>+7.3f}"
              f" {(vals > 1).mean()*100:>7.1f}% {(vals > 2).mean()*100:>7.1f}%")

    post_mfe_ts = ts["post_mfe_60"].dropna().values
    print(f"\n  Post-exit MFE (60 bars):")
    print(f"    Mean: {post_mfe_ts.mean():.3f}R  Median: {np.median(post_mfe_ts):.3f}R")
    print(f"    Reached +2R: {(post_mfe_ts >= 2).mean()*100:.1f}%")
    print(f"    Reached +5R: {(post_mfe_ts >= 5).mean()*100:.1f}%")

    # Lost profit: sum of post_mfe for trail stops that continued
    continued = post_mfe_ts[post_mfe_ts >= 2]
    print(f"    Trades that continued +2R: {len(continued)} ({len(continued)/len(ts)*100:.1f}%)")
    print(f"    Potential lost R from these: {continued.sum():.1f}R")

    resumed_2r = (post_mfe_ts >= 2).mean() * 100
    print(f"\n  ★ VERDICT 0B: {resumed_2r:.1f}% of trail stops continued +2R")
    if resumed_2r > 40:
        print(f"    → GO: Phase 1B full (chandelier 过早止出)")
    elif resumed_2r > 20:
        print(f"    → PARTIAL: Phase 1B reduced (仅测渐进 chandelier)")
    else:
        print(f"    → NO-GO: Phase 1B minimal (chandelier 已最优)")

    # ══════════════════════════════════════════════════════════════
    # 0C: DAILY CHOP REGIME
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*90}")
    print(f"  0C. DAILY CHOP REGIME — Does day-level classification work?")
    print(f"{'='*90}")

    daily_stats = compute_daily_stats(df)

    # Merge trade data with daily stats
    tdf["date_key"] = tdf["date"]
    daily_stats.index.name = "date_key"
    merged = tdf.merge(daily_stats, left_on="date_key", right_index=True, how="left")

    # Classify days
    # Method 1: Range/ATR
    ra_median = daily_stats["range_atr"].median()
    print(f"\n  Daily range/ATR median: {ra_median:.2f}")

    print(f"\n  Method 1: Daily Range/ATR split")
    print(f"  {'Category':<25} {'Days':>5} {'Trades':>7} {'PF':>7} {'AvgR':>8} {'5R+':>5} {'TotalR':>8}")
    print(f"  {'-'*70}")

    for label, mask_fn in [
        ("Low range (<P25)", lambda x: x["range_atr"] <= daily_stats["range_atr"].quantile(0.25)),
        ("Below median", lambda x: x["range_atr"] <= ra_median),
        ("Above median", lambda x: x["range_atr"] > ra_median),
        ("High range (>P75)", lambda x: x["range_atr"] > daily_stats["range_atr"].quantile(0.75)),
    ]:
        day_mask = mask_fn(daily_stats)
        good_dates = set(daily_stats[day_mask].index)
        sub = merged[merged["date_key"].isin(good_dates)]
        if len(sub) < 20: continue
        gw = sub.loc[sub["r"] > 0, "r"].sum()
        gl = abs(sub.loc[sub["r"] <= 0, "r"].sum())
        pf = gw / gl if gl > 0 else 0
        nd = len(good_dates)
        print(f"  {label:<25} {nd:>4} {len(sub):>6} {pf:>6.3f} {sub['r'].mean():>+7.3f}"
              f" {(sub['r']>=5).sum():>4} {sub['r'].sum():>+7.1f}R")

    # Method 2: EMA cross count
    cross_median = daily_stats["ema_crosses"].median()
    print(f"\n  Method 2: Daily EMA cross count (median={cross_median:.0f})")
    print(f"  {'Category':<25} {'Days':>5} {'Trades':>7} {'PF':>7} {'AvgR':>8} {'5R+':>5} {'TotalR':>8}")
    print(f"  {'-'*70}")

    for label, lo, hi in [
        ("Low crosses (0-P25)", 0, daily_stats["ema_crosses"].quantile(0.25)),
        ("Below median", 0, cross_median),
        ("Above median", cross_median + 1, 999),
        ("High crosses (>P75)", daily_stats["ema_crosses"].quantile(0.75) + 1, 999),
    ]:
        day_mask = (daily_stats["ema_crosses"] >= lo) & (daily_stats["ema_crosses"] <= hi)
        good_dates = set(daily_stats[day_mask].index)
        sub = merged[merged["date_key"].isin(good_dates)]
        if len(sub) < 20: continue
        gw = sub.loc[sub["r"] > 0, "r"].sum()
        gl = abs(sub.loc[sub["r"] <= 0, "r"].sum())
        pf = gw / gl if gl > 0 else 0
        nd = len(good_dates)
        print(f"  {label:<25} {nd:>4} {len(sub):>6} {pf:>6.3f} {sub['r'].mean():>+7.3f}"
              f" {(sub['r']>=5).sum():>4} {sub['r'].sum():>+7.1f}R")

    # Method 3: Combined (range + crosses)
    print(f"\n  Method 3: Combined regime (range/ATR > median AND crosses < median)")
    trend_days = daily_stats[(daily_stats["range_atr"] > ra_median) &
                              (daily_stats["ema_crosses"] <= cross_median)]
    chop_days = daily_stats[(daily_stats["range_atr"] <= ra_median) &
                             (daily_stats["ema_crosses"] > cross_median)]
    neutral_dates = set(daily_stats.index) - set(trend_days.index) - set(chop_days.index)

    print(f"  {'Category':<25} {'Days':>5} {'Trades':>7} {'PF':>7} {'AvgR':>8} {'5R+':>5} {'TotalR':>8}")
    print(f"  {'-'*70}")
    for label, day_set in [("Trend days", set(trend_days.index)),
                           ("Chop days", set(chop_days.index)),
                           ("Neutral days", neutral_dates)]:
        sub = merged[merged["date_key"].isin(day_set)]
        if len(sub) < 20: continue
        gw = sub.loc[sub["r"] > 0, "r"].sum()
        gl = abs(sub.loc[sub["r"] <= 0, "r"].sum())
        pf = gw / gl if gl > 0 else 0
        print(f"  {label:<25} {len(day_set):>4} {len(sub):>6} {pf:>6.3f} {sub['r'].mean():>+7.3f}"
              f" {(sub['r']>=5).sum():>4} {sub['r'].sum():>+7.1f}R")

    # Verdict
    trend_sub = merged[merged["date_key"].isin(set(trend_days.index))]
    chop_sub = merged[merged["date_key"].isin(set(chop_days.index))]
    trend_pf = trend_sub.loc[trend_sub["r"]>0,"r"].sum() / abs(trend_sub.loc[trend_sub["r"]<=0,"r"].sum()) if len(trend_sub) > 0 else 0
    chop_pf = chop_sub.loc[chop_sub["r"]>0,"r"].sum() / abs(chop_sub.loc[chop_sub["r"]<=0,"r"].sum()) if len(chop_sub) > 0 else 0

    print(f"\n  ★ VERDICT 0C: Trend PF={trend_pf:.3f}, Chop PF={chop_pf:.3f}")
    if trend_pf > 2.0 and chop_pf < 1.0:
        print(f"    → GO: 日级别 chop 过滤纳入 Phase 2")
    elif trend_pf / max(chop_pf, 0.01) > 1.5:
        print(f"    → PARTIAL: 有差异但不够大，作为 Phase 2 可选项")
    else:
        print(f"    → NO-GO: 日级别也没用，彻底放弃 chop 过滤")

    # ══════════════════════════════════════════════════════════════
    # 0D: TIME FILTERS
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*90}")
    print(f"  0D. TIME FILTERS")
    print(f"{'='*90}")

    configs = [
        ("Baseline (no filter)", None, 0),
        ("No entry after 14:00", dt.time(14, 0), 0),
        ("No entry after 13:30", dt.time(13, 30), 0),
        ("Skip 1 after win", None, 1),
        ("Skip 2 after win", None, 2),
        ("14:00 cutoff + skip 1", dt.time(14, 0), 1),
    ]

    print(f"\n  {'Config':<30} {'PF':>7} {'Ret%':>8} {'Trades':>7} {'5R+':>5} {'MaxDD':>7}")
    print(f"  {'-'*70}")
    for label, cutoff, skip in configs:
        t, eq = run_detailed(df, no_entry_after=cutoff, skip_after_win=skip)
        gw = t.loc[t["r"]>0,"r"].sum(); gl = abs(t.loc[t["r"]<=0,"r"].sum())
        pf = gw/gl if gl > 0 else 0
        ret = (eq - 100_000) / 100_000 * 100
        big5 = (t["r"] >= 5).sum()
        mdd = t["max_dd_r"].max()
        print(f"  {label:<30} {pf:>6.3f} {ret:>+7.2f}% {len(t):>6} {big5:>4} {mdd:>6.2f}R")

    # ══════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*90}")
    print(f"  PHASE 0 SUMMARY — GO / NO-GO DECISIONS")
    print(f"{'='*90}")
    print(f"\n  See verdicts above for 0A, 0B, 0C, 0D.")
    print(f"  Update RESEARCH_PLAN.md decision tree based on these results.")


if __name__ == "__main__":
    main()
