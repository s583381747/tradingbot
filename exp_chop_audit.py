"""
Chop Filter Audit — 5 critical validations before accepting the finding.

1. Look-ahead removal: only count trades AFTER 10:30
2. Date set overlap: 60min crosses vs retrospective trend days
3. Intraday consistency: first 60min regime vs rest-of-day regime
4. OOS validation: Barchart 2022-2024
5. Threshold stability: crosses 3-12 sweep on IS + OOS
"""
from __future__ import annotations
import functools, datetime as dt
import numpy as np, pandas as pd
from entry_signal import add_indicators, detect_trend, check_touch

print = functools.partial(print, flush=True)
IS_PATH = "data/QQQ_1Min_Polygon_2y_clean.csv"
OOS_PATH = "data/QQQ_1Min_Barchart_2y_2022-2024_clean.csv"

BASE = {
    "ema_fast": 20, "ema_slow": 50, "atr_period": 14,
    "touch_tol": 0.15, "touch_below_max": 0.5,
    "stop_buffer": 0.3,
    "lock_rr": 0.1, "lock_pct": 0.05,
    "chand_bars": 40, "chand_mult": 0.5,
    "max_hold_bars": 180,
    "risk_pct": 0.01, "max_pos_pct": 0.25,
    "force_close_at": dt.time(15, 58),
    "commission_per_share": 0.005, "daily_loss_r": 2.5,
    "skip_after_win": 1,
}


def compute_daily(df):
    df_ind = add_indicators(df, BASE)
    daily = {}
    for date, day_df in df_ind.groupby(df_ind.index.date):
        if len(day_df) < 100: continue
        c = day_df["Close"].values; ema20 = day_df["ema20"].values
        atr = day_df["atr"].values; h = day_df["High"].values; l = day_df["Low"].values
        mean_atr = np.nanmean(atr)
        if mean_atr <= 0 or np.isnan(mean_atr): continue

        above = c > ema20
        day_range = h.max() - l.min()

        # Full day stats (retrospective ground truth)
        all_crosses = sum(1 for i in range(1, len(above)) if above[i] != above[i-1])
        range_atr = day_range / mean_atr

        # First 60 min
        b60 = min(60, len(c))
        crosses_60 = sum(1 for i in range(1, b60) if above[i] != above[i-1])

        # Rest of day (after 60 min)
        rest_crosses = sum(1 for i in range(b60, len(above)) if above[i] != above[i-1]) if len(above) > b60 else 0
        rest_range = (max(h[b60:]) - min(l[b60:])) / mean_atr if len(h) > b60 else 0

        daily[date] = {
            "range_atr": range_atr, "all_crosses": all_crosses,
            "crosses_60": crosses_60, "rest_crosses": rest_crosses,
            "rest_range_atr": rest_range, "mean_atr": mean_atr,
        }
    return pd.DataFrame(daily).T


def run_backtest(df, capital=100_000, allowed_dates=None, no_entry_after=dt.time(14, 0),
                 no_entry_before=None):
    """Run with optional date filter and time window."""
    p = BASE.copy()
    p["no_entry_after"] = no_entry_after
    df = add_indicators(df, p)
    high = df["High"].values; low = df["Low"].values; close = df["Close"].values
    ema = df["ema20"].values; ema_s = df["ema50"].values; atr_v = df["atr"].values
    times = df.index.time; dates = df.index.date; n = len(df)
    comm = p["commission_per_share"]

    equity = capital; trades = []
    bar = max(p["ema_slow"], p["atr_period"]) + 5
    daily_r_loss = 0.0; current_date = None; skip_count = 0

    while bar < n - p["max_hold_bars"] - 5:
        a = atr_v[bar]
        if np.isnan(a) or a <= 0 or np.isnan(ema[bar]) or np.isnan(ema_s[bar]):
            bar += 1; continue
        if times[bar] >= p["no_entry_after"]: bar += 1; continue
        if no_entry_before and times[bar] < no_entry_before: bar += 1; continue
        d = dates[bar]
        if current_date != d: current_date = d; daily_r_loss = 0.0
        if daily_r_loss >= p["daily_loss_r"]: bar += 1; continue
        if allowed_dates is not None and d not in allowed_dates: bar += 1; continue

        trend = detect_trend(close[bar], ema[bar], ema_s[bar])
        if trend == 0: bar += 1; continue
        if not check_touch(trend, low[bar], high[bar], ema[bar], a,
                           p["touch_tol"], p["touch_below_max"]):
            bar += 1; continue
        if skip_count > 0: skip_count -= 1; bar += 1; continue

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

        for k in range(1, p["max_hold_bars"] + 1):
            bi = entry_bar + k
            if bi >= n: break
            h = high[bi]; l = low[bi]
            ca = atr_v[bi] if not np.isnan(atr_v[bi]) else a
            if times[bi] >= p["force_close_at"]:
                trade_pnl += remaining * (close[bi] - actual_entry) * trend - remaining * comm
                end_bar = bi; break
            stopped = (trend == 1 and l <= runner_stop) or (trend == -1 and h >= runner_stop)
            if stopped:
                trade_pnl += remaining * (runner_stop - actual_entry) * trend - remaining * comm
                end_bar = bi; break
            if not lock_done and remaining > lock_sh:
                target = actual_entry + p["lock_rr"] * risk * trend
                if (trend == 1 and h >= target) or (trend == -1 and l <= target):
                    trade_pnl += lock_sh * p["lock_rr"] * risk - lock_sh * comm
                    remaining -= lock_sh; lock_done = True
                    if trend == 1: runner_stop = max(runner_stop, actual_entry)
                    else: runner_stop = min(runner_stop, actual_entry)
            if lock_done and k >= p["chand_bars"]:
                sk = max(1, k - p["chand_bars"] + 1)
                hv = [high[entry_bar+kk] for kk in range(sk, k) if entry_bar+kk < n]
                lv = [low[entry_bar+kk] for kk in range(sk, k) if entry_bar+kk < n]
                if hv and lv:
                    if trend == 1: runner_stop = max(runner_stop, max(hv) - p["chand_mult"] * ca)
                    else: runner_stop = min(runner_stop, min(lv) + p["chand_mult"] * ca)
        else:
            ep = close[min(entry_bar + p["max_hold_bars"], n - 1)]
            trade_pnl += remaining * (ep - actual_entry) * trend - remaining * comm
            end_bar = min(entry_bar + p["max_hold_bars"], n - 1)

        equity += trade_pnl
        r_mult = trade_pnl / (shares * risk) if shares * risk > 0 else 0
        if trade_pnl < 0 and shares * risk > 0: daily_r_loss += abs(r_mult)
        if r_mult > 0: skip_count = 1

        trades.append({"pnl": trade_pnl, "r": r_mult, "dir": trend,
                        "shares": shares, "risk": risk})
        bar = end_bar + 1

    tdf = pd.DataFrame(trades) if trades else pd.DataFrame()
    total = len(tdf)
    if total == 0:
        return {"pf": 0, "ret": 0, "trades": 0, "big5": 0, "lpf": 0, "spf": 0, "tpd": 0}
    gw = tdf.loc[tdf["r"]>0, "r"].sum() if (tdf["r"]>0).any() else 0
    gl = abs(tdf.loc[tdf["r"]<=0, "r"].sum()) if (tdf["r"]<=0).any() else 0
    pf = gw / gl if gl > 0 else 0; ret = (equity - capital) / capital * 100
    r_arr = tdf["r"].values
    longs = tdf[tdf["dir"]==1]; shorts = tdf[tdf["dir"]==-1]
    def spf(s):
        if len(s)==0: return 0
        w = s.loc[s["pnl"]>0,"pnl"].sum(); l = abs(s.loc[s["pnl"]<=0,"pnl"].sum())
        return round(w/l,3) if l>0 else 0
    days = len(set(df.index.date[df.index.date >= df.index.date[0]]))
    return {"pf": round(pf,3), "ret": round(ret,2), "trades": total,
            "big5": int((r_arr>=5).sum()), "lpf": spf(longs), "spf": spf(shorts),
            "tpd": round(total / max(days, 1), 1)}


def main():
    df_is = pd.read_csv(IS_PATH, index_col="timestamp", parse_dates=True)
    df_oos = pd.read_csv(OOS_PATH, index_col="timestamp", parse_dates=True)
    print(f"IS: {len(df_is):,} bars | OOS: {len(df_oos):,} bars\n")

    daily_is = compute_daily(df_is)
    daily_oos = compute_daily(df_oos)

    # Retrospective ground truth (IS)
    ra_med = daily_is["range_atr"].median()
    cross_med = daily_is["all_crosses"].median()
    trend_retro = set(daily_is[(daily_is["range_atr"] > ra_med) & (daily_is["all_crosses"] <= cross_med)].index)
    chop_retro = set(daily_is[(daily_is["range_atr"] <= ra_med) & (daily_is["all_crosses"] > cross_med)].index)

    hdr = f"  {'Filter':<50} {'PF':>6} {'Ret%':>8} {'Trades':>7} {'5R+':>5} {'L.PF':>6} {'S.PF':>6}"
    sep = f"  {'-'*90}"

    # ═══════════════════════════════════════════════════════════════
    # AUDIT 1: Look-ahead removal
    # ═══════════════════════════════════════════════════════════════
    print(f"{'='*95}")
    print(f"  AUDIT 1: LOOK-AHEAD REMOVAL — Only trades after 10:30")
    print(f"{'='*95}")
    print(f"\n  Compare: full day entry vs 10:30+ only entry\n")
    print(hdr); print(sep)

    # Baseline: all day (9:30-14:00)
    r = run_backtest(df_is)
    print(f"  {'Baseline (9:30-14:00, no filter)':<50} {r['pf']:>6.3f} {r['ret']:>+7.2f}% {r['trades']:>6} {r['big5']:>4} {r['lpf']:>5.2f} {r['spf']:>5.2f}")

    # Baseline: 10:30+ only (no chop filter)
    r = run_backtest(df_is, no_entry_before=dt.time(10, 30))
    print(f"  {'Baseline (10:30-14:00, no filter)':<50} {r['pf']:>6.3f} {r['ret']:>+7.2f}% {r['trades']:>6} {r['big5']:>4} {r['lpf']:>5.2f} {r['spf']:>5.2f}")

    print()
    for thresh in [5, 8, 10]:
        good = set(daily_is[daily_is["crosses_60"] <= thresh].index)

        # Full day (HAS look-ahead for 9:30-10:30 trades)
        r_full = run_backtest(df_is, allowed_dates=good)
        print(f"  {'crosses60 ≤ '+str(thresh)+' (FULL day, has look-ahead)':<50} {r_full['pf']:>6.3f} {r_full['ret']:>+7.2f}% {r_full['trades']:>6} {r_full['big5']:>4} {r_full['lpf']:>5.2f} {r_full['spf']:>5.2f}")

        # 10:30+ only (NO look-ahead)
        r_clean = run_backtest(df_is, allowed_dates=good, no_entry_before=dt.time(10, 30))
        print(f"  {'crosses60 ≤ '+str(thresh)+' (10:30+ only, CLEAN)':<50} {r_clean['pf']:>6.3f} {r_clean['ret']:>+7.2f}% {r_clean['trades']:>6} {r_clean['big5']:>4} {r_clean['lpf']:>5.2f} {r_clean['spf']:>5.2f}")

        delta = r_clean['pf'] - r_full['pf']
        print(f"  {'  → Look-ahead impact: ΔPF = ':<50} {delta:>+.3f}")
        print()

    # ═══════════════════════════════════════════════════════════════
    # AUDIT 2: Date set overlap
    # ═══════════════════════════════════════════════════════════════
    print(f"{'='*95}")
    print(f"  AUDIT 2: DATE SET OVERLAP — 60min crosses vs retrospective")
    print(f"{'='*95}")

    for thresh in [5, 8]:
        cross_dates = set(daily_is[daily_is["crosses_60"] <= thresh].index)
        overlap = cross_dates & trend_retro
        only_cross = cross_dates - trend_retro
        only_retro = trend_retro - cross_dates

        print(f"\n  crosses_60 ≤ {thresh}:")
        print(f"    60min filter dates: {len(cross_dates)}")
        print(f"    Retro trend dates:  {len(trend_retro)}")
        print(f"    Overlap:            {len(overlap)} ({len(overlap)/len(cross_dates)*100:.1f}% of 60min)")
        print(f"    Only in 60min:      {len(only_cross)}")
        print(f"    Only in retro:      {len(only_retro)}")

        # How many chop days does 60min include?
        chop_leaked = cross_dates & chop_retro
        print(f"    Retro CHOP days included: {len(chop_leaked)} ({len(chop_leaked)/len(cross_dates)*100:.1f}%)")

    # ═══════════════════════════════════════════════════════════════
    # AUDIT 3: Intraday regime consistency
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*95}")
    print(f"  AUDIT 3: INTRADAY CONSISTENCY — first 60min vs rest of day")
    print(f"{'='*95}")

    # Correlation between first 60 min crosses and rest-of-day crosses
    valid = daily_is.dropna(subset=["crosses_60", "rest_crosses"])
    corr = valid["crosses_60"].corr(valid["rest_crosses"])
    print(f"\n  Correlation(crosses_60, rest_crosses): {corr:.3f}")

    # Bucketize
    print(f"\n  {'First 60min crosses':<25} {'Days':>5} {'Rest crosses mean':>18} {'Rest range/ATR':>15}")
    print(f"  {'-'*65}")
    for lo, hi in [(0,3), (3,5), (5,8), (8,12), (12,99)]:
        sub = valid[(valid["crosses_60"] >= lo) & (valid["crosses_60"] < hi)]
        if len(sub) < 5: continue
        print(f"  {lo:>3} - {hi:<3}               {len(sub):>5} {sub['rest_crosses'].mean():>17.1f} {sub['rest_range_atr'].mean():>14.2f}")

    # Key question: if first 60min is calm, does rest-of-day trend?
    calm_first = valid[valid["crosses_60"] <= 5]
    busy_first = valid[valid["crosses_60"] > 8]
    print(f"\n  Calm first hour (≤5 crosses): {len(calm_first)} days")
    print(f"    Rest-of-day crosses mean: {calm_first['rest_crosses'].mean():.1f}")
    print(f"    Rest-of-day range/ATR:    {calm_first['rest_range_atr'].mean():.2f}")
    print(f"  Busy first hour (>8 crosses): {len(busy_first)} days")
    print(f"    Rest-of-day crosses mean: {busy_first['rest_crosses'].mean():.1f}")
    print(f"    Rest-of-day range/ATR:    {busy_first['rest_range_atr'].mean():.2f}")

    # ═══════════════════════════════════════════════════════════════
    # AUDIT 4: OOS VALIDATION (Barchart 2022-2024)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*95}")
    print(f"  AUDIT 4: OOS VALIDATION — Barchart 2022-2024")
    print(f"{'='*95}")
    print(f"\n{hdr}"); print(sep)

    r = run_backtest(df_oos)
    print(f"  {'OOS Baseline (no filter)':<50} {r['pf']:>6.3f} {r['ret']:>+7.2f}% {r['trades']:>6} {r['big5']:>4} {r['lpf']:>5.2f} {r['spf']:>5.2f}")

    r = run_backtest(df_oos, no_entry_before=dt.time(10, 30))
    print(f"  {'OOS Baseline (10:30+, no filter)':<50} {r['pf']:>6.3f} {r['ret']:>+7.2f}% {r['trades']:>6} {r['big5']:>4} {r['lpf']:>5.2f} {r['spf']:>5.2f}")

    for thresh in [5, 6, 7, 8, 9, 10]:
        good = set(daily_oos[daily_oos["crosses_60"] <= thresh].index)
        r_full = run_backtest(df_oos, allowed_dates=good)
        r_clean = run_backtest(df_oos, allowed_dates=good, no_entry_before=dt.time(10, 30))
        print(f"  {'OOS crosses60 ≤ '+str(thresh)+' (full day)':<50} {r_full['pf']:>6.3f} {r_full['ret']:>+7.2f}% {r_full['trades']:>6} {r_full['big5']:>4} {r_full['lpf']:>5.2f} {r_full['spf']:>5.2f}")
        print(f"  {'OOS crosses60 ≤ '+str(thresh)+' (10:30+ CLEAN)':<50} {r_clean['pf']:>6.3f} {r_clean['ret']:>+7.2f}% {r_clean['trades']:>6} {r_clean['big5']:>4} {r_clean['lpf']:>5.2f} {r_clean['spf']:>5.2f}")

    # ═══════════════════════════════════════════════════════════════
    # AUDIT 5: Threshold stability sweep
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*95}")
    print(f"  AUDIT 5: THRESHOLD STABILITY — IS vs OOS")
    print(f"{'='*95}")
    print(f"\n  {'Thresh':<8} {'IS PF':>7} {'IS Ret':>8} {'IS Trd':>7} {'IS 5R+':>7}"
          f" {'OOS PF':>8} {'OOS Ret':>9} {'OOS Trd':>8} {'OOS 5R+':>8} {'ΔPF':>6}")
    print(f"  {'-'*80}")

    for thresh in range(3, 13):
        good_is = set(daily_is[daily_is["crosses_60"] <= thresh].index)
        good_oos = set(daily_oos[daily_oos["crosses_60"] <= thresh].index)

        r_is = run_backtest(df_is, allowed_dates=good_is, no_entry_before=dt.time(10, 30))
        r_oos = run_backtest(df_oos, allowed_dates=good_oos, no_entry_before=dt.time(10, 30))

        delta = r_oos["pf"] - r_is["pf"]
        print(f"  ≤ {thresh:<5} {r_is['pf']:>6.3f} {r_is['ret']:>+7.2f}% {r_is['trades']:>6} {r_is['big5']:>6}"
              f" {r_oos['pf']:>7.3f} {r_oos['ret']:>+8.2f}% {r_oos['trades']:>7} {r_oos['big5']:>7} {delta:>+5.3f}")

    # No filter baseline for comparison
    r_is_bl = run_backtest(df_is, no_entry_before=dt.time(10, 30))
    r_oos_bl = run_backtest(df_oos, no_entry_before=dt.time(10, 30))
    print(f"  {'None':<7} {r_is_bl['pf']:>6.3f} {r_is_bl['ret']:>+7.2f}% {r_is_bl['trades']:>6} {r_is_bl['big5']:>6}"
          f" {r_oos_bl['pf']:>7.3f} {r_oos_bl['ret']:>+8.2f}% {r_oos_bl['trades']:>7} {r_oos_bl['big5']:>7}")

    # ═══════════════════════════════════════════════════════════════
    # VERDICT
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*95}")
    print(f"  VERDICT")
    print(f"{'='*95}")
    print(f"\n  Check each audit item and conclude GO/NO-GO for chop filter.")


if __name__ == "__main__":
    main()
