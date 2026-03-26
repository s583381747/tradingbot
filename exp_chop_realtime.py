"""
Chop Regime — Real-Time Detection

Phase 0C proved: trend days PF=2.28, chop days PF=0.65 (retrospective).
But we need REAL-TIME detection. Methods:

1. Previous day carryover: yesterday trend → today likely trend
2. Opening 30/60 min classification: first N minutes determine regime
3. Rolling intraday: continuously update regime estimate
4. Multi-day momentum: last 3/5 days trend → today likely trend
5. VIX proxy (ATR regime): high ATR periods = trending
6. Opening range size: wide open range = trend day
7. EMA alignment at open: strong trend structure at 9:31 = trend day
8. Hybrid: combine multiple signals

For each method, test:
  - Can we classify at 9:30 (pre-market)?
  - Can we classify at 10:00 (after 30 min)?
  - Rolling re-classification during the day?
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
    "no_entry_after": dt.time(14, 0),
    "force_close_at": dt.time(15, 58),
    "commission_per_share": 0.005, "daily_loss_r": 2.5,
    "skip_after_win": 1,
}


def compute_daily_regime(df):
    """Compute per-day regime indicators (used as ground truth and for retrospective methods)."""
    df_ind = add_indicators(df, BASE)

    daily = {}
    for date, day_df in df_ind.groupby(df_ind.index.date):
        if len(day_df) < 100:
            continue
        h = day_df["High"].values
        l = day_df["Low"].values
        c = day_df["Close"].values
        ema20 = day_df["ema20"].values
        atr = day_df["atr"].values
        mean_atr = np.nanmean(atr)
        if mean_atr <= 0 or np.isnan(mean_atr):
            continue

        day_range = h.max() - l.min()
        range_atr = day_range / mean_atr

        # EMA crosses
        above = c > ema20
        crosses = sum(1 for i in range(1, len(above)) if above[i] != above[i-1])

        # Opening range (first 30 min = first 30 bars)
        or_bars = min(30, len(h))
        open_range = max(h[:or_bars]) - min(l[:or_bars])
        open_range_atr = open_range / mean_atr if mean_atr > 0 else 0

        # First 60 min stats
        or60_bars = min(60, len(h))
        or60_range = max(h[:or60_bars]) - min(l[:or60_bars])
        or60_range_atr = or60_range / mean_atr if mean_atr > 0 else 0
        or60_crosses = sum(1 for i in range(1, or60_bars) if above[i] != above[i-1])

        # EMA alignment at open (9:31ish)
        ema50 = day_df["ema50"].values
        ema_aligned = abs(ema20[5] - ema50[5]) / mean_atr if len(ema20) > 5 else 0

        # ATR at open vs recent
        atr_open = atr[5] if len(atr) > 5 and not np.isnan(atr[5]) else mean_atr

        daily[date] = {
            "range_atr": range_atr,
            "crosses": crosses,
            "open_range_atr": open_range_atr,
            "or60_range_atr": or60_range_atr,
            "or60_crosses": or60_crosses,
            "ema_aligned": ema_aligned,
            "atr_open": atr_open,
            "mean_atr": mean_atr,
        }

    return pd.DataFrame(daily).T


def run_filtered(df, capital=100_000, allowed_dates=None):
    """Run backtest only on allowed dates."""
    p = BASE.copy()
    df = add_indicators(df, p)
    high = df["High"].values; low = df["Low"].values
    close = df["Close"].values
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
        d = dates[bar]
        if current_date != d: current_date = d; daily_r_loss = 0.0
        if daily_r_loss >= p["daily_loss_r"]: bar += 1; continue

        # Date filter
        if allowed_dates is not None and d not in allowed_dates:
            bar += 1; continue

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
                vals_h = [high[entry_bar + kk] for kk in range(sk, k) if entry_bar + kk < n]
                vals_l = [low[entry_bar + kk] for kk in range(sk, k) if entry_bar + kk < n]
                if vals_h and vals_l:
                    if trend == 1:
                        runner_stop = max(runner_stop, max(vals_h) - p["chand_mult"] * ca)
                    else:
                        runner_stop = min(runner_stop, min(vals_l) + p["chand_mult"] * ca)
        else:
            ep = close[min(entry_bar + p["max_hold_bars"], n - 1)]
            trade_pnl += remaining * (ep - actual_entry) * trend - remaining * comm
            end_bar = min(entry_bar + p["max_hold_bars"], n - 1)

        equity += trade_pnl
        r_mult = trade_pnl / (shares * risk) if shares * risk > 0 else 0
        if trade_pnl < 0 and shares * risk > 0:
            daily_r_loss += abs(r_mult)
        if r_mult > 0: skip_count = p.get("skip_after_win", 0)

        trades.append({"pnl": trade_pnl, "r": r_mult, "dir": trend,
                        "shares": shares, "risk": risk, "date": d})
        bar = end_bar + 1

    tdf = pd.DataFrame(trades) if trades else pd.DataFrame()
    total = len(tdf)
    if total == 0:
        return {"pf": 0, "ret": 0, "trades": 0, "tpd": 0, "big5": 0, "lpf": 0, "spf": 0}
    gw = tdf.loc[tdf["r"] > 0, "r"].sum() if (tdf["r"] > 0).any() else 0
    gl = abs(tdf.loc[tdf["r"] <= 0, "r"].sum()) if (tdf["r"] <= 0).any() else 0
    pf = gw / gl if gl > 0 else 0; ret = (equity - capital) / capital * 100
    days = len(set(tdf["date"])) if "date" in tdf.columns else 1
    longs = tdf[tdf["dir"] == 1]; shorts = tdf[tdf["dir"] == -1]
    def spf(s):
        if len(s) == 0: return 0
        w = s.loc[s["pnl"] > 0, "pnl"].sum(); l = abs(s.loc[s["pnl"] <= 0, "pnl"].sum())
        return round(w / l, 3) if l > 0 else 0
    r_arr = tdf["r"].values
    return {
        "pf": round(pf, 3), "ret": round(ret, 2),
        "trades": total, "tpd": round(total / max(days, 1), 1),
        "big5": int((r_arr >= 5).sum()),
        "lpf": spf(longs), "spf": spf(shorts),
        "days_traded": days,
    }


def main():
    df = pd.read_csv(DATA_PATH, index_col="timestamp", parse_dates=True)
    print(f"Data: {len(df):,} bars\n")

    # Baseline
    bl = run_filtered(df)
    all_dates = sorted(set(df.index.date))
    print(f"Baseline: PF={bl['pf']:.3f}, ret={bl['ret']:+.2f}%, trades={bl['trades']}, 5R+={bl['big5']}\n")

    # Compute daily stats
    print("Computing daily regime indicators...")
    daily = compute_daily_regime(df)
    print(f"Days analyzed: {len(daily)}\n")

    # ═══════════════════════════════════════════════════════════════
    # Ground truth: retrospective classification (from Phase 0C)
    # ═══════════════════════════════════════════════════════════════
    ra_median = daily["range_atr"].median()
    cross_median = daily["crosses"].median()

    trend_days_retro = set(daily[(daily["range_atr"] > ra_median) & (daily["crosses"] <= cross_median)].index)
    chop_days_retro = set(daily[(daily["range_atr"] <= ra_median) & (daily["crosses"] > cross_median)].index)

    print(f"Retrospective: {len(trend_days_retro)} trend days, {len(chop_days_retro)} chop days\n")

    # ═══════════════════════════════════════════════════════════════
    # METHOD 1: Previous day carryover
    # ═══════════════════════════════════════════════════════════════
    print(f"{'='*95}")
    print(f"  METHOD 1: Previous Day Carryover")
    print(f"{'='*95}")

    sorted_dates = sorted(daily.index)

    print(f"\n  {'Filter':<45} {'PF':>6} {'Ret%':>8} {'Trades':>7} {'Days':>5} {'5R+':>5} {'L.PF':>6} {'S.PF':>6}")
    print(f"  {'-'*90}")

    # 1a: yesterday was trend → trade today
    prev_trend = set()
    for i in range(1, len(sorted_dates)):
        if sorted_dates[i-1] in trend_days_retro:
            prev_trend.add(sorted_dates[i])
    r = run_filtered(df, allowed_dates=prev_trend)
    print(f"  {'Yesterday was trend → trade today':<45} {r['pf']:>6.3f} {r['ret']:>+7.2f}% {r['trades']:>6} {r['days_traded']:>4} {r['big5']:>4} {r['lpf']:>5.2f} {r['spf']:>5.2f}")

    # 1b: yesterday NOT chop → trade today
    prev_not_chop = set()
    for i in range(1, len(sorted_dates)):
        if sorted_dates[i-1] not in chop_days_retro:
            prev_not_chop.add(sorted_dates[i])
    r = run_filtered(df, allowed_dates=prev_not_chop)
    print(f"  {'Yesterday NOT chop → trade today':<45} {r['pf']:>6.3f} {r['ret']:>+7.2f}% {r['trades']:>6} {r['days_traded']:>4} {r['big5']:>4} {r['lpf']:>5.2f} {r['spf']:>5.2f}")

    # 1c: last 3 days average range/ATR > median
    multi_trend = set()
    for i in range(3, len(sorted_dates)):
        avg_ra = np.mean([daily.loc[sorted_dates[j], "range_atr"] for j in range(i-3, i)
                          if sorted_dates[j] in daily.index])
        if avg_ra > ra_median:
            multi_trend.add(sorted_dates[i])
    r = run_filtered(df, allowed_dates=multi_trend)
    print(f"  {'Last 3 days avg range/ATR > median':<45} {r['pf']:>6.3f} {r['ret']:>+7.2f}% {r['trades']:>6} {r['days_traded']:>4} {r['big5']:>4} {r['lpf']:>5.2f} {r['spf']:>5.2f}")

    # 1d: last 5 days average range/ATR > median
    multi5_trend = set()
    for i in range(5, len(sorted_dates)):
        avg_ra = np.mean([daily.loc[sorted_dates[j], "range_atr"] for j in range(i-5, i)
                          if sorted_dates[j] in daily.index])
        if avg_ra > ra_median:
            multi5_trend.add(sorted_dates[i])
    r = run_filtered(df, allowed_dates=multi5_trend)
    print(f"  {'Last 5 days avg range/ATR > median':<45} {r['pf']:>6.3f} {r['ret']:>+7.2f}% {r['trades']:>6} {r['days_traded']:>4} {r['big5']:>4} {r['lpf']:>5.2f} {r['spf']:>5.2f}")

    # ═══════════════════════════════════════════════════════════════
    # METHOD 2: Opening range classification
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*95}")
    print(f"  METHOD 2: Opening Range Classification (first 30/60 min)")
    print(f"{'='*95}")

    print(f"\n  {'Filter':<45} {'PF':>6} {'Ret%':>8} {'Trades':>7} {'Days':>5} {'5R+':>5} {'L.PF':>6} {'S.PF':>6}")
    print(f"  {'-'*90}")

    # 2a-d: opening 30-min range/ATR thresholds
    or_median = daily["open_range_atr"].median()
    for pct_label, pct in [("P25", 0.25), ("P40", 0.40), ("P50 (median)", 0.50), ("P60", 0.60)]:
        threshold = daily["open_range_atr"].quantile(pct)
        good = set(daily[daily["open_range_atr"] > threshold].index)
        r = run_filtered(df, allowed_dates=good)
        print(f"  {'30min range/ATR > ' + pct_label:<45} {r['pf']:>6.3f} {r['ret']:>+7.2f}% {r['trades']:>6} {r['days_traded']:>4} {r['big5']:>4} {r['lpf']:>5.2f} {r['spf']:>5.2f}")

    # 2e-h: opening 60-min range/ATR
    or60_median = daily["or60_range_atr"].median()
    for pct_label, pct in [("P25", 0.25), ("P40", 0.40), ("P50", 0.50), ("P60", 0.60)]:
        threshold = daily["or60_range_atr"].quantile(pct)
        good = set(daily[daily["or60_range_atr"] > threshold].index)
        r = run_filtered(df, allowed_dates=good)
        print(f"  {'60min range/ATR > ' + pct_label:<45} {r['pf']:>6.3f} {r['ret']:>+7.2f}% {r['trades']:>6} {r['days_traded']:>4} {r['big5']:>4} {r['lpf']:>5.2f} {r['spf']:>5.2f}")

    # 2i: 60-min low crosses
    for max_crosses in [3, 5, 8]:
        good = set(daily[daily["or60_crosses"] <= max_crosses].index)
        r = run_filtered(df, allowed_dates=good)
        print(f"  {'60min crosses <= ' + str(max_crosses):<45} {r['pf']:>6.3f} {r['ret']:>+7.2f}% {r['trades']:>6} {r['days_traded']:>4} {r['big5']:>4} {r['lpf']:>5.2f} {r['spf']:>5.2f}")

    # ═══════════════════════════════════════════════════════════════
    # METHOD 3: EMA alignment at open
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*95}")
    print(f"  METHOD 3: EMA Alignment at Market Open")
    print(f"{'='*95}")

    print(f"\n  {'Filter':<45} {'PF':>6} {'Ret%':>8} {'Trades':>7} {'Days':>5} {'5R+':>5} {'L.PF':>6} {'S.PF':>6}")
    print(f"  {'-'*90}")

    for pct_label, pct in [("P25", 0.25), ("P40", 0.40), ("P50", 0.50), ("P60", 0.60), ("P75", 0.75)]:
        threshold = daily["ema_aligned"].quantile(pct)
        good = set(daily[daily["ema_aligned"] > threshold].index)
        r = run_filtered(df, allowed_dates=good)
        print(f"  {'EMA gap at open > ' + pct_label:<45} {r['pf']:>6.3f} {r['ret']:>+7.2f}% {r['trades']:>6} {r['days_traded']:>4} {r['big5']:>4} {r['lpf']:>5.2f} {r['spf']:>5.2f}")

    # ═══════════════════════════════════════════════════════════════
    # METHOD 4: ATR regime (previous day ATR)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*95}")
    print(f"  METHOD 4: ATR Regime (previous day)")
    print(f"{'='*95}")

    print(f"\n  {'Filter':<45} {'PF':>6} {'Ret%':>8} {'Trades':>7} {'Days':>5} {'5R+':>5} {'L.PF':>6} {'S.PF':>6}")
    print(f"  {'-'*90}")

    for pct_label, pct in [("P25", 0.25), ("P40", 0.40), ("P50", 0.50)]:
        threshold = daily["mean_atr"].quantile(pct)
        prev_high_atr = set()
        for i in range(1, len(sorted_dates)):
            if sorted_dates[i-1] in daily.index and daily.loc[sorted_dates[i-1], "mean_atr"] > threshold:
                prev_high_atr.add(sorted_dates[i])
        r = run_filtered(df, allowed_dates=prev_high_atr)
        print(f"  {'Yesterday ATR > ' + pct_label:<45} {r['pf']:>6.3f} {r['ret']:>+7.2f}% {r['trades']:>6} {r['days_traded']:>4} {r['big5']:>4} {r['lpf']:>5.2f} {r['spf']:>5.2f}")

    # ═══════════════════════════════════════════════════════════════
    # METHOD 5: Hybrids (combine best signals)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*95}")
    print(f"  METHOD 5: Hybrid Combinations")
    print(f"{'='*95}")

    print(f"\n  {'Filter':<45} {'PF':>6} {'Ret%':>8} {'Trades':>7} {'Days':>5} {'5R+':>5} {'L.PF':>6} {'S.PF':>6}")
    print(f"  {'-'*90}")

    # 5a: prev day trend + OR > P40
    or_p40 = daily["open_range_atr"].quantile(0.40)
    hybrid_a = prev_trend & set(daily[daily["open_range_atr"] > or_p40].index)
    r = run_filtered(df, allowed_dates=hybrid_a)
    print(f"  {'Prev trend + OR30 > P40':<45} {r['pf']:>6.3f} {r['ret']:>+7.2f}% {r['trades']:>6} {r['days_traded']:>4} {r['big5']:>4} {r['lpf']:>5.2f} {r['spf']:>5.2f}")

    # 5b: prev NOT chop + OR > P40
    hybrid_b = prev_not_chop & set(daily[daily["open_range_atr"] > or_p40].index)
    r = run_filtered(df, allowed_dates=hybrid_b)
    print(f"  {'Prev NOT chop + OR30 > P40':<45} {r['pf']:>6.3f} {r['ret']:>+7.2f}% {r['trades']:>6} {r['days_traded']:>4} {r['big5']:>4} {r['lpf']:>5.2f} {r['spf']:>5.2f}")

    # 5c: prev trend + EMA gap > P40
    ema_p40 = daily["ema_aligned"].quantile(0.40)
    hybrid_c = prev_trend & set(daily[daily["ema_aligned"] > ema_p40].index)
    r = run_filtered(df, allowed_dates=hybrid_c)
    print(f"  {'Prev trend + EMA gap > P40':<45} {r['pf']:>6.3f} {r['ret']:>+7.2f}% {r['trades']:>6} {r['days_traded']:>4} {r['big5']:>4} {r['lpf']:>5.2f} {r['spf']:>5.2f}")

    # 5d: OR60 > P40 + 60min crosses <= 5
    hybrid_d = set(daily[(daily["or60_range_atr"] > daily["or60_range_atr"].quantile(0.40)) &
                          (daily["or60_crosses"] <= 5)].index)
    r = run_filtered(df, allowed_dates=hybrid_d)
    print(f"  {'OR60 > P40 + crosses60 <= 5':<45} {r['pf']:>6.3f} {r['ret']:>+7.2f}% {r['trades']:>6} {r['days_traded']:>4} {r['big5']:>4} {r['lpf']:>5.2f} {r['spf']:>5.2f}")

    # 5e: prev NOT chop + EMA gap > P40 + OR > P25
    or_p25 = daily["open_range_atr"].quantile(0.25)
    hybrid_e = prev_not_chop & set(daily[(daily["ema_aligned"] > ema_p40) &
                                          (daily["open_range_atr"] > or_p25)].index)
    r = run_filtered(df, allowed_dates=hybrid_e)
    print(f"  {'Prev!chop + EMA>P40 + OR>P25':<45} {r['pf']:>6.3f} {r['ret']:>+7.2f}% {r['trades']:>6} {r['days_traded']:>4} {r['big5']:>4} {r['lpf']:>5.2f} {r['spf']:>5.2f}")

    # 5f: 3-day trend momentum + OR > P40
    hybrid_f = multi_trend & set(daily[daily["open_range_atr"] > or_p40].index)
    r = run_filtered(df, allowed_dates=hybrid_f)
    print(f"  {'3d trend + OR30 > P40':<45} {r['pf']:>6.3f} {r['ret']:>+7.2f}% {r['trades']:>6} {r['days_traded']:>4} {r['big5']:>4} {r['lpf']:>5.2f} {r['spf']:>5.2f}")

    # 5g: just exclude chop (prev chop → don't trade)
    not_prev_chop = set()
    for i in range(1, len(sorted_dates)):
        if sorted_dates[i-1] not in chop_days_retro:
            not_prev_chop.add(sorted_dates[i])
    r = run_filtered(df, allowed_dates=not_prev_chop)
    print(f"  {'Exclude if yesterday was chop':<45} {r['pf']:>6.3f} {r['ret']:>+7.2f}% {r['trades']:>6} {r['days_traded']:>4} {r['big5']:>4} {r['lpf']:>5.2f} {r['spf']:>5.2f}")

    # 5h: 60min intraday: only trade after 10:00 if OR60 looks good
    # This one simulates starting to trade at 10:00 only on good days
    or60_p40 = daily["or60_range_atr"].quantile(0.40)
    good_after_60 = set(daily[daily["or60_range_atr"] > or60_p40].index)
    r = run_filtered(df, allowed_dates=good_after_60)
    print(f"  {'Trade only if OR60 > P40 (start 10:00)':<45} {r['pf']:>6.3f} {r['ret']:>+7.2f}% {r['trades']:>6} {r['days_traded']:>4} {r['big5']:>4} {r['lpf']:>5.2f} {r['spf']:>5.2f}")

    # ═══════════════════════════════════════════════════════════════
    # RETROSPECTIVE CEILING (how good can it get?)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*95}")
    print(f"  CEILING: Retrospective (perfect foresight)")
    print(f"{'='*95}")
    print(f"\n  {'Filter':<45} {'PF':>6} {'Ret%':>8} {'Trades':>7} {'Days':>5} {'5R+':>5}")
    print(f"  {'-'*80}")

    r = run_filtered(df, allowed_dates=trend_days_retro)
    print(f"  {'Only retrospective trend days':<45} {r['pf']:>6.3f} {r['ret']:>+7.2f}% {r['trades']:>6} {r['days_traded']:>4} {r['big5']:>4}")

    r = run_filtered(df, allowed_dates=set(all_dates) - chop_days_retro)
    print(f"  {'Exclude retrospective chop days':<45} {r['pf']:>6.3f} {r['ret']:>+7.2f}% {r['trades']:>6} {r['days_traded']:>4} {r['big5']:>4}")

    r = run_filtered(df)
    print(f"  {'No filter (baseline)':<45} {r['pf']:>6.3f} {r['ret']:>+7.2f}% {r['trades']:>6} {r['days_traded']:>4} {r['big5']:>4}")


if __name__ == "__main__":
    main()
