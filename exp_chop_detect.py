"""
Chop Detection — Predictive Power Analysis

For every trade, compute chop indicators at entry, then measure:
  1. Distribution: indicator value for winners vs losers vs 5R+
  2. Separation: can the indicator distinguish good from bad trades?
  3. Optimal threshold: at what cutoff does filter maximize PF?
  4. Cost: how many 5R+ winners does each threshold sacrifice?

Chop indicators tested:
  A. ADX (Average Directional Index)
  B. EMA20 slope (normalized)
  C. Range compression (N-bar range / ATR)
  D. EMA cross frequency (crosses in last N bars)
  E. Choppiness Index
  F. EMA20/EMA50 gap (already have from edge analysis)
  G. Bollinger Bandwidth
  H. Efficiency Ratio (net move / total path)
  I. Consecutive same-side bars (price staying above/below EMA)
  J. Volume trend (rising vol = trend, flat = chop)
"""
from __future__ import annotations
import functools, datetime as dt
import numpy as np, pandas as pd
from entry_signal import add_indicators as base_indicators, detect_trend, check_touch

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


def add_chop_indicators(df):
    """Add all chop detection indicators to dataframe."""
    h = df["High"].values; l = df["Low"].values; c = df["Close"].values
    v = df["Volume"].values if "Volume" in df.columns else np.ones(len(df))
    ema20 = df["ema20"].values; ema50 = df["ema50"].values; atr = df["atr"].values
    n = len(df)

    # A. ADX (14-period)
    plus_dm = np.zeros(n); minus_dm = np.zeros(n)
    for i in range(1, n):
        up = h[i] - h[i-1]; down = l[i-1] - l[i]
        plus_dm[i] = up if up > down and up > 0 else 0
        minus_dm[i] = down if down > up and down > 0 else 0

    tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
    tr[0] = h[0] - l[0]

    period = 14
    atr_adx = np.zeros(n); plus_di = np.zeros(n); minus_di = np.zeros(n)
    atr_adx[period] = tr[1:period+1].sum()
    sm_plus = plus_dm[1:period+1].sum(); sm_minus = minus_dm[1:period+1].sum()
    for i in range(period + 1, n):
        atr_adx[i] = atr_adx[i-1] - atr_adx[i-1]/period + tr[i]
        sm_plus = sm_plus - sm_plus/period + plus_dm[i]
        sm_minus = sm_minus - sm_minus/period + minus_dm[i]
        if atr_adx[i] > 0:
            plus_di[i] = sm_plus / atr_adx[i] * 100
            minus_di[i] = sm_minus / atr_adx[i] * 100

    dx = np.zeros(n)
    for i in range(period, n):
        s = plus_di[i] + minus_di[i]
        dx[i] = abs(plus_di[i] - minus_di[i]) / s * 100 if s > 0 else 0

    adx = np.zeros(n)
    if n > 2 * period:
        adx[2*period] = dx[period+1:2*period+1].mean()
        for i in range(2*period+1, n):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    df["adx"] = adx

    # B. EMA20 slope (5-bar, normalized by price)
    slope = np.zeros(n)
    for i in range(5, n):
        if c[i] > 0:
            slope[i] = (ema20[i] - ema20[i-5]) / 5 / c[i] * 10000  # basis points per bar
    df["ema_slope"] = slope
    df["ema_slope_abs"] = np.abs(slope)

    # C. Range compression (20-bar range / ATR)
    for lookback in [20, 30, 50]:
        rc = np.zeros(n)
        for i in range(lookback, n):
            rng = max(h[i-lookback:i+1]) - min(l[i-lookback:i+1])
            rc[i] = rng / atr[i] if atr[i] > 0 else 0
        df[f"range_atr_{lookback}"] = rc

    # D. EMA cross frequency
    above = c > ema20
    for lookback in [20, 30, 50]:
        crosses = np.zeros(n)
        for i in range(lookback, n):
            cnt = 0
            for j in range(i-lookback+1, i+1):
                if above[j] != above[j-1]:
                    cnt += 1
            crosses[i] = cnt
        df[f"ema_crosses_{lookback}"] = crosses

    # E. Choppiness Index (14-period)
    ci = np.zeros(n)
    for i in range(14, n):
        hh = max(h[i-13:i+1]); ll = min(l[i-13:i+1])
        atr_sum = sum(max(h[j]-l[j], abs(h[j]-c[j-1]), abs(l[j]-c[j-1])) for j in range(i-13, i+1))
        rng = hh - ll
        if rng > 0 and atr_sum > 0:
            ci[i] = 100 * np.log10(atr_sum / rng) / np.log10(14)
    df["chop_index"] = ci

    # F. EMA gap (already available from detect, but compute explicitly)
    df["ema_gap"] = np.abs(ema20 - ema50) / np.where(atr > 0, atr, 1)

    # G. Bollinger Bandwidth (20-period, 2 std)
    sma20 = pd.Series(c).rolling(20).mean().values
    std20 = pd.Series(c).rolling(20).std().values
    df["bb_width"] = np.where(sma20 > 0, 2 * std20 / sma20 * 100, 0)  # as % of price

    # H. Efficiency Ratio (10-bar)
    for lookback in [10, 20, 30]:
        er = np.zeros(n)
        for i in range(lookback, n):
            net_move = abs(c[i] - c[i-lookback])
            total_path = sum(abs(c[j] - c[j-1]) for j in range(i-lookback+1, i+1))
            er[i] = net_move / total_path if total_path > 0 else 0
        df[f"eff_ratio_{lookback}"] = er

    # I. Consecutive same-side bars (bars since last EMA cross)
    same_side = np.zeros(n)
    for i in range(1, n):
        if above[i] == above[i-1]:
            same_side[i] = same_side[i-1] + 1
        else:
            same_side[i] = 0
    df["same_side_bars"] = same_side

    return df


def run_with_indicators(df, capital=100_000):
    """Run backtest collecting chop indicator values at each entry."""
    p = BASE.copy()
    df = base_indicators(df, p)
    df = add_chop_indicators(df)

    high = df["High"].values; low = df["Low"].values; close = df["Close"].values
    ema = df["ema20"].values; ema_s = df["ema50"].values; atr_v = df["atr"].values
    times = df.index.time; dates = df.index.date; n = len(df)
    comm = p["commission_per_share"]

    equity = capital; trades = []
    bar = max(p["ema_slow"], p["atr_period"]) + 50  # extra warmup for indicators
    daily_r_loss = 0.0; current_date = None

    # Pre-extract indicator arrays
    adx = df["adx"].values
    ema_slope_abs = df["ema_slope_abs"].values
    range_atr_20 = df["range_atr_20"].values
    range_atr_30 = df["range_atr_30"].values
    range_atr_50 = df["range_atr_50"].values
    ema_crosses_20 = df["ema_crosses_20"].values
    ema_crosses_30 = df["ema_crosses_30"].values
    ema_crosses_50 = df["ema_crosses_50"].values
    chop_index = df["chop_index"].values
    ema_gap = df["ema_gap"].values
    bb_width = df["bb_width"].values
    eff_ratio_10 = df["eff_ratio_10"].values
    eff_ratio_20 = df["eff_ratio_20"].values
    eff_ratio_30 = df["eff_ratio_30"].values
    same_side = df["same_side_bars"].values

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

        # Collect indicator values at entry
        indicators = {
            "adx": adx[bar],
            "ema_slope_abs": ema_slope_abs[bar],
            "range_atr_20": range_atr_20[bar],
            "range_atr_30": range_atr_30[bar],
            "range_atr_50": range_atr_50[bar],
            "ema_crosses_20": ema_crosses_20[bar],
            "ema_crosses_30": ema_crosses_30[bar],
            "ema_crosses_50": ema_crosses_50[bar],
            "chop_index": chop_index[bar],
            "ema_gap": ema_gap[bar],
            "bb_width": bb_width[bar],
            "eff_ratio_10": eff_ratio_10[bar],
            "eff_ratio_20": eff_ratio_20[bar],
            "eff_ratio_30": eff_ratio_30[bar],
            "same_side": same_side[bar],
        }

        # Execute trade (same logic)
        lock_sh = max(1, int(shares * p["lock_pct"]))
        remaining = shares; runner_stop = stop; lock_done = False
        trade_pnl = -shares * comm; end_bar = entry_bar
        exit_reason = "timeout"

        for k in range(1, p["max_hold_bars"] + 1):
            bi = entry_bar + k
            if bi >= n: break
            h = high[bi]; l = low[bi]
            ca = atr_v[bi] if not np.isnan(atr_v[bi]) else a

            if times[bi] >= p["force_close_at"]:
                trade_pnl += remaining * (close[bi] - actual_entry) * trend - remaining * comm
                end_bar = bi; exit_reason = "session_close"; break

            stopped = (trend == 1 and l <= runner_stop) or (trend == -1 and h >= runner_stop)
            if stopped:
                trade_pnl += remaining * (runner_stop - actual_entry) * trend - remaining * comm
                end_bar = bi
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

        equity += trade_pnl
        r_mult = trade_pnl / (shares * risk) if shares * risk > 0 else 0
        if trade_pnl < 0 and shares * risk > 0:
            daily_r_loss += abs(r_mult)

        trades.append({
            **indicators,
            "pnl": trade_pnl, "r": r_mult, "dir": trend,
            "exit": exit_reason, "shares": shares, "risk": risk,
        })
        bar = end_bar + 1

    return pd.DataFrame(trades)


def analyze_indicator(tdf, col, name, direction="higher_is_trend", n_bins=10):
    """
    Analyze one indicator's predictive power.
    direction: "higher_is_trend" or "lower_is_trend"
    Returns: best_threshold, best_pf, trades_kept, big5_kept
    """
    vals = tdf[col].values
    r_vals = tdf["r"].values
    is_5r = r_vals >= 5

    # Decile analysis
    try:
        quantiles = np.percentile(vals[~np.isnan(vals)], np.linspace(0, 100, n_bins + 1))
        quantiles = np.unique(quantiles)
    except:
        return None

    if len(quantiles) < 3:
        return None

    print(f"\n  {'Bin':<25} {'Trades':>7} {'PF':>7} {'AvgR':>8} {'5R+':>5} {'TotalR':>9}")
    print(f"  {'-'*65}")

    for i in range(len(quantiles) - 1):
        lo, hi = quantiles[i], quantiles[i+1]
        if i == len(quantiles) - 2:
            mask = (vals >= lo) & (vals <= hi)
        else:
            mask = (vals >= lo) & (vals < hi)
        sub_r = r_vals[mask]
        if len(sub_r) < 10: continue
        gw = sub_r[sub_r > 0].sum(); gl = abs(sub_r[sub_r <= 0].sum())
        pf = gw / gl if gl > 0 else 0
        big5 = (sub_r >= 5).sum()
        print(f"  {lo:>10.2f} - {hi:<10.2f} {mask.sum():>6} {pf:>6.3f} {sub_r.mean():>+7.3f}"
              f" {big5:>4} {sub_r.sum():>+8.1f}R")

    # Threshold sweep: find best PF by filtering
    total_5r = is_5r.sum()
    best = {"threshold": 0, "pf": 0, "kept": 0, "big5_kept": 0, "big5_pct": 0}

    thresholds = np.percentile(vals[~np.isnan(vals)], np.linspace(5, 95, 30))
    for thresh in thresholds:
        if direction == "higher_is_trend":
            mask = vals >= thresh
        else:
            mask = vals <= thresh

        sub_r = r_vals[mask]
        if len(sub_r) < 100: continue
        gw = sub_r[sub_r > 0].sum(); gl = abs(sub_r[sub_r <= 0].sum())
        pf = gw / gl if gl > 0 else 0
        big5_kept = is_5r[mask].sum()
        big5_pct = big5_kept / total_5r * 100 if total_5r > 0 else 0

        if pf > best["pf"] and big5_pct >= 70:  # must keep 70% of big wins
            best = {"threshold": thresh, "pf": pf, "kept": mask.sum(),
                    "big5_kept": big5_kept, "big5_pct": big5_pct}

    return best


def main():
    df = pd.read_csv(DATA_PATH, index_col="timestamp", parse_dates=True)
    print(f"Data: {len(df):,} bars\n")

    print("Running backtest with indicator collection...")
    tdf = run_with_indicators(df)
    total = len(tdf)
    total_5r = (tdf["r"] >= 5).sum()
    gw = tdf.loc[tdf["r"] > 0, "r"].sum()
    gl = abs(tdf.loc[tdf["r"] <= 0, "r"].sum())
    base_pf = gw / gl if gl > 0 else 0
    print(f"Trades: {total}, 5R+: {total_5r}, Baseline PF: {base_pf:.3f}\n")

    # Analyze each indicator
    indicators = [
        ("adx",            "A. ADX(14)",                  "higher_is_trend"),
        ("ema_slope_abs",  "B. |EMA20 slope|",            "higher_is_trend"),
        ("range_atr_20",   "C. Range/ATR (20 bars)",      "higher_is_trend"),
        ("range_atr_30",   "C2. Range/ATR (30 bars)",     "higher_is_trend"),
        ("range_atr_50",   "C3. Range/ATR (50 bars)",     "higher_is_trend"),
        ("ema_crosses_20", "D. EMA crosses (20 bars)",    "lower_is_trend"),
        ("ema_crosses_30", "D2. EMA crosses (30 bars)",   "lower_is_trend"),
        ("ema_crosses_50", "D3. EMA crosses (50 bars)",   "lower_is_trend"),
        ("chop_index",     "E. Choppiness Index",         "lower_is_trend"),
        ("ema_gap",        "F. EMA20/50 gap (ATR)",       "higher_is_trend"),
        ("bb_width",       "G. Bollinger Bandwidth",      "higher_is_trend"),
        ("eff_ratio_10",   "H. Efficiency Ratio (10)",    "higher_is_trend"),
        ("eff_ratio_20",   "H2. Efficiency Ratio (20)",   "higher_is_trend"),
        ("eff_ratio_30",   "H3. Efficiency Ratio (30)",   "higher_is_trend"),
        ("same_side",      "I. Same-side bars",           "higher_is_trend"),
    ]

    results = []
    for col, name, direction in indicators:
        print(f"\n{'='*75}")
        print(f"  {name}")
        print(f"{'='*75}")

        best = analyze_indicator(tdf, col, name, direction)
        if best is None:
            print(f"  Insufficient data for analysis")
            continue

        print(f"\n  Best filter: {col} {'≥' if direction == 'higher_is_trend' else '≤'} {best['threshold']:.3f}")
        print(f"  PF: {base_pf:.3f} → {best['pf']:.3f} ({(best['pf']-base_pf)/base_pf*100:+.1f}%)")
        print(f"  Trades: {total} → {best['kept']} ({best['kept']/total*100:.1f}%)")
        print(f"  5R+ kept: {best['big5_kept']}/{total_5r} ({best['big5_pct']:.1f}%)")

        results.append({
            "name": name, "col": col, "direction": direction,
            "threshold": best["threshold"],
            "pf": best["pf"], "pf_gain": best["pf"] - base_pf,
            "kept": best["kept"], "kept_pct": best["kept"] / total * 100,
            "big5_kept": best["big5_kept"], "big5_pct": best["big5_pct"],
        })

    # ═══ RANKING ═══
    print(f"\n{'='*90}")
    print(f"  CHOP INDICATOR RANKING — by PF improvement")
    print(f"{'='*90}")

    results.sort(key=lambda x: x["pf"], reverse=True)
    print(f"\n  {'#':>2} {'Indicator':<30} {'PF':>6} {'ΔPF':>7} {'Trades':>7} {'Kept%':>6} {'5R+ kept':>9}")
    print(f"  {'-'*75}")
    print(f"  {'':>2} {'BASELINE':<30} {base_pf:>6.3f} {'':>7} {total:>6} {'100%':>6} {total_5r:>8}")
    for i, r in enumerate(results):
        op = "≥" if r["direction"] == "higher_is_trend" else "≤"
        print(f"  {i+1:>2} {r['name']:<30} {r['pf']:>6.3f} {r['pf_gain']:>+6.3f}"
              f" {r['kept']:>6} {r['kept_pct']:>5.1f}% {r['big5_kept']:>4}/{total_5r}"
              f"  ({r['col']} {op} {r['threshold']:.2f})")

    # ═══ SEPARATION QUALITY ═══
    print(f"\n{'='*90}")
    print(f"  SEPARATION QUALITY — how well does each indicator split winners from losers?")
    print(f"{'='*90}")

    trail_trades = tdf[tdf["exit"] == "trail_stop"]
    loser_trades = tdf[tdf["exit"].isin(["initial_stop", "be_stop"])]

    print(f"\n  {'Indicator':<30} {'Winners mean':>13} {'Losers mean':>13} {'Separation':>11}")
    print(f"  {'-'*70}")
    for col, name, direction in indicators:
        w_mean = trail_trades[col].mean()
        l_mean = loser_trades[col].mean()
        # Cohen's d (effect size)
        pooled_std = np.sqrt((trail_trades[col].std()**2 + loser_trades[col].std()**2) / 2)
        if pooled_std > 0:
            cohens_d = abs(w_mean - l_mean) / pooled_std
        else:
            cohens_d = 0
        quality = "★★★" if cohens_d > 0.5 else ("★★" if cohens_d > 0.3 else ("★" if cohens_d > 0.15 else "—"))
        print(f"  {name:<30} {w_mean:>12.3f} {l_mean:>12.3f} {cohens_d:>8.3f}  {quality}")


if __name__ == "__main__":
    main()
