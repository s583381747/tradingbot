"""
Trend Alive Filter — detect when the trend has stalled even though EMAs still align.

Core thesis: runner needs room to run. If the trend has stalled,
there's no room → entry is a guaranteed loss. We need to verify
the trend is ALIVE, not just that EMAs are aligned.

Step 1: ANALYZE — for each trade, compute structural trend health indicators.
        Compare winners (trail_stop) vs losers (initial_stop + be_stop).
        Find which indicators actually separate them.

Step 2: FILTER — use the best separators as entry conditions.

Structural trend health indicators:
  1. Higher timeframe trend: 5min/15min EMA alignment
  2. Recent touch failure rate: last 3/5 touches, how many stopped out?
  3. Price vs swing high: distance from recent N-bar high (trend momentum)
  4. EMA slope acceleration: is slope increasing or flattening?
  5. Multi-EMA alignment: EMA10 > EMA20 > EMA50 (tighter alignment)
  6. Bars since last new high/low: if no new high in 30 bars, trend stalled
  7. Price compression: last N bars range shrinking vs expanding
  8. Trend age: how many bars has EMA20 > EMA50 been true? (old trends fail more)
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
    "no_entry_after": dt.time(14, 0),
    "force_close_at": dt.time(15, 58),
    "commission_per_share": 0.005, "daily_loss_r": 2.5,
    "skip_after_win": 1,
}


def add_structural_indicators(df):
    """Add higher-level trend structure indicators."""
    p = BASE
    c = df["Close"].values; h = df["High"].values; l = df["Low"].values
    o = df["Open"].values
    ema20 = df["ema20"].values; ema50 = df["ema50"].values
    atr = df["atr"].values
    n = len(df)

    # EMA10
    ema10 = pd.Series(c).ewm(span=10, adjust=False).mean().values

    # 5-min EMA20: every 5th bar's close, then interpolate
    # Simpler: use EMA with span=100 (=20 * 5min equivalent)
    ema100 = pd.Series(c).ewm(span=100, adjust=False).mean().values
    # 15-min EMA20 equivalent: span=300
    ema300 = pd.Series(c).ewm(span=300, adjust=False).mean().values

    indicators = {}

    # 1. Higher TF alignment: Close > EMA100 > EMA300
    htf_aligned = np.zeros(n)
    for i in range(300, n):
        if c[i] > ema100[i] and ema100[i] > ema300[i]: htf_aligned[i] = 1
        elif c[i] < ema100[i] and ema100[i] < ema300[i]: htf_aligned[i] = -1
    indicators["htf_aligned"] = htf_aligned

    # Does 1min trend match HTF trend?
    indicators["htf_agree"] = np.zeros(n)
    for i in range(300, n):
        trend_1m = 1 if c[i] > ema20[i] and ema20[i] > ema50[i] else (-1 if c[i] < ema20[i] and ema20[i] < ema50[i] else 0)
        indicators["htf_agree"][i] = 1 if trend_1m == htf_aligned[i] and trend_1m != 0 else 0

    # 2. Multi-EMA alignment: EMA10 > EMA20 > EMA50
    indicators["triple_ema"] = np.zeros(n)
    for i in range(50, n):
        if ema10[i] > ema20[i] > ema50[i]: indicators["triple_ema"][i] = 1
        elif ema10[i] < ema20[i] < ema50[i]: indicators["triple_ema"][i] = -1

    # 3. Price vs recent swing high (for longs: how far below the N-bar high)
    for lookback in [20, 40, 60]:
        dist = np.zeros(n)
        for i in range(lookback, n):
            if atr[i] > 0:
                hh = max(h[i-lookback:i+1])
                ll = min(l[i-lookback:i+1])
                # Normalized distance from high (0 = at high, 1 = at low)
                rng = hh - ll
                if rng > 0:
                    dist[i] = (hh - c[i]) / rng  # 0 = at top, 1 = at bottom
        indicators[f"swing_pos_{lookback}"] = dist

    # 4. Bars since new N-bar high/low
    for lookback in [20, 40]:
        bars_since = np.zeros(n)
        for i in range(lookback, n):
            hh = max(h[i-lookback:i+1])
            # How many bars ago was the high?
            for j in range(i, i-lookback-1, -1):
                if h[j] == hh:
                    bars_since[i] = i - j
                    break
        indicators[f"bars_since_high_{lookback}"] = bars_since

    # 5. EMA20 slope acceleration (is slope increasing or decreasing?)
    slope5 = np.zeros(n)
    slope_accel = np.zeros(n)
    for i in range(10, n):
        slope5[i] = (ema20[i] - ema20[i-5]) / 5
        if i >= 15:
            prev_slope = (ema20[i-5] - ema20[i-10]) / 5
            slope_accel[i] = slope5[i] - prev_slope  # positive = accelerating
    indicators["ema_slope"] = slope5
    indicators["ema_slope_accel"] = slope_accel

    # Normalized slope acceleration
    indicators["slope_accel_norm"] = np.zeros(n)
    for i in range(15, n):
        if atr[i] > 0:
            indicators["slope_accel_norm"][i] = slope_accel[i] / atr[i] * 1000

    # 6. Trend age: consecutive bars where EMA20 > EMA50
    trend_age = np.zeros(n)
    for i in range(1, n):
        if ema20[i] > ema50[i] and ema20[i-1] > ema50[i-1]:
            trend_age[i] = trend_age[i-1] + 1
        elif ema20[i] < ema50[i] and ema20[i-1] < ema50[i-1]:
            trend_age[i] = trend_age[i-1] + 1
        else:
            trend_age[i] = 0
    indicators["trend_age"] = trend_age

    # 7. Range expansion/contraction: ratio of last 10 bars range vs last 30 bars range
    indicators["range_expansion"] = np.zeros(n)
    for i in range(30, n):
        r10 = max(h[i-9:i+1]) - min(l[i-9:i+1])
        r30 = max(h[i-29:i+1]) - min(l[i-29:i+1])
        indicators["range_expansion"][i] = r10 / r30 if r30 > 0 else 1

    # 8. Price momentum: close vs close N bars ago, normalized
    for lookback in [10, 20, 40]:
        mom = np.zeros(n)
        for i in range(lookback, n):
            if atr[i] > 0:
                mom[i] = (c[i] - c[i-lookback]) / (atr[i] * lookback) * 100
        indicators[f"momentum_{lookback}"] = mom

    return indicators


def run_analysis(df):
    """Collect all structural indicators for every trade."""
    p = BASE.copy()
    df = add_indicators(df, p)
    indicators = add_structural_indicators(df)

    high = df["High"].values; low = df["Low"].values; close = df["Close"].values
    ema = df["ema20"].values; ema_s = df["ema50"].values; atr_v = df["atr"].values
    times = df.index.time; dates = df.index.date; n = len(df)
    comm = p["commission_per_share"]

    equity = 100_000; trades = []
    bar = 310  # warmup for all indicators
    daily_r_loss = 0.0; current_date = None; skip_count = 0

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
        if skip_count > 0: skip_count -= 1; bar += 1; continue

        actual_entry = close[bar]
        stop = low[bar] - p["stop_buffer"] * a if trend == 1 else high[bar] + p["stop_buffer"] * a
        risk = abs(actual_entry - stop)
        if risk <= 0: bar += 1; continue
        entry_bar = bar

        # Collect indicators
        ind_vals = {k: v[bar] for k, v in indicators.items()}
        # For directional indicators, flip sign for shorts
        if trend == -1:
            for key in ["ema_slope", "ema_slope_accel", "slope_accel_norm",
                        "momentum_10", "momentum_20", "momentum_40"]:
                ind_vals[key] = -ind_vals[key]
            # swing_pos: for shorts, being near swing LOW is good (invert)
            for lb in [20, 40, 60]:
                ind_vals[f"swing_pos_{lb}"] = 1 - ind_vals[f"swing_pos_{lb}"]

        shares = max(1, int(equity * p["risk_pct"] / risk))
        if shares * abs(actual_entry) > equity * p["max_pos_pct"]:
            shares = max(1, int(equity * p["max_pos_pct"] / abs(actual_entry)))
        if equity < shares * risk or shares < 1: bar += 1; continue

        lock_sh = max(1, int(shares * p["lock_pct"]))
        remaining = shares; runner_stop = stop; lock_done = False
        trade_pnl = -shares * comm; end_bar = entry_bar; exit_reason = "timeout"

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
                exit_reason = "be_stop" if lock_done and abs(runner_stop - actual_entry) < 0.02 else ("trail_stop" if lock_done else "initial_stop")
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

        trades.append({"r": r_mult, "exit": exit_reason, "dir": trend, **ind_vals})
        bar = end_bar + 1

    return pd.DataFrame(trades)


def run_filtered(df, capital=100_000, filters=None):
    """Run backtest with structural filters."""
    p = BASE.copy()
    df_ind = add_indicators(df, p)
    indicators = add_structural_indicators(df_ind)

    high = df_ind["High"].values; low = df_ind["Low"].values; close = df_ind["Close"].values
    ema = df_ind["ema20"].values; ema_s = df_ind["ema50"].values; atr_v = df_ind["atr"].values
    times = df_ind.index.time; dates = df_ind.index.date; n = len(df_ind)
    comm = p["commission_per_share"]

    equity = capital; trades = []
    bar = 310; daily_r_loss = 0.0; current_date = None; skip_count = 0; filtered = 0

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

        # Apply structural filters
        if filters:
            passed = True
            for filt in filters:
                col = filt["col"]; op = filt["op"]; val = filt["val"]
                ind_val = indicators[col][bar]
                # Flip directional indicators for shorts
                if trend == -1 and col in ["ema_slope", "ema_slope_accel", "slope_accel_norm",
                                            "momentum_10", "momentum_20", "momentum_40"]:
                    ind_val = -ind_val
                if trend == -1 and "swing_pos" in col:
                    ind_val = 1 - ind_val

                if op == ">=" and ind_val < val: passed = False; break
                elif op == "<=" and ind_val > val: passed = False; break
                elif op == "==" and ind_val != val: passed = False; break
            if not passed:
                filtered += 1; bar += 1; continue

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
        trades.append({"r": r_mult, "dir": trend})
        bar = end_bar + 1

    tdf = pd.DataFrame(trades) if trades else pd.DataFrame()
    total = len(tdf)
    if total == 0: return {"pf": 0, "ret": 0, "trades": 0, "big5": 0, "lpf": 0, "spf": 0, "filtered": filtered}
    gw = tdf.loc[tdf["r"]>0,"r"].sum(); gl = abs(tdf.loc[tdf["r"]<=0,"r"].sum())
    pf = gw/gl if gl > 0 else 0; ret = (equity - capital) / capital * 100
    r_arr = tdf["r"].values
    longs = tdf[tdf["dir"]==1]; shorts = tdf[tdf["dir"]==-1]
    def spf(s):
        if len(s)==0: return 0
        w = s.loc[s["pnl" if "pnl" in s.columns else "r"]>0,"r"].sum()
        l = abs(s.loc[s["r"]<=0,"r"].sum())
        return round(w/l,3) if l>0 else 0
    return {"pf": round(pf,3), "ret": round(ret,2), "trades": total,
            "big5": int((r_arr>=5).sum()), "lpf": spf(longs), "spf": spf(shorts), "filtered": filtered}


def main():
    df_is = pd.read_csv(IS_PATH, index_col="timestamp", parse_dates=True)
    df_oos = pd.read_csv(OOS_PATH, index_col="timestamp", parse_dates=True)
    print(f"IS: {len(df_is):,} bars | OOS: {len(df_oos):,} bars\n")

    # ═══ STEP 1: ANALYSIS ═══
    print(f"{'='*100}")
    print(f"  STEP 1: SEPARATION ANALYSIS — which structural indicators distinguish winners?")
    print(f"{'='*100}\n")

    tdf = run_analysis(df_is)
    trail = tdf[tdf["exit"] == "trail_stop"]
    losers = tdf[tdf["exit"].isin(["initial_stop", "be_stop"])]
    big5 = tdf[tdf["r"] >= 5]

    ind_cols = [c for c in tdf.columns if c not in ["r", "exit", "dir"]]

    print(f"  Trades: {len(tdf)} | Trail stops: {len(trail)} | Losers: {len(losers)} | 5R+: {len(big5)}\n")
    print(f"  {'Indicator':<25} {'Winners':>10} {'Losers':>10} {'5R+':>10} {'Cohen d':>9} {'Signal':>7}")
    print(f"  {'-'*75}")

    good_indicators = []
    for col in sorted(ind_cols):
        w_mean = trail[col].mean()
        l_mean = losers[col].mean()
        b_mean = big5[col].mean()
        pooled_std = np.sqrt((trail[col].std()**2 + losers[col].std()**2) / 2)
        d = abs(w_mean - l_mean) / pooled_std if pooled_std > 0 else 0
        sig = "★★★" if d > 0.4 else ("★★" if d > 0.25 else ("★" if d > 0.15 else ""))
        print(f"  {col:<25} {w_mean:>10.3f} {l_mean:>10.3f} {b_mean:>10.3f} {d:>8.3f}  {sig}")
        if d > 0.15:
            good_indicators.append((col, d, "higher" if w_mean > l_mean else "lower"))

    # ═══ STEP 2: FILTER EXPERIMENTS ═══
    print(f"\n{'='*100}")
    print(f"  STEP 2: FILTER EXPERIMENTS — IS + immediate OOS check")
    print(f"{'='*100}\n")

    bl_is = run_filtered(df_is)
    bl_oos = run_filtered(df_oos)
    print(f"  IS Baseline:  PF={bl_is['pf']:.3f}, ret={bl_is['ret']:+.2f}%, trades={bl_is['trades']}, 5R+={bl_is['big5']}")
    print(f"  OOS Baseline: PF={bl_oos['pf']:.3f}, ret={bl_oos['ret']:+.2f}%, trades={bl_oos['trades']}, 5R+={bl_oos['big5']}\n")

    hdr = f"  {'Filter':<50} {'IS PF':>7} {'IS Ret':>8} {'IS Trd':>7} {'IS 5R+':>7} {'OOS PF':>8} {'OOS 5R+':>8}"
    print(hdr)
    print(f"  {'-'*100}")

    # Test each good indicator at multiple thresholds
    configs = []

    # Always test these regardless of Cohen's d
    all_tests = [
        # HTF alignment
        ("HTF agree (1min matches 5/15min)", [{"col": "htf_agree", "op": "==", "val": 1}]),
        # Triple EMA
        ("Triple EMA aligned", [{"col": "triple_ema", "op": ">=", "val": 0.5}]),
        # HTF + Triple
        ("HTF agree + Triple EMA", [{"col": "htf_agree", "op": "==", "val": 1}, {"col": "triple_ema", "op": ">=", "val": 0.5}]),
    ]

    # Swing position at various thresholds
    for lb in [20, 40, 60]:
        for th in [0.3, 0.4, 0.5]:
            all_tests.append(
                (f"Swing pos {lb}b < {th} (near high)",
                 [{"col": f"swing_pos_{lb}", "op": "<=", "val": th}]))

    # Bars since high
    for lb in [20, 40]:
        for th in [5, 10, 15]:
            all_tests.append(
                (f"Bars since {lb}b high <= {th}",
                 [{"col": f"bars_since_high_{lb}", "op": "<=", "val": th}]))

    # Slope acceleration
    for th in [0, -0.5, -1.0]:
        all_tests.append(
            (f"Slope accel (norm) >= {th}",
             [{"col": "slope_accel_norm", "op": ">=", "val": th}]))

    # Trend age
    for th in [20, 50, 100, 200]:
        all_tests.append(
            (f"Trend age >= {th} bars",
             [{"col": "trend_age", "op": ">=", "val": th}]))
    for th in [200, 400, 600]:
        all_tests.append(
            (f"Trend age <= {th} bars (not stale)",
             [{"col": "trend_age", "op": "<=", "val": th}]))

    # Range expansion
    for th in [0.4, 0.5, 0.6]:
        all_tests.append(
            (f"Range expansion > {th}",
             [{"col": "range_expansion", "op": ">=", "val": th}]))

    # Momentum
    for lb in [10, 20, 40]:
        for th in [0, 0.5, 1.0]:
            all_tests.append(
                (f"Momentum {lb}b >= {th}",
                 [{"col": f"momentum_{lb}", "op": ">=", "val": th}]))

    # Combos
    all_tests.extend([
        ("HTF agree + swing40 < 0.4",
         [{"col": "htf_agree", "op": "==", "val": 1},
          {"col": "swing_pos_40", "op": "<=", "val": 0.4}]),
        ("Triple EMA + bars_since_high_20 <= 10",
         [{"col": "triple_ema", "op": ">=", "val": 0.5},
          {"col": "bars_since_high_20", "op": "<=", "val": 10}]),
        ("HTF + momentum_20 >= 0.5",
         [{"col": "htf_agree", "op": "==", "val": 1},
          {"col": "momentum_20", "op": ">=", "val": 0.5}]),
        ("Swing40<0.4 + trend_age>=50",
         [{"col": "swing_pos_40", "op": "<=", "val": 0.4},
          {"col": "trend_age", "op": ">=", "val": 50}]),
        ("Triple + swing40<0.4 + trend_age>=50",
         [{"col": "triple_ema", "op": ">=", "val": 0.5},
          {"col": "swing_pos_40", "op": "<=", "val": 0.4},
          {"col": "trend_age", "op": ">=", "val": 50}]),
    ])

    results = []
    for label, filt in all_tests:
        r_is = run_filtered(df_is, filters=filt)
        r_oos = run_filtered(df_oos, filters=filt)
        results.append((label, r_is, r_oos, filt))
        if r_is["trades"] > 0:
            print(f"  {label:<50} {r_is['pf']:>6.3f} {r_is['ret']:>+7.2f}% {r_is['trades']:>6} {r_is['big5']:>6}"
                  f" {r_oos['pf']:>7.3f} {r_oos['big5']:>7}")

    # ═══ TOP RESULTS ═══
    print(f"\n{'='*100}")
    print(f"  TOP 10 BY IS PF (with OOS cross-check)")
    print(f"{'='*100}")
    results.sort(key=lambda x: x[1]["pf"], reverse=True)
    print(f"\n  {'#':>2} {'Filter':<50} {'IS PF':>7} {'OOS PF':>8} {'IS Trd':>7} {'IS 5R+':>7} {'OOS 5R+':>8} {'Holds':>6}")
    print(f"  {'-'*95}")
    for i, (label, r_is, r_oos, _) in enumerate(results[:15]):
        holds = "✅" if r_oos["pf"] > bl_oos["pf"] and r_is["pf"] > bl_is["pf"] else "❌"
        print(f"  {i+1:>2} {label:<50} {r_is['pf']:>6.3f} {r_oos['pf']:>7.3f}"
              f" {r_is['trades']:>6} {r_is['big5']:>6} {r_oos['big5']:>7}  {holds}")


if __name__ == "__main__":
    main()
