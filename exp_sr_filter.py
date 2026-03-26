"""
Support/Resistance Filter — Do EMA20 touches near key S/R levels perform better?

Hypothesis: EMA20 touches near key S/R = double confirmation → higher quality.
Anti-hypothesis: touches near opposing S/R (e.g., long near resistance) are bad.

S/R detection methods:
  1. Prior day high/low — yesterday's H/L are natural S/R
  2. N-bar swing highs/lows — local peaks/troughs (50/100/200 bars)
  3. Round numbers — $5 and $10 increments
  4. Volume nodes — bars with unusually high volume mark key price levels
  5. Opening range — first 30-min H/L as S/R

For each, check if touch is NEAR a confirming S/R level:
  - Longs near support = good?  Longs near resistance = bad?
  - Shorts near resistance = good?  Shorts near support = bad?

"Near" = within 0.5/1.0/1.5 × ATR of the S/R level.

Step 1: Separation analysis (Cohen's d) for each S/R indicator
Step 2: Filter results — IS + OOS with 3-bar MFE gate
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
    "stop_buffer": 0.3, "lock_rr": 0.1, "lock_pct": 0.05,
    "chand_bars": 40, "chand_mult": 0.5, "max_hold_bars": 180,
    "risk_pct": 0.01, "max_pos_pct": 0.25,
    "no_entry_after": dt.time(14, 0), "force_close_at": dt.time(15, 58),
    "commission_per_share": 0.005, "daily_loss_r": 2.5, "skip_after_win": 1,
}


# ══════════════════════════════════════════════════════════════════
# S/R DETECTION HELPERS
# ══════════════════════════════════════════════════════════════════

def compute_sr_levels(high, low, close, vol, times, dates, bar, atr_val):
    """
    Compute all S/R levels relevant to bar.
    Returns dict of {method: {"support": [prices], "resistance": [prices]}}.
    """
    sr = {}

    # 1. Prior day high/low
    cur_date = dates[bar]
    prev_highs = []
    prev_lows = []
    # Scan backwards to find previous trading day
    for j in range(bar - 1, max(bar - 500, 0), -1):
        if dates[j] != cur_date:
            prev_date = dates[j]
            # Collect all bars of that day
            for k in range(j, max(j - 500, 0), -1):
                if dates[k] == prev_date:
                    prev_highs.append(high[k])
                    prev_lows.append(low[k])
                else:
                    break
            break
    if prev_highs:
        sr["prior_day"] = {
            "support": [min(prev_lows)],
            "resistance": [max(prev_highs)],
        }
    else:
        sr["prior_day"] = {"support": [], "resistance": []}

    # 2. N-bar swing highs/lows (local peaks/troughs)
    for lookback in [50, 100, 200]:
        supports = []
        resistances = []
        start = max(0, bar - lookback)
        if bar - start < 10:
            sr[f"swing_{lookback}"] = {"support": [], "resistance": []}
            continue
        # Find swing lows (local minima with 5-bar shoulder)
        for j in range(start + 5, bar - 4):
            if low[j] == min(low[j-5:j+6]):
                supports.append(low[j])
        # Find swing highs (local maxima with 5-bar shoulder)
        for j in range(start + 5, bar - 4):
            if high[j] == max(high[j-5:j+6]):
                resistances.append(high[j])
        sr[f"swing_{lookback}"] = {"support": supports, "resistance": resistances}

    # 3. Round numbers ($5 and $10 increments)
    price = close[bar]
    round5_levels = []
    round10_levels = []
    base5 = int(price / 5) * 5
    for offset in range(-3, 4):
        round5_levels.append(base5 + offset * 5)
    base10 = int(price / 10) * 10
    for offset in range(-2, 3):
        round10_levels.append(base10 + offset * 10)
    # For round numbers, they serve as both support and resistance
    sr["round_5"] = {"support": round5_levels, "resistance": round5_levels}
    sr["round_10"] = {"support": round10_levels, "resistance": round10_levels}

    # 4. Volume nodes — find bars with unusually high volume in lookback
    vol_lookback = 200
    start = max(0, bar - vol_lookback)
    if bar - start >= 20:
        vol_slice = vol[start:bar]
        vol_mean = np.mean(vol_slice)
        vol_std = np.std(vol_slice)
        threshold = vol_mean + 2 * vol_std  # 2-sigma volume spikes
        vol_supports = []
        vol_resistances = []
        for j in range(start, bar):
            if vol[j] > threshold:
                # High volume bar — its high is resistance, low is support
                vol_supports.append(low[j])
                vol_resistances.append(high[j])
        sr["vol_node"] = {"support": vol_supports, "resistance": vol_resistances}
    else:
        sr["vol_node"] = {"support": [], "resistance": []}

    # 5. Opening range — first 30 min H/L of today
    or_high = None
    or_low = None
    # Find start of today
    for j in range(bar, max(bar - 500, 0), -1):
        if dates[j] != cur_date:
            day_start = j + 1
            break
    else:
        day_start = max(0, bar - 400)
    # First 30 bars (approx 30 min for 1-min data)
    or_end = min(day_start + 30, bar)
    if or_end > day_start:
        or_high = max(high[day_start:or_end])
        or_low = min(low[day_start:or_end])
    if or_high is not None:
        sr["open_range"] = {"support": [or_low], "resistance": [or_high]}
    else:
        sr["open_range"] = {"support": [], "resistance": []}

    return sr


def nearest_sr_distance(price, levels, atr_val):
    """Return distance to nearest level in ATR units. Returns large value if no levels."""
    if not levels:
        return 999.0
    dists = [abs(price - lv) / atr_val for lv in levels if atr_val > 0]
    return min(dists) if dists else 999.0


# ══════════════════════════════════════════════════════════════════
# STEP 1: ANALYSIS — collect S/R indicators per trade
# ══════════════════════════════════════════════════════════════════

def run_sr_analysis(df):
    """Collect S/R distance indicators for every trade."""
    p = BASE.copy()
    df = add_indicators(df, p)
    high = df["High"].values; low = df["Low"].values; close = df["Close"].values
    vol = df["Volume"].values
    ema = df["ema20"].values; ema_s = df["ema50"].values; atr_v = df["atr"].values
    times = df.index.time; dates = df.index.date; n = len(df)
    comm = p["commission_per_share"]

    trades = []
    bar = max(p["ema_slow"], p["atr_period"]) + 200  # need lookback for swing detection
    daily_r_loss = 0.0; current_date = None; skip_count = 0; equity = 100_000

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

        # Compute S/R levels
        sr = compute_sr_levels(high, low, close, vol, times, dates, bar, a)

        touch_price = low[bar] if trend == 1 else high[bar]

        # Compute indicator values: distance to nearest S/R in ATR units
        vi = {}
        for method, levels in sr.items():
            vi[f"{method}_sup_dist"] = nearest_sr_distance(touch_price, levels["support"], a)
            vi[f"{method}_res_dist"] = nearest_sr_distance(touch_price, levels["resistance"], a)
            # Confirming S/R: for longs, distance to support; for shorts, distance to resistance
            if trend == 1:
                vi[f"{method}_confirm_dist"] = vi[f"{method}_sup_dist"]
                vi[f"{method}_oppose_dist"] = vi[f"{method}_res_dist"]
            else:
                vi[f"{method}_confirm_dist"] = vi[f"{method}_res_dist"]
                vi[f"{method}_oppose_dist"] = vi[f"{method}_sup_dist"]
            # Binary: is there a confirming S/R within 1 ATR?
            vi[f"{method}_confirm_1atr"] = 1 if vi[f"{method}_confirm_dist"] <= 1.0 else 0
            vi[f"{method}_confirm_05atr"] = 1 if vi[f"{method}_confirm_dist"] <= 0.5 else 0
            vi[f"{method}_oppose_1atr"] = 1 if vi[f"{method}_oppose_dist"] <= 1.0 else 0

        # Any S/R method has confirming level nearby?
        any_confirm_1atr = 1 if any(vi.get(f"{m}_confirm_1atr", 0) for m in sr) else 0
        any_confirm_05atr = 1 if any(vi.get(f"{m}_confirm_05atr", 0) for m in sr) else 0
        vi["any_confirm_1atr"] = any_confirm_1atr
        vi["any_confirm_05atr"] = any_confirm_05atr

        # Count how many methods have confirming S/R within 1 ATR
        vi["confirm_count_1atr"] = sum(1 for m in sr if vi.get(f"{m}_confirm_1atr", 0))
        vi["confirm_count_05atr"] = sum(1 for m in sr if vi.get(f"{m}_confirm_05atr", 0))

        # Run full trade
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
        trade_pnl = -shares * comm; end_bar = entry_bar; exit_reason = "timeout"

        for k in range(1, p["max_hold_bars"] + 1):
            bi = entry_bar + k
            if bi >= n: break
            h = high[bi]; l = low[bi]; ca = atr_v[bi] if not np.isnan(atr_v[bi]) else a
            if times[bi] >= p["force_close_at"]:
                trade_pnl += remaining * (close[bi] - actual_entry) * trend - remaining * comm
                end_bar = bi; exit_reason = "session_close"; break
            stopped = (trend == 1 and l <= runner_stop) or (trend == -1 and h >= runner_stop)
            if stopped:
                trade_pnl += remaining * (runner_stop - actual_entry) * trend - remaining * comm
                end_bar = bi
                exit_reason = "be_stop" if lock_done and abs(runner_stop - actual_entry) < 0.02 else \
                              ("trail_stop" if lock_done else "initial_stop")
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

        trades.append({"r": r_mult, "exit": exit_reason, "dir": trend, **vi})
        bar = end_bar + 1

    return pd.DataFrame(trades)


# ══════════════════════════════════════════════════════════════════
# STEP 2: FILTERED BACKTEST — with S/R filter + 3-bar MFE gate
# ══════════════════════════════════════════════════════════════════

def run_sr_filtered(df, capital=100_000, sr_filter=None, gate_bars=3, gate_mfe=0.3, gate_tighten=-0.3):
    """
    Run backtest with S/R filter + 3b MFE gate.

    sr_filter: dict with keys:
        "method": str — S/R method name (e.g., "prior_day", "swing_50", etc.)
        "side": "confirm" or "oppose" — check confirming or opposing S/R
        "max_dist": float — max distance in ATR units (e.g., 0.5, 1.0, 1.5)
        "mode": "require" or "avoid" — require proximity or avoid it
    If sr_filter is None, runs baseline.
    Can also be a list of filters (all must pass).
    """
    p = BASE.copy()
    df = add_indicators(df, p)
    high = df["High"].values; low = df["Low"].values; close = df["Close"].values
    vol = df["Volume"].values
    ema = df["ema20"].values; ema_s = df["ema50"].values; atr_v = df["atr"].values
    times = df.index.time; dates = df.index.date; n = len(df)
    comm = p["commission_per_share"]

    equity = capital; trades = []
    bar = max(p["ema_slow"], p["atr_period"]) + 200
    daily_r_loss = 0.0; current_date = None; skip_count = 0; sr_filtered = 0

    # Normalize filter to list
    if sr_filter is not None and not isinstance(sr_filter, list):
        sr_filters = [sr_filter]
    else:
        sr_filters = sr_filter

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

        # S/R filter check
        if sr_filters:
            sr = compute_sr_levels(high, low, close, vol, times, dates, bar, a)
            touch_price = low[bar] if trend == 1 else high[bar]
            passed = True

            for filt in sr_filters:
                method = filt["method"]
                side = filt["side"]  # "confirm" or "oppose"
                max_dist = filt["max_dist"]
                mode = filt["mode"]  # "require" or "avoid"

                if method == "any":
                    # Check all methods
                    methods_to_check = list(sr.keys())
                else:
                    methods_to_check = [method]

                # Compute the relevant distance
                found_near = False
                for m in methods_to_check:
                    if m not in sr:
                        continue
                    if side == "confirm":
                        levels = sr[m]["support"] if trend == 1 else sr[m]["resistance"]
                    else:  # oppose
                        levels = sr[m]["resistance"] if trend == 1 else sr[m]["support"]
                    dist = nearest_sr_distance(touch_price, levels, a)
                    if dist <= max_dist:
                        found_near = True
                        break

                if mode == "require" and not found_near:
                    passed = False; break
                elif mode == "avoid" and found_near:
                    passed = False; break

            if not passed:
                sr_filtered += 1; bar += 1; continue

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
        trade_pnl = -shares * comm; end_bar = entry_bar; mfe_so_far = 0

        for k in range(1, p["max_hold_bars"] + 1):
            bi = entry_bar + k
            if bi >= n: break
            h = high[bi]; l = low[bi]; ca = atr_v[bi] if not np.isnan(atr_v[bi]) else a

            if trend == 1: mfe_so_far = max(mfe_so_far, (h - actual_entry) / risk)
            else: mfe_so_far = max(mfe_so_far, (actual_entry - l) / risk)

            if times[bi] >= p["force_close_at"]:
                trade_pnl += remaining * (close[bi] - actual_entry) * trend - remaining * comm
                end_bar = bi; break

            # 3-bar MFE gate
            if gate_bars > 0 and k == gate_bars and not lock_done:
                if mfe_so_far < gate_mfe:
                    new_stop = actual_entry + gate_tighten * risk * trend
                    if trend == 1: runner_stop = max(runner_stop, new_stop)
                    else: runner_stop = min(runner_stop, new_stop)

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
    if total == 0:
        return {"pf": 0, "ret": 0, "trades": 0, "big5": 0, "wr": 0, "avg_r": 0,
                "sr_filt": sr_filtered, "equity": 0}
    gw = tdf.loc[tdf["r"] > 0, "r"].sum()
    gl = abs(tdf.loc[tdf["r"] <= 0, "r"].sum())
    pf = gw / gl if gl > 0 else 0
    ret = (equity - capital) / capital * 100
    r_arr = tdf["r"].values
    wr = (r_arr > 0).sum() / total * 100
    avg_r = r_arr.mean()
    return {"pf": round(pf, 3), "ret": round(ret, 2), "trades": total,
            "big5": int((r_arr >= 5).sum()), "wr": round(wr, 1), "avg_r": round(avg_r, 3),
            "sr_filt": sr_filtered, "equity": round(equity, 0)}


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    df_is = pd.read_csv(IS_PATH, index_col="timestamp", parse_dates=True)
    df_oos = pd.read_csv(OOS_PATH, index_col="timestamp", parse_dates=True)
    print(f"IS: {len(df_is)} bars | OOS: {len(df_oos)} bars\n")

    # ═══════════════════════════════════════════════════════════════
    # STEP 1: SEPARATION ANALYSIS
    # ═══════════════════════════════════════════════════════════════
    print(f"{'='*100}")
    print(f"  STEP 1: S/R INDICATOR SEPARATION — Winners vs Losers")
    print(f"{'='*100}\n")

    tdf = run_sr_analysis(df_is)
    trail = tdf[tdf["exit"] == "trail_stop"]
    init = tdf[tdf["exit"] == "initial_stop"]
    be = tdf[tdf["exit"] == "be_stop"]
    big5 = tdf[tdf["r"] >= 5]
    losers = tdf[tdf["exit"].isin(["initial_stop", "be_stop"])]
    winners = tdf[tdf["r"] > 0]

    print(f"  Trades: {len(tdf)} | Trail: {len(trail)} | InitStop: {len(init)} | BE: {len(be)} | 5R+: {len(big5)}\n")

    # Focus on key distance metrics (confirm_dist and oppose_dist)
    sr_methods = ["prior_day", "swing_50", "swing_100", "swing_200",
                  "round_5", "round_10", "vol_node", "open_range"]
    key_cols = []
    for m in sr_methods:
        key_cols.extend([f"{m}_confirm_dist", f"{m}_oppose_dist",
                         f"{m}_confirm_1atr", f"{m}_confirm_05atr"])
    key_cols.extend(["any_confirm_1atr", "any_confirm_05atr",
                     "confirm_count_1atr", "confirm_count_05atr"])

    # Filter to columns that exist
    key_cols = [c for c in key_cols if c in tdf.columns]

    print(f"  {'Indicator':<30} {'Winners':>10} {'Losers':>10} {'5R+':>10} {'Cohen d':>9} {'Sig':>5}")
    print(f"  {'-'*80}")

    scored_cols = []
    for col in key_cols:
        w_mean = winners[col].mean() if len(winners) > 0 else 0
        l_mean = losers[col].mean() if len(losers) > 0 else 0
        b5_mean = big5[col].mean() if len(big5) > 0 else 0

        w_std = winners[col].std() if len(winners) > 1 else 1
        l_std = losers[col].std() if len(losers) > 1 else 1
        pooled = np.sqrt((w_std**2 + l_std**2) / 2)
        d = abs(w_mean - l_mean) / pooled if pooled > 0 else 0

        sig = "***" if d > 0.4 else ("**" if d > 0.25 else ("*" if d > 0.15 else ""))
        scored_cols.append((col, d))
        print(f"  {col:<30} {w_mean:>10.3f} {l_mean:>10.3f} {b5_mean:>10.3f} {d:>8.3f}  {sig}")

    # Sort by Cohen's d
    print(f"\n  TOP 10 by Cohen's d:")
    scored_cols.sort(key=lambda x: x[1], reverse=True)
    for i, (col, d) in enumerate(scored_cols[:10]):
        print(f"  {i+1}. {col:<30} d={d:.3f}")

    # ═══════════════════════════════════════════════════════════════
    # STEP 1B: CONDITIONAL WIN RATE ANALYSIS
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*100}")
    print(f"  STEP 1B: CONDITIONAL WIN RATE — Does S/R proximity predict outcome?")
    print(f"{'='*100}\n")

    for method in sr_methods:
        confirm_col = f"{method}_confirm_dist"
        oppose_col = f"{method}_oppose_dist"
        if confirm_col not in tdf.columns:
            continue

        # Split by confirming S/R proximity
        for threshold, label in [(0.5, "0.5 ATR"), (1.0, "1.0 ATR"), (1.5, "1.5 ATR")]:
            near = tdf[tdf[confirm_col] <= threshold]
            far = tdf[tdf[confirm_col] > threshold]
            if len(near) < 5 or len(far) < 5:
                continue
            wr_near = (near["r"] > 0).mean() * 100
            wr_far = (far["r"] > 0).mean() * 100
            avg_r_near = near["r"].mean()
            avg_r_far = far["r"].mean()
            pf_near_w = near.loc[near["r"] > 0, "r"].sum()
            pf_near_l = abs(near.loc[near["r"] <= 0, "r"].sum())
            pf_far_w = far.loc[far["r"] > 0, "r"].sum()
            pf_far_l = abs(far.loc[far["r"] <= 0, "r"].sum())
            pf_near = pf_near_w / pf_near_l if pf_near_l > 0 else 0
            pf_far = pf_far_w / pf_far_l if pf_far_l > 0 else 0
            tag = " <-- better" if pf_near > pf_far else ""
            print(f"  {method:<15} confirm<{label}: n={len(near):>4} WR={wr_near:.1f}% avgR={avg_r_near:+.3f} PF={pf_near:.3f}"
                  f"  |  far: n={len(far):>4} WR={wr_far:.1f}% avgR={avg_r_far:+.3f} PF={pf_far:.3f}{tag}")

        # Also check opposing S/R (long near resistance = bad?)
        for threshold, label in [(0.5, "0.5 ATR"), (1.0, "1.0 ATR")]:
            near = tdf[tdf[oppose_col] <= threshold]
            far = tdf[tdf[oppose_col] > threshold]
            if len(near) < 5 or len(far) < 5:
                continue
            wr_near = (near["r"] > 0).mean() * 100
            wr_far = (far["r"] > 0).mean() * 100
            pf_near_w = near.loc[near["r"] > 0, "r"].sum()
            pf_near_l = abs(near.loc[near["r"] <= 0, "r"].sum())
            pf_far_w = far.loc[far["r"] > 0, "r"].sum()
            pf_far_l = abs(far.loc[far["r"] <= 0, "r"].sum())
            pf_near = pf_near_w / pf_near_l if pf_near_l > 0 else 0
            pf_far = pf_far_w / pf_far_l if pf_far_l > 0 else 0
            tag = " <-- avoid" if pf_near < pf_far else ""
            print(f"  {method:<15} OPPOSE<{label}: n={len(near):>4} WR={wr_near:.1f}% PF={pf_near:.3f}"
                  f"  |  far: n={len(far):>4} WR={wr_far:.1f}% PF={pf_far:.3f}{tag}")
        print()

    # ═══════════════════════════════════════════════════════════════
    # STEP 2: FILTER EXPERIMENTS — IS + OOS
    # ═══════════════════════════════════════════════════════════════
    print(f"{'='*100}")
    print(f"  STEP 2: S/R FILTERS — IS + OOS (with 3-bar MFE gate)")
    print(f"{'='*100}\n")

    # Baseline
    bl_is = run_sr_filtered(df_is)
    bl_oos = run_sr_filtered(df_oos)
    print(f"  BASELINE IS:  PF={bl_is['pf']:.3f}  ret={bl_is['ret']:+.2f}%  trades={bl_is['trades']}  5R+={bl_is['big5']}  WR={bl_is['wr']:.1f}%")
    print(f"  BASELINE OOS: PF={bl_oos['pf']:.3f}  ret={bl_oos['ret']:+.2f}%  trades={bl_oos['trades']}  5R+={bl_oos['big5']}  WR={bl_oos['wr']:.1f}%\n")

    # Define all filter configs to test
    filters_to_test = []

    # For each S/R method: require confirming S/R nearby at various distances
    for method in sr_methods:
        for dist in [0.5, 1.0, 1.5]:
            filters_to_test.append((
                f"Require {method} confirm <{dist}ATR",
                {"method": method, "side": "confirm", "max_dist": dist, "mode": "require"}
            ))

    # Anti-filters: avoid opposing S/R nearby
    for method in ["prior_day", "swing_50", "swing_100", "open_range"]:
        for dist in [0.5, 1.0]:
            filters_to_test.append((
                f"Avoid {method} oppose <{dist}ATR",
                {"method": method, "side": "oppose", "max_dist": dist, "mode": "avoid"}
            ))

    # "Any" confirming S/R
    for dist in [0.5, 1.0, 1.5]:
        filters_to_test.append((
            f"Require ANY confirm <{dist}ATR",
            {"method": "any", "side": "confirm", "max_dist": dist, "mode": "require"}
        ))

    # Combos: require confirming AND avoid opposing
    for method in ["prior_day", "swing_100", "open_range"]:
        filters_to_test.append((
            f"{method}: confirm<1ATR + avoid oppose<0.5ATR",
            [
                {"method": method, "side": "confirm", "max_dist": 1.0, "mode": "require"},
                {"method": method, "side": "oppose", "max_dist": 0.5, "mode": "avoid"},
            ]
        ))

    # Multi-method combos
    filters_to_test.append((
        "ANY confirm<1ATR + avoid prior_day oppose<0.5ATR",
        [
            {"method": "any", "side": "confirm", "max_dist": 1.0, "mode": "require"},
            {"method": "prior_day", "side": "oppose", "max_dist": 0.5, "mode": "avoid"},
        ]
    ))
    filters_to_test.append((
        "swing_100 OR open_range confirm<1ATR",
        [
            {"method": "any", "side": "confirm", "max_dist": 1.0, "mode": "require"},
        ]
    ))

    print(f"  {'Filter':<50} {'IS PF':>7} {'IS Ret':>8} {'IS Trd':>7} {'IS 5R+':>6} {'IS WR':>6}"
          f" {'OOS PF':>8} {'OOS Trd':>8} {'OOS 5R+':>8} {'Holds':>6}")
    print(f"  {'-'*115}")

    results = []
    for label, filt in filters_to_test:
        r_is = run_sr_filtered(df_is, sr_filter=filt)
        r_oos = run_sr_filtered(df_oos, sr_filter=filt)
        # "Holds" = IS PF > baseline AND OOS PF > baseline AND OOS trades > 50
        holds = "YES" if (r_is["pf"] > bl_is["pf"] and r_oos["pf"] > bl_oos["pf"]
                          and r_oos["trades"] >= 50) else "no"
        results.append((label, r_is, r_oos, holds))
        print(f"  {label:<50} {r_is['pf']:>6.3f} {r_is['ret']:>+7.2f}% {r_is['trades']:>6} {r_is['big5']:>5} {r_is['wr']:>5.1f}%"
              f" {r_oos['pf']:>7.3f} {r_oos['trades']:>7} {r_oos['big5']:>7}  {holds}")

    # ═══════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*100}")
    print(f"  SUMMARY: TOP FILTERS (sorted by IS PF, showing OOS confirmation)")
    print(f"{'='*100}\n")

    results.sort(key=lambda x: x[1]["pf"], reverse=True)
    print(f"  {'#':<3} {'Filter':<50} {'IS PF':>7} {'OOS PF':>8} {'IS Trd':>7} {'OOS Trd':>8} {'IS 5R+':>6} {'OOS 5R+':>8} {'Holds':>6}")
    print(f"  {'-'*100}")
    for i, (label, r_is, r_oos, holds) in enumerate(results[:15]):
        print(f"  {i+1:<3} {label:<50} {r_is['pf']:>6.3f} {r_oos['pf']:>7.3f} {r_is['trades']:>6} {r_oos['trades']:>7} {r_is['big5']:>5} {r_oos['big5']:>7}  {holds}")

    # Filters that hold on OOS
    oos_winners = [(l, i, o, h) for l, i, o, h in results if h == "YES"]
    if oos_winners:
        print(f"\n  FILTERS THAT HOLD ON OOS ({len(oos_winners)}):")
        print(f"  {'-'*100}")
        for label, r_is, r_oos, holds in oos_winners:
            delta_is = r_is["pf"] - bl_is["pf"]
            delta_oos = r_oos["pf"] - bl_oos["pf"]
            print(f"  {label:<50} IS: {r_is['pf']:.3f} ({delta_is:+.3f})  OOS: {r_oos['pf']:.3f} ({delta_oos:+.3f})  trades: {r_is['trades']}/{r_oos['trades']}")
    else:
        print(f"\n  NO FILTERS HOLD ON OOS -- S/R proximity does not improve edge.")

    print(f"\n  CONCLUSION: ", end="")
    if oos_winners:
        best = max(oos_winners, key=lambda x: x[2]["pf"])
        print(f"Best OOS-confirmed filter: {best[0]}")
        print(f"           IS PF: {best[1]['pf']:.3f} -> OOS PF: {best[2]['pf']:.3f}")
    else:
        print(f"S/R filters do NOT reliably improve edge. EMA20 touch quality is independent of S/R proximity.")


if __name__ == "__main__":
    main()
