"""
Volume-Price Analysis — Can volume distinguish good from bad touches?

Core hypotheses:
  1. Healthy pullback: touch on LOW volume → trend continues
  2. Selling pressure: touch on HIGH volume → trend fails
  3. Volume trend: declining volume during trend → weakening
  4. Touch bar relative volume: touch bar vol vs recent average
  5. Directional volume: up-bar volume vs down-bar volume balance
  6. Volume acceleration: is volume increasing or decreasing into the touch?

Step 1: Analyze — compute volume indicators for every trade, measure separation
Step 2: Filter — use best separators as entry conditions
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


def run_vol_analysis(df):
    """Collect volume indicators for every trade."""
    p = BASE.copy()
    df = add_indicators(df, p)
    high = df["High"].values; low = df["Low"].values; close = df["Close"].values
    vol = df["Volume"].values
    ema = df["ema20"].values; ema_s = df["ema50"].values; atr_v = df["atr"].values
    times = df.index.time; dates = df.index.date; n = len(df)
    comm = p["commission_per_share"]

    # Precompute volume indicators
    vol_sma20 = pd.Series(vol).rolling(20).mean().values
    vol_sma50 = pd.Series(vol).rolling(50).mean().values

    # Up/down volume
    up_vol = np.where(close > np.roll(close, 1), vol, 0)
    down_vol = np.where(close < np.roll(close, 1), vol, 0)
    up_vol_sma20 = pd.Series(up_vol).rolling(20).mean().values
    down_vol_sma20 = pd.Series(down_vol).rolling(20).mean().values

    trades = []
    bar = max(p["ema_slow"], p["atr_period"]) + 50
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

        actual_entry = close[bar]
        stop = low[bar] - p["stop_buffer"] * a if trend == 1 else high[bar] + p["stop_buffer"] * a
        risk = abs(actual_entry - stop)
        if risk <= 0: bar += 1; continue

        # Volume indicators at touch bar
        vi = {}
        # 1. Relative volume: touch bar vs SMA
        vi["rvol_20"] = vol[bar] / vol_sma20[bar] if vol_sma20[bar] > 0 else 1
        vi["rvol_50"] = vol[bar] / vol_sma50[bar] if vol_sma50[bar] > 0 else 1

        # 2. Volume of last 3/5 bars relative to SMA (pullback volume)
        vi["pullback_vol_3"] = np.mean(vol[bar-2:bar+1]) / vol_sma20[bar] if vol_sma20[bar] > 0 else 1
        vi["pullback_vol_5"] = np.mean(vol[bar-4:bar+1]) / vol_sma20[bar] if vol_sma20[bar] > 0 else 1

        # 3. Volume trend: SMA20/SMA50 (>1 = volume increasing)
        vi["vol_trend"] = vol_sma20[bar] / vol_sma50[bar] if vol_sma50[bar] > 0 else 1

        # 4. Up/down volume ratio (last 20 bars)
        uv = up_vol_sma20[bar]; dv = down_vol_sma20[bar]
        vi["up_down_ratio"] = uv / dv if dv > 0 else 2
        # For shorts, flip: we want down-volume dominance
        if trend == -1:
            vi["up_down_ratio"] = dv / uv if uv > 0 else 2

        # 5. Volume on trend bars vs counter-trend bars (last 10 bars)
        trend_vol = 0; counter_vol = 0; trend_count = 0; counter_count = 0
        for j in range(bar-9, bar+1):
            if j < 0: continue
            if trend == 1:
                if close[j] > close[j-1]: trend_vol += vol[j]; trend_count += 1
                else: counter_vol += vol[j]; counter_count += 1
            else:
                if close[j] < close[j-1]: trend_vol += vol[j]; trend_count += 1
                else: counter_vol += vol[j]; counter_count += 1
        avg_trend_vol = trend_vol / max(trend_count, 1)
        avg_counter_vol = counter_vol / max(counter_count, 1)
        vi["trend_vs_counter_vol"] = avg_trend_vol / avg_counter_vol if avg_counter_vol > 0 else 2

        # 6. Volume acceleration (last 5 bars vs 5 bars before that)
        recent_vol = np.mean(vol[bar-4:bar+1])
        prev_vol = np.mean(vol[bar-9:bar-4])
        vi["vol_accel"] = recent_vol / prev_vol if prev_vol > 0 else 1

        # 7. Touch bar is low volume bar? (below 20-bar min)
        vol_min20 = min(vol[bar-19:bar+1]) if bar >= 19 else vol[bar]
        vi["is_low_vol"] = 1 if vol[bar] <= vol_min20 * 1.2 else 0

        # 8. Volume dry-up: count of bars in last 10 with vol < 0.5 * sma20
        vi["dry_up_count"] = sum(1 for j in range(bar-9, bar+1)
                                  if j >= 0 and vol_sma20[j] > 0 and vol[j] < 0.5 * vol_sma20[j])

        # 9. Climax detection: any bar in last 5 with vol > 3 * sma20
        vi["recent_climax"] = 1 if any(vol[j] > 3 * vol_sma20[j]
                                        for j in range(bar-4, bar+1)
                                        if j >= 0 and vol_sma20[j] > 0) else 0

        # 10. OBV slope (On Balance Volume direction, last 20 bars)
        obv_change = 0
        for j in range(bar-19, bar+1):
            if j <= 0: continue
            if close[j] > close[j-1]: obv_change += vol[j]
            elif close[j] < close[j-1]: obv_change -= vol[j]
        vi["obv_slope"] = obv_change / (vol_sma20[bar] * 20) if vol_sma20[bar] > 0 else 0
        if trend == -1: vi["obv_slope"] = -vi["obv_slope"]  # flip for shorts

        # Run full trade
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


def run_vol_filtered(df, capital=100_000, filters=None, gate_bars=3, gate_mfe=0.3, gate_tighten=-0.3):
    """Run with volume filters + 3b gate."""
    p = BASE.copy()
    df = add_indicators(df, p)
    high = df["High"].values; low = df["Low"].values; close = df["Close"].values
    vol = df["Volume"].values
    ema = df["ema20"].values; ema_s = df["ema50"].values; atr_v = df["atr"].values
    times = df.index.time; dates = df.index.date; n = len(df)
    comm = p["commission_per_share"]

    vol_sma20 = pd.Series(vol).rolling(20).mean().values
    vol_sma50 = pd.Series(vol).rolling(50).mean().values
    up_vol = np.where(close > np.roll(close, 1), vol, 0)
    down_vol = np.where(close < np.roll(close, 1), vol, 0)
    up_vol_sma20 = pd.Series(up_vol).rolling(20).mean().values
    down_vol_sma20 = pd.Series(down_vol).rolling(20).mean().values

    equity = capital; trades = []
    bar = max(p["ema_slow"], p["atr_period"]) + 50
    daily_r_loss = 0.0; current_date = None; skip_count = 0; vol_filtered = 0

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

        # Volume filters
        if filters:
            passed = True
            for f in filters:
                col = f["col"]; op = f["op"]; val = f["val"]
                # Compute the indicator value
                if col == "rvol_20":
                    iv = vol[bar] / vol_sma20[bar] if vol_sma20[bar] > 0 else 1
                elif col == "pullback_vol_3":
                    iv = np.mean(vol[bar-2:bar+1]) / vol_sma20[bar] if vol_sma20[bar] > 0 else 1
                elif col == "pullback_vol_5":
                    iv = np.mean(vol[bar-4:bar+1]) / vol_sma20[bar] if vol_sma20[bar] > 0 else 1
                elif col == "vol_trend":
                    iv = vol_sma20[bar] / vol_sma50[bar] if vol_sma50[bar] > 0 else 1
                elif col == "up_down_ratio":
                    uv = up_vol_sma20[bar]; dv = down_vol_sma20[bar]
                    iv = (uv / dv if dv > 0 else 2) if trend == 1 else (dv / uv if uv > 0 else 2)
                elif col == "trend_vs_counter_vol":
                    tv = 0; cv = 0; tc = 0; cc = 0
                    for j in range(bar-9, bar+1):
                        if j < 0: continue
                        if (trend == 1 and close[j] > close[j-1]) or (trend == -1 and close[j] < close[j-1]):
                            tv += vol[j]; tc += 1
                        else:
                            cv += vol[j]; cc += 1
                    iv = (tv/max(tc,1)) / (cv/max(cc,1)) if cv > 0 else 2
                elif col == "vol_accel":
                    iv = np.mean(vol[bar-4:bar+1]) / np.mean(vol[bar-9:bar-4]) if np.mean(vol[bar-9:bar-4]) > 0 else 1
                elif col == "obv_slope":
                    obv = 0
                    for j in range(bar-19, bar+1):
                        if j <= 0: continue
                        if close[j] > close[j-1]: obv += vol[j]
                        elif close[j] < close[j-1]: obv -= vol[j]
                    iv = obv / (vol_sma20[bar] * 20) if vol_sma20[bar] > 0 else 0
                    if trend == -1: iv = -iv
                else:
                    iv = 0

                if op == "<=" and iv > val: passed = False; break
                elif op == ">=" and iv < val: passed = False; break
            if not passed:
                vol_filtered += 1; bar += 1; continue

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

            # 3b gate
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
                hv2 = [high[entry_bar+kk] for kk in range(sk, k) if entry_bar+kk < n]
                lv2 = [low[entry_bar+kk] for kk in range(sk, k) if entry_bar+kk < n]
                if hv2 and lv2:
                    if trend == 1: runner_stop = max(runner_stop, max(hv2) - p["chand_mult"] * ca)
                    else: runner_stop = min(runner_stop, min(lv2) + p["chand_mult"] * ca)
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
    if total == 0: return {"pf":0,"ret":0,"trades":0,"big5":0,"lpf":0,"spf":0,"vfilt":0,"equity":0}
    gw = tdf.loc[tdf["r"]>0,"r"].sum(); gl = abs(tdf.loc[tdf["r"]<=0,"r"].sum())
    pf = gw/gl if gl>0 else 0; ret = (equity-capital)/capital*100
    r_arr = tdf["r"].values
    longs = tdf[tdf["dir"]==1]; shorts = tdf[tdf["dir"]==-1]
    def spf(s):
        if len(s)==0: return 0
        w=s.loc[s["r"]>0,"r"].sum(); l=abs(s.loc[s["r"]<=0,"r"].sum())
        return round(w/l,3) if l>0 else 0
    return {"pf":round(pf,3),"ret":round(ret,2),"trades":total,
            "big5":int((r_arr>=5).sum()),"lpf":spf(longs),"spf":spf(shorts),
            "vfilt":vol_filtered,"equity":round(equity,0)}


def main():
    df_is = pd.read_csv(IS_PATH, index_col="timestamp", parse_dates=True)
    df_oos = pd.read_csv(OOS_PATH, index_col="timestamp", parse_dates=True)

    # ═══ STEP 1: SEPARATION ANALYSIS ═══
    print(f"{'='*90}")
    print(f"  STEP 1: VOLUME INDICATOR SEPARATION — Winners vs Losers")
    print(f"{'='*90}\n")

    tdf = run_vol_analysis(df_is)
    trail = tdf[tdf["exit"] == "trail_stop"]
    init = tdf[tdf["exit"] == "initial_stop"]
    be = tdf[tdf["exit"] == "be_stop"]
    big5 = tdf[tdf["r"] >= 5]
    losers = tdf[tdf["exit"].isin(["initial_stop", "be_stop"])]

    vol_cols = [c for c in tdf.columns if c not in ["r", "exit", "dir"]]

    print(f"  Trades: {len(tdf)} | Trail: {len(trail)} | InitStop: {len(init)} | BE: {len(be)} | 5R+: {len(big5)}\n")
    print(f"  {'Indicator':<25} {'Trail(win)':>10} {'InitStop':>10} {'BE stop':>10} {'5R+':>10} {'Cohen d':>9}")
    print(f"  {'-'*80}")

    for col in sorted(vol_cols):
        tw = trail[col].mean(); ti = init[col].mean(); tb = be[col].mean(); t5 = big5[col].mean()
        pooled = np.sqrt((trail[col].std()**2 + losers[col].std()**2) / 2)
        d = abs(tw - losers[col].mean()) / pooled if pooled > 0 else 0
        sig = "★★★" if d > 0.4 else ("★★" if d > 0.25 else ("★" if d > 0.15 else ""))
        print(f"  {col:<25} {tw:>10.3f} {ti:>10.3f} {tb:>10.3f} {t5:>10.3f} {d:>8.3f}  {sig}")

    # ═══ STEP 2: FILTER EXPERIMENTS ═══
    print(f"\n{'='*90}")
    print(f"  STEP 2: VOLUME FILTERS — IS + OOS (with 3b MFE gate baked in)")
    print(f"{'='*90}\n")

    bl_is = run_vol_filtered(df_is)
    bl_oos = run_vol_filtered(df_oos)
    print(f"  Baseline IS:  PF={bl_is['pf']:.3f} ret={bl_is['ret']:+.2f}% trades={bl_is['trades']} 5R+={bl_is['big5']}")
    print(f"  Baseline OOS: PF={bl_oos['pf']:.3f} ret={bl_oos['ret']:+.2f}% trades={bl_oos['trades']} 5R+={bl_oos['big5']}\n")

    print(f"  {'Filter':<45} {'IS PF':>7} {'IS Ret':>8} {'IS Trd':>7} {'IS 5R+':>6}"
          f" {'OOS PF':>8} {'OOS 5R+':>8} {'Holds':>6}")
    print(f"  {'-'*95}")

    filters = [
        # Low volume pullback (touch bar below average)
        ("Touch bar rvol < 0.8 (low vol touch)", [{"col":"rvol_20","op":"<=","val":0.8}]),
        ("Touch bar rvol < 1.0", [{"col":"rvol_20","op":"<=","val":1.0}]),
        ("Touch bar rvol < 1.2", [{"col":"rvol_20","op":"<=","val":1.2}]),
        ("Touch bar rvol < 1.5", [{"col":"rvol_20","op":"<=","val":1.5}]),
        ("Touch bar rvol > 0.5 (not dead)", [{"col":"rvol_20","op":">=","val":0.5}]),

        # Pullback volume
        ("Pullback 3b vol < 0.8×avg", [{"col":"pullback_vol_3","op":"<=","val":0.8}]),
        ("Pullback 3b vol < 1.0×avg", [{"col":"pullback_vol_3","op":"<=","val":1.0}]),
        ("Pullback 5b vol < 0.8×avg", [{"col":"pullback_vol_5","op":"<=","val":0.8}]),
        ("Pullback 5b vol < 1.0×avg", [{"col":"pullback_vol_5","op":"<=","val":1.0}]),

        # Volume trend
        ("Vol trend SMA20/50 < 1.0 (declining)", [{"col":"vol_trend","op":"<=","val":1.0}]),
        ("Vol trend SMA20/50 > 0.8", [{"col":"vol_trend","op":">=","val":0.8}]),

        # Directional volume
        ("Up/down ratio > 1.0 (trend vol dominant)", [{"col":"up_down_ratio","op":">=","val":1.0}]),
        ("Up/down ratio > 1.2", [{"col":"up_down_ratio","op":">=","val":1.2}]),
        ("Up/down ratio > 1.5", [{"col":"up_down_ratio","op":">=","val":1.5}]),
        ("Trend vs counter vol > 1.0", [{"col":"trend_vs_counter_vol","op":">=","val":1.0}]),
        ("Trend vs counter vol > 1.2", [{"col":"trend_vs_counter_vol","op":">=","val":1.2}]),

        # Volume acceleration
        ("Vol accel < 0.8 (decreasing into touch)", [{"col":"vol_accel","op":"<=","val":0.8}]),
        ("Vol accel < 1.0", [{"col":"vol_accel","op":"<=","val":1.0}]),

        # OBV
        ("OBV slope > 0 (volume confirms trend)", [{"col":"obv_slope","op":">=","val":0}]),
        ("OBV slope > 0.1", [{"col":"obv_slope","op":">=","val":0.1}]),
        ("OBV slope > 0.2", [{"col":"obv_slope","op":">=","val":0.2}]),

        # Combos
        ("Low touch vol + trend vol dominant", [{"col":"rvol_20","op":"<=","val":1.0},
                                                 {"col":"up_down_ratio","op":">=","val":1.0}]),
        ("Pullback dry + OBV confirms", [{"col":"pullback_vol_3","op":"<=","val":0.8},
                                          {"col":"obv_slope","op":">=","val":0}]),
        ("Low pullback + declining vol", [{"col":"pullback_vol_5","op":"<=","val":1.0},
                                           {"col":"vol_accel","op":"<=","val":1.0}]),
    ]

    results = []
    for label, filt in filters:
        r_is = run_vol_filtered(df_is, filters=filt)
        r_oos = run_vol_filtered(df_oos, filters=filt)
        holds = "✅" if r_oos["pf"] > bl_oos["pf"] and r_is["pf"] > bl_is["pf"] else "❌"
        results.append((label, r_is, r_oos))
        print(f"  {label:<45} {r_is['pf']:>6.3f} {r_is['ret']:>+7.2f}% {r_is['trades']:>6} {r_is['big5']:>5}"
              f" {r_oos['pf']:>7.3f} {r_oos['big5']:>7}  {holds}")

    # Top results
    print(f"\n  TOP 5 (IS PF):")
    results.sort(key=lambda x: x[1]["pf"], reverse=True)
    for i, (label, r_is, r_oos) in enumerate(results[:5]):
        holds = "✅" if r_oos["pf"] > bl_oos["pf"] and r_is["pf"] > bl_is["pf"] else "❌"
        print(f"  {i+1}. {label:<45} IS={r_is['pf']:.3f} OOS={r_oos['pf']:.3f}  {holds}")


if __name__ == "__main__":
    main()
