"""
Close Position Analysis — Can where the candle closes within the H-L range
distinguish good from bad touches?

Hypothesis: The close position within the High-Low range indicates buying/selling
pressure (proxy for bid-ask/order flow without L2 data).

Indicators:
  1. DCP (Directional Close Position): for longs, (Close-Low)/(High-Low).
     1.0 = closed at high (strong). For shorts, invert: (High-Close)/(High-Low).
  2. DCP of touch bar itself
  3. Average DCP of last 5/10 bars (recent close position trend)
  4. DCP slope: linear regression slope over last 5/10 bars
  5. Consecutive strong closes: how many bars in a row closed in upper half?

Step 1: Separation analysis — compare winners vs losers (Cohen's d)
Step 2: Filter experiments — only enter on strong close patterns. IS + OOS.
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


# ══════════════════════════════════════════════════════════════════
# DCP HELPERS
# ══════════════════════════════════════════════════════════════════

def _dcp_at(bar, trend, high, low, close):
    """Compute directional close position at a single bar."""
    hl = high[bar] - low[bar]
    if hl <= 0:
        return 0.5
    if trend == 1:
        return (close[bar] - low[bar]) / hl
    else:
        return (high[bar] - close[bar]) / hl


# ══════════════════════════════════════════════════════════════════
# STEP 1: SEPARATION ANALYSIS
# ══════════════════════════════════════════════════════════════════

def run_dcp_analysis(df):
    """Collect DCP indicators for every trade, return DataFrame."""
    p = BASE.copy()
    df = add_indicators(df, p)
    high = df["High"].values; low = df["Low"].values; close = df["Close"].values
    opn = df["Open"].values
    ema = df["ema20"].values; ema_s = df["ema50"].values; atr_v = df["atr"].values
    times = df.index.time; dates = df.index.date; n = len(df)
    comm = p["commission_per_share"]

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

        # ─── DCP indicators at touch bar ───
        vi = {}

        # 1. DCP of touch bar (directional)
        vi["dcp_touch"] = _dcp_at(bar, trend, high, low, close)

        # 2. Average DCP of last 5 bars (directional)
        dcp_vals_5 = [_dcp_at(j, trend, high, low, close) for j in range(max(0, bar - 4), bar + 1)]
        vi["dcp_avg5"] = np.mean(dcp_vals_5) if dcp_vals_5 else 0.5

        # 3. Average DCP of last 10 bars (directional)
        dcp_vals_10 = [_dcp_at(j, trend, high, low, close) for j in range(max(0, bar - 9), bar + 1)]
        vi["dcp_avg10"] = np.mean(dcp_vals_10) if dcp_vals_10 else 0.5

        # 4. DCP slope over last 5 bars
        seg5 = np.array(dcp_vals_5)
        if len(seg5) >= 3:
            x = np.arange(len(seg5), dtype=float); x -= x.mean()
            y = seg5 - seg5.mean(); denom = (x * x).sum()
            vi["dcp_slope5"] = (x * y).sum() / denom if denom > 0 else 0.0
        else:
            vi["dcp_slope5"] = 0.0

        # 5. DCP slope over last 10 bars
        seg10 = np.array(dcp_vals_10)
        if len(seg10) >= 3:
            x = np.arange(len(seg10), dtype=float); x -= x.mean()
            y = seg10 - seg10.mean(); denom = (x * x).sum()
            vi["dcp_slope10"] = (x * y).sum() / denom if denom > 0 else 0.0
        else:
            vi["dcp_slope10"] = 0.0

        # 6. Consecutive strong closes (DCP > 0.5)
        consec = 0
        for j in range(bar, max(bar - 20, -1), -1):
            if j < 0: break
            if _dcp_at(j, trend, high, low, close) > 0.5:
                consec += 1
            else:
                break
        vi["consec_strong"] = consec

        # 7. Consecutive strong closes at DCP > 0.6
        consec60 = 0
        for j in range(bar, max(bar - 20, -1), -1):
            if j < 0: break
            if _dcp_at(j, trend, high, low, close) > 0.6:
                consec60 += 1
            else:
                break
        vi["consec_strong60"] = consec60

        # 8. DCP of bar before touch (bar - 1)
        vi["dcp_prev1"] = _dcp_at(bar - 1, trend, high, low, close) if bar >= 1 else 0.5

        # 9. DCP change (touch bar vs prev bar)
        vi["dcp_change"] = vi["dcp_touch"] - vi["dcp_prev1"]

        # 10. Fraction of last 10 bars with DCP > 0.5
        strong_count = sum(1 for j in range(max(0, bar - 9), bar + 1)
                           if _dcp_at(j, trend, high, low, close) > 0.5)
        total_count = min(10, bar + 1)
        vi["strong_ratio10"] = strong_count / max(total_count, 1)

        # 11. Min DCP in last 3 bars
        dcp_last3 = [_dcp_at(j, trend, high, low, close) for j in range(max(0, bar - 2), bar + 1)]
        vi["dcp_min3"] = min(dcp_last3) if dcp_last3 else 0.5

        # 12. Touch bar body size relative to H-L range
        hl = high[bar] - low[bar]
        vi["body_ratio"] = abs(close[bar] - opn[bar]) / hl if hl > 0 else 0.0

        # ─── Run full trade ───
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
                hv = [high[entry_bar + kk] for kk in range(sk, k) if entry_bar + kk < n]
                lv = [low[entry_bar + kk] for kk in range(sk, k) if entry_bar + kk < n]
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
# STEP 2: FILTERED BACKTEST (with 3-bar MFE gate)
# ══════════════════════════════════════════════════════════════════

def _compute_dcp_indicator(col, bar, trend, high, low, close, opn):
    """Compute a single DCP indicator value at given bar."""
    if col == "dcp_touch":
        return _dcp_at(bar, trend, high, low, close)

    elif col == "dcp_avg5":
        vals = [_dcp_at(j, trend, high, low, close) for j in range(max(0, bar - 4), bar + 1)]
        return np.mean(vals) if vals else 0.5

    elif col == "dcp_avg10":
        vals = [_dcp_at(j, trend, high, low, close) for j in range(max(0, bar - 9), bar + 1)]
        return np.mean(vals) if vals else 0.5

    elif col == "dcp_slope5":
        vals = np.array([_dcp_at(j, trend, high, low, close) for j in range(max(0, bar - 4), bar + 1)])
        if len(vals) < 3: return 0.0
        x = np.arange(len(vals), dtype=float); x -= x.mean()
        y = vals - vals.mean(); denom = (x * x).sum()
        return (x * y).sum() / denom if denom > 0 else 0.0

    elif col == "dcp_slope10":
        vals = np.array([_dcp_at(j, trend, high, low, close) for j in range(max(0, bar - 9), bar + 1)])
        if len(vals) < 3: return 0.0
        x = np.arange(len(vals), dtype=float); x -= x.mean()
        y = vals - vals.mean(); denom = (x * x).sum()
        return (x * y).sum() / denom if denom > 0 else 0.0

    elif col == "consec_strong":
        count = 0
        for j in range(bar, max(bar - 20, -1), -1):
            if j < 0: break
            if _dcp_at(j, trend, high, low, close) > 0.5: count += 1
            else: break
        return count

    elif col == "consec_strong60":
        count = 0
        for j in range(bar, max(bar - 20, -1), -1):
            if j < 0: break
            if _dcp_at(j, trend, high, low, close) > 0.6: count += 1
            else: break
        return count

    elif col == "dcp_prev1":
        return _dcp_at(bar - 1, trend, high, low, close) if bar >= 1 else 0.5

    elif col == "dcp_change":
        touch = _dcp_at(bar, trend, high, low, close)
        prev = _dcp_at(bar - 1, trend, high, low, close) if bar >= 1 else 0.5
        return touch - prev

    elif col == "strong_ratio10":
        cnt = sum(1 for j in range(max(0, bar - 9), bar + 1)
                  if _dcp_at(j, trend, high, low, close) > 0.5)
        total = min(10, bar + 1)
        return cnt / max(total, 1)

    elif col == "dcp_min3":
        vals = [_dcp_at(j, trend, high, low, close) for j in range(max(0, bar - 2), bar + 1)]
        return min(vals) if vals else 0.5

    elif col == "body_ratio":
        hl = high[bar] - low[bar]
        return abs(close[bar] - opn[bar]) / hl if hl > 0 else 0.0

    return 0.5


def run_dcp_filtered(df, capital=100_000, filters=None,
                     gate_bars=3, gate_mfe=0.3, gate_tighten=-0.3):
    """Run backtest with DCP filters + 3-bar MFE gate."""
    p = BASE.copy()
    df = add_indicators(df, p)
    high = df["High"].values; low = df["Low"].values; close = df["Close"].values
    opn = df["Open"].values
    ema = df["ema20"].values; ema_s = df["ema50"].values; atr_v = df["atr"].values
    times = df.index.time; dates = df.index.date; n = len(df)
    comm = p["commission_per_share"]

    equity = capital; trades = []
    bar = max(p["ema_slow"], p["atr_period"]) + 50
    daily_r_loss = 0.0; current_date = None; skip_count = 0; dcp_filtered = 0

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

        # ─── DCP filters ───
        if filters:
            passed = True
            for f in filters:
                col = f["col"]; op = f["op"]; val = f["val"]
                iv = _compute_dcp_indicator(col, bar, trend, high, low, close, opn)
                if op == "<=" and iv > val: passed = False; break
                elif op == ">=" and iv < val: passed = False; break
            if not passed:
                dcp_filtered += 1; bar += 1; continue

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
                hv2 = [high[entry_bar + kk] for kk in range(sk, k) if entry_bar + kk < n]
                lv2 = [low[entry_bar + kk] for kk in range(sk, k) if entry_bar + kk < n]
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
    if total == 0:
        return {"pf": 0, "ret": 0, "trades": 0, "big5": 0, "lpf": 0, "spf": 0,
                "dfilt": 0, "equity": 0}
    gw = tdf.loc[tdf["r"] > 0, "r"].sum()
    gl = abs(tdf.loc[tdf["r"] <= 0, "r"].sum())
    pf = gw / gl if gl > 0 else 0
    ret = (equity - capital) / capital * 100
    r_arr = tdf["r"].values
    longs = tdf[tdf["dir"] == 1]; shorts = tdf[tdf["dir"] == -1]

    def spf(s):
        if len(s) == 0: return 0
        w = s.loc[s["r"] > 0, "r"].sum(); l = abs(s.loc[s["r"] <= 0, "r"].sum())
        return round(w / l, 3) if l > 0 else 0

    return {"pf": round(pf, 3), "ret": round(ret, 2), "trades": total,
            "big5": int((r_arr >= 5).sum()), "lpf": spf(longs), "spf": spf(shorts),
            "dfilt": dcp_filtered, "equity": round(equity, 0)}


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    df_is = pd.read_csv(IS_PATH, index_col="timestamp", parse_dates=True)
    df_oos = pd.read_csv(OOS_PATH, index_col="timestamp", parse_dates=True)

    # ═══ STEP 1: SEPARATION ANALYSIS ═══
    print(f"{'=' * 90}")
    print(f"  STEP 1: CLOSE POSITION (DCP) SEPARATION -- Winners vs Losers")
    print(f"{'=' * 90}\n")

    tdf = run_dcp_analysis(df_is)
    trail = tdf[tdf["exit"] == "trail_stop"]
    init = tdf[tdf["exit"] == "initial_stop"]
    be = tdf[tdf["exit"] == "be_stop"]
    big5 = tdf[tdf["r"] >= 5]
    winners = tdf[tdf["r"] > 0]
    all_losers = tdf[tdf["r"] <= 0]

    dcp_cols = [c for c in tdf.columns if c not in ["r", "exit", "dir"]]

    print(f"  Total trades: {len(tdf)} | Trail stops: {len(trail)} | Init stops: {len(init)}"
          f" | BE stops: {len(be)} | 5R+: {len(big5)}")
    print(f"  Winners: {len(winners)} | Losers: {len(all_losers)}\n")

    print(f"  {'Indicator':<22} {'Winners':>9} {'Losers':>9} {'Trail':>9} {'InitStop':>9}"
          f" {'BE':>9} {'5R+':>9} {'Cohen d':>9}  Sep")
    print(f"  {'-' * 95}")

    for col in sorted(dcp_cols):
        wm = winners[col].mean()
        lm = all_losers[col].mean()
        tw = trail[col].mean()
        ti = init[col].mean()
        tb = be[col].mean()
        t5 = big5[col].mean() if len(big5) > 0 else 0
        ws = winners[col].std()
        ls = all_losers[col].std()
        pooled = np.sqrt((ws ** 2 + ls ** 2) / 2)
        d = abs(wm - lm) / pooled if pooled > 0 else 0
        direction = "+" if wm > lm else "-"
        sig = "***" if d > 0.4 else ("** " if d > 0.25 else ("*  " if d > 0.15 else "   "))
        print(f"  {col:<22} {wm:>9.4f} {lm:>9.4f} {tw:>9.4f} {ti:>9.4f}"
              f" {tb:>9.4f} {t5:>9.4f} {d:>8.3f}{direction} {sig}")

    # Show OOS separation too
    print(f"\n  OOS Separation Check:")
    tdf_oos = run_dcp_analysis(df_oos)
    winners_oos = tdf_oos[tdf_oos["r"] > 0]
    losers_oos = tdf_oos[tdf_oos["r"] <= 0]
    print(f"  OOS trades: {len(tdf_oos)} | W: {len(winners_oos)} | L: {len(losers_oos)}\n")

    print(f"  {'Indicator':<22} {'IS d':>8} {'IS dir':>7} {'OOS d':>8} {'OOS dir':>8} {'Stable?':>8}")
    print(f"  {'-' * 60}")

    oos_dcp_cols = [c for c in tdf_oos.columns if c not in ["r", "exit", "dir"]]

    for col in sorted(dcp_cols):
        if col not in oos_dcp_cols:
            continue
        # IS
        wm_is = winners[col].mean(); lm_is = all_losers[col].mean()
        ws_is = winners[col].std(); ls_is = all_losers[col].std()
        p_is = np.sqrt((ws_is ** 2 + ls_is ** 2) / 2)
        d_is = abs(wm_is - lm_is) / p_is if p_is > 0 else 0
        dir_is = "+" if wm_is > lm_is else "-"
        # OOS
        wm_oos = winners_oos[col].mean(); lm_oos = losers_oos[col].mean()
        ws_oos = winners_oos[col].std(); ls_oos = losers_oos[col].std()
        p_oos = np.sqrt((ws_oos ** 2 + ls_oos ** 2) / 2)
        d_oos = abs(wm_oos - lm_oos) / p_oos if p_oos > 0 else 0
        dir_oos = "+" if wm_oos > lm_oos else "-"
        stable = "YES" if dir_is == dir_oos and d_oos > 0.1 else ("weak" if dir_is == dir_oos else "FLIP")
        print(f"  {col:<22} {d_is:>8.3f} {dir_is:>6} {d_oos:>8.3f} {dir_oos:>7} {stable:>8}")

    # ═══ STEP 2: FILTER EXPERIMENTS ═══
    print(f"\n{'=' * 90}")
    print(f"  STEP 2: DCP FILTERS -- IS + OOS (with 3-bar MFE gate)")
    print(f"{'=' * 90}\n")

    bl_is = run_dcp_filtered(df_is)
    bl_oos = run_dcp_filtered(df_oos)
    print(f"  Baseline IS:  PF={bl_is['pf']:.3f}  ret={bl_is['ret']:+.2f}%"
          f"  trades={bl_is['trades']}  5R+={bl_is['big5']}")
    print(f"  Baseline OOS: PF={bl_oos['pf']:.3f}  ret={bl_oos['ret']:+.2f}%"
          f"  trades={bl_oos['trades']}  5R+={bl_oos['big5']}\n")

    print(f"  {'Filter':<50} {'IS PF':>7} {'IS Ret':>8} {'IS Trd':>7} {'IS 5R+':>6}"
          f" {'OOS PF':>8} {'OOS Trd':>8} {'OOS 5R+':>8} {'Holds':>6}")
    print(f"  {'-' * 110}")

    filter_configs = [
        # Touch bar DCP
        ("Touch DCP >= 0.4 (not closing at low)", [{"col": "dcp_touch", "op": ">=", "val": 0.4}]),
        ("Touch DCP >= 0.5 (upper half)", [{"col": "dcp_touch", "op": ">=", "val": 0.5}]),
        ("Touch DCP >= 0.6 (strong close)", [{"col": "dcp_touch", "op": ">=", "val": 0.6}]),
        ("Touch DCP >= 0.7 (very strong)", [{"col": "dcp_touch", "op": ">=", "val": 0.7}]),
        ("Touch DCP <= 0.3 (weak close)", [{"col": "dcp_touch", "op": "<=", "val": 0.3}]),

        # Average DCP
        ("Avg DCP 5b >= 0.45", [{"col": "dcp_avg5", "op": ">=", "val": 0.45}]),
        ("Avg DCP 5b >= 0.5", [{"col": "dcp_avg5", "op": ">=", "val": 0.5}]),
        ("Avg DCP 5b >= 0.55", [{"col": "dcp_avg5", "op": ">=", "val": 0.55}]),
        ("Avg DCP 10b >= 0.45", [{"col": "dcp_avg10", "op": ">=", "val": 0.45}]),
        ("Avg DCP 10b >= 0.5", [{"col": "dcp_avg10", "op": ">=", "val": 0.5}]),
        ("Avg DCP 10b >= 0.55", [{"col": "dcp_avg10", "op": ">=", "val": 0.55}]),

        # DCP slope
        ("DCP slope 5b > 0 (improving)", [{"col": "dcp_slope5", "op": ">=", "val": 0.0}]),
        ("DCP slope 5b > 0.02", [{"col": "dcp_slope5", "op": ">=", "val": 0.02}]),
        ("DCP slope 5b > 0.05", [{"col": "dcp_slope5", "op": ">=", "val": 0.05}]),
        ("DCP slope 10b > 0 (improving)", [{"col": "dcp_slope10", "op": ">=", "val": 0.0}]),
        ("DCP slope 10b > 0.01", [{"col": "dcp_slope10", "op": ">=", "val": 0.01}]),

        # Consecutive strong closes
        ("Consec strong (>0.5) >= 1", [{"col": "consec_strong", "op": ">=", "val": 1}]),
        ("Consec strong (>0.5) >= 2", [{"col": "consec_strong", "op": ">=", "val": 2}]),
        ("Consec strong (>0.5) >= 3", [{"col": "consec_strong", "op": ">=", "val": 3}]),
        ("Consec strong60 (>0.6) >= 1", [{"col": "consec_strong60", "op": ">=", "val": 1}]),
        ("Consec strong60 (>0.6) >= 2", [{"col": "consec_strong60", "op": ">=", "val": 2}]),

        # DCP change (momentum building)
        ("DCP change > 0 (improving)", [{"col": "dcp_change", "op": ">=", "val": 0.0}]),
        ("DCP change > 0.1", [{"col": "dcp_change", "op": ">=", "val": 0.1}]),
        ("DCP change > 0.2", [{"col": "dcp_change", "op": ">=", "val": 0.2}]),

        # Strong ratio
        ("Strong ratio 10b >= 0.5", [{"col": "strong_ratio10", "op": ">=", "val": 0.5}]),
        ("Strong ratio 10b >= 0.6", [{"col": "strong_ratio10", "op": ">=", "val": 0.6}]),
        ("Strong ratio 10b >= 0.7", [{"col": "strong_ratio10", "op": ">=", "val": 0.7}]),

        # Min DCP in last 3 bars (floor quality)
        ("Min DCP 3b >= 0.3", [{"col": "dcp_min3", "op": ">=", "val": 0.3}]),
        ("Min DCP 3b >= 0.4", [{"col": "dcp_min3", "op": ">=", "val": 0.4}]),

        # Body ratio
        ("Body ratio >= 0.4 (decisive bar)", [{"col": "body_ratio", "op": ">=", "val": 0.4}]),
        ("Body ratio >= 0.5", [{"col": "body_ratio", "op": ">=", "val": 0.5}]),
        ("Body ratio >= 0.6", [{"col": "body_ratio", "op": ">=", "val": 0.6}]),

        # Combos
        ("Touch DCP>=0.5 + Avg5>=0.5", [
            {"col": "dcp_touch", "op": ">=", "val": 0.5},
            {"col": "dcp_avg5", "op": ">=", "val": 0.5}]),
        ("Touch DCP>=0.5 + slope5>0", [
            {"col": "dcp_touch", "op": ">=", "val": 0.5},
            {"col": "dcp_slope5", "op": ">=", "val": 0.0}]),
        ("Touch DCP>=0.6 + consec>=2", [
            {"col": "dcp_touch", "op": ">=", "val": 0.6},
            {"col": "consec_strong", "op": ">=", "val": 2}]),
        ("Avg5>=0.5 + slope5>0 + body>=0.4", [
            {"col": "dcp_avg5", "op": ">=", "val": 0.5},
            {"col": "dcp_slope5", "op": ">=", "val": 0.0},
            {"col": "body_ratio", "op": ">=", "val": 0.4}]),
        ("Strong ratio>=0.6 + touch>=0.5", [
            {"col": "strong_ratio10", "op": ">=", "val": 0.6},
            {"col": "dcp_touch", "op": ">=", "val": 0.5}]),
        ("Consec>=2 + DCP change>0", [
            {"col": "consec_strong", "op": ">=", "val": 2},
            {"col": "dcp_change", "op": ">=", "val": 0.0}]),
        ("Min3>=0.3 + avg10>=0.5", [
            {"col": "dcp_min3", "op": ">=", "val": 0.3},
            {"col": "dcp_avg10", "op": ">=", "val": 0.5}]),
        ("Touch>=0.5 + body>=0.5 + slope5>0", [
            {"col": "dcp_touch", "op": ">=", "val": 0.5},
            {"col": "body_ratio", "op": ">=", "val": 0.5},
            {"col": "dcp_slope5", "op": ">=", "val": 0.0}]),
    ]

    results = []
    for label, filt in filter_configs:
        r_is = run_dcp_filtered(df_is, filters=filt)
        r_oos = run_dcp_filtered(df_oos, filters=filt)
        holds = "YES" if r_oos["pf"] > bl_oos["pf"] and r_is["pf"] > bl_is["pf"] else "no"
        results.append((label, r_is, r_oos, holds))
        print(f"  {label:<50} {r_is['pf']:>6.3f} {r_is['ret']:>+7.2f}% {r_is['trades']:>6}"
              f" {r_is['big5']:>5} {r_oos['pf']:>7.3f} {r_oos['trades']:>7}"
              f" {r_oos['big5']:>7}  {holds}")

    # ═══ SUMMARY ═══
    print(f"\n{'=' * 90}")
    print(f"  SUMMARY: TOP 5 BY IS PF (that also beat baseline OOS)")
    print(f"{'=' * 90}\n")

    both_beat = [r for r in results if r[3] == "YES"]
    both_beat.sort(key=lambda x: x[1]["pf"], reverse=True)

    if both_beat:
        for i, (label, r_is, r_oos, _) in enumerate(both_beat[:5]):
            print(f"  {i + 1}. {label}")
            print(f"     IS:  PF={r_is['pf']:.3f}  ret={r_is['ret']:+.2f}%"
                  f"  trades={r_is['trades']}  5R+={r_is['big5']}")
            print(f"     OOS: PF={r_oos['pf']:.3f}  ret={r_oos['ret']:+.2f}%"
                  f"  trades={r_oos['trades']}  5R+={r_oos['big5']}")
    else:
        print("  No filters beat baseline on both IS and OOS.")

    print(f"\n  TOP 5 BY IS PF (regardless of OOS):")
    results.sort(key=lambda x: x[1]["pf"], reverse=True)
    for i, (label, r_is, r_oos, holds) in enumerate(results[:5]):
        print(f"  {i + 1}. {label}")
        print(f"     IS:  PF={r_is['pf']:.3f}  ret={r_is['ret']:+.2f}%  trades={r_is['trades']}")
        print(f"     OOS: PF={r_oos['pf']:.3f}  ret={r_oos['ret']:+.2f}%  trades={r_oos['trades']}  {holds}")


if __name__ == "__main__":
    main()
