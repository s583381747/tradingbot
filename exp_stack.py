"""
Stack Test — Combine all OOS-validated improvements.

Layer 0: Baseline (touch close + 14:00 cutoff + skip-after-win + 3b gate)
Layer 1: + Close Position (DCP >= 0.5 + slope5 > 0)
Layer 2: + RE-ENTRY after BE (w10, cd1, re1)
Layer 3: + Volume Node S/R (within 1.0 ATR of high-vol price)
Layer 4: + Gate pass add (+50% on gate pass)

Test each layer individually AND stacked incrementally.
Run on IS, OOS, and 4-year combined.
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
    "commission_per_share": 0.005,
    "daily_loss_r": 2.5,
    "skip_after_win": 1,
}


def precompute_vol_nodes(df, lookback=200, top_n=20):
    """Precompute high-volume price levels for each bar."""
    vol = df["Volume"].values
    close = df["Close"].values
    atr = df["atr"].values
    n = len(df)
    # For each bar, find the top_n highest volume bars in the lookback
    # and return their close prices as S/R levels
    vol_node_levels = [None] * n
    for i in range(lookback, n):
        window_vol = vol[i-lookback:i]
        window_close = close[i-lookback:i]
        top_idx = np.argsort(window_vol)[-top_n:]
        vol_node_levels[i] = window_close[top_idx]
    return vol_node_levels


def near_vol_node(price, levels, atr_val, threshold=1.0):
    """Check if price is within threshold*ATR of any volume node."""
    if levels is None or atr_val <= 0:
        return False
    for level in levels:
        if abs(price - level) <= threshold * atr_val:
            return True
    return False


def compute_dcp(close_val, high_val, low_val, trend):
    """Directional close position: 1.0 = closed at favorable extreme."""
    rng = high_val - low_val
    if rng <= 0:
        return 0.5
    if trend == 1:
        return (close_val - low_val) / rng
    else:
        return (high_val - close_val) / rng


def execute_trade(trend, actual_entry, stop, risk, shares, entry_bar,
                  high, low, close, atr_v, times, n, p,
                  gate_bars=3, gate_mfe=0.3, gate_tighten=-0.3,
                  add_on_pass=False, add_pct=0.5, equity=100000):
    """Execute one trade with optional gate add. Returns (pnl, end_bar, exit_reason, r_mult, was_be)."""
    comm = p["commission_per_share"]
    lock_sh = max(1, int(shares * p["lock_pct"]))
    remaining = shares; runner_stop = stop; lock_done = False
    trade_pnl = -shares * comm; end_bar = entry_bar
    exit_reason = "timeout"; mfe = 0.0; added = False

    for k in range(1, p["max_hold_bars"] + 1):
        bi = entry_bar + k
        if bi >= n: break
        h = high[bi]; l = low[bi]
        ca = atr_v[bi] if not np.isnan(atr_v[bi]) else atr_v[entry_bar]

        if trend == 1: mfe = max(mfe, (h - actual_entry) / risk)
        else: mfe = max(mfe, (actual_entry - l) / risk)

        if times[bi] >= p["force_close_at"]:
            trade_pnl += remaining * (close[bi] - actual_entry) * trend - remaining * comm
            end_bar = bi; exit_reason = "session_close"; break

        # MFE gate
        if gate_bars > 0 and k == gate_bars and not lock_done:
            if mfe < gate_mfe:
                new_stop = actual_entry + gate_tighten * risk * trend
                if trend == 1: runner_stop = max(runner_stop, new_stop)
                else: runner_stop = min(runner_stop, new_stop)
            elif add_on_pass and not added:
                # Gate passed — add to position
                add_shares = max(1, int(shares * add_pct))
                add_entry = close[bi]
                trade_pnl -= add_shares * comm
                remaining += add_shares
                added = True
                # Recalculate lock shares for new total
                lock_sh = max(1, int(remaining * p["lock_pct"]))

        stopped = (trend == 1 and l <= runner_stop) or (trend == -1 and h >= runner_stop)
        if stopped:
            trade_pnl += remaining * (runner_stop - actual_entry) * trend - remaining * comm
            end_bar = bi
            was_be = lock_done and abs(runner_stop - actual_entry) < 0.02
            exit_reason = "be_stop" if was_be else ("trail_stop" if lock_done else "initial_stop")
            r_mult = trade_pnl / (shares * risk) if shares * risk > 0 else 0
            return trade_pnl, end_bar, exit_reason, r_mult, was_be

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

    r_mult = trade_pnl / (shares * risk) if shares * risk > 0 else 0
    return trade_pnl, end_bar, exit_reason, r_mult, False


def run(df, capital=100_000, use_dcp=False, use_reentry=False, use_volnode=False, use_gate_add=False):
    p = BASE.copy()
    df = add_indicators(df, p)
    high = df["High"].values; low = df["Low"].values; close = df["Close"].values
    open_p = df["Open"].values; vol = df["Volume"].values
    ema = df["ema20"].values; ema_s = df["ema50"].values; atr_v = df["atr"].values
    times = df.index.time; dates = df.index.date; n = len(df)
    comm = p["commission_per_share"]

    # Precompute DCP slope (5-bar)
    dcp_raw = np.zeros(n)
    for i in range(n):
        rng = high[i] - low[i]
        dcp_raw[i] = (close[i] - low[i]) / rng if rng > 0 else 0.5
    dcp_slope = np.zeros(n)
    for i in range(5, n):
        dcp_slope[i] = (dcp_raw[i] - dcp_raw[i-5]) / 5

    # Precompute volume nodes
    vol_nodes = None
    if use_volnode:
        vol_nodes = precompute_vol_nodes(df)

    equity = capital; peak_eq = capital; max_dd = 0
    trades = []; bar = max(p["ema_slow"], p["atr_period"]) + 5
    daily_r_loss = 0.0; current_date = None; skip_count = 0
    re_pending = None  # for re-entry

    while bar < n - p["max_hold_bars"] - 5:
        a = atr_v[bar]
        if np.isnan(a) or a <= 0 or np.isnan(ema[bar]) or np.isnan(ema_s[bar]):
            bar += 1; continue
        if times[bar] >= p["no_entry_after"]: bar += 1; continue
        d = dates[bar]
        if current_date != d:
            current_date = d; daily_r_loss = 0.0; re_pending = None
        if daily_r_loss >= p["daily_loss_r"]: bar += 1; continue

        trend = detect_trend(close[bar], ema[bar], ema_s[bar])

        # RE-ENTRY check
        if use_reentry and re_pending is not None:
            if bar > re_pending["window_until"]:
                re_pending = None
            elif bar < re_pending["cooldown_until"]:
                bar += 1; continue
            elif trend == re_pending["trend"]:
                if check_touch(trend, low[bar], high[bar], ema[bar], a,
                               p["touch_tol"], p["touch_below_max"]):
                    # DCP filter on re-entry too
                    if use_dcp:
                        dcp_val = compute_dcp(close[bar], high[bar], low[bar], trend)
                        if dcp_val < 0.5 or dcp_slope[bar] <= 0:
                            bar += 1; continue

                    actual_entry = close[bar]
                    stop = low[bar] - p["stop_buffer"] * a if trend == 1 else high[bar] + p["stop_buffer"] * a
                    risk = abs(actual_entry - stop)
                    if risk > 0:
                        shares = max(1, int(equity * p["risk_pct"] / risk))
                        if shares * abs(actual_entry) > equity * p["max_pos_pct"]:
                            shares = max(1, int(equity * p["max_pos_pct"] / abs(actual_entry)))
                        if equity >= shares * risk and shares >= 1:
                            pnl, end_bar, exit_reason, r_mult, was_be = execute_trade(
                                trend, actual_entry, stop, risk, shares, bar,
                                high, low, close, atr_v, times, n, p,
                                add_on_pass=use_gate_add, add_pct=0.5, equity=equity)
                            equity += pnl
                            peak_eq = max(peak_eq, equity)
                            max_dd = max(max_dd, (peak_eq - equity) / peak_eq * 100)
                            if pnl < 0 and shares * risk > 0:
                                daily_r_loss += abs(r_mult)
                            trades.append({"r": r_mult, "exit": exit_reason, "dir": trend, "re": True})
                            if r_mult > 0: skip_count = 1
                            if was_be:
                                re_pending = {"trend": trend, "end_bar": end_bar,
                                              "cooldown_until": end_bar + 1, "window_until": end_bar + 10}
                            else:
                                re_pending = None
                            bar = end_bar + 1; continue
                bar += 1; continue
            else:
                bar += 1; continue

        # Normal entry
        if trend == 0: bar += 1; continue
        if not check_touch(trend, low[bar], high[bar], ema[bar], a,
                           p["touch_tol"], p["touch_below_max"]):
            bar += 1; continue

        # DCP filter
        if use_dcp:
            dcp_val = compute_dcp(close[bar], high[bar], low[bar], trend)
            if dcp_val < 0.5 or dcp_slope[bar] <= 0:
                bar += 1; continue

        # Volume node filter
        if use_volnode and vol_nodes is not None:
            entry_price = close[bar]
            if not near_vol_node(entry_price, vol_nodes[bar], a, 1.0):
                bar += 1; continue

        if skip_count > 0: skip_count -= 1; bar += 1; continue

        actual_entry = close[bar]
        stop = low[bar] - p["stop_buffer"] * a if trend == 1 else high[bar] + p["stop_buffer"] * a
        risk = abs(actual_entry - stop)
        if risk <= 0: bar += 1; continue

        shares = max(1, int(equity * p["risk_pct"] / risk))
        if shares * abs(actual_entry) > equity * p["max_pos_pct"]:
            shares = max(1, int(equity * p["max_pos_pct"] / abs(actual_entry)))
        if equity < shares * risk or shares < 1: bar += 1; continue

        pnl, end_bar, exit_reason, r_mult, was_be = execute_trade(
            trend, actual_entry, stop, risk, shares, bar,
            high, low, close, atr_v, times, n, p,
            add_on_pass=use_gate_add, add_pct=0.5, equity=equity)
        equity += pnl
        peak_eq = max(peak_eq, equity)
        max_dd = max(max_dd, (peak_eq - equity) / peak_eq * 100)
        if pnl < 0 and shares * risk > 0:
            daily_r_loss += abs(r_mult)
        trades.append({"r": r_mult, "exit": exit_reason, "dir": trend, "re": False})
        if r_mult > 0: skip_count = 1

        # Set up re-entry if BE
        if use_reentry and was_be:
            re_pending = {"trend": trend, "end_bar": end_bar,
                          "cooldown_until": end_bar + 1, "window_until": end_bar + 10}

        bar = end_bar + 1

    tdf = pd.DataFrame(trades) if trades else pd.DataFrame()
    total = len(tdf)
    if total == 0:
        return {"pf":0,"ret":0,"trades":0,"big5":0,"lpf":0,"spf":0,"max_dd":0,"equity":capital,"tpd":0}
    gw = tdf.loc[tdf["r"]>0,"r"].sum(); gl = abs(tdf.loc[tdf["r"]<=0,"r"].sum())
    pf = gw/gl if gl>0 else 0; ret = (equity-capital)/capital*100
    r_arr = tdf["r"].values; days = df.index.normalize().nunique()
    longs = tdf[tdf["dir"]==1]; shorts = tdf[tdf["dir"]==-1]
    def spf(s):
        if len(s)==0: return 0
        w=s.loc[s["r"]>0,"r"].sum(); l=abs(s.loc[s["r"]<=0,"r"].sum())
        return round(w/l,3) if l>0 else 0
    re_count = tdf["re"].sum() if "re" in tdf.columns else 0
    return {"pf":round(pf,3),"ret":round(ret,2),"trades":total,
            "big5":int((r_arr>=5).sum()),"lpf":spf(longs),"spf":spf(shorts),
            "max_dd":round(max_dd,2),"equity":round(equity,0),
            "tpd":round(total/max(days,1),1),"re_count":int(re_count)}


def main():
    df_is = pd.read_csv(IS_PATH, index_col="timestamp", parse_dates=True)
    df_oos = pd.read_csv(OOS_PATH, index_col="timestamp", parse_dates=True)
    df_4y = pd.concat([df_oos, df_is]).sort_index()
    df_4y = df_4y[~df_4y.index.duplicated(keep='first')]

    hdr = (f"  {'Stack':<45} {'PF':>6} {'Ret%':>8} {'$Equity':>10} {'Trd':>5} {'5R+':>4}"
           f" {'L.PF':>6} {'S.PF':>6} {'DD%':>6} {'Re#':>4}")
    sep = f"  {'-'*105}"

    layers = [
        ("L0: Baseline",                    {}),
        ("L1: + DCP filter",                {"use_dcp": True}),
        ("L2: + RE-ENTRY",                  {"use_reentry": True}),
        ("L3: + Volume Node",               {"use_volnode": True}),
        ("L4: + Gate Add",                  {"use_gate_add": True}),
        ("L1+L2: DCP + RE-ENTRY",           {"use_dcp": True, "use_reentry": True}),
        ("L1+L3: DCP + VolNode",            {"use_dcp": True, "use_volnode": True}),
        ("L1+L4: DCP + GateAdd",            {"use_dcp": True, "use_gate_add": True}),
        ("L1+L2+L4: DCP+RE+GateAdd",        {"use_dcp": True, "use_reentry": True, "use_gate_add": True}),
        ("L1+L2+L3: DCP+RE+VolNode",        {"use_dcp": True, "use_reentry": True, "use_volnode": True}),
        ("L1+L3+L4: DCP+VolNode+GateAdd",   {"use_dcp": True, "use_volnode": True, "use_gate_add": True}),
        ("ALL: DCP+RE+VolNode+GateAdd",      {"use_dcp": True, "use_reentry": True, "use_volnode": True, "use_gate_add": True}),
    ]

    for dataset_label, df_data in [("IN-SAMPLE (Polygon 2024-2026)", df_is),
                                     ("OUT-OF-SAMPLE (Barchart 2022-2024)", df_oos),
                                     ("4-YEAR COMBINED", df_4y)]:
        print(f"\n{'='*110}")
        print(f"  {dataset_label}")
        print(f"{'='*110}")
        print(hdr); print(sep)

        for label, flags in layers:
            r = run(df_data, **flags)
            print(f"  {label:<45} {r['pf']:>6.3f} {r['ret']:>+7.2f}% ${r['equity']:>9,.0f}"
                  f" {r['trades']:>4} {r['big5']:>3} {r['lpf']:>5.2f} {r['spf']:>5.2f}"
                  f" {r['max_dd']:>5.2f}% {r['re_count']:>3}")

    # Annual breakdown for best stack
    print(f"\n{'='*110}")
    print(f"  ANNUAL BREAKDOWN — Best stack on 4-year data")
    print(f"{'='*110}")

    for label, flags in [("Baseline", {}),
                          ("DCP+RE+GateAdd", {"use_dcp": True, "use_reentry": True, "use_gate_add": True}),
                          ("ALL", {"use_dcp": True, "use_reentry": True, "use_volnode": True, "use_gate_add": True})]:
        print(f"\n  {label}:")
        for year_start, year_end, yr_label in [
            ("2022-03-22","2023-03-22","Y1(22-23)"),
            ("2023-03-22","2024-03-22","Y2(23-24)"),
            ("2024-03-22","2025-03-22","Y3(24-25)"),
            ("2025-03-22","2026-03-21","Y4(25-26)")]:
            sub = df_4y[(df_4y.index >= year_start) & (df_4y.index < year_end)]
            if len(sub) < 500: continue
            r = run(sub, **flags)
            print(f"    {yr_label}: PF={r['pf']:.3f} ret={r['ret']:+.2f}% trades={r['trades']} 5R+={r['big5']}"
                  f" L={r['lpf']:.2f} S={r['spf']:.2f}")


if __name__ == "__main__":
    main()
