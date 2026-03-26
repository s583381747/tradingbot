"""
RE-ENTRY v2 — Touch Close entry (not old signal trigger).

After BE exit, if price touches EMA20 again within N bars, re-enter.
Uses the correct touch close entry (no gap bias).

Sweep:
  - window: 10/20/30 bars after BE to look for re-touch
  - cooldown: 1/3/5 bars before scanning
  - max_reentries: 1/2

Includes 3-bar MFE gate on all entries (original + re-entry).
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


def execute_trade(trend, actual_entry, stop, risk, shares, entry_bar,
                  high, low, close, atr_v, times, n, p,
                  gate_bars=3, gate_mfe=0.3, gate_tighten=-0.3):
    """Execute one trade. Returns (pnl, end_bar, exit_reason, r_mult)."""
    comm = p["commission_per_share"]
    lock_sh = max(1, int(shares * p["lock_pct"]))
    remaining = shares; runner_stop = stop; lock_done = False
    trade_pnl = -shares * comm; end_bar = entry_bar
    exit_reason = "timeout"; mfe = 0.0

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
    return trade_pnl, end_bar, exit_reason, r_mult


def run(df, capital=100_000, reentry_window=0, reentry_cooldown=1, max_reentries=0):
    p = BASE.copy()
    df = add_indicators(df, p)
    high = df["High"].values; low = df["Low"].values; close = df["Close"].values
    ema = df["ema20"].values; ema_s = df["ema50"].values; atr_v = df["atr"].values
    times = df.index.time; dates = df.index.date; n = len(df)
    comm = p["commission_per_share"]

    equity = capital; trades = []
    bar = max(p["ema_slow"], p["atr_period"]) + 5
    daily_r_loss = 0.0; current_date = None; skip_count = 0
    reentry_count = 0; total_reentries = 0

    # Pending re-entry state
    re_pending = None  # {"trend": int, "end_bar": int, "cooldown_until": int, "window_until": int, "chain": int}

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

        # Check for re-entry opportunity
        if re_pending is not None:
            if bar > re_pending["window_until"]:
                re_pending = None  # window expired
            elif bar < re_pending["cooldown_until"]:
                bar += 1; continue  # still in cooldown
            elif trend == re_pending["trend"]:
                # Check touch for re-entry
                if check_touch(trend, low[bar], high[bar], ema[bar], a,
                               p["touch_tol"], p["touch_below_max"]):
                    # RE-ENTER at touch close
                    actual_entry = close[bar]
                    stop = low[bar] - p["stop_buffer"] * a if trend == 1 else high[bar] + p["stop_buffer"] * a
                    risk = abs(actual_entry - stop)
                    if risk > 0:
                        shares = max(1, int(equity * p["risk_pct"] / risk))
                        if shares * abs(actual_entry) > equity * p["max_pos_pct"]:
                            shares = max(1, int(equity * p["max_pos_pct"] / abs(actual_entry)))
                        if equity >= shares * risk and shares >= 1:
                            pnl, end_bar, exit_reason, r_mult = execute_trade(
                                trend, actual_entry, stop, risk, shares, bar,
                                high, low, close, atr_v, times, n, p)
                            equity += pnl
                            if pnl < 0 and shares * risk > 0:
                                daily_r_loss += abs(r_mult)
                            trades.append({"r": r_mult, "exit": exit_reason,
                                           "dir": trend, "is_reentry": True})
                            total_reentries += 1

                            if r_mult > 0: skip_count = 1

                            # Chain: if this also BE'd, allow another re-entry
                            if exit_reason == "be_stop" and re_pending["chain"] < max_reentries:
                                re_pending = {
                                    "trend": trend, "end_bar": end_bar,
                                    "cooldown_until": end_bar + reentry_cooldown,
                                    "window_until": end_bar + reentry_window,
                                    "chain": re_pending["chain"] + 1,
                                }
                            else:
                                re_pending = None

                            bar = end_bar + 1; continue
                # Not a touch, keep scanning
                bar += 1; continue
            else:
                # Trend doesn't match anymore
                bar += 1; continue

        # Normal entry logic
        if trend == 0: bar += 1; continue
        if not check_touch(trend, low[bar], high[bar], ema[bar], a,
                           p["touch_tol"], p["touch_below_max"]):
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

        pnl, end_bar, exit_reason, r_mult = execute_trade(
            trend, actual_entry, stop, risk, shares, bar,
            high, low, close, atr_v, times, n, p)
        equity += pnl
        if pnl < 0 and shares * risk > 0:
            daily_r_loss += abs(r_mult)
        trades.append({"r": r_mult, "exit": exit_reason, "dir": trend, "is_reentry": False})

        if r_mult > 0: skip_count = 1

        # If BE exit and re-entry enabled, set up pending
        if exit_reason == "be_stop" and reentry_window > 0 and max_reentries > 0:
            re_pending = {
                "trend": trend, "end_bar": end_bar,
                "cooldown_until": end_bar + reentry_cooldown,
                "window_until": end_bar + reentry_window,
                "chain": 1,
            }

        bar = end_bar + 1

    # Stats
    tdf = pd.DataFrame(trades) if trades else pd.DataFrame()
    total = len(tdf)
    if total == 0:
        return {"pf": 0, "ret": 0, "trades": 0, "big5": 0, "lpf": 0, "spf": 0,
                "reentries": 0, "re_pf": 0}
    gw = tdf.loc[tdf["r"] > 0, "r"].sum()
    gl = abs(tdf.loc[tdf["r"] <= 0, "r"].sum())
    pf = gw / gl if gl > 0 else 0
    ret = (equity - capital) / capital * 100
    r_arr = tdf["r"].values
    days = df.index.normalize().nunique()
    longs = tdf[tdf["dir"] == 1]; shorts = tdf[tdf["dir"] == -1]
    def spf(s):
        if len(s) == 0: return 0
        w = s.loc[s["r"] > 0, "r"].sum(); l = abs(s.loc[s["r"] <= 0, "r"].sum())
        return round(w / l, 3) if l > 0 else 0

    # Re-entry specific stats
    re_trades = tdf[tdf["is_reentry"] == True] if "is_reentry" in tdf.columns else pd.DataFrame()
    re_gw = re_trades.loc[re_trades["r"] > 0, "r"].sum() if len(re_trades) > 0 else 0
    re_gl = abs(re_trades.loc[re_trades["r"] <= 0, "r"].sum()) if len(re_trades) > 0 else 0
    re_pf = re_gw / re_gl if re_gl > 0 else 0

    return {
        "pf": round(pf, 3), "ret": round(ret, 2),
        "trades": total, "tpd": round(total / max(days, 1), 1),
        "big5": int((r_arr >= 5).sum()),
        "lpf": spf(longs), "spf": spf(shorts),
        "reentries": len(re_trades),
        "re_pf": round(re_pf, 3),
        "re_avg_r": round(re_trades["r"].mean(), 3) if len(re_trades) > 0 else 0,
    }


def main():
    df_is = pd.read_csv(IS_PATH, index_col="timestamp", parse_dates=True)
    df_oos = pd.read_csv(OOS_PATH, index_col="timestamp", parse_dates=True)
    print(f"IS: {len(df_is):,} bars | OOS: {len(df_oos):,} bars\n")

    # Baseline
    bl_is = run(df_is)
    bl_oos = run(df_oos)
    print(f"Baseline IS:  PF={bl_is['pf']:.3f} ret={bl_is['ret']:+.2f}% trades={bl_is['trades']} 5R+={bl_is['big5']}")
    print(f"Baseline OOS: PF={bl_oos['pf']:.3f} ret={bl_oos['ret']:+.2f}% trades={bl_oos['trades']} 5R+={bl_oos['big5']}\n")

    print(f"{'Config':<30} {'IS PF':>7} {'IS Ret':>8} {'IS Trd':>6} {'IS 5R+':>6} {'IS Re#':>6} {'RePF':>6}"
          f" {'OOS PF':>8} {'OOS Ret':>9} {'OOS Trd':>8} {'OOS Re#':>8} {'Holds':>6}")
    print(f"{'-'*115}")

    for window in [10, 20, 30]:
        for cooldown in [1, 3, 5]:
            for max_re in [1, 2]:
                label = f"w{window}_cd{cooldown}_re{max_re}"
                r_is = run(df_is, reentry_window=window, reentry_cooldown=cooldown, max_reentries=max_re)
                r_oos = run(df_oos, reentry_window=window, reentry_cooldown=cooldown, max_reentries=max_re)
                holds = "✅" if r_is["pf"] > bl_is["pf"] and r_oos["pf"] > bl_oos["pf"] else "❌"
                print(f"{label:<30} {r_is['pf']:>6.3f} {r_is['ret']:>+7.2f}% {r_is['trades']:>5} {r_is['big5']:>5}"
                      f" {r_is['reentries']:>5} {r_is['re_pf']:>5.3f}"
                      f" {r_oos['pf']:>7.3f} {r_oos['ret']:>+8.2f}% {r_oos['trades']:>7} {r_oos['reentries']:>7}  {holds}")

    # Re-entry quality analysis on best config
    print(f"\n{'='*80}")
    print(f"  RE-ENTRY TRADE QUALITY (w20_cd5_re2)")
    print(f"{'='*80}")
    r = run(df_is, reentry_window=20, reentry_cooldown=5, max_reentries=2)
    print(f"  Total trades: {r['trades']}, Re-entries: {r['reentries']}")
    print(f"  Overall PF: {r['pf']:.3f}")
    print(f"  Re-entry PF: {r['re_pf']:.3f}")
    print(f"  Re-entry avg R: {r['re_avg_r']:+.3f}")
    print(f"  Re-entry trades are {'PROFITABLE' if r['re_pf'] > 1.0 else 'NOT profitable'}")


if __name__ == "__main__":
    main()
