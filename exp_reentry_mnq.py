"""
RE-ENTRY after BE exit — MNQ 3min framework.

Hypothesis: When a trade reaches BE trigger (0.25R), then gets stopped at BE (0.15R),
the trend is still valid. Re-entering on the NEXT touch of EMA20 within N bars
captures the continuation move that the original trade missed.

Logic:
  - After a BE exit, set a "re-entry window" of N bars
  - During this window, if price touches EMA20 again (same trend), re-enter
  - Re-entry uses the NEW touch bar's stop (fresh risk)
  - Same exit management applies (gate, BE, chandelier)
  - Max 1 re-entry per original trade (no infinite loops)

Test params: window = [5, 10, 15, 20, 30] bars after BE exit
"""
from __future__ import annotations
import functools, datetime as dt
import numpy as np, pandas as pd

print = functools.partial(print, flush=True)

IS_PATH = "data/QQQ_1Min_Polygon_2y_clean.csv"
OOS_PATH = "data/QQQ_1Min_Barchart_2y_2022-2024_clean.csv"

S = {
    "tf_minutes": 3, "ema_fast": 20, "ema_slow": 50, "atr_period": 14,
    "touch_tol": 0.15, "touch_below_max": 0.5, "no_entry_after": dt.time(14, 0),
    "stop_buffer": 0.4, "gate_bars": 3, "gate_mfe": 0.2, "gate_tighten": -0.1,
    "be_trigger_r": 0.25, "be_stop_r": 0.15, "chand_bars": 25, "chand_mult": 0.3,
    "max_hold_bars": 180, "force_close_at": dt.time(15, 58),
    "daily_loss_r": 2.0, "skip_after_win": 1, "n_contracts": 2,
}

QQQ_TO_NQ = 40; MNQ_PER_POINT = 2.0
COMM_RT = 2.46; SPREAD = 0.50; STOP_SLIP = 1.00; BE_SLIP = 1.00


def resample(df, m):
    if m <= 1: return df
    return df.resample(f"{m}min").agg(
        {"Open": "first", "High": "max", "Low": "min",
         "Close": "last", "Volume": "sum"}).dropna()


def add_indicators(df, s):
    df = df.copy()
    df["ema_f"] = df["Close"].ewm(span=s["ema_fast"], adjust=False).mean()
    df["ema_s"] = df["Close"].ewm(span=s["ema_slow"], adjust=False).mean()
    tr = np.maximum(df["High"] - df["Low"],
                    np.maximum((df["High"] - df["Close"].shift(1)).abs(),
                               (df["Low"] - df["Close"].shift(1)).abs()))
    df["atr"] = tr.rolling(s["atr_period"]).mean()
    return df


def execute_trade(high, low, close, atr, times, entry_bar, entry, stop, trend, risk_qqq, s, n, tf):
    """Execute a single trade and return (trade_r, end_bar, exit_reason, net_pnl)."""
    nc = s["n_contracts"]
    max_hold = max(20, s["max_hold_bars"] // tf)
    chand_b = max(5, s["chand_bars"] // tf)
    gate_b = max(1, s["gate_bars"] // tf) if s["gate_bars"] > 0 else 0
    risk_mnq = risk_qqq * QQQ_TO_NQ * MNQ_PER_POINT * nc
    entry_cost = COMM_RT * nc / 2 + SPREAD

    runner_stop = stop; be_triggered = False; mfe = 0.0
    trade_r = 0.0; end_bar = entry_bar; exit_reason = "timeout"

    for k in range(1, max_hold + 1):
        bi = entry_bar + k
        if bi >= n: break
        h = high[bi]; l = low[bi]
        a = atr[bi] if not np.isnan(atr[bi]) else abs(entry - stop) / s["stop_buffer"]

        if trend == 1: mfe = max(mfe, (h - entry) / risk_qqq)
        else: mfe = max(mfe, (entry - l) / risk_qqq)

        if times[bi] >= s["force_close_at"]:
            trade_r = (close[bi] - entry) / risk_qqq * trend
            end_bar = bi; exit_reason = "close"; break

        if gate_b > 0 and k == gate_b and not be_triggered:
            if mfe < s["gate_mfe"]:
                ns = entry + s["gate_tighten"] * risk_qqq * trend
                if trend == 1: runner_stop = max(runner_stop, ns)
                else: runner_stop = min(runner_stop, ns)

        stopped = (trend == 1 and l <= runner_stop) or (trend == -1 and h >= runner_stop)
        if stopped:
            trade_r = (runner_stop - entry) / risk_qqq * trend
            end_bar = bi
            if be_triggered:
                be_ref = entry + s["be_stop_r"] * risk_qqq * trend
                exit_reason = "be" if abs(runner_stop - be_ref) < 0.05 * risk_qqq else "trail"
            else: exit_reason = "stop"
            break

        if not be_triggered and s["be_trigger_r"] > 0:
            tp = entry + s["be_trigger_r"] * risk_qqq * trend
            if (trend == 1 and h >= tp) or (trend == -1 and l <= tp):
                be_triggered = True
                bl = entry + s["be_stop_r"] * risk_qqq * trend
                if trend == 1: runner_stop = max(runner_stop, bl)
                else: runner_stop = min(runner_stop, bl)

        if be_triggered and k >= chand_b:
            sk = max(1, k - chand_b + 1)
            hv = [high[entry_bar + kk] for kk in range(sk, k) if entry_bar + kk < n]
            lv = [low[entry_bar + kk] for kk in range(sk, k) if entry_bar + kk < n]
            if hv and lv:
                if trend == 1: runner_stop = max(runner_stop, max(hv) - s["chand_mult"] * a)
                else: runner_stop = min(runner_stop, min(lv) + s["chand_mult"] * a)
    else:
        trade_r = (close[min(entry_bar + max_hold, n - 1)] - entry) / risk_qqq * trend
        end_bar = min(entry_bar + max_hold, n - 1)

    raw_pnl = trade_r * risk_mnq
    exit_comm = COMM_RT * nc / 2
    exit_slip = STOP_SLIP if exit_reason in ("stop", "trail") else 0
    be_slip_cost = BE_SLIP if exit_reason == "be" else 0
    net_pnl = raw_pnl - (entry_cost + exit_comm + exit_slip + be_slip_cost)

    return trade_r, end_bar, exit_reason, net_pnl, risk_mnq


def run(df_1min, s=S, reentry_window=0):
    """Run strategy with optional re-entry after BE."""
    df = resample(df_1min, s["tf_minutes"])
    df = add_indicators(df, s)
    high = df["High"].values; low = df["Low"].values; close = df["Close"].values
    ema_f = df["ema_f"].values; ema_s = df["ema_s"].values; atr_arr = df["atr"].values
    times = df.index.time; dates = df.index.date; n = len(df)
    tf = max(1, s["tf_minutes"])

    trades = []; bar = max(s["ema_slow"], s["atr_period"]) + 5
    daily_r_loss = 0.0; current_date = None; skip_count = 0
    cum_pnl = 0.0; peak_pnl = 0.0; max_dd = 0.0
    reentry_trades = 0; reentry_wins = 0; reentry_pnl = 0.0

    while bar < n - 200:
        a = atr_arr[bar]
        if np.isnan(a) or a <= 0 or np.isnan(ema_f[bar]) or np.isnan(ema_s[bar]):
            bar += 1; continue
        if times[bar] >= s["no_entry_after"]:
            bar += 1; continue
        d = dates[bar]
        if current_date != d:
            current_date = d; daily_r_loss = 0.0
        if daily_r_loss >= s["daily_loss_r"]:
            bar += 1; continue

        c = close[bar]
        if c > ema_f[bar] and ema_f[bar] > ema_s[bar]: trend = 1
        elif c < ema_f[bar] and ema_f[bar] < ema_s[bar]: trend = -1
        else: bar += 1; continue

        tol = a * s["touch_tol"]
        if trend == 1:
            touch = low[bar] <= ema_f[bar] + tol and low[bar] >= ema_f[bar] - a * s["touch_below_max"]
        else:
            touch = high[bar] >= ema_f[bar] - tol and high[bar] <= ema_f[bar] + a * s["touch_below_max"]
        if not touch: bar += 1; continue
        if skip_count > 0: skip_count -= 1; bar += 1; continue

        # Execute primary trade
        entry = close[bar]
        stop = low[bar] - s["stop_buffer"] * a if trend == 1 else high[bar] + s["stop_buffer"] * a
        risk_qqq = abs(entry - stop)
        if risk_qqq <= 0: bar += 1; continue

        trade_r, end_bar, exit_reason, net_pnl, risk_mnq = execute_trade(
            high, low, close, atr_arr, times, bar, entry, stop, trend, risk_qqq, s, n, tf)

        cum_pnl += net_pnl; peak_pnl = max(peak_pnl, cum_pnl)
        max_dd = max(max_dd, peak_pnl - cum_pnl)
        trades.append({"net_pnl": net_pnl, "raw_r": trade_r, "exit": exit_reason,
                        "risk_$": risk_mnq, "type": "primary"})
        if trade_r < 0: daily_r_loss += abs(trade_r)
        if trade_r > 0: skip_count = s.get("skip_after_win", 0)

        # ─── RE-ENTRY after BE exit ───
        if reentry_window > 0 and exit_reason == "be":
            re_start = end_bar + 1
            re_end = min(re_start + reentry_window, n - 200)
            found_reentry = False

            for rb in range(re_start, re_end):
                ra = atr_arr[rb]
                if np.isnan(ra) or ra <= 0 or np.isnan(ema_f[rb]) or np.isnan(ema_s[rb]):
                    continue
                if times[rb] >= s["no_entry_after"]:
                    continue
                if dates[rb] != current_date:
                    break  # don't re-enter on a different day

                # Check trend still valid
                rc = close[rb]
                if trend == 1 and not (rc > ema_f[rb] and ema_f[rb] > ema_s[rb]):
                    continue
                if trend == -1 and not (rc < ema_f[rb] and ema_f[rb] < ema_s[rb]):
                    continue

                # Check touch
                rtol = ra * s["touch_tol"]
                if trend == 1:
                    rtouch = low[rb] <= ema_f[rb] + rtol and low[rb] >= ema_f[rb] - ra * s["touch_below_max"]
                else:
                    rtouch = high[rb] >= ema_f[rb] - rtol and high[rb] <= ema_f[rb] + ra * s["touch_below_max"]

                if not rtouch:
                    continue

                # Daily loss check
                if daily_r_loss >= s["daily_loss_r"]:
                    break

                # Execute re-entry trade
                re_entry = close[rb]
                re_stop = low[rb] - s["stop_buffer"] * ra if trend == 1 else high[rb] + s["stop_buffer"] * ra
                re_risk = abs(re_entry - re_stop)
                if re_risk <= 0: continue

                re_r, re_end_bar, re_exit, re_net, re_risk_mnq = execute_trade(
                    high, low, close, atr_arr, times, rb, re_entry, re_stop, trend, re_risk, s, n, tf)

                cum_pnl += re_net; peak_pnl = max(peak_pnl, cum_pnl)
                max_dd = max(max_dd, peak_pnl - cum_pnl)
                trades.append({"net_pnl": re_net, "raw_r": re_r, "exit": re_exit,
                                "risk_$": re_risk_mnq, "type": "reentry"})
                if re_r < 0: daily_r_loss += abs(re_r)
                if re_r > 0: skip_count = s.get("skip_after_win", 0)

                reentry_trades += 1
                if re_r > 0: reentry_wins += 1
                reentry_pnl += re_net
                end_bar = max(end_bar, re_end_bar)
                found_reentry = True
                break  # max 1 re-entry

        bar = end_bar + 1

    # Stats
    tdf = pd.DataFrame(trades) if trades else pd.DataFrame()
    total = len(tdf)
    days = len(set(dates))
    if total == 0:
        return {"net_pf": 0, "daily_pnl": 0, "max_dd": 0, "trades": 0, "re": 0, "re_pnl": 0}
    gw = tdf.loc[tdf["net_pnl"] > 0, "net_pnl"].sum()
    gl = abs(tdf.loc[tdf["net_pnl"] <= 0, "net_pnl"].sum())
    exits = tdf["exit"].value_counts().to_dict()

    return {
        "net_pf": round(gw / gl if gl > 0 else 0, 3),
        "daily_pnl": round(cum_pnl / max(days, 1), 1),
        "max_dd": round(max_dd, 0),
        "total_pnl": round(cum_pnl, 0),
        "trades": total,
        "tpd": round(total / max(days, 1), 1),
        "big5": int((tdf["raw_r"] >= 5).sum()),
        "exits": exits,
        "re": reentry_trades,
        "re_win": reentry_wins,
        "re_pnl": round(reentry_pnl, 0),
    }


if __name__ == "__main__":
    df_is = pd.read_csv(IS_PATH, index_col="timestamp", parse_dates=True)
    df_oos = pd.read_csv(OOS_PATH, index_col="timestamp", parse_dates=True)

    print("=" * 75)
    print("RE-ENTRY AFTER BE — MNQ 3min")
    print("=" * 75)

    # Baseline (no re-entry)
    ri = run(df_is, reentry_window=0)
    ro = run(df_oos, reentry_window=0)
    print(f"\n  BASE  IS: PF={ri['net_pf']:.3f} $/d={ri['daily_pnl']:+.1f} DD=${ri['max_dd']:.0f} "
          f"N={ri['trades']} 5R+={ri['big5']}")
    print(f"  BASE OOS: PF={ro['net_pf']:.3f} $/d={ro['daily_pnl']:+.1f} DD=${ro['max_dd']:.0f} "
          f"N={ro['trades']} 5R+={ro['big5']}")

    # Test different re-entry windows
    print(f"\n  {'Window':>8} {'IS_PF':>7} {'IS_$/d':>8} {'IS_DD':>7} {'IS_N':>6} "
          f"{'OOS_PF':>7} {'OOS_$/d':>8} {'OOS_DD':>7} {'OOS_N':>6} "
          f"{'RE_n':>5} {'RE_wr%':>7} {'RE_PnL':>8}")

    for window in [3, 5, 10, 15, 20, 30, 50]:
        ri = run(df_is, reentry_window=window)
        ro = run(df_oos, reentry_window=window)
        re_wr = round(ri['re_win'] / max(ri['re'], 1) * 100, 1)
        print(f"  {window:>8} {ri['net_pf']:>7.3f} {ri['daily_pnl']:>+8.1f} {ri['max_dd']:>7.0f} {ri['trades']:>6} "
              f"{ro['net_pf']:>7.3f} {ro['daily_pnl']:>+8.1f} {ro['max_dd']:>7.0f} {ro['trades']:>6} "
              f"{ri['re']:>5} {re_wr:>6.1f}% {ri['re_pnl']:>+8.0f}")

    # Best window detail
    print(f"\n--- Best window detail ---")
    for window in [10, 15, 20]:
        ri = run(df_is, reentry_window=window)
        ro = run(df_oos, reentry_window=window)
        print(f"\n  Window={window}")
        print(f"    IS:  PF={ri['net_pf']:.3f} $/d={ri['daily_pnl']:+.1f} DD=${ri['max_dd']:.0f} "
              f"N={ri['trades']} 5R+={ri['big5']} Exits={ri['exits']}")
        print(f"    OOS: PF={ro['net_pf']:.3f} $/d={ro['daily_pnl']:+.1f} DD=${ro['max_dd']:.0f} "
              f"N={ro['trades']} 5R+={ro['big5']} Exits={ro['exits']}")
        print(f"    RE stats: IS {ri['re']} trades, PnL=${ri['re_pnl']:+.0f} | "
              f"OOS {ro['re']} trades, PnL=${ro['re_pnl']:+.0f}")
