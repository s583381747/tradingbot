"""
MNQ Prop Firm Strategy — Touch Close EMA20 with MNQ cost model.

Outputs: Net PF, Daily $, MaxDD, trades, for IS and OOS.
Configurable via STRATEGY dict at top.
"""
from __future__ import annotations
import functools, datetime as dt, sys
import numpy as np, pandas as pd

print = functools.partial(print, flush=True)

IS_PATH = "data/QQQ_1Min_Polygon_2y_clean.csv"
OOS_PATH = "data/QQQ_1Min_Barchart_2y_2022-2024_clean.csv"

# ══════════════════════════════════════════════════════════════
# STRATEGY PARAMETERS (modify these for experiments)
# ══════════════════════════════════════════════════════════════

STRATEGY = {
    # Timeframe
    "tf_minutes": 1,          # resample to N-min bars (1=no resample)

    # Entry
    "ema_fast": 20,
    "ema_slow": 50,
    "atr_period": 14,
    "touch_tol": 0.15,
    "touch_below_max": 0.5,
    "no_entry_after": dt.time(14, 0),

    # Stop
    "stop_buffer": 0.3,       # ATR multiplier below touch low

    # Exit: Lock + BE
    "lock_rr": 0.1,           # lock R:R target (0 = no lock)
    "lock_pct": 0.05,         # fraction of position to lock

    # Exit: Chandelier
    "chand_bars": 40,
    "chand_mult": 0.5,

    # Exit: MFE gate
    "gate_bars": 3,           # check MFE after N bars (0 = off)
    "gate_mfe": 0.3,          # minimum MFE to pass gate
    "gate_tighten": -0.3,     # tighten stop to this R on fail

    # Position
    "max_hold_bars": 180,
    "force_close_at": dt.time(15, 58),
    "daily_loss_r": 2.5,
    "skip_after_win": 1,

    # MNQ sizing
    "n_contracts": 2,

    # DD control
    "dd_threshold_r": 0,      # if trailing DD > this R, reduce contracts (0=off)
    "dd_reduce_to": 1,        # reduce to N contracts when in DD
    "dd_cooldown": 0,         # min trades before restoring
    "dd_recovery_r": 0,       # DD must recover to < this R to restore
}

# ══════════════════════════════════════════════════════════════
# MNQ COST MODEL
# ══════════════════════════════════════════════════════════════

COMM_PER_CONTRACT_RT = 2.46
SPREAD_PER_TRADE = 0.50
STOP_SLIP = 0.50
BE_SLIP = 0.50
QQQ_TO_NQ = 40
MNQ_PER_POINT = 2.0


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════

def resample(df, minutes):
    if minutes <= 1:
        return df
    return df.resample(f"{minutes}min").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    ).dropna()


def add_indicators(df, s):
    df = df.copy()
    df["ema_f"] = df["Close"].ewm(span=s["ema_fast"], adjust=False).mean()
    df["ema_s"] = df["Close"].ewm(span=s["ema_slow"], adjust=False).mean()
    tr = np.maximum(df["High"] - df["Low"],
                     np.maximum((df["High"] - df["Close"].shift(1)).abs(),
                                (df["Low"] - df["Close"].shift(1)).abs()))
    df["atr"] = tr.rolling(s["atr_period"]).mean()
    return df


# ══════════════════════════════════════════════════════════════
# BACKTEST
# ══════════════════════════════════════════════════════════════

def run(df_1min, s=None):
    if s is None:
        s = STRATEGY
    df = resample(df_1min, s["tf_minutes"])
    df = add_indicators(df, s)

    high = df["High"].values; low = df["Low"].values; close = df["Close"].values
    ema_f = df["ema_f"].values; ema_s_arr = df["ema_s"].values; atr = df["atr"].values
    times = df.index.time; dates = df.index.date; n = len(df)

    adj = max(1, s["tf_minutes"])
    max_hold = max(20, s["max_hold_bars"] // adj)
    chand_b = max(5, s["chand_bars"] // adj)
    gate_b = max(1, s["gate_bars"] // adj) if s["gate_bars"] > 0 else 0

    base_contracts = s["n_contracts"]
    cum_pnl = 0.0; peak_pnl = 0.0; max_dd = 0.0
    trades = []; bar = max(s["ema_slow"], s["atr_period"]) + 5
    daily_r_loss = 0.0; current_date = None; skip_count = 0
    dd_reduced = False; dd_trade_count = 0

    while bar < n - max_hold - 5:
        a = atr[bar]
        if np.isnan(a) or a <= 0 or np.isnan(ema_f[bar]) or np.isnan(ema_s_arr[bar]):
            bar += 1; continue
        if times[bar] >= s["no_entry_after"]: bar += 1; continue
        d = dates[bar]
        if current_date != d: current_date = d; daily_r_loss = 0.0
        if daily_r_loss >= s["daily_loss_r"]: bar += 1; continue

        # Trend
        c = close[bar]
        if c > ema_f[bar] and ema_f[bar] > ema_s_arr[bar]:
            trend = 1
        elif c < ema_f[bar] and ema_f[bar] < ema_s_arr[bar]:
            trend = -1
        else:
            bar += 1; continue

        # Touch
        tol = a * s["touch_tol"]
        if trend == 1:
            touch = (low[bar] <= ema_f[bar] + tol) and (low[bar] >= ema_f[bar] - a * s["touch_below_max"])
        else:
            touch = (high[bar] >= ema_f[bar] - tol) and (high[bar] <= ema_f[bar] + a * s["touch_below_max"])
        if not touch: bar += 1; continue

        if skip_count > 0: skip_count -= 1; bar += 1; continue

        # DD control: determine contract count
        nc = base_contracts
        if s["dd_threshold_r"] > 0:
            # Compute current DD in R (approximate: use avg risk)
            dd_now = peak_pnl - cum_pnl
            avg_risk_approx = a * s["stop_buffer"] * QQQ_TO_NQ * MNQ_PER_POINT * base_contracts
            dd_r_approx = dd_now / avg_risk_approx if avg_risk_approx > 0 else 0
            if not dd_reduced and dd_r_approx > s["dd_threshold_r"]:
                dd_reduced = True; dd_trade_count = 0
            if dd_reduced:
                nc = s["dd_reduce_to"]
                dd_trade_count += 1
                if dd_r_approx < s["dd_recovery_r"] and dd_trade_count >= s["dd_cooldown"]:
                    dd_reduced = False; nc = base_contracts

        entry = close[bar]
        stop = low[bar] - s["stop_buffer"] * a if trend == 1 else high[bar] + s["stop_buffer"] * a
        risk_qqq = abs(entry - stop)
        if risk_qqq <= 0: bar += 1; continue

        risk_nq = risk_qqq * QQQ_TO_NQ
        risk_mnq = risk_nq * MNQ_PER_POINT * nc

        # Costs
        entry_comm = COMM_PER_CONTRACT_RT * nc / 2
        entry_cost = entry_comm + SPREAD_PER_TRADE

        entry_bar = bar; runner_stop = stop; lock_done = False; mfe = 0.0
        trade_r = 0.0; end_bar = bar; exit_reason = "timeout"

        for k in range(1, max_hold + 1):
            bi = entry_bar + k
            if bi >= n: break
            h = high[bi]; l = low[bi]; ca = atr[bi] if not np.isnan(atr[bi]) else a

            if trend == 1: mfe = max(mfe, (h - entry) / risk_qqq)
            else: mfe = max(mfe, (entry - l) / risk_qqq)

            if times[bi] >= s["force_close_at"]:
                trade_r = (close[bi] - entry) / risk_qqq * trend
                end_bar = bi; exit_reason = "close"; break

            # Gate
            if gate_b > 0 and k == gate_b and not lock_done and mfe < s["gate_mfe"]:
                ns = entry + s["gate_tighten"] * risk_qqq * trend
                if trend == 1: runner_stop = max(runner_stop, ns)
                else: runner_stop = min(runner_stop, ns)

            stopped = (trend == 1 and l <= runner_stop) or (trend == -1 and h >= runner_stop)
            if stopped:
                trade_r = (runner_stop - entry) / risk_qqq * trend
                end_bar = bi
                exit_reason = "be" if lock_done and abs(runner_stop - entry) < 0.02 else (
                    "trail" if lock_done else "stop")
                break

            if not lock_done and s["lock_rr"] > 0:
                tgt = entry + s["lock_rr"] * risk_qqq * trend
                if (trend == 1 and h >= tgt) or (trend == -1 and l <= tgt):
                    lock_done = True
                    if trend == 1: runner_stop = max(runner_stop, entry)
                    else: runner_stop = min(runner_stop, entry)

            if lock_done and k >= chand_b:
                sk = max(1, k - chand_b + 1)
                hv = [high[entry_bar + kk] for kk in range(sk, k) if entry_bar + kk < n]
                lv = [low[entry_bar + kk] for kk in range(sk, k) if entry_bar + kk < n]
                if hv and lv:
                    if trend == 1: runner_stop = max(runner_stop, max(hv) - s["chand_mult"] * ca)
                    else: runner_stop = min(runner_stop, min(lv) + s["chand_mult"] * ca)
        else:
            trade_r = (close[min(entry_bar + max_hold, n - 1)] - entry) / risk_qqq * trend
            end_bar = min(entry_bar + max_hold, n - 1)

        # P&L in dollars
        raw_pnl = trade_r * risk_mnq
        exit_comm = COMM_PER_CONTRACT_RT * nc / 2
        exit_slip = STOP_SLIP if exit_reason in ("stop", "trail") else 0
        be_slip = BE_SLIP if exit_reason == "be" else 0
        total_cost = entry_cost + exit_comm + exit_slip + be_slip
        net_pnl = raw_pnl - total_cost

        cum_pnl += net_pnl
        peak_pnl = max(peak_pnl, cum_pnl)
        dd = peak_pnl - cum_pnl
        max_dd = max(max_dd, dd)

        trades.append({"net_pnl": net_pnl, "raw_r": trade_r, "cost": total_cost,
                        "exit": exit_reason, "risk_$": risk_mnq, "nc": nc})
        if trade_r < 0: daily_r_loss += abs(trade_r)
        if trade_r > 0: skip_count = s.get("skip_after_win", 0)
        bar = end_bar + 1

    # Stats
    tdf = pd.DataFrame(trades) if trades else pd.DataFrame()
    total = len(tdf)
    days = len(set(dates))
    if total == 0:
        return {"net_pf": 0, "daily_pnl": 0, "max_dd": 0, "trades": 0, "tpd": 0,
                "total_pnl": 0, "cost_pct": 0, "big5": 0}
    gw = tdf.loc[tdf["net_pnl"] > 0, "net_pnl"].sum()
    gl = abs(tdf.loc[tdf["net_pnl"] <= 0, "net_pnl"].sum())
    net_pf = gw / gl if gl > 0 else 0
    big5 = int((tdf["raw_r"] >= 5).sum())

    return {
        "net_pf": round(net_pf, 3),
        "daily_pnl": round(cum_pnl / max(days, 1), 1),
        "max_dd": round(max_dd, 0),
        "total_pnl": round(cum_pnl, 0),
        "trades": total,
        "tpd": round(total / max(days, 1), 1),
        "cost_pct": round(tdf["cost"].mean() / tdf["risk_$"].mean() * 100, 1),
        "big5": big5,
    }


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    df_is = pd.read_csv(IS_PATH, index_col="timestamp", parse_dates=True)
    df_oos = pd.read_csv(OOS_PATH, index_col="timestamp", parse_dates=True)

    ri = run(df_is)
    ro = run(df_oos)

    print(f"IS:  NetPF={ri['net_pf']:.3f}  $/day={ri['daily_pnl']:+.1f}  MaxDD=${ri['max_dd']:.0f}"
          f"  Trades={ri['trades']}({ri['tpd']}/d)  5R+={ri['big5']}  Cost={ri['cost_pct']}%")
    print(f"OOS: NetPF={ro['net_pf']:.3f}  $/day={ro['daily_pnl']:+.1f}  MaxDD=${ro['max_dd']:.0f}"
          f"  Trades={ro['trades']}({ro['tpd']}/d)  5R+={ro['big5']}  Cost={ro['cost_pct']}%")

    # Key metrics for researcher
    print(f"\nPRIMARY: {ri['net_pf']:.6f}")
    dd_ok = "PASS" if ri["max_dd"] <= 2500 and ro["max_dd"] <= 2500 else "FAIL"
    oos_ok = "PASS" if ro["net_pf"] > 1.0 else "FAIL"
    print(f"DD_CHECK: {dd_ok} (IS=${ri['max_dd']:.0f} OOS=${ro['max_dd']:.0f})")
    print(f"OOS_CHECK: {oos_ok} (OOS PF={ro['net_pf']:.3f})")
