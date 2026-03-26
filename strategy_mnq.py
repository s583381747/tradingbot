"""
MNQ Prop Firm Strategy — Touch Close EMA20.

NQ-tuned v9 (2026-03-26):
  gate_tighten: -0.1 → 0.0 (BE on gate fail, not small loss — NQ noise too wide)
  gate_mfe: 0.20 → 0.25 (stricter gate threshold)
  be_trigger_r: 0.25 → 0.20 (earlier BE protection)
  skip_after_win: 1 → 2 (reduce overtrading)
  slippage: $0.50 → $1.00 (real Tradovate MNQ data)

Result on real NQ 4Y: PF=3.84, $94/day, MaxDD=$429
"""
from __future__ import annotations
import functools, datetime as dt
import numpy as np, pandas as pd

print = functools.partial(print, flush=True)

IS_PATH = "data/QQQ_1Min_Polygon_2y_clean.csv"
OOS_PATH = "data/QQQ_1Min_Barchart_2y_2022-2024_clean.csv"

# ══════════════════════════════════════════════════════════════
# STRATEGY PARAMETERS
# ══════════════════════════════════════════════════════════════

STRATEGY = {
    # Timeframe
    "tf_minutes": 3,

    # Entry
    "ema_fast": 20, "ema_slow": 50, "atr_period": 14,
    "touch_tol": 0.15, "touch_below_max": 0.5,
    "no_entry_after": dt.time(14, 0),

    # Stop
    "stop_buffer": 0.4,          # ATR mult below touch low

    # MFE gate: move to BE if price doesn't move fast enough
    "gate_bars": 3,              # check after N bars (0=off)
    "gate_mfe": 0.25,            # minimum MFE in R to pass (NQ-tuned: 0.25 vs 0.20)
    "gate_tighten": 0.0,         # move stop to entry + 0R on fail = breakeven (NQ-tuned: 0.0 vs -0.1)

    # BE mechanism: NO partial exit. Just move the stop order.
    # When price reaches +be_trigger_r, move stop to entry + be_stop_r.
    # be_stop_r MUST be < be_trigger_r to leave room.
    "be_trigger_r": 0.20,        # price must reach +0.20R to trigger (NQ-tuned: 0.20 vs 0.25)
    "be_stop_r": 0.15,           # stop moves to entry + 0.15R (room = 0.05R)

    # Chandelier trail (activates after BE trigger)
    "chand_bars": 25,            # lookback in ORIGINAL timeframe bars
    "chand_mult": 0.3,

    # Position management
    "max_hold_bars": 180,
    "force_close_at": dt.time(15, 58),
    "daily_loss_r": 2.0,
    "skip_after_win": 2,          # NQ-tuned: skip 2 after win (was 1)

    # DCP filter (directional close position)
    "use_dcp": False,            # enable DCP filter
    "dcp_min": 0.5,              # touch bar DCP must be >= this
    "dcp_slope_min": 0.0,        # 5-bar DCP slope must be > this

    # MNQ contracts (whole number, no partial)
    "n_contracts": 2,

    # DD control
    "dd_reduce_dollar": 0,       # reduce to dd_reduce_to when DD > this (0=off)
    "dd_reduce_to": 1,
    "dd_cooldown": 20,
    "dd_recovery_dollar": 500,
}

# ══════════════════════════════════════════════════════════════
# MNQ COST MODEL
# ══════════════════════════════════════════════════════════════

COMM_PER_CONTRACT_RT = 2.46
SPREAD_PER_TRADE = 0.50
STOP_SLIP = 1.00              # real: Tradovate MNQ mean 1.94 ticks = $0.97 ≈ $1.00
BE_SLIP = 1.00                # conservative estimate
QQQ_TO_NQ = 40
MNQ_PER_POINT = 2.0


def resample(df, minutes):
    if minutes <= 1:
        return df
    return df.resample(f"{minutes}min").agg(
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


def run(df_1min, s=None):
    if s is None:
        s = STRATEGY
    df = resample(df_1min, s["tf_minutes"])
    df = add_indicators(df, s)

    high = df["High"].values; low = df["Low"].values; close = df["Close"].values
    ema_f = df["ema_f"].values; ema_s_arr = df["ema_s"].values; atr = df["atr"].values
    times = df.index.time; dates = df.index.date; n = len(df)

    # Precompute DCP (raw, not directional — direction applied at entry)
    dcp_raw = np.zeros(n)
    for i in range(n):
        rng = high[i] - low[i]
        dcp_raw[i] = (close[i] - low[i]) / rng if rng > 0 else 0.5
    dcp_slope = np.zeros(n)
    for i in range(5, n):
        dcp_slope[i] = (dcp_raw[i] - dcp_raw[i - 5]) / 5

    tf = max(1, s["tf_minutes"])
    max_hold = max(20, s["max_hold_bars"] // tf)
    chand_b = max(5, s["chand_bars"] // tf)
    gate_b = max(1, s["gate_bars"] // tf) if s["gate_bars"] > 0 else 0

    nc = s["n_contracts"]
    cum_pnl = 0.0; peak_pnl = 0.0; max_dd = 0.0
    trades = []; bar = max(s["ema_slow"], s["atr_period"]) + 5
    daily_r_loss = 0.0; current_date = None; skip_count = 0
    dd_reduced = False; dd_trade_count = 0

    while bar < n - max_hold - 5:
        a = atr[bar]
        if np.isnan(a) or a <= 0 or np.isnan(ema_f[bar]) or np.isnan(ema_s_arr[bar]):
            bar += 1; continue
        if times[bar] >= s["no_entry_after"]:
            bar += 1; continue
        d = dates[bar]
        if current_date != d:
            current_date = d; daily_r_loss = 0.0
        if daily_r_loss >= s["daily_loss_r"]:
            bar += 1; continue

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
            touch = low[bar] <= ema_f[bar] + tol and low[bar] >= ema_f[bar] - a * s["touch_below_max"]
        else:
            touch = high[bar] >= ema_f[bar] - tol and high[bar] <= ema_f[bar] + a * s["touch_below_max"]
        if not touch:
            bar += 1; continue
        if skip_count > 0:
            skip_count -= 1; bar += 1; continue

        # DCP filter
        if s.get("use_dcp", False):
            bar_rng = high[bar] - low[bar]
            if bar_rng > 0:
                dcp_val = (close[bar] - low[bar]) / bar_rng if trend == 1 else (high[bar] - close[bar]) / bar_rng
            else:
                dcp_val = 0.5
            if dcp_val < s.get("dcp_min", 0.5):
                bar += 1; continue
            # DCP slope: for longs use raw slope, for shorts invert
            slope_val = dcp_slope[bar] if trend == 1 else -dcp_slope[bar]
            if slope_val <= s.get("dcp_slope_min", 0.0):
                bar += 1; continue

        # DD control
        active_nc = nc
        dd_now = peak_pnl - cum_pnl
        if s["dd_reduce_dollar"] > 0:
            if not dd_reduced and dd_now > s["dd_reduce_dollar"]:
                dd_reduced = True; dd_trade_count = 0
            if dd_reduced:
                active_nc = s["dd_reduce_to"]
                dd_trade_count += 1
                if dd_now < s["dd_recovery_dollar"] and dd_trade_count >= s["dd_cooldown"]:
                    dd_reduced = False; active_nc = nc

        entry = close[bar]
        stop = low[bar] - s["stop_buffer"] * a if trend == 1 else high[bar] + s["stop_buffer"] * a
        risk_qqq = abs(entry - stop)
        if risk_qqq <= 0:
            bar += 1; continue

        risk_mnq = risk_qqq * QQQ_TO_NQ * MNQ_PER_POINT * active_nc

        # Entry cost
        entry_cost = COMM_PER_CONTRACT_RT * active_nc / 2 + SPREAD_PER_TRADE

        # ─── Trade execution (no partial exit, whole position) ───
        entry_bar = bar
        runner_stop = stop
        be_triggered = False    # has price reached be_trigger_r?
        mfe = 0.0
        trade_r = 0.0
        end_bar = bar
        exit_reason = "timeout"

        for k in range(1, max_hold + 1):
            bi = entry_bar + k
            if bi >= n:
                break
            h = high[bi]; l = low[bi]
            ca = atr[bi] if not np.isnan(atr[bi]) else a

            # Track MFE
            if trend == 1:
                mfe = max(mfe, (h - entry) / risk_qqq)
            else:
                mfe = max(mfe, (entry - l) / risk_qqq)

            # Force close
            if times[bi] >= s["force_close_at"]:
                trade_r = (close[bi] - entry) / risk_qqq * trend
                end_bar = bi; exit_reason = "close"; break

            # MFE gate: tighten stop if price didn't move
            if gate_b > 0 and k == gate_b and not be_triggered:
                if mfe < s["gate_mfe"]:
                    ns = entry + s["gate_tighten"] * risk_qqq * trend
                    if trend == 1:
                        runner_stop = max(runner_stop, ns)
                    else:
                        runner_stop = min(runner_stop, ns)

            # Stop check
            stopped = (trend == 1 and l <= runner_stop) or (trend == -1 and h >= runner_stop)
            if stopped:
                trade_r = (runner_stop - entry) / risk_qqq * trend
                end_bar = bi
                if be_triggered:
                    be_ref = entry + s["be_stop_r"] * risk_qqq * trend
                    exit_reason = "be" if abs(runner_stop - be_ref) < 0.05 * risk_qqq else "trail"
                else:
                    exit_reason = "stop"
                break

            # BE trigger: price reached threshold → move stop (NO partial exit)
            if not be_triggered and s["be_trigger_r"] > 0:
                trigger_price = entry + s["be_trigger_r"] * risk_qqq * trend
                if (trend == 1 and h >= trigger_price) or (trend == -1 and l <= trigger_price):
                    be_triggered = True
                    be_level = entry + s["be_stop_r"] * risk_qqq * trend
                    if trend == 1:
                        runner_stop = max(runner_stop, be_level)
                    else:
                        runner_stop = min(runner_stop, be_level)

            # Chandelier trail (only after BE triggered)
            if be_triggered and k >= chand_b:
                sk = max(1, k - chand_b + 1)
                hv = [high[entry_bar + kk] for kk in range(sk, k) if entry_bar + kk < n]
                lv = [low[entry_bar + kk] for kk in range(sk, k) if entry_bar + kk < n]
                if hv and lv:
                    if trend == 1:
                        runner_stop = max(runner_stop, max(hv) - s["chand_mult"] * ca)
                    else:
                        runner_stop = min(runner_stop, min(lv) + s["chand_mult"] * ca)
        else:
            trade_r = (close[min(entry_bar + max_hold, n - 1)] - entry) / risk_qqq * trend
            end_bar = min(entry_bar + max_hold, n - 1)

        # P&L
        raw_pnl = trade_r * risk_mnq
        exit_comm = COMM_PER_CONTRACT_RT * active_nc / 2
        exit_slip = STOP_SLIP if exit_reason in ("stop", "trail") else 0
        be_slip = BE_SLIP if exit_reason == "be" else 0
        total_cost = entry_cost + exit_comm + exit_slip + be_slip
        net_pnl = raw_pnl - total_cost

        cum_pnl += net_pnl
        peak_pnl = max(peak_pnl, cum_pnl)
        dd = peak_pnl - cum_pnl
        max_dd = max(max_dd, dd)

        trades.append({"net_pnl": net_pnl, "raw_r": trade_r, "cost": total_cost,
                        "exit": exit_reason, "risk_$": risk_mnq, "nc": active_nc})
        if trade_r < 0:
            daily_r_loss += abs(trade_r)
        if trade_r > 0:
            skip_count = s.get("skip_after_win", 0)
        bar = end_bar + 1

    # Stats
    tdf = pd.DataFrame(trades) if trades else pd.DataFrame()
    total = len(tdf)
    days = len(set(dates))
    if total == 0:
        return {"net_pf": 0, "daily_pnl": 0, "max_dd": 0, "trades": 0,
                "tpd": 0, "total_pnl": 0, "cost_pct": 0, "big5": 0}
    gw = tdf.loc[tdf["net_pnl"] > 0, "net_pnl"].sum()
    gl = abs(tdf.loc[tdf["net_pnl"] <= 0, "net_pnl"].sum())
    net_pf = gw / gl if gl > 0 else 0

    # Exit breakdown
    exits = tdf["exit"].value_counts().to_dict()

    return {
        "net_pf": round(net_pf, 3),
        "daily_pnl": round(cum_pnl / max(days, 1), 1),
        "max_dd": round(max_dd, 0),
        "total_pnl": round(cum_pnl, 0),
        "trades": total,
        "tpd": round(total / max(days, 1), 1),
        "cost_pct": round(tdf["cost"].mean() / tdf["risk_$"].mean() * 100, 1),
        "big5": int((tdf["raw_r"] >= 5).sum()),
        "exits": exits,
    }


if __name__ == "__main__":
    df_is = pd.read_csv(IS_PATH, index_col="timestamp", parse_dates=True)
    df_oos = pd.read_csv(OOS_PATH, index_col="timestamp", parse_dates=True)

    ri = run(df_is)
    ro = run(df_oos)

    print(f"IS:  NetPF={ri['net_pf']:.3f}  $/day={ri['daily_pnl']:+.1f}  MaxDD=${ri['max_dd']:.0f}"
          f"  Trades={ri['trades']}({ri['tpd']}/d)  5R+={ri['big5']}  Cost={ri['cost_pct']}%")
    print(f"     Exits: {ri['exits']}")
    print(f"OOS: NetPF={ro['net_pf']:.3f}  $/day={ro['daily_pnl']:+.1f}  MaxDD=${ro['max_dd']:.0f}"
          f"  Trades={ro['trades']}({ro['tpd']}/d)  5R+={ro['big5']}  Cost={ro['cost_pct']}%")
    print(f"     Exits: {ro['exits']}")

    dd_ok = "PASS" if ri["max_dd"] <= 2500 and ro["max_dd"] <= 2500 else "FAIL"
    oos_ok = "PASS" if ro["net_pf"] > 1.0 else "FAIL"
    print(f"\nDD_CHECK: {dd_ok} (IS=${ri['max_dd']:.0f} OOS=${ro['max_dd']:.0f})")
    print(f"OOS_CHECK: {oos_ok} (OOS PF={ro['net_pf']:.3f})")
