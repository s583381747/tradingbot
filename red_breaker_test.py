"""
RED BREAKER — Destruction Test for EMA20 Touch NQ Strategy (v11, 10min, gate=0.0)

Attack vectors:
1. Worst-case scenario construction (worst days, worst DD, worst streak)
2. Regime stress tests (per half-year)
3. Cost sensitivity bomb (double costs, triple slippage, cost bomb threshold)
4. Execution delay attack (1-bar entry delay)
5. Random baseline test (1000 permutations)
6. Starting-day sensitivity (what if you start on the worst day?)

All costs PER CONTRACT. Gap-through stop fill.
IS = 2022-2023, OOS = 2024-2026.
"""
from __future__ import annotations
import functools, datetime as dt, math, time, random, os, json
import numpy as np, pandas as pd

print = functools.partial(print, flush=True)

NQ_PATH = "data/barchart_nq/NQ_1min_continuous_RTH.csv"

# Strategy params (v11: 10min, gate=0.0)
S = {
    "tf_minutes": 10,
    "ema_fast": 20, "ema_slow": 50, "atr_period": 14,
    "touch_tol": 0.15, "touch_below_max": 0.5,
    "no_entry_after": dt.time(14, 0),
    "stop_buffer": 0.4,
    "gate_bars": 3, "gate_mfe": 0.2, "gate_tighten": 0.0,
    "be_trigger_r": 0.25, "be_stop_r": 0.15,
    "chand_bars": 25, "chand_mult": 0.3,
    "max_hold_bars": 180,
    "force_close_at": dt.time(15, 58),
    "daily_loss_r": 2.0,
    "skip_after_win": 1,
    "n_contracts": 2,
}

# MNQ cost model -- all costs PER CONTRACT
MNQ_PER_POINT = 2.0
COMM_PER_CONTRACT_RT = 2.46
SPREAD_PER_CONTRACT = 0.50
STOP_SLIP_PER_CONTRACT = 1.00
BE_SLIP_PER_CONTRACT = 1.00
STARTING_EQUITY = 50_000


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


def run_strategy(df_1min, s=None, cost_mult=1.0, slip_mult=1.0, entry_delay=0):
    """
    Run strategy on NQ data with configurable cost/slippage multipliers and entry delay.
    Returns (trades_df, daily_pnl_dict).
    """
    if s is None:
        s = S
    df = resample(df_1min, s["tf_minutes"])
    df = add_indicators(df, s)

    high = df["High"].values; low = df["Low"].values; close = df["Close"].values
    opn = df["Open"].values
    ema_f = df["ema_f"].values; ema_s = df["ema_s"].values; atr = df["atr"].values
    times = df.index.time; dates = df.index.date; n = len(df)

    tf = max(1, s["tf_minutes"])
    max_hold = max(20, s["max_hold_bars"] // tf)
    chand_b = max(5, s["chand_bars"] // tf)
    gate_b = max(1, s["gate_bars"] // tf) if s["gate_bars"] > 0 else 0
    nc = s["n_contracts"]

    # Cost model with multipliers
    comm_rt = COMM_PER_CONTRACT_RT * cost_mult
    spread = SPREAD_PER_CONTRACT * cost_mult
    stop_slip = STOP_SLIP_PER_CONTRACT * slip_mult
    be_slip = BE_SLIP_PER_CONTRACT * slip_mult

    equity = float(STARTING_EQUITY)
    peak_eq = equity
    cum_pnl = 0.0; peak_pnl = 0.0; max_dd = 0.0
    trades = []; daily_pnl = {}
    bar = max(s["ema_slow"], s["atr_period"]) + 5
    daily_r_loss = 0.0; current_date = None; skip_count = 0

    while bar < n - max_hold - 5:
        a = atr[bar]
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
        if c > ema_f[bar] and ema_f[bar] > ema_s[bar]:
            trend = 1
        elif c < ema_f[bar] and ema_f[bar] < ema_s[bar]:
            trend = -1
        else:
            bar += 1; continue

        tol = a * s["touch_tol"]
        if trend == 1:
            touch = low[bar] <= ema_f[bar] + tol and low[bar] >= ema_f[bar] - a * s["touch_below_max"]
        else:
            touch = high[bar] >= ema_f[bar] - tol and high[bar] <= ema_f[bar] + a * s["touch_below_max"]
        if not touch:
            bar += 1; continue
        if skip_count > 0:
            skip_count -= 1; bar += 1; continue

        # Entry delay: shift entry bar forward
        actual_entry_bar = bar + entry_delay
        if actual_entry_bar >= n - max_hold - 5:
            bar += 1; continue

        entry = close[actual_entry_bar] if entry_delay > 0 else close[bar]
        if entry_delay > 0:
            # Use original signal bar for stop, but entry at delayed bar
            stop = low[bar] - s["stop_buffer"] * a if trend == 1 else high[bar] + s["stop_buffer"] * a
        else:
            stop = low[bar] - s["stop_buffer"] * a if trend == 1 else high[bar] + s["stop_buffer"] * a

        risk_price = abs(entry - stop)
        if risk_price <= 0:
            bar += 1; continue

        risk_mnq = risk_price * MNQ_PER_POINT * nc
        entry_cost = comm_rt * nc / 2 + spread * nc

        entry_bar = actual_entry_bar; runner_stop = stop; be_triggered = False
        mfe = 0.0; trade_r = 0.0; end_bar = entry_bar; exit_reason = "timeout"

        for k in range(1, max_hold + 1):
            bi = entry_bar + k
            if bi >= n: break
            h = high[bi]; l = low[bi]
            ca = atr[bi] if not np.isnan(atr[bi]) else a

            if trend == 1: mfe = max(mfe, (h - entry) / risk_price)
            else: mfe = max(mfe, (entry - l) / risk_price)

            if times[bi] >= s["force_close_at"]:
                trade_r = (close[bi] - entry) / risk_price * trend
                end_bar = bi; exit_reason = "close"; break

            if gate_b > 0 and k == gate_b and not be_triggered:
                if mfe < s["gate_mfe"]:
                    ns = entry + s["gate_tighten"] * risk_price * trend
                    if trend == 1: runner_stop = max(runner_stop, ns)
                    else: runner_stop = min(runner_stop, ns)

            stopped = (trend == 1 and l <= runner_stop) or (trend == -1 and h >= runner_stop)
            if stopped:
                if trend == 1:
                    fill_price = min(runner_stop, opn[bi]) if opn[bi] < runner_stop else runner_stop
                else:
                    fill_price = max(runner_stop, opn[bi]) if opn[bi] > runner_stop else runner_stop
                trade_r = (fill_price - entry) / risk_price * trend
                end_bar = bi
                if be_triggered:
                    be_ref = entry + s["be_stop_r"] * risk_price * trend
                    exit_reason = "be" if abs(runner_stop - be_ref) < 0.05 * risk_price else "trail"
                else: exit_reason = "stop"
                break

            if not be_triggered and s["be_trigger_r"] > 0:
                tp = entry + s["be_trigger_r"] * risk_price * trend
                if (trend == 1 and h >= tp) or (trend == -1 and l <= tp):
                    be_triggered = True
                    bl = entry + s["be_stop_r"] * risk_price * trend
                    if trend == 1: runner_stop = max(runner_stop, bl)
                    else: runner_stop = min(runner_stop, bl)

            if be_triggered and k >= chand_b:
                sk = max(1, k - chand_b + 1)
                hv = [high[entry_bar + kk] for kk in range(sk, k) if entry_bar + kk < n]
                lv = [low[entry_bar + kk] for kk in range(sk, k) if entry_bar + kk < n]
                if hv and lv:
                    if trend == 1: runner_stop = max(runner_stop, max(hv) - s["chand_mult"] * ca)
                    else: runner_stop = min(runner_stop, min(lv) + s["chand_mult"] * ca)
        else:
            trade_r = (close[min(entry_bar + max_hold, n - 1)] - entry) / risk_price * trend
            end_bar = min(entry_bar + max_hold, n - 1)

        raw_pnl = trade_r * risk_mnq
        exit_comm = comm_rt * nc / 2
        exit_slip = stop_slip * nc if exit_reason in ("stop", "trail") else 0
        be_s = be_slip * nc if exit_reason == "be" else 0
        total_cost = entry_cost + exit_comm + exit_slip + be_s
        net_pnl = raw_pnl - total_cost

        cum_pnl += net_pnl
        equity += net_pnl
        peak_eq = max(peak_eq, equity)
        peak_pnl = max(peak_pnl, cum_pnl)
        max_dd = max(max_dd, peak_pnl - cum_pnl)

        d_str = str(d)
        daily_pnl[d_str] = daily_pnl.get(d_str, 0) + net_pnl

        trades.append({
            "pnl": net_pnl, "r": trade_r, "exit": exit_reason,
            "cost": total_cost, "risk": risk_mnq,
            "date": d_str, "mfe": mfe, "equity": equity,
            "entry_bar": entry_bar, "end_bar": end_bar,
        })
        if trade_r < 0: daily_r_loss += abs(trade_r)
        if trade_r > 0: skip_count = s.get("skip_after_win", 0)
        bar = end_bar + 1

    tdf = pd.DataFrame(trades) if trades else pd.DataFrame()
    return tdf, daily_pnl


def compute_pf(tdf):
    if len(tdf) == 0: return 0
    gw = tdf.loc[tdf["pnl"] > 0, "pnl"].sum()
    gl = abs(tdf.loc[tdf["pnl"] <= 0, "pnl"].sum())
    return round(gw / gl, 3) if gl > 0 else 0


def compute_full_stats(tdf, daily_pnl):
    if len(tdf) == 0:
        return {"pf": 0, "n": 0, "pnl": 0, "dd": 0}
    gw = tdf.loc[tdf["pnl"] > 0, "pnl"].sum()
    gl = abs(tdf.loc[tdf["pnl"] <= 0, "pnl"].sum())
    cum = tdf["pnl"].cumsum()
    dd = (cum.cummax() - cum).max()
    pf = gw / gl if gl > 0 else 0
    wr = (tdf["r"] > 0).mean() * 100

    days_arr = np.array(list(daily_pnl.values()))
    n_days = len(days_arr)
    years = n_days / 252
    sharpe = (days_arr.mean() / days_arr.std()) * 252**0.5 if days_arr.std() > 0 else 0
    apr = cum.iloc[-1] / STARTING_EQUITY / max(years, 0.1) * 100

    # Max consecutive losses
    is_loss = (tdf["r"] <= 0).values
    mcl = cur = 0
    for x in is_loss:
        if x: cur += 1; mcl = max(mcl, cur)
        else: cur = 0

    return {
        "pf": round(pf, 3), "wr": round(wr, 1), "dd": round(dd, 0),
        "pnl": round(cum.iloc[-1], 0), "n": len(tdf),
        "sharpe": round(sharpe, 2), "apr": round(apr, 1),
        "mcl": mcl,
        "dpnl": round(cum.iloc[-1] / max(n_days, 1), 1),
        "b5": int((tdf["r"] >= 5).sum()),
        "cost_pct": round(tdf["cost"].mean() / tdf["risk"].mean() * 100, 1) if tdf["risk"].mean() > 0 else 0,
    }


def run_random_strategy(df_1min, s, seed, n_trades_target, mode="full_exit"):
    """
    Random baseline: same number of trades, random entries.

    Modes:
    - "full_exit": Same exit system (BE, chandelier, gate) -- tests entry edge
    - "simple_exit": Fixed 2.5 ATR stop, no trail -- tests if market has inherent bias
    - "shuffled": Shuffle the strategy's actual trade PnLs -- tests if ordering matters
    """
    rng = random.Random(seed)
    df = resample(df_1min, s["tf_minutes"])
    df = add_indicators(df, s)

    high = df["High"].values; low = df["Low"].values; close = df["Close"].values
    opn = df["Open"].values
    ema_f = df["ema_f"].values; ema_s = df["ema_s"].values; atr = df["atr"].values
    times = df.index.time; dates = df.index.date; n = len(df)

    tf = max(1, s["tf_minutes"])
    max_hold = max(20, s["max_hold_bars"] // tf)
    chand_b = max(5, s["chand_bars"] // tf)
    gate_b = max(1, s["gate_bars"] // tf) if s["gate_bars"] > 0 else 0
    nc = s["n_contracts"]

    min_bar = max(s["ema_slow"], s["atr_period"]) + 5

    # Generate random entry bars within valid range
    valid_bars = []
    for i in range(min_bar, n - max_hold - 5):
        a = atr[i]
        if np.isnan(a) or a <= 0: continue
        if times[i] >= s["no_entry_after"]: continue
        if times[i] >= s["force_close_at"]: continue
        valid_bars.append(i)

    if len(valid_bars) < n_trades_target:
        n_trades_target = len(valid_bars) // 2

    entry_bars = sorted(rng.sample(valid_bars, min(n_trades_target * 3, len(valid_bars))))

    # Remove bars too close together
    filtered = []
    last = -999
    for b in entry_bars:
        if b - last > max_hold // 2:
            filtered.append(b)
            last = b
    filtered = filtered[:n_trades_target]

    trades = []
    for bar in filtered:
        a = atr[bar]
        trend = 1 if rng.random() < 0.5 else -1
        entry = close[bar]

        # Use IDENTICAL stop calculation to real strategy
        if trend == 1:
            stop = low[bar] - s["stop_buffer"] * a
        else:
            stop = high[bar] + s["stop_buffer"] * a
        risk_price = abs(entry - stop)
        if risk_price <= 0: continue
        risk_mnq = risk_price * MNQ_PER_POINT * nc
        entry_cost = COMM_PER_CONTRACT_RT * nc / 2 + SPREAD_PER_CONTRACT * nc

        if mode == "simple_exit":
            # Simple exit: fixed stop at entry - 2.5*ATR, exit at max_hold bars
            trade_r = 0.0; exit_reason = "timeout"
            for k in range(1, max_hold + 1):
                bi = bar + k
                if bi >= n: break
                h = high[bi]; l = low[bi]
                if times[bi] >= s["force_close_at"]:
                    trade_r = (close[bi] - entry) / risk_price * trend
                    exit_reason = "close"; break
                stopped = (trend == 1 and l <= stop) or (trend == -1 and h >= stop)
                if stopped:
                    fill_price = stop
                    trade_r = (fill_price - entry) / risk_price * trend
                    exit_reason = "stop"; break
            else:
                trade_r = (close[min(bar + max_hold, n - 1)] - entry) / risk_price * trend

            raw_pnl = trade_r * risk_mnq
            exit_comm = COMM_PER_CONTRACT_RT * nc / 2
            exit_slip = STOP_SLIP_PER_CONTRACT * nc if exit_reason == "stop" else 0
            total_cost = entry_cost + exit_comm + exit_slip
            net_pnl = raw_pnl - total_cost
        else:
            # Full exit system (same as real strategy)
            runner_stop = stop; be_triggered = False; mfe = 0.0
            trade_r = 0.0; end_bar = bar; exit_reason = "timeout"

            for k in range(1, max_hold + 1):
                bi = bar + k
                if bi >= n: break
                h = high[bi]; l = low[bi]
                ca = atr[bi] if not np.isnan(atr[bi]) else a

                if trend == 1: mfe = max(mfe, (h - entry) / risk_price)
                else: mfe = max(mfe, (entry - l) / risk_price)

                if times[bi] >= s["force_close_at"]:
                    trade_r = (close[bi] - entry) / risk_price * trend
                    end_bar = bi; exit_reason = "close"; break

                if gate_b > 0 and k == gate_b and not be_triggered:
                    if mfe < s["gate_mfe"]:
                        ns = entry + s["gate_tighten"] * risk_price * trend
                        if trend == 1: runner_stop = max(runner_stop, ns)
                        else: runner_stop = min(runner_stop, ns)

                stopped = (trend == 1 and l <= runner_stop) or (trend == -1 and h >= runner_stop)
                if stopped:
                    if trend == 1:
                        fill_price = min(runner_stop, opn[bi]) if opn[bi] < runner_stop else runner_stop
                    else:
                        fill_price = max(runner_stop, opn[bi]) if opn[bi] > runner_stop else runner_stop
                    trade_r = (fill_price - entry) / risk_price * trend
                    end_bar = bi
                    if be_triggered:
                        be_ref = entry + s["be_stop_r"] * risk_price * trend
                        exit_reason = "be" if abs(runner_stop - be_ref) < 0.05 * risk_price else "trail"
                    else: exit_reason = "stop"
                    break

                if not be_triggered and s["be_trigger_r"] > 0:
                    tp = entry + s["be_trigger_r"] * risk_price * trend
                    if (trend == 1 and h >= tp) or (trend == -1 and l <= tp):
                        be_triggered = True
                        bl = entry + s["be_stop_r"] * risk_price * trend
                        if trend == 1: runner_stop = max(runner_stop, bl)
                        else: runner_stop = min(runner_stop, bl)

                if be_triggered and k >= chand_b:
                    sk = max(1, k - chand_b + 1)
                    hv = [high[bar + kk] for kk in range(sk, k) if bar + kk < n]
                    lv = [low[bar + kk] for kk in range(sk, k) if bar + kk < n]
                    if hv and lv:
                        if trend == 1: runner_stop = max(runner_stop, max(hv) - s["chand_mult"] * ca)
                        else: runner_stop = min(runner_stop, min(lv) + s["chand_mult"] * ca)
            else:
                trade_r = (close[min(bar + max_hold, n - 1)] - entry) / risk_price * trend
                end_bar = min(bar + max_hold, n - 1)

            raw_pnl = trade_r * risk_mnq
            exit_comm = COMM_PER_CONTRACT_RT * nc / 2
            exit_slip = STOP_SLIP_PER_CONTRACT * nc if exit_reason in ("stop", "trail") else 0
            be_s = BE_SLIP_PER_CONTRACT * nc if exit_reason == "be" else 0
            total_cost = entry_cost + exit_comm + exit_slip + be_s
            net_pnl = raw_pnl - total_cost

        d_str = str(dates[bar])
        trades.append({"pnl": net_pnl, "r": trade_r, "exit": exit_reason,
                        "cost": total_cost, "risk": risk_mnq, "date": d_str})

    tdf = pd.DataFrame(trades) if trades else pd.DataFrame()
    return tdf


# ============================================================================
#  MAIN
# ============================================================================

def main():
    t_start = time.time()
    results = {}

    print("=" * 80)
    print("RED BREAKER -- DESTRUCTION TEST: EMA20 Touch NQ v11 (10min, gate=0.0)")
    print("=" * 80)

    # Load data
    print("\n[0] Loading NQ data...")
    nq = pd.read_csv(NQ_PATH, parse_dates=["Time"], index_col="Time")
    nq.index.name = "timestamp"
    nq.index = nq.index + pd.Timedelta(hours=1)  # CT -> ET
    nq_full = nq[nq.index >= "2022-01-01"]
    nq_is = nq[(nq.index >= "2022-01-01") & (nq.index < "2024-01-01")]
    nq_oos = nq[nq.index >= "2024-01-01"]
    print(f"    Full: {nq_full.index[0].date()} to {nq_full.index[-1].date()}")
    print(f"    IS (2022-2023): {len(nq_is):,} bars")
    print(f"    OOS (2024-2026): {len(nq_oos):,} bars")

    # ===================================================================
    #  BASELINE RUN
    # ===================================================================
    print(f"\n{'='*80}")
    print("BASELINE RUN (standard costs)")
    print(f"{'='*80}")

    tdf_full, dpnl_full = run_strategy(nq_full)
    tdf_is, dpnl_is = run_strategy(nq_is)
    tdf_oos, dpnl_oos = run_strategy(nq_oos)

    stats_full = compute_full_stats(tdf_full, dpnl_full)
    stats_is = compute_full_stats(tdf_is, dpnl_is)
    stats_oos = compute_full_stats(tdf_oos, dpnl_oos)

    print(f"\n  {'Period':>12} {'PF':>7} {'PnL':>9} {'DD':>7} {'N':>5} {'WR':>6} {'Sharpe':>7} {'MCL':>4} {'5R+':>4}")
    for label, st in [("Full 4Y", stats_full), ("IS 22-23", stats_is), ("OOS 24-26", stats_oos)]:
        print(f"  {label:>12} {st['pf']:>7.3f} {st['pnl']:>+9,.0f} {st['dd']:>7,.0f} {st['n']:>5} {st['wr']:>5.1f}% {st['sharpe']:>7.2f} {st['mcl']:>4} {st['b5']:>4}")

    results["baseline"] = {"full": stats_full, "is": stats_is, "oos": stats_oos}

    # ===================================================================
    #  ATTACK 1: WORST CASE SCENARIOS
    # ===================================================================
    print(f"\n{'='*80}")
    print("ATTACK 1: WORST-CASE SCENARIO CONSTRUCTION")
    print(f"{'='*80}")

    # --- 1a. Worst 10 trading days ---
    print("\n  --- 1a. Worst 10 Trading Days (full period) ---")
    daily_series = pd.Series(dpnl_full)
    worst_days = daily_series.nsmallest(10)
    print(f"  {'Rank':>4} {'Date':>12} {'PnL':>10}")
    for i, (date, pnl) in enumerate(worst_days.items(), 1):
        print(f"  {i:>4} {date:>12} ${pnl:>+9,.2f}")
    results["worst_10_days"] = {str(d): round(p, 2) for d, p in worst_days.items()}

    # --- 1b. Worst drawdown period ---
    print("\n  --- 1b. Worst Drawdown Period ---")
    cum_pnl = tdf_full["pnl"].cumsum()
    cum_pnl_daily = daily_series.cumsum()
    peak_daily = cum_pnl_daily.cummax()
    dd_daily = peak_daily - cum_pnl_daily

    # Find max DD
    max_dd_idx = dd_daily.idxmax()
    max_dd_val = dd_daily.max()

    # Find DD start (peak before max DD)
    pre_dd = cum_pnl_daily[:max_dd_idx]
    dd_start = pre_dd.idxmax() if len(pre_dd) > 0 else cum_pnl_daily.index[0]

    # Find recovery date
    post_dd = cum_pnl_daily[max_dd_idx:]
    peak_val = peak_daily[max_dd_idx]
    recovery_mask = post_dd >= peak_val
    if recovery_mask.any():
        dd_end = post_dd[recovery_mask].index[0]
        recovery_days = len(pd.bdate_range(max_dd_idx, dd_end))
    else:
        dd_end = "NEVER (still underwater)"
        recovery_days = "N/A"

    dd_depth_pct = max_dd_val / STARTING_EQUITY * 100
    print(f"  Peak date:     {dd_start}")
    print(f"  Trough date:   {max_dd_idx}")
    print(f"  Recovery date: {dd_end}")
    print(f"  Depth:         ${max_dd_val:,.0f} ({dd_depth_pct:.1f}% of starting equity)")
    print(f"  Recovery time: {recovery_days} business days")

    # Drawdown in R
    cum_r = tdf_full["r"].cumsum()
    dd_r = (cum_r.cummax() - cum_r).max()
    print(f"  Max DD in R:   {dd_r:.2f}R")

    results["worst_drawdown"] = {
        "peak_date": str(dd_start), "trough_date": str(max_dd_idx),
        "recovery_date": str(dd_end), "depth_usd": round(max_dd_val, 0),
        "depth_pct": round(dd_depth_pct, 1), "recovery_days": str(recovery_days),
        "dd_r": round(dd_r, 2),
    }

    # --- 1c. Maximum consecutive losing streak ---
    print("\n  --- 1c. Maximum Consecutive Losing Streak ---")
    is_loss = (tdf_full["r"] <= 0).values
    streaks = []
    cur_streak = 0; cur_start = 0
    for i, x in enumerate(is_loss):
        if x:
            if cur_streak == 0: cur_start = i
            cur_streak += 1
        else:
            if cur_streak > 0:
                streaks.append((cur_streak, cur_start, i - 1))
            cur_streak = 0
    if cur_streak > 0:
        streaks.append((cur_streak, cur_start, len(is_loss) - 1))

    streaks.sort(key=lambda x: -x[0])
    print(f"  {'Rank':>4} {'Length':>7} {'Start Date':>12} {'End Date':>12} {'Total PnL':>10}")
    for rank, (length, s_idx, e_idx) in enumerate(streaks[:5], 1):
        streak_pnl = tdf_full.iloc[s_idx:e_idx+1]["pnl"].sum()
        s_date = tdf_full.iloc[s_idx]["date"]
        e_date = tdf_full.iloc[e_idx]["date"]
        print(f"  {rank:>4} {length:>7} {s_date:>12} {e_date:>12} ${streak_pnl:>+9,.2f}")

    results["worst_streak"] = streaks[0][0] if streaks else 0

    # --- 1d. Starting-day sensitivity ---
    print("\n  --- 1d. What if you start on the worst possible day? ---")
    # Find the worst starting point: max trailing DD from any start
    equity_curve = STARTING_EQUITY + tdf_full["pnl"].cumsum()
    n_trades_total = len(tdf_full)

    worst_start_dd = 0; worst_start_idx = 0
    for start_i in range(0, n_trades_total - 10):
        sub_equity = equity_curve.iloc[start_i:] - equity_curve.iloc[start_i] + STARTING_EQUITY
        sub_dd = (sub_equity.cummax() - sub_equity).max()
        if sub_dd > worst_start_dd:
            worst_start_dd = sub_dd
            worst_start_idx = start_i

    worst_start_date = tdf_full.iloc[worst_start_idx]["date"]
    sub_eq = equity_curve.iloc[worst_start_idx:] - equity_curve.iloc[worst_start_idx] + STARTING_EQUITY
    sub_pnl_after_50 = (equity_curve.iloc[min(worst_start_idx + 50, n_trades_total - 1)]
                         - equity_curve.iloc[worst_start_idx])

    print(f"  Worst start date:     {worst_start_date}")
    print(f"  Max DD from start:    ${worst_start_dd:,.0f} ({worst_start_dd/STARTING_EQUITY*100:.1f}%)")
    print(f"  PnL after 50 trades:  ${sub_pnl_after_50:+,.0f}")

    # Check: at worst start, does equity go negative in first 100 trades?
    sub_100 = tdf_full.iloc[worst_start_idx:worst_start_idx + 100]["pnl"]
    min_eq = STARTING_EQUITY + sub_100.cumsum().min()
    print(f"  Min equity (100 trades): ${min_eq:,.0f}")
    topstep_blown = "YES" if worst_start_dd > 2000 else "NO"
    print(f"  Topstep $2K DD blown:  {topstep_blown}")

    results["worst_start"] = {
        "date": worst_start_date,
        "max_dd_from_start": round(worst_start_dd, 0),
        "topstep_blown": topstep_blown,
    }

    # ===================================================================
    #  ATTACK 2: REGIME STRESS TESTS
    # ===================================================================
    print(f"\n{'='*80}")
    print("ATTACK 2: REGIME STRESS TESTS")
    print(f"{'='*80}")

    regimes = [
        ("2022 H1 (bear crash)", "2022-01-01", "2022-07-01"),
        ("2022 H2 (bear rally)", "2022-07-01", "2023-01-01"),
        ("2023 H1 (AI rally)", "2023-01-01", "2023-07-01"),
        ("2023 H2 (range/pullback)", "2023-07-01", "2024-01-01"),
        ("2024 H1 (steady up)", "2024-01-01", "2024-07-01"),
        ("2024 H2 (election vol)", "2024-07-01", "2025-01-01"),
        ("2025 H1 (tariff chaos)", "2025-01-01", "2025-07-01"),
        ("2025 H2", "2025-07-01", "2026-01-01"),
        ("2026 Q1", "2026-01-01", "2026-04-01"),
    ]

    print(f"\n  {'Regime':>30} {'PF':>7} {'PnL':>9} {'DD':>7} {'N':>5} {'WR':>6} {'MCL':>4} {'Verdict':>8}")
    regime_results = {}
    n_profitable = 0; n_regimes = 0
    for name, start, end in regimes:
        sub = nq_full[(nq_full.index >= start) & (nq_full.index < end)]
        if len(sub) < 1000: continue
        tdf_r, dpnl_r = run_strategy(sub)
        if len(tdf_r) == 0: continue
        n_regimes += 1
        st = compute_full_stats(tdf_r, dpnl_r)
        verdict = "PASS" if st["pf"] >= 1.0 else "FAIL"
        if st["pf"] >= 1.0: n_profitable += 1
        print(f"  {name:>30} {st['pf']:>7.3f} {st['pnl']:>+9,.0f} {st['dd']:>7,.0f} {st['n']:>5} {st['wr']:>5.1f}% {st['mcl']:>4} {verdict:>8}")
        regime_results[name] = {"pf": st["pf"], "verdict": verdict}

    print(f"\n  Profitable regimes: {n_profitable}/{n_regimes}")
    results["regimes"] = {"profitable": n_profitable, "total": n_regimes, "details": regime_results}

    # ===================================================================
    #  ATTACK 3: COST SENSITIVITY BOMB
    # ===================================================================
    print(f"\n{'='*80}")
    print("ATTACK 3: COST SENSITIVITY BOMB")
    print(f"{'='*80}")

    print(f"\n  --- 3a. Escalating costs (full 4Y) ---")
    print(f"  {'Scenario':>30} {'PF':>7} {'PnL':>9} {'DD':>7} {'N':>5} {'Verdict':>8}")

    cost_scenarios = [
        ("Baseline (1x costs)", 1.0, 1.0),
        ("1.5x all costs", 1.5, 1.5),
        ("2x all costs", 2.0, 2.0),
        ("3x all costs", 3.0, 3.0),
        ("2x costs + 3x slip", 2.0, 3.0),
        ("5x slippage only", 1.0, 5.0),
    ]

    cost_results = {}
    for name, cm, sm in cost_scenarios:
        tdf_c, dpnl_c = run_strategy(nq_full, cost_mult=cm, slip_mult=sm)
        pf = compute_pf(tdf_c)
        pnl = tdf_c["pnl"].sum() if len(tdf_c) > 0 else 0
        dd = (tdf_c["pnl"].cumsum().cummax() - tdf_c["pnl"].cumsum()).max() if len(tdf_c) > 0 else 0
        n_t = len(tdf_c)
        verdict = "PASS" if pf >= 1.0 else "DEAD"
        print(f"  {name:>30} {pf:>7.3f} {pnl:>+9,.0f} {dd:>7,.0f} {n_t:>5} {verdict:>8}")
        cost_results[name] = {"pf": pf, "verdict": verdict}

    # Find breakeven cost multiplier
    print(f"\n  --- 3b. Cost bomb threshold (what multiplier kills the strategy?) ---")
    for mult in [1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0]:
        tdf_c, _ = run_strategy(nq_full, cost_mult=mult, slip_mult=mult)
        pf = compute_pf(tdf_c)
        if pf < 1.0:
            print(f"  Strategy DIES at {mult}x costs (PF={pf:.3f})")
            results["cost_bomb_threshold"] = mult
            break
    else:
        print(f"  Strategy survives even 10x costs!")
        results["cost_bomb_threshold"] = ">10x"

    # Specifically what user asked:
    print(f"\n  --- 3c. Specific tests ---")
    tdf_2x, _ = run_strategy(nq_full, cost_mult=2.0, slip_mult=1.0)
    tdf_3slip, _ = run_strategy(nq_full, cost_mult=1.0, slip_mult=3.0)
    print(f"  Double ALL costs:     PF = {compute_pf(tdf_2x):.3f}")
    print(f"  Triple slippage:      PF = {compute_pf(tdf_3slip):.3f}")

    results["double_costs_pf"] = compute_pf(tdf_2x)
    results["triple_slip_pf"] = compute_pf(tdf_3slip)
    results["cost_scenarios"] = cost_results

    # ===================================================================
    #  ATTACK 4: EXECUTION DELAY
    # ===================================================================
    print(f"\n{'='*80}")
    print("ATTACK 4: EXECUTION DELAY ATTACK")
    print(f"{'='*80}")

    print(f"\n  {'Delay':>20} {'PF':>7} {'PnL':>9} {'DD':>7} {'N':>5}")
    delay_results = {}
    for delay in [0, 1, 2, 3]:
        tdf_d, dpnl_d = run_strategy(nq_full, entry_delay=delay)
        st = compute_full_stats(tdf_d, dpnl_d)
        label = f"{delay} bar ({delay * 10}min)" if delay > 0 else "0 (baseline)"
        print(f"  {label:>20} {st['pf']:>7.3f} {st['pnl']:>+9,.0f} {st['dd']:>7,.0f} {st['n']:>5}")
        delay_results[delay] = st["pf"]

    results["delay_1bar_pf"] = delay_results.get(1, 0)
    results["delay_results"] = delay_results

    # ===================================================================
    #  ATTACK 5: RANDOM BASELINE TEST
    # ===================================================================
    print(f"\n{'='*80}")
    print("ATTACK 5: RANDOM BASELINE (1000 iterations)")
    print(f"{'='*80}")

    # Get target trade count from baseline
    n_target = stats_full["n"]
    print(f"  Target trades: ~{n_target} (same as baseline)")

    # --- 5a. Random entries + FULL exit system (tests entry edge) ---
    print(f"\n  --- 5a. Random entries + SAME exit system (entry edge test) ---")
    random_pfs_full = []
    t0 = time.time()
    for i in range(1000):
        tdf_rand = run_random_strategy(nq_full, S, seed=i, n_trades_target=n_target, mode="full_exit")
        if len(tdf_rand) > 0:
            rpf = compute_pf(tdf_rand)
        else:
            rpf = 0
        random_pfs_full.append(rpf)
        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            print(f"    [{i+1}/1000] {elapsed:.0f}s elapsed...")

    random_pfs_full = np.array(random_pfs_full)
    real_pf = stats_full["pf"]
    pct_beat_full = (random_pfs_full >= real_pf).mean() * 100
    percentile_full = (random_pfs_full < real_pf).mean() * 100

    print(f"    Mean PF:       {random_pfs_full.mean():.3f}")
    print(f"    Median PF:     {np.median(random_pfs_full):.3f}")
    print(f"    Max PF:        {random_pfs_full.max():.3f}")
    print(f"    % with PF>1:   {(random_pfs_full > 1).mean()*100:.1f}%")
    print(f"    % beat strat:  {pct_beat_full:.1f}%")
    print(f"    Strategy PF:   {real_pf:.3f}")
    print(f"    Percentile:    {percentile_full:.1f}th")

    # --- 5b. Random entries + SIMPLE exit (market bias test) ---
    print(f"\n  --- 5b. Random entries + SIMPLE exit (market bias test) ---")
    random_pfs_simple = []
    t0 = time.time()
    for i in range(1000):
        tdf_rand = run_random_strategy(nq_full, S, seed=i, n_trades_target=n_target, mode="simple_exit")
        if len(tdf_rand) > 0:
            rpf = compute_pf(tdf_rand)
        else:
            rpf = 0
        random_pfs_simple.append(rpf)
        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            print(f"    [{i+1}/1000] {elapsed:.0f}s elapsed...")

    random_pfs_simple = np.array(random_pfs_simple)
    pct_beat_simple = (random_pfs_simple >= real_pf).mean() * 100

    print(f"    Mean PF:       {random_pfs_simple.mean():.3f}")
    print(f"    Median PF:     {np.median(random_pfs_simple):.3f}")
    print(f"    % with PF>1:   {(random_pfs_simple > 1).mean()*100:.1f}%")
    print(f"    % beat strat:  {pct_beat_simple:.1f}%")

    # --- 5c. PnL Shuffle (bootstrap) ---
    print(f"\n  --- 5c. PnL Shuffle bootstrap (trade-order significance) ---")
    real_trades_pnl = tdf_full["pnl"].values
    np.random.seed(42)
    shuffled_pfs = []
    for _ in range(10000):
        shuffled = np.random.choice(real_trades_pnl, size=len(real_trades_pnl), replace=True)
        w = shuffled[shuffled > 0].sum()
        l = abs(shuffled[shuffled <= 0].sum())
        shuffled_pfs.append(w / l if l > 0 else 0)
    shuffled_pfs = np.array(shuffled_pfs)
    p_value = (shuffled_pfs >= real_pf).mean()
    print(f"    Bootstrap p-value (PF >= {real_pf:.3f}): {p_value:.4f}")
    print(f"    Bootstrap mean PF: {shuffled_pfs.mean():.3f}")
    print(f"    Bootstrap 95th pct: {np.percentile(shuffled_pfs, 95):.3f}")

    # --- 5d. Edge decomposition ---
    print(f"\n  --- 5d. Edge decomposition ---")
    exit_edge = random_pfs_full.mean()  # random entry + real exit
    market_bias = random_pfs_simple.mean()  # random entry + simple exit
    total_edge = real_pf

    print(f"    Market bias (simple exit):    PF = {market_bias:.3f}")
    print(f"    Exit system (full exit):      PF = {exit_edge:.3f} (+{exit_edge - market_bias:.3f} over market)")
    print(f"    Strategy (entry + exit):      PF = {total_edge:.3f} (+{total_edge - exit_edge:.3f} over random entry)")
    entry_alpha = total_edge - exit_edge
    exit_alpha = exit_edge - market_bias
    print(f"    Entry alpha:                  {entry_alpha:+.3f} PF")
    print(f"    Exit alpha:                   {exit_alpha:+.3f} PF")

    # Determine verdict based on all tests
    if pct_beat_full < 5:
        random_verdict = "REAL ENTRY EDGE -- < 5% random entries beat it"
    elif pct_beat_full < 20:
        random_verdict = "MARGINAL ENTRY EDGE -- 5-20% random entries beat it"
    else:
        if exit_edge > 1.1:
            random_verdict = f"EXIT-DRIVEN -- entry adds no edge, but exit system PF={exit_edge:.3f}"
        else:
            random_verdict = "NO EDGE -- > 20% random strategies beat it"

    print(f"\n  Verdict:       {random_verdict}")
    pct_beat = pct_beat_full  # use full exit for overall assessment

    results["random_baseline"] = {
        "full_exit_mean_pf": round(random_pfs_full.mean(), 3),
        "full_exit_median_pf": round(np.median(random_pfs_full), 3),
        "full_exit_max_pf": round(random_pfs_full.max(), 3),
        "full_exit_pct_pf_gt1": round((random_pfs_full > 1).mean() * 100, 1),
        "simple_exit_mean_pf": round(random_pfs_simple.mean(), 3),
        "simple_exit_pct_pf_gt1": round((random_pfs_simple > 1).mean() * 100, 1),
        "pct_beat_strategy": round(pct_beat_full, 1),
        "strategy_percentile": round(percentile_full, 1),
        "bootstrap_p_value": round(p_value, 4),
        "entry_alpha": round(entry_alpha, 3),
        "exit_alpha": round(exit_alpha, 3),
        "verdict": random_verdict,
    }

    # ===================================================================
    #  ATTACK 6: YEAR-BY-YEAR CONSISTENCY
    # ===================================================================
    print(f"\n{'='*80}")
    print("ATTACK 6: YEAR-BY-YEAR CONSISTENCY")
    print(f"{'='*80}")

    print(f"\n  {'Year':>6} {'PF':>7} {'PnL':>9} {'DD':>7} {'N':>5} {'WR':>6} {'MCL':>4} {'Verdict':>8}")
    yearly_results = {}
    for yr in range(2022, 2027):
        sub = nq_full[(nq_full.index >= f"{yr}-01-01") & (nq_full.index < f"{yr+1}-01-01")]
        if len(sub) < 1000: continue
        tdf_y, dpnl_y = run_strategy(sub)
        if len(tdf_y) == 0: continue
        st = compute_full_stats(tdf_y, dpnl_y)
        verdict = "PASS" if st["pf"] >= 1.0 else "FAIL"
        print(f"  {yr:>6} {st['pf']:>7.3f} {st['pnl']:>+9,.0f} {st['dd']:>7,.0f} {st['n']:>5} {st['wr']:>5.1f}% {st['mcl']:>4} {verdict:>8}")
        yearly_results[yr] = {"pf": st["pf"], "pnl": st["pnl"], "verdict": verdict}

    results["yearly"] = yearly_results

    # ===================================================================
    #  FINAL VERDICT
    # ===================================================================
    print(f"\n{'='*80}")
    print("FINAL DESTRUCTION ASSESSMENT")
    print(f"{'='*80}")

    # Tally kills
    kills = []
    survives = []

    # Cost bomb
    if isinstance(results.get("cost_bomb_threshold"), (int, float)) and results["cost_bomb_threshold"] <= 2.0:
        kills.append(f"Dies at {results['cost_bomb_threshold']}x costs")
    else:
        survives.append(f"Survives up to {results.get('cost_bomb_threshold', '?')}x costs")

    # Delay
    if results.get("delay_1bar_pf", 0) < 1.0:
        kills.append(f"1-bar delay kills it (PF={results['delay_1bar_pf']:.3f})")
    else:
        survives.append(f"Survives 1-bar delay (PF={results['delay_1bar_pf']:.3f})")

    # Random
    pct_beat = results.get("random_baseline", {}).get("pct_beat_strategy", 100)
    if pct_beat > 20:
        kills.append(f"No edge vs random ({pct_beat:.0f}% beat it)")
    elif pct_beat > 5:
        kills.append(f"Marginal edge ({pct_beat:.0f}% random beat it)")
    else:
        survives.append(f"Real edge ({pct_beat:.1f}% random beat it)")

    # Regimes
    regime_data = results.get("regimes", {})
    if regime_data.get("profitable", 0) < regime_data.get("total", 1) * 0.6:
        kills.append(f"Only {regime_data['profitable']}/{regime_data['total']} regimes profitable")
    else:
        survives.append(f"{regime_data['profitable']}/{regime_data['total']} regimes profitable")

    # Worst start
    if results.get("worst_start", {}).get("topstep_blown") == "YES":
        kills.append("Starting on worst day blows Topstep $2K DD limit")

    # Max consecutive losses
    if stats_full["mcl"] >= 15:
        kills.append(f"Max consecutive losses: {stats_full['mcl']} (psych blow)")
    else:
        survives.append(f"Max consecutive losses: {stats_full['mcl']}")

    # DD
    if stats_full["dd"] > STARTING_EQUITY * 0.10:
        kills.append(f"Max DD ${stats_full['dd']:,.0f} ({stats_full['dd']/STARTING_EQUITY*100:.1f}% of equity)")

    print("\n  KILL SHOTS:")
    for k in kills:
        print(f"    [X] {k}")

    print("\n  SURVIVED:")
    for s in survives:
        print(f"    [+] {s}")

    # Overall verdict
    if len(kills) >= 3:
        overall = "DEAD"
    elif len(kills) >= 2:
        overall = "FRAGILE"
    elif len(kills) == 1:
        overall = "BRUISED"
    else:
        overall = "ROBUST"

    print(f"\n  OVERALL VERDICT: {'='*10} {overall} {'='*10}")
    print(f"  Kill count: {len(kills)} / Survive count: {len(survives)}")

    results["verdict"] = overall
    results["kills"] = kills
    results["survives"] = survives

    elapsed = time.time() - t_start
    print(f"\n  Total runtime: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # ===================================================================
    #  WRITE REPORT
    # ===================================================================
    write_report(results, stats_full, stats_is, stats_oos)

    return results


def write_report(results, stats_full, stats_is, stats_oos):
    """Write the destruction report to docs/RED_BREAKER_REPORT.md"""

    lines = []
    a = lines.append

    a("# RED BREAKER REPORT -- Strategy Destruction Test")
    a("")
    a(f"**Strategy:** EMA20 Touch, NQ v11 (10min bars, gate=0.0, MNQ costs per contract)")
    a(f"**Generated:** {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    a(f"**Verdict: {results['verdict']}**")
    a("")

    a("## Baseline Performance")
    a("")
    a("| Period | PF | PnL | Max DD | Trades | Win Rate | Sharpe | MCL | 5R+ |")
    a("|--------|---:|----:|-------:|-------:|---------:|-------:|----:|----:|")
    for label, st in [("Full 4Y", stats_full), ("IS 2022-23", stats_is), ("OOS 2024-26", stats_oos)]:
        a(f"| {label} | {st['pf']:.3f} | ${st['pnl']:+,.0f} | ${st['dd']:,.0f} | {st['n']} | {st['wr']:.1f}% | {st['sharpe']:.2f} | {st['mcl']} | {st['b5']} |")
    a("")

    a("## Attack 1: Worst-Case Scenarios")
    a("")

    a("### Worst 10 Trading Days")
    a("")
    a("| Rank | Date | PnL |")
    a("|-----:|------|----:|")
    for i, (date, pnl) in enumerate(results.get("worst_10_days", {}).items(), 1):
        a(f"| {i} | {date} | ${pnl:+,.2f} |")
    a("")

    dd = results.get("worst_drawdown", {})
    a("### Worst Drawdown Period")
    a("")
    a(f"- **Peak date:** {dd.get('peak_date', '?')}")
    a(f"- **Trough date:** {dd.get('trough_date', '?')}")
    a(f"- **Recovery date:** {dd.get('recovery_date', '?')}")
    a(f"- **Depth:** ${dd.get('depth_usd', 0):,.0f} ({dd.get('depth_pct', 0):.1f}% of equity)")
    a(f"- **Max DD in R:** {dd.get('dd_r', 0):.2f}R")
    a(f"- **Recovery time:** {dd.get('recovery_days', '?')} business days")
    a("")

    a(f"### Max Consecutive Losing Streak: {results.get('worst_streak', '?')} trades")
    a("")

    ws = results.get("worst_start", {})
    a("### Worst Starting Day")
    a("")
    a(f"- **Date:** {ws.get('date', '?')}")
    a(f"- **Max DD from start:** ${ws.get('max_dd_from_start', 0):,.0f}")
    a(f"- **Topstep $2K DD blown:** {ws.get('topstep_blown', '?')}")
    a("")

    a("## Attack 2: Regime Stress Tests")
    a("")
    a("| Regime | PF | Verdict |")
    a("|--------|---:|--------:|")
    for name, data in results.get("regimes", {}).get("details", {}).items():
        a(f"| {name} | {data['pf']:.3f} | {data['verdict']} |")
    a("")
    reg = results.get("regimes", {})
    a(f"**Profitable regimes: {reg.get('profitable', 0)}/{reg.get('total', 0)}**")
    a("")

    a("## Attack 3: Cost Sensitivity Bomb")
    a("")
    a("| Scenario | PF | Verdict |")
    a("|----------|---:|--------:|")
    for name, data in results.get("cost_scenarios", {}).items():
        a(f"| {name} | {data['pf']:.3f} | {data['verdict']} |")
    a("")
    a(f"- **Double all costs:** PF = {results.get('double_costs_pf', 0):.3f}")
    a(f"- **Triple slippage:** PF = {results.get('triple_slip_pf', 0):.3f}")
    a(f"- **Cost bomb threshold:** {results.get('cost_bomb_threshold', '?')}x")
    a("")

    a("## Attack 4: Execution Delay")
    a("")
    a("| Delay | PF |")
    a("|------:|---:|")
    for delay, pf in results.get("delay_results", {}).items():
        a(f"| {delay} bar ({delay * 10}min) | {pf:.3f} |")
    a("")
    a(f"- **1-bar delay PF:** {results.get('delay_1bar_pf', 0):.3f}")
    a("")

    a("## Attack 5: Random Baseline (1000 iterations x 2 modes + bootstrap)")
    a("")
    rb = results.get("random_baseline", {})
    a("### 5a. Random entries + SAME exit system (entry edge test)")
    a(f"- **Mean random PF:** {rb.get('full_exit_mean_pf', 0):.3f}")
    a(f"- **Median random PF:** {rb.get('full_exit_median_pf', 0):.3f}")
    a(f"- **Max random PF:** {rb.get('full_exit_max_pf', 0):.3f}")
    a(f"- **% random PF > 1:** {rb.get('full_exit_pct_pf_gt1', 0):.1f}%")
    a(f"- **% random beat strategy:** {rb.get('pct_beat_strategy', 0):.1f}%")
    a(f"- **Strategy percentile:** {rb.get('strategy_percentile', 0):.1f}th")
    a("")
    a("### 5b. Random entries + SIMPLE exit (market bias test)")
    a(f"- **Mean random PF (simple exit):** {rb.get('simple_exit_mean_pf', 0):.3f}")
    a(f"- **% with PF > 1:** {rb.get('simple_exit_pct_pf_gt1', 0):.1f}%")
    a("")
    a("### 5c. Bootstrap p-value")
    a(f"- **p-value:** {rb.get('bootstrap_p_value', 0):.4f}")
    a("")
    a("### 5d. Edge decomposition")
    a(f"- **Entry alpha (strategy PF - random entry PF):** {rb.get('entry_alpha', 0):+.3f}")
    a(f"- **Exit alpha (exit system PF - simple exit PF):** {rb.get('exit_alpha', 0):+.3f}")
    a(f"- **Verdict:** {rb.get('verdict', '?')}")
    a("")

    a("## Attack 6: Year-by-Year Consistency")
    a("")
    a("| Year | PF | PnL | Verdict |")
    a("|-----:|---:|----:|--------:|")
    for yr, data in sorted(results.get("yearly", {}).items()):
        a(f"| {yr} | {data['pf']:.3f} | ${data['pnl']:+,.0f} | {data['verdict']} |")
    a("")

    a("## Final Verdict")
    a("")
    a(f"### {results['verdict']}")
    a("")
    a("**Kill Shots:**")
    for k in results.get("kills", []):
        a(f"- {k}")
    a("")
    a("**Survived:**")
    for s in results.get("survives", []):
        a(f"- {s}")
    a("")

    report_path = "docs/RED_BREAKER_REPORT.md"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n  Report written to {report_path}")


if __name__ == "__main__":
    main()
