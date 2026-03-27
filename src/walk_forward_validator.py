"""
Walk-Forward Validator — Blue Team backtesting framework.

Validates any strategy function through:
1. Rolling walk-forward (train 12m, test 6m, roll 6m)
2. Parameter sensitivity (+-20% on each numeric param)
3. Cost stress test (1.5x and 2x costs)
4. 1-bar execution delay test
5. Full stats table (IS, OOS, 4Y, walk-forward aggregate)
6. Yearly breakdown

Uses the CORRECT cost model from src/backtest_engine.py:
  NQ: comm=$2.46 RT, spread=$5.00, stop_slip=$1.25, be_slip=$1.25
  All costs PER CONTRACT. Gap-through stop fill.

Data: NQ 1min continuous RTH (CT+1h = ET)
IS = 2022-2023, OOS = 2024-2026
"""
from __future__ import annotations
import datetime as dt, copy, functools, sys, os
import numpy as np, pandas as pd

# ── Cost model (NQ, per contract) ──────────────────────────────────────
NQ_PT_VAL    = 20.0
COMM_RT      = 2.46   # round-trip commission per contract
SPREAD       = 5.00   # spread cost per contract on entry
SLIP_STOP    = 1.25   # slippage per contract on stop exit
SLIP_BE      = 1.25   # slippage per contract on BE exit

NQ_PATH = "data/barchart_nq/NQ_1min_continuous_RTH.csv"


# ═══════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

def load_nq(path=NQ_PATH, start="2022-01-01"):
    """Load NQ continuous RTH data, convert CT to ET."""
    nq = pd.read_csv(path, parse_dates=["Time"], index_col="Time")
    nq.index.name = "timestamp"
    nq.index = nq.index + pd.Timedelta(hours=1)  # CT -> ET
    return nq[nq.index >= start]


def resample(df, minutes):
    if minutes <= 1:
        return df
    return df.resample(f"{minutes}min").agg(
        {"Open": "first", "High": "max", "Low": "min",
         "Close": "last", "Volume": "sum"}).dropna()


def add_indicators(df, ema_fast=20, ema_slow=50, atr_period=14):
    df = df.copy()
    df["ema_f"] = df["Close"].ewm(span=ema_fast, adjust=False).mean()
    df["ema_s"] = df["Close"].ewm(span=ema_slow, adjust=False).mean()
    tr = np.maximum(df["High"] - df["Low"],
        np.maximum((df["High"] - df["Close"].shift(1)).abs(),
                   (df["Low"] - df["Close"].shift(1)).abs()))
    df["atr"] = tr.rolling(atr_period).mean()
    return df


# ═══════════════════════════════════════════════════════════════════════
#  BASELINE EMA20 TOUCH STRATEGY (NQ, matching exp_nq_tune.py logic)
# ═══════════════════════════════════════════════════════════════════════

BASELINE_PARAMS = {
    "tf_minutes": 10,
    "ema_fast": 20, "ema_slow": 50, "atr_period": 14,
    "touch_tol": 0.15, "touch_below_max": 0.5,
    "no_entry_after": dt.time(14, 0),
    "stop_buffer": 0.4,
    "gate_bars": 3, "gate_mfe": 0.2, "gate_tighten": 0.0,
    "be_trigger_r": 0.25, "be_stop_r": 0.15,
    "chand_bars": 25, "chand_mult": 0.3,
    "max_hold_bars": 180, "force_close_at": dt.time(15, 58),
    "daily_loss_r": 2.0, "skip_after_win": 1,
    "n_contracts": 2,
}

# Keys that are numeric and should be perturbed in sensitivity analysis
SENSITIVITY_KEYS = [
    "touch_tol", "touch_below_max", "stop_buffer",
    "gate_mfe", "gate_tighten",
    "be_trigger_r", "be_stop_r",
    "chand_mult", "daily_loss_r",
]


def run_baseline(df_1min, params=None, cost_mult=1.0, delay_bars=0):
    """
    Run the EMA20 touch strategy on NQ 1-min data.
    Returns a DataFrame of trades with columns:
        pnl, r, exit, date, cost, risk, year, mfe, nc

    Parameters
    ----------
    df_1min : DataFrame with Open/High/Low/Close/Volume, ET timestamp index
    params  : strategy params dict (default: BASELINE_PARAMS)
    cost_mult : multiply all costs by this factor (for stress test)
    delay_bars : delay entry by N bars (execution delay test)
    """
    s = params if params is not None else BASELINE_PARAMS
    df = resample(df_1min, s["tf_minutes"])
    df = add_indicators(df, s["ema_fast"], s["ema_slow"], s["atr_period"])

    H = df["High"].values; L = df["Low"].values; C = df["Close"].values
    O = df["Open"].values
    ef = df["ema_f"].values; es = df["ema_s"].values; atr = df["atr"].values
    T = df.index.time; D = df.index.date; n = len(df)

    tf = max(1, s["tf_minutes"])
    mh = max(20, s["max_hold_bars"] // tf)
    cb = max(5, s["chand_bars"] // tf)
    gb = max(1, s["gate_bars"] // tf) if s["gate_bars"] > 0 else 0
    nc = s["n_contracts"]

    # Costs per contract, scaled
    c_comm   = COMM_RT * cost_mult
    c_spread = SPREAD * cost_mult
    c_slip_s = SLIP_STOP * cost_mult
    c_slip_b = SLIP_BE * cost_mult

    trades = []
    bar = max(s["ema_slow"], s["atr_period"]) + 5
    dlr = 0.0; cd = None; sk = 0

    while bar < n - mh - 5:
        a = atr[bar]
        if np.isnan(a) or a <= 0 or np.isnan(ef[bar]) or np.isnan(es[bar]):
            bar += 1; continue
        if T[bar] >= s["no_entry_after"]:
            bar += 1; continue
        d = D[bar]
        if cd != d:
            cd = d; dlr = 0.0
        if dlr >= s["daily_loss_r"]:
            bar += 1; continue

        c = C[bar]
        if c > ef[bar] and ef[bar] > es[bar]:
            trend = 1
        elif c < ef[bar] and ef[bar] < es[bar]:
            trend = -1
        else:
            bar += 1; continue

        tol = a * s["touch_tol"]
        if trend == 1:
            touch = L[bar] <= ef[bar] + tol and L[bar] >= ef[bar] - a * s["touch_below_max"]
        else:
            touch = H[bar] >= ef[bar] - tol and H[bar] <= ef[bar] + a * s["touch_below_max"]
        if not touch:
            bar += 1; continue
        if sk > 0:
            sk -= 1; bar += 1; continue

        # Apply execution delay
        entry_signal_bar = bar
        if delay_bars > 0:
            bar = bar + delay_bars
            if bar >= n - mh - 5:
                break

        entry = C[bar]
        stop = L[bar] - s["stop_buffer"] * a if trend == 1 else H[bar] + s["stop_buffer"] * a
        rp = abs(entry - stop)
        if rp <= 0:
            bar += 1; continue
        rm = rp * NQ_PT_VAL * nc  # risk in $

        # Entry cost
        ec = c_comm * nc / 2 + c_spread * nc

        eb = bar; rs = stop; bt = False; mfe = 0.0; r = 0.0
        endb = bar; ex = "timeout"

        for k in range(1, mh + 1):
            bi = eb + k
            if bi >= n:
                break
            h = H[bi]; l = L[bi]; ca = atr[bi] if not np.isnan(atr[bi]) else a

            # MFE
            if trend == 1:
                mfe = max(mfe, (h - entry) / rp)
            else:
                mfe = max(mfe, (entry - l) / rp)

            # Force close
            if T[bi] >= s["force_close_at"]:
                r = (C[bi] - entry) / rp * trend
                endb = bi; ex = "close"; break

            # MFE gate
            if gb > 0 and k == gb and not bt:
                if mfe < s["gate_mfe"]:
                    ns = entry + s["gate_tighten"] * rp * trend
                    if trend == 1:
                        rs = max(rs, ns)
                    else:
                        rs = min(rs, ns)

            # Stop check with gap-through
            stopped = (trend == 1 and l <= rs) or (trend == -1 and h >= rs)
            if stopped:
                if trend == 1:
                    fill = min(rs, O[bi]) if O[bi] < rs else rs
                else:
                    fill = max(rs, O[bi]) if O[bi] > rs else rs
                r = (fill - entry) / rp * trend
                endb = bi
                if bt:
                    ref = entry + s["be_stop_r"] * rp * trend
                    ex = "be" if abs(rs - ref) < 0.05 * rp else "trail"
                else:
                    ex = "stop"
                break

            # BE trigger
            if not bt and s["be_trigger_r"] > 0:
                tp = entry + s["be_trigger_r"] * rp * trend
                if (trend == 1 and h >= tp) or (trend == -1 and l <= tp):
                    bt = True
                    bl = entry + s["be_stop_r"] * rp * trend
                    if trend == 1:
                        rs = max(rs, bl)
                    else:
                        rs = min(rs, bl)

            # Chandelier trail
            if bt and k >= cb:
                skk = max(1, k - cb + 1)
                hv = [H[eb + kk] for kk in range(skk, k) if eb + kk < n]
                lv = [L[eb + kk] for kk in range(skk, k) if eb + kk < n]
                if hv and lv:
                    if trend == 1:
                        rs = max(rs, max(hv) - s["chand_mult"] * ca)
                    else:
                        rs = min(rs, min(lv) + s["chand_mult"] * ca)
        else:
            r = (C[min(eb + mh, n - 1)] - entry) / rp * trend
            endb = min(eb + mh, n - 1)

        # P&L
        raw = r * rm
        xc = c_comm * nc / 2
        xs = c_slip_s * nc if ex in ("stop", "trail") else 0
        bs = c_slip_b * nc if ex == "be" else 0
        total_cost = ec + xc + xs + bs
        net = raw - total_cost

        trades.append({
            "pnl": net, "r": r, "exit": ex,
            "date": str(d), "cost": total_cost, "risk": rm,
            "year": d.year, "mfe": mfe, "nc": nc,
        })
        if r < 0:
            dlr += abs(r)
        if r > 0:
            sk = s.get("skip_after_win", 0)
        bar = endb + 1

    return pd.DataFrame(trades) if trades else pd.DataFrame(
        columns=["pnl", "r", "exit", "date", "cost", "risk", "year", "mfe", "nc"])


# ═══════════════════════════════════════════════════════════════════════
#  STATS COMPUTATION
# ═══════════════════════════════════════════════════════════════════════

def compute_stats(tdf, starting_equity=50000):
    """Full stats from a trades DataFrame. Matches backtest_engine.compute_stats."""
    if len(tdf) == 0:
        return {k: 0 for k in [
            "pf", "wr", "dd", "dd_r", "pnl", "n", "dpnl", "b5",
            "sharpe", "sortino", "apr", "calmar", "mcl",
            "win_days_pct", "worst_day", "best_day",
            "avg_win_r", "avg_loss_r", "cost_pct"]}

    gw = tdf.loc[tdf["pnl"] > 0, "pnl"].sum()
    gl = abs(tdf.loc[tdf["pnl"] <= 0, "pnl"].sum())
    cum = tdf["pnl"].cumsum()
    dd = (cum.cummax() - cum).max()

    daily = tdf.groupby("date")["pnl"].sum()
    days_arr = daily.values
    n_days = len(days_arr)
    years = n_days / 252

    sharpe = (days_arr.mean() / days_arr.std()) * 252**0.5 if days_arr.std() > 0 else 0
    down = days_arr[days_arr < 0]
    sortino = (days_arr.mean() / down.std()) * 252**0.5 if len(down) > 0 and down.std() > 0 else 0

    apr = cum.iloc[-1] / starting_equity / max(years, 0.1) * 100
    calmar = apr / (dd / starting_equity * 100) if dd > 0 else 0

    is_loss = (tdf["r"] <= 0).values
    mcl = cur = 0
    for x in is_loss:
        if x:
            cur += 1; mcl = max(mcl, cur)
        else:
            cur = 0

    cum_r = tdf["r"].cumsum()
    dd_r = (cum_r.cummax() - cum_r).max()

    return {
        "pf": round(gw / gl, 3) if gl > 0 else 0,
        "wr": round((tdf["r"] > 0).mean() * 100, 1),
        "dd": round(dd, 0),
        "dd_r": round(dd_r, 2),
        "pnl": round(cum.iloc[-1], 0),
        "n": len(tdf),
        "dpnl": round(cum.iloc[-1] / max(n_days, 1), 1),
        "b5": int((tdf["r"] >= 5).sum()),
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "apr": round(apr, 1),
        "calmar": round(calmar, 2),
        "mcl": mcl,
        "win_days_pct": round((days_arr > 0).mean() * 100, 0),
        "worst_day": round(days_arr.min(), 0),
        "best_day": round(days_arr.max(), 0),
        "avg_win_r": round(tdf.loc[tdf["r"] > 0, "r"].mean(), 3) if (tdf["r"] > 0).any() else 0,
        "avg_loss_r": round(tdf.loc[tdf["r"] <= 0, "r"].mean(), 3) if (tdf["r"] <= 0).any() else 0,
        "cost_pct": round(tdf["cost"].mean() / tdf["risk"].mean() * 100, 1) if tdf["risk"].mean() > 0 else 0,
    }


# ═══════════════════════════════════════════════════════════════════════
#  WALK-FORWARD ENGINE
# ═══════════════════════════════════════════════════════════════════════

def walk_forward(nq_1min, strategy_fn, params=None,
                 train_months=12, test_months=6, roll_months=6,
                 cost_mult=1.0, delay_bars=0):
    """
    Rolling walk-forward validation.

    Parameters
    ----------
    nq_1min : full NQ 1-min DataFrame (ET timestamps)
    strategy_fn : callable(df_1min, params, cost_mult, delay_bars) -> trades DataFrame
    params : strategy params dict
    train_months, test_months, roll_months : window sizes
    cost_mult, delay_bars : passed through to strategy_fn

    Returns
    -------
    List of dicts with period results, each containing:
        train_start, train_end, test_start, test_end, stats, trades
    """
    start = nq_1min.index[0]
    end = nq_1min.index[-1]

    results = []
    test_start = start + pd.DateOffset(months=train_months)

    while test_start + pd.DateOffset(months=test_months) <= end + pd.Timedelta(days=30):
        test_end = test_start + pd.DateOffset(months=test_months)
        train_start = test_start - pd.DateOffset(months=train_months)

        test_data = nq_1min[(nq_1min.index >= test_start) & (nq_1min.index < test_end)]
        if len(test_data) < 1000:
            test_start += pd.DateOffset(months=roll_months)
            continue

        trades = strategy_fn(test_data, params, cost_mult=cost_mult, delay_bars=delay_bars)
        stats = compute_stats(trades)

        results.append({
            "train_start": train_start.strftime("%Y-%m"),
            "train_end": test_start.strftime("%Y-%m"),
            "test_start": test_start.strftime("%Y-%m"),
            "test_end": test_end.strftime("%Y-%m"),
            "label": f"{test_start.strftime('%Y-%m')} -> {(test_end - pd.Timedelta(days=1)).strftime('%Y-%m')}",
            "stats": stats,
            "trades": trades,
        })

        test_start += pd.DateOffset(months=roll_months)

    return results


def wf_aggregate_stats(wf_results):
    """Aggregate walk-forward results: combine all OOS trades, compute stats."""
    if not wf_results:
        return compute_stats(pd.DataFrame(
            columns=["pnl", "r", "exit", "date", "cost", "risk", "year", "mfe", "nc"]))
    all_trades = pd.concat([r["trades"] for r in wf_results], ignore_index=True)
    return compute_stats(all_trades)


# ═══════════════════════════════════════════════════════════════════════
#  PARAMETER SENSITIVITY
# ═══════════════════════════════════════════════════════════════════════

def sensitivity_test(nq_1min, strategy_fn, params, keys=None, pct=0.20):
    """
    Perturb each key by +/-pct, run strategy, compare PF and DD.

    Returns list of dicts: param, direction, value, pf, dd, pnl, n
    """
    if keys is None:
        keys = SENSITIVITY_KEYS

    base_trades = strategy_fn(nq_1min, params)
    base_stats = compute_stats(base_trades)

    results = [{"param": "BASE", "direction": "", "value": "-",
                "pf": base_stats["pf"], "dd": base_stats["dd"],
                "pnl": base_stats["pnl"], "n": base_stats["n"],
                "sharpe": base_stats["sharpe"]}]

    for key in keys:
        if key not in params:
            continue
        val = params[key]
        if not isinstance(val, (int, float)):
            continue
        if val == 0:
            # For zero values, try small absolute perturbations
            perturbations = [("lo", -0.05), ("hi", +0.05)]
        else:
            perturbations = [("lo", val * (1 - pct)), ("hi", val * (1 + pct))]

        for direction, new_val in perturbations:
            p = copy.deepcopy(params)
            p[key] = round(new_val, 4) if isinstance(val, float) else int(round(new_val))
            trades = strategy_fn(nq_1min, p)
            stats = compute_stats(trades)
            results.append({
                "param": key, "direction": direction,
                "value": p[key],
                "pf": stats["pf"], "dd": stats["dd"],
                "pnl": stats["pnl"], "n": stats["n"],
                "sharpe": stats["sharpe"],
            })

    return results, base_stats


# ═══════════════════════════════════════════════════════════════════════
#  COST STRESS TEST
# ═══════════════════════════════════════════════════════════════════════

def cost_stress_test(nq_1min, strategy_fn, params):
    """Run at 1x, 1.5x, 2x cost multipliers."""
    results = []
    for mult in [1.0, 1.5, 2.0]:
        trades = strategy_fn(nq_1min, params, cost_mult=mult)
        stats = compute_stats(trades)
        results.append({"cost_mult": mult, "stats": stats})
    return results


# ═══════════════════════════════════════════════════════════════════════
#  EXECUTION DELAY TEST
# ═══════════════════════════════════════════════════════════════════════

def delay_test(nq_1min, strategy_fn, params):
    """Run at 0, 1 bar delay."""
    results = []
    for delay in [0, 1]:
        trades = strategy_fn(nq_1min, params, delay_bars=delay)
        stats = compute_stats(trades)
        results.append({"delay": delay, "stats": stats})
    return results


# ═══════════════════════════════════════════════════════════════════════
#  YEARLY BREAKDOWN
# ═══════════════════════════════════════════════════════════════════════

def yearly_breakdown(trades_df):
    """PF, PnL, DD, N per year."""
    if len(trades_df) == 0:
        return []
    results = []
    for yr in sorted(trades_df["year"].unique()):
        yt = trades_df[trades_df["year"] == yr]
        stats = compute_stats(yt)
        results.append({"year": yr, **stats})
    return results


# ═══════════════════════════════════════════════════════════════════════
#  FULL VALIDATION PIPELINE
# ═══════════════════════════════════════════════════════════════════════

def validate(strategy_fn, params, name="Strategy",
             nq_data=None, sensitivity_keys=None):
    """
    Run the complete validation pipeline:
    1. IS (2022-2023), OOS (2024-2026), 4Y (2022-2026)
    2. Walk-forward (train 12m, test 6m, roll 6m)
    3. Parameter sensitivity (+/-20%)
    4. Cost stress (1.5x, 2x)
    5. 1-bar execution delay
    6. Yearly breakdown

    Returns a dict with all results.
    """
    if nq_data is None:
        nq_data = load_nq()

    nq_is = nq_data[(nq_data.index >= "2022-01-01") & (nq_data.index < "2024-01-01")]
    nq_oos = nq_data[nq_data.index >= "2024-01-01"]
    nq_4y = nq_data[nq_data.index >= "2022-01-01"]

    print(f"\n{'=' * 80}")
    print(f"  VALIDATING: {name}")
    print(f"{'=' * 80}")
    print(f"  Data: {nq_data.index[0].date()} to {nq_data.index[-1].date()}")
    print(f"  IS: 2022-01 to 2023-12  |  OOS: 2024-01 to 2026-03")

    # 1. IS / OOS / 4Y
    print(f"\n  --- Period Results ---")
    is_trades = strategy_fn(nq_is, params)
    oos_trades = strategy_fn(nq_oos, params)
    full_trades = strategy_fn(nq_4y, params)

    is_stats = compute_stats(is_trades)
    oos_stats = compute_stats(oos_trades)
    full_stats = compute_stats(full_trades)

    _print_stats_row("IS 22-23", is_stats)
    _print_stats_row("OOS 24-26", oos_stats)
    _print_stats_row("Full 4Y", full_stats)

    # Decay ratio
    if is_stats["pf"] > 0:
        decay = (1 - oos_stats["pf"] / is_stats["pf"]) * 100
        print(f"  OOS/IS PF decay: {decay:+.1f}%")

    # 2. Walk-forward
    print(f"\n  --- Walk-Forward (train 12m, test 6m, roll 6m) ---")
    wf = walk_forward(nq_4y, strategy_fn, params)
    profitable = 0
    for r in wf:
        st = r["stats"]
        win = "+" if st["pnl"] > 0 else "-"
        if st["pnl"] > 0:
            profitable += 1
        print(f"  {win} {r['label']:>22}  PF={st['pf']:.3f}  PnL=${st['pnl']:>+8,.0f}  "
              f"DD=${st['dd']:>6,.0f}  N={st['n']:>4}  Sharpe={st['sharpe']:>5.2f}")

    wf_stats = wf_aggregate_stats(wf)
    print(f"  WF Aggregate: PF={wf_stats['pf']:.3f}  PnL=${wf_stats['pnl']:>+8,.0f}  "
          f"DD=${wf_stats['dd']:>6,.0f}  N={wf_stats['n']:>4}")
    print(f"  Profitable periods: {profitable}/{len(wf)}")

    # 3. Parameter sensitivity
    print(f"\n  --- Parameter Sensitivity (+/-20%) on OOS ---")
    sens, _ = sensitivity_test(nq_oos, strategy_fn, params, keys=sensitivity_keys)
    base_pf = sens[0]["pf"]
    print(f"  {'Param':>20} {'Dir':>4} {'Value':>8} {'PF':>7} {'DD':>7} {'PnL':>9} {'N':>5} {'Sharpe':>7}")
    for row in sens:
        delta_pf = row["pf"] - base_pf if row["param"] != "BASE" else 0
        flag = " !" if abs(delta_pf) > 0.15 else ""
        print(f"  {row['param']:>20} {row['direction']:>4} {str(row['value']):>8} "
              f"{row['pf']:>7.3f} {row['dd']:>7.0f} {row['pnl']:>+9,.0f} {row['n']:>5} "
              f"{row['sharpe']:>7.2f}{flag}")

    pf_range = max(r["pf"] for r in sens) - min(r["pf"] for r in sens)
    print(f"  PF range across perturbations: {pf_range:.3f}")
    robust = pf_range < 0.3
    print(f"  Sensitivity verdict: {'ROBUST' if robust else 'FRAGILE'}")

    # 4. Cost stress
    print(f"\n  --- Cost Stress Test on OOS ---")
    cost_res = cost_stress_test(nq_oos, strategy_fn, params)
    for cr in cost_res:
        st = cr["stats"]
        print(f"  Cost {cr['cost_mult']:.1f}x: PF={st['pf']:.3f}  PnL=${st['pnl']:>+8,.0f}  "
              f"DD=${st['dd']:>6,.0f}  Sharpe={st['sharpe']:.2f}")
    survives_2x = cost_res[-1]["stats"]["pf"] > 1.0
    print(f"  Survives 2x costs: {'YES' if survives_2x else 'NO'}")

    # 5. Execution delay
    print(f"\n  --- Execution Delay Test on OOS ---")
    delay_res = delay_test(nq_oos, strategy_fn, params)
    for dr in delay_res:
        st = dr["stats"]
        print(f"  Delay {dr['delay']} bar: PF={st['pf']:.3f}  PnL=${st['pnl']:>+8,.0f}  "
              f"DD=${st['dd']:>6,.0f}  N={st['n']:>4}")
    if len(delay_res) >= 2 and delay_res[0]["stats"]["pf"] > 0:
        delay_decay = (1 - delay_res[1]["stats"]["pf"] / delay_res[0]["stats"]["pf"]) * 100
        print(f"  PF decay with 1-bar delay: {delay_decay:+.1f}%")

    # 6. Yearly breakdown
    print(f"\n  --- Yearly Breakdown (full 4Y) ---")
    yearly = yearly_breakdown(full_trades)
    print(f"  {'Year':>6} {'PF':>7} {'PnL':>9} {'DD':>7} {'N':>5} {'Sharpe':>7} {'WR%':>5} {'5R+':>4} {'MCL':>4}")
    for yr in yearly:
        print(f"  {yr['year']:>6} {yr['pf']:>7.3f} {yr['pnl']:>+9,.0f} {yr['dd']:>7,.0f} "
              f"{yr['n']:>5} {yr['sharpe']:>7.2f} {yr['wr']:>5.1f} {yr['b5']:>4} {yr['mcl']:>4}")

    return {
        "name": name,
        "params": params,
        "is_stats": is_stats,
        "oos_stats": oos_stats,
        "full_stats": full_stats,
        "wf_results": wf,
        "wf_stats": wf_stats,
        "wf_profitable": profitable,
        "wf_total": len(wf),
        "sensitivity": sens,
        "sensitivity_robust": robust,
        "cost_stress": cost_res,
        "survives_2x": survives_2x,
        "delay_results": delay_res,
        "yearly": yearly,
        "is_trades": is_trades,
        "oos_trades": oos_trades,
        "full_trades": full_trades,
    }


def _print_stats_row(label, s):
    print(f"  {label:>12}: PF={s['pf']:.3f}  PnL=${s['pnl']:>+9,.0f}  DD=${s['dd']:>6,.0f}  "
          f"N={s['n']:>4}  Sharpe={s['sharpe']:.2f}  Sortino={s['sortino']:.2f}  "
          f"WR={s['wr']:.1f}%  5R+={s['b5']}  MCL={s['mcl']}")


# ═══════════════════════════════════════════════════════════════════════
#  COMPARISON TABLE
# ═══════════════════════════════════════════════════════════════════════

def compare_strategies(results_list):
    """Print side-by-side comparison of multiple validated strategies."""
    print(f"\n{'=' * 100}")
    print(f"  STRATEGY COMPARISON")
    print(f"{'=' * 100}")

    header = f"  {'Metric':>20}"
    for r in results_list:
        header += f" | {r['name']:>14}"
    print(header)
    print(f"  {'-' * (22 + 17 * len(results_list))}")

    metrics = [
        ("IS PF", lambda r: f"{r['is_stats']['pf']:.3f}"),
        ("OOS PF", lambda r: f"{r['oos_stats']['pf']:.3f}"),
        ("4Y PF", lambda r: f"{r['full_stats']['pf']:.3f}"),
        ("WF PF", lambda r: f"{r['wf_stats']['pf']:.3f}"),
        ("IS PnL", lambda r: f"${r['is_stats']['pnl']:>+,.0f}"),
        ("OOS PnL", lambda r: f"${r['oos_stats']['pnl']:>+,.0f}"),
        ("4Y PnL", lambda r: f"${r['full_stats']['pnl']:>+,.0f}"),
        ("IS MaxDD", lambda r: f"${r['is_stats']['dd']:>,.0f}"),
        ("OOS MaxDD", lambda r: f"${r['oos_stats']['dd']:>,.0f}"),
        ("IS Sharpe", lambda r: f"{r['is_stats']['sharpe']:.2f}"),
        ("OOS Sharpe", lambda r: f"{r['oos_stats']['sharpe']:.2f}"),
        ("OOS Sortino", lambda r: f"{r['oos_stats']['sortino']:.2f}"),
        ("OOS Calmar", lambda r: f"{r['oos_stats']['calmar']:.2f}"),
        ("OOS WR%", lambda r: f"{r['oos_stats']['wr']:.1f}"),
        ("OOS 5R+ trades", lambda r: f"{r['oos_stats']['b5']}"),
        ("OOS MCL", lambda r: f"{r['oos_stats']['mcl']}"),
        ("OOS WorstDay", lambda r: f"${r['oos_stats']['worst_day']:>,.0f}"),
        ("OOS N trades", lambda r: f"{r['oos_stats']['n']}"),
        ("WF win periods", lambda r: f"{r['wf_profitable']}/{r['wf_total']}"),
        ("Sens. robust", lambda r: "YES" if r["sensitivity_robust"] else "NO"),
        ("Surv. 2x cost", lambda r: "YES" if r["survives_2x"] else "NO"),
        ("1-bar delay PF", lambda r: f"{r['delay_results'][1]['stats']['pf']:.3f}" if len(r['delay_results']) >= 2 else "N/A"),
    ]

    for metric_name, extractor in metrics:
        row = f"  {metric_name:>20}"
        for r in results_list:
            row += f" | {extractor(r):>14}"
        print(row)


# ═══════════════════════════════════════════════════════════════════════
#  MARKDOWN REPORT GENERATION
# ═══════════════════════════════════════════════════════════════════════

def generate_report_md(results_list, output_path="docs/BLUE_VALIDATION.md"):
    """Generate markdown validation report."""
    lines = ["# Blue Team Validation Report\n"]
    lines.append(f"Generated: {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    lines.append("## Cost Model\n")
    lines.append("All costs PER CONTRACT (NQ):\n")
    lines.append(f"- Commission: ${COMM_RT:.2f} RT\n")
    lines.append(f"- Spread: ${SPREAD:.2f}\n")
    lines.append(f"- Stop slippage: ${SLIP_STOP:.2f}\n")
    lines.append(f"- BE slippage: ${SLIP_BE:.2f}\n")
    lines.append(f"- Point value: ${NQ_PT_VAL:.0f}\n")
    lines.append(f"- Gap-through stop fills: YES\n")
    lines.append("\n## Strategy Comparison\n")

    # Main comparison table
    lines.append("| Metric |")
    for r in results_list:
        lines.append(f" {r['name']} |")
    lines.append("\n|--------|")
    for _ in results_list:
        lines.append("--------|")
    lines.append("\n")

    metrics = [
        ("IS PF", lambda r: f"{r['is_stats']['pf']:.3f}"),
        ("OOS PF", lambda r: f"{r['oos_stats']['pf']:.3f}"),
        ("4Y PF", lambda r: f"{r['full_stats']['pf']:.3f}"),
        ("WF Agg PF", lambda r: f"{r['wf_stats']['pf']:.3f}"),
        ("IS PnL", lambda r: f"${r['is_stats']['pnl']:>+,.0f}"),
        ("OOS PnL", lambda r: f"${r['oos_stats']['pnl']:>+,.0f}"),
        ("IS MaxDD", lambda r: f"${r['is_stats']['dd']:>,.0f}"),
        ("OOS MaxDD", lambda r: f"${r['oos_stats']['dd']:>,.0f}"),
        ("IS Sharpe", lambda r: f"{r['is_stats']['sharpe']:.2f}"),
        ("OOS Sharpe", lambda r: f"{r['oos_stats']['sharpe']:.2f}"),
        ("OOS Sortino", lambda r: f"{r['oos_stats']['sortino']:.2f}"),
        ("OOS Calmar", lambda r: f"{r['oos_stats']['calmar']:.2f}"),
        ("OOS WR%", lambda r: f"{r['oos_stats']['wr']:.1f}%"),
        ("OOS 5R+", lambda r: f"{r['oos_stats']['b5']}"),
        ("OOS MCL", lambda r: f"{r['oos_stats']['mcl']}"),
        ("OOS Worst Day", lambda r: f"${r['oos_stats']['worst_day']:>,.0f}"),
        ("OOS N", lambda r: f"{r['oos_stats']['n']}"),
        ("WF Profitable", lambda r: f"{r['wf_profitable']}/{r['wf_total']}"),
        ("Param Robust", lambda r: "YES" if r["sensitivity_robust"] else "NO"),
        ("Survives 2x Cost", lambda r: "YES" if r["survives_2x"] else "NO"),
        ("1-bar Delay PF", lambda r: f"{r['delay_results'][1]['stats']['pf']:.3f}" if len(r['delay_results']) >= 2 else "N/A"),
    ]

    for metric_name, extractor in metrics:
        lines.append(f"| {metric_name} |")
        for r in results_list:
            lines.append(f" {extractor(r)} |")
        lines.append("\n")

    # Walk-forward details per strategy
    for r in results_list:
        lines.append(f"\n## {r['name']} - Walk-Forward Detail\n\n")
        lines.append("| Period | PF | PnL | DD | N | Sharpe |\n")
        lines.append("|--------|-----|------|-----|---|--------|\n")
        for wr in r["wf_results"]:
            st = wr["stats"]
            lines.append(f"| {wr['label']} | {st['pf']:.3f} | ${st['pnl']:>+,.0f} | "
                         f"${st['dd']:>,.0f} | {st['n']} | {st['sharpe']:.2f} |\n")
        lines.append(f"| **Aggregate** | **{r['wf_stats']['pf']:.3f}** | "
                     f"**${r['wf_stats']['pnl']:>+,.0f}** | **${r['wf_stats']['dd']:>,.0f}** | "
                     f"**{r['wf_stats']['n']}** | **{r['wf_stats']['sharpe']:.2f}** |\n")

    # Yearly breakdown per strategy
    for r in results_list:
        lines.append(f"\n## {r['name']} - Yearly Breakdown\n\n")
        lines.append("| Year | PF | PnL | DD | N | Sharpe | WR% | 5R+ | MCL |\n")
        lines.append("|------|-----|------|-----|---|--------|-----|-----|-----|\n")
        for yr in r["yearly"]:
            lines.append(f"| {yr['year']} | {yr['pf']:.3f} | ${yr['pnl']:>+,.0f} | "
                         f"${yr['dd']:>,.0f} | {yr['n']} | {yr['sharpe']:.2f} | "
                         f"{yr['wr']:.1f}% | {yr['b5']} | {yr['mcl']} |\n")

    # Cost stress per strategy
    for r in results_list:
        lines.append(f"\n## {r['name']} - Cost Stress (OOS)\n\n")
        lines.append("| Cost Mult | PF | PnL | DD | Sharpe |\n")
        lines.append("|-----------|-----|------|-----|--------|\n")
        for cr in r["cost_stress"]:
            st = cr["stats"]
            lines.append(f"| {cr['cost_mult']:.1f}x | {st['pf']:.3f} | ${st['pnl']:>+,.0f} | "
                         f"${st['dd']:>,.0f} | {st['sharpe']:.2f} |\n")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.writelines(lines)
    print(f"\n  Report written to {output_path}")


# ═══════════════════════════════════════════════════════════════════════
#  MAIN: validate baseline EMA20 touch
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Loading NQ data...")
    nq = load_nq()
    print(f"  Loaded: {nq.index[0].date()} to {nq.index[-1].date()} ({len(nq):,} bars)")

    # Validate baseline
    baseline_result = validate(
        strategy_fn=run_baseline,
        params=BASELINE_PARAMS,
        name="EMA20-Touch-v11",
        nq_data=nq,
        sensitivity_keys=SENSITIVITY_KEYS,
    )

    # Check for blue strategies
    import glob as glob_mod
    blue_files = sorted(glob_mod.glob("src/blue_strategy_*.py"))
    blue_results = [baseline_result]

    if blue_files:
        print(f"\n  Found {len(blue_files)} blue strategy file(s)")
        for bf in blue_files:
            mod_name = os.path.splitext(os.path.basename(bf))[0]
            print(f"\n  Loading {mod_name}...")
            import importlib
            spec = importlib.util.spec_from_file_location(mod_name, bf)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

            # Expect each blue strategy to expose:
            #   run_strategy(df_1min, params=None, cost_mult=1.0, delay_bars=0) -> trades DataFrame
            #   PARAMS dict
            #   NAME string
            #   SENSITIVITY_KEYS list (optional)
            if hasattr(mod, "run_strategy") and hasattr(mod, "PARAMS"):
                strat_name = getattr(mod, "NAME", mod_name)
                strat_keys = getattr(mod, "SENSITIVITY_KEYS", None)
                result = validate(
                    strategy_fn=mod.run_strategy,
                    params=mod.PARAMS,
                    name=strat_name,
                    nq_data=nq,
                    sensitivity_keys=strat_keys,
                )
                blue_results.append(result)
            else:
                print(f"  WARNING: {mod_name} missing run_strategy() or PARAMS, skipping")
    else:
        print("\n  No blue_strategy_*.py files found yet. Only baseline validated.")

    # Comparison
    if len(blue_results) > 1:
        compare_strategies(blue_results)

    # Generate report
    generate_report_md(blue_results)
