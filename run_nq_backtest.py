"""
Run MNQ strategy on REAL NQ continuous 1min data (Panama Canal adjusted).

Key change vs QQQ proxy:
  - Prices are NQ points directly
  - risk_mnq = risk_nq_points * MNQ_PER_POINT * n_contracts
  - No more QQQ_TO_NQ=40 approximation
  - Split: IS=2024-01 to 2026-03, OOS=2022-01 to 2023-12

Also compares: QQQ×40 proxy vs real NQ results.
"""
from __future__ import annotations
import functools, datetime as dt
import numpy as np, pandas as pd

print = functools.partial(print, flush=True)

NQ_PATH = "data/barchart_nq/NQ_1min_continuous_RTH.csv"
QQQ_IS_PATH = "data/QQQ_1Min_Polygon_2y_clean.csv"
QQQ_OOS_PATH = "data/QQQ_1Min_Barchart_2y_2022-2024_clean.csv"

# Strategy params (same as strategy_mnq.py v8 BEST)
S = {
    "tf_minutes": 3,
    "ema_fast": 20, "ema_slow": 50, "atr_period": 14,
    "touch_tol": 0.15, "touch_below_max": 0.5,
    "no_entry_after": dt.time(14, 0),
    "stop_buffer": 0.4,
    "gate_bars": 3, "gate_mfe": 0.2, "gate_tighten": -0.1,
    "be_trigger_r": 0.25, "be_stop_r": 0.15,
    "chand_bars": 25, "chand_mult": 0.3,
    "max_hold_bars": 180,
    "force_close_at": dt.time(15, 58),
    "daily_loss_r": 2.0,
    "skip_after_win": 1,
    "n_contracts": 2,
}

# MNQ cost model (real slippage)
MNQ_PER_POINT = 2.0        # $2 per NQ point per MNQ contract
COMM_PER_CONTRACT_RT = 2.46
SPREAD_PER_TRADE = 0.50
STOP_SLIP = 1.00            # real: mean 1.94 ticks = $0.97 ≈ $1.00
BE_SLIP = 1.00


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


def run_nq(df_1min, s=None, use_qqq_proxy=False):
    """
    Run strategy on NQ data.
    If use_qqq_proxy=True, applies QQQ_TO_NQ=40 conversion (for comparison).
    """
    if s is None:
        s = S
    df = resample(df_1min, s["tf_minutes"])
    df = add_indicators(df, s)

    high = df["High"].values; low = df["Low"].values; close = df["Close"].values
    ema_f = df["ema_f"].values; ema_s = df["ema_s"].values; atr = df["atr"].values
    times = df.index.time; dates = df.index.date; n = len(df)

    tf = max(1, s["tf_minutes"])
    max_hold = max(20, s["max_hold_bars"] // tf)
    chand_b = max(5, s["chand_bars"] // tf)
    gate_b = max(1, s["gate_bars"] // tf) if s["gate_bars"] > 0 else 0
    nc = s["n_contracts"]

    # Price-to-MNQ conversion
    if use_qqq_proxy:
        price_to_mnq = 40 * MNQ_PER_POINT  # QQQ$ → NQ points → MNQ $
    else:
        price_to_mnq = MNQ_PER_POINT  # NQ points → MNQ $

    cum_pnl = 0.0; peak_pnl = 0.0; max_dd = 0.0
    trades = []; bar = max(s["ema_slow"], s["atr_period"]) + 5
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

        entry = close[bar]
        stop = low[bar] - s["stop_buffer"] * a if trend == 1 else high[bar] + s["stop_buffer"] * a
        risk_price = abs(entry - stop)  # in price units (NQ points or QQQ $)
        if risk_price <= 0:
            bar += 1; continue

        risk_mnq = risk_price * price_to_mnq * nc  # in USD
        entry_cost = COMM_PER_CONTRACT_RT * nc / 2 + SPREAD_PER_TRADE

        entry_bar = bar; runner_stop = stop; be_triggered = False
        mfe = 0.0; trade_r = 0.0; end_bar = bar; exit_reason = "timeout"

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
                trade_r = (runner_stop - entry) / risk_price * trend
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
        exit_comm = COMM_PER_CONTRACT_RT * nc / 2
        exit_slip = STOP_SLIP if exit_reason in ("stop", "trail") else 0
        be_slip = BE_SLIP if exit_reason == "be" else 0
        total_cost = entry_cost + exit_comm + exit_slip + be_slip
        net_pnl = raw_pnl - total_cost

        cum_pnl += net_pnl; peak_pnl = max(peak_pnl, cum_pnl); max_dd = max(max_dd, peak_pnl - cum_pnl)
        trades.append({"net_pnl": net_pnl, "raw_r": trade_r, "cost": total_cost,
                        "exit": exit_reason, "risk_$": risk_mnq, "date": str(d)})
        if trade_r < 0: daily_r_loss += abs(trade_r)
        if trade_r > 0: skip_count = s.get("skip_after_win", 0)
        bar = end_bar + 1

    tdf = pd.DataFrame(trades) if trades else pd.DataFrame()
    total = len(tdf); days = len(set(dates))
    if total == 0:
        return {"net_pf": 0, "daily_pnl": 0, "max_dd": 0, "trades": 0, "total_pnl": 0,
                "tpd": 0, "cost_pct": 0, "big5": 0, "exits": {}}
    gw = tdf.loc[tdf["net_pnl"] > 0, "net_pnl"].sum()
    gl = abs(tdf.loc[tdf["net_pnl"] <= 0, "net_pnl"].sum())
    exits = tdf["exit"].value_counts().to_dict()

    # Yearly breakdown
    tdf["year"] = pd.to_datetime(tdf["date"]).dt.year
    yearly = {}
    for yr, grp in tdf.groupby("year"):
        yw = grp.loc[grp["net_pnl"] > 0, "net_pnl"].sum()
        yl = abs(grp.loc[grp["net_pnl"] <= 0, "net_pnl"].sum())
        yearly[yr] = {
            "pf": round(yw / yl, 3) if yl > 0 else 0,
            "pnl": round(grp["net_pnl"].sum(), 0),
            "n": len(grp),
        }

    return {
        "net_pf": round(gw / gl if gl > 0 else 0, 3),
        "daily_pnl": round(cum_pnl / max(days, 1), 1),
        "max_dd": round(max_dd, 0),
        "total_pnl": round(cum_pnl, 0),
        "trades": total,
        "tpd": round(total / max(days, 1), 1),
        "cost_pct": round(tdf["cost"].mean() / tdf["risk_$"].mean() * 100, 1),
        "big5": int((tdf["raw_r"] >= 5).sum()),
        "exits": exits,
        "yearly": yearly,
    }


def main():
    print("=" * 75)
    print("REAL NQ vs QQQ×40 PROXY — MNQ Strategy Backtest")
    print("=" * 75)

    # Load NQ data
    print("\n[1] Loading NQ continuous RTH data...")
    nq = pd.read_csv(NQ_PATH, parse_dates=["Time"], index_col="Time")
    nq.index.name = "timestamp"

    # Convert CT to ET (+1 hour) — Barchart uses CT
    nq.index = nq.index + pd.Timedelta(hours=1)

    print(f"    Range: {nq.index[0]} → {nq.index[-1]}")
    print(f"    Rows: {len(nq):,}")
    print(f"    Days: {len(nq.index.date):,} unique dates")

    # Split NQ: IS=2024+, OOS=2022-2023
    nq_is = nq[nq.index >= "2024-01-01"]
    nq_oos = nq[(nq.index >= "2022-01-01") & (nq.index < "2024-01-01")]
    print(f"    NQ IS: {len(nq_is):,} rows ({nq_is.index[0].date()} → {nq_is.index[-1].date()})")
    print(f"    NQ OOS: {len(nq_oos):,} rows ({nq_oos.index[0].date()} → {nq_oos.index[-1].date()})")

    # Run on NQ data (direct — no QQQ_TO_NQ)
    print("\n[2] Running strategy on REAL NQ data...")
    ri_nq = run_nq(nq_is, use_qqq_proxy=False)
    ro_nq = run_nq(nq_oos, use_qqq_proxy=False)

    print(f"\n  ╔══ REAL NQ RESULTS ══════════════════════════════════════════╗")
    print(f"  ║ IS:  NetPF={ri_nq['net_pf']:.3f}  $/day={ri_nq['daily_pnl']:+.1f}  MaxDD=${ri_nq['max_dd']:.0f}")
    print(f"  ║      Trades={ri_nq['trades']}({ri_nq['tpd']}/d)  5R+={ri_nq['big5']}  Cost={ri_nq['cost_pct']}%")
    print(f"  ║      Exits: {ri_nq['exits']}")
    print(f"  ║ OOS: NetPF={ro_nq['net_pf']:.3f}  $/day={ro_nq['daily_pnl']:+.1f}  MaxDD=${ro_nq['max_dd']:.0f}")
    print(f"  ║      Trades={ro_nq['trades']}({ro_nq['tpd']}/d)  5R+={ro_nq['big5']}  Cost={ro_nq['cost_pct']}%")
    print(f"  ║      Exits: {ro_nq['exits']}")
    print(f"  ╚══════════════════════════════════════════════════════════════╝")

    # Yearly breakdown
    print(f"\n  Yearly breakdown (NQ real):")
    print(f"  {'Year':>6} {'PF':>7} {'PnL':>9} {'Trades':>7}")
    for period_data, label in [(ri_nq, "IS"), (ro_nq, "OOS")]:
        for yr, yd in sorted(period_data["yearly"].items()):
            print(f"  {yr:>6} {yd['pf']:>7.3f} {yd['pnl']:>+9.0f} {yd['n']:>7}")

    # Run on QQQ proxy for comparison
    print("\n[3] Running strategy on QQQ×40 proxy (for comparison)...")
    df_is_qqq = pd.read_csv(QQQ_IS_PATH, index_col="timestamp", parse_dates=True)
    df_oos_qqq = pd.read_csv(QQQ_OOS_PATH, index_col="timestamp", parse_dates=True)
    ri_qqq = run_nq(df_is_qqq, use_qqq_proxy=True)
    ro_qqq = run_nq(df_oos_qqq, use_qqq_proxy=True)

    print(f"\n  QQQ×40: IS PF={ri_qqq['net_pf']:.3f} $/d={ri_qqq['daily_pnl']:+.1f} DD=${ri_qqq['max_dd']:.0f} N={ri_qqq['trades']}")
    print(f"  QQQ×40: OOS PF={ro_qqq['net_pf']:.3f} $/d={ro_qqq['daily_pnl']:+.1f} DD=${ro_qqq['max_dd']:.0f} N={ro_qqq['trades']}")

    # ═══ Comparison table ═══
    print(f"\n  ╔══ COMPARISON: REAL NQ vs QQQ×40 PROXY ═════════════════════╗")
    print(f"  ║ {'':15s} {'NQ Real':>12} {'QQQ×40':>12} {'Delta':>10}  ║")
    print(f"  ╠═══════════════════════════════════════════════════════════════╣")

    for label, nq_r, qqq_r in [("IS", ri_nq, ri_qqq), ("OOS", ro_nq, ro_qqq)]:
        print(f"  ║ {label} PF        {nq_r['net_pf']:>12.3f} {qqq_r['net_pf']:>12.3f} {nq_r['net_pf']-qqq_r['net_pf']:>+10.3f}  ║")
        print(f"  ║ {label} $/day     {nq_r['daily_pnl']:>12.1f} {qqq_r['daily_pnl']:>12.1f} {nq_r['daily_pnl']-qqq_r['daily_pnl']:>+10.1f}  ║")
        print(f"  ║ {label} MaxDD     {nq_r['max_dd']:>12.0f} {qqq_r['max_dd']:>12.0f} {nq_r['max_dd']-qqq_r['max_dd']:>+10.0f}  ║")
        print(f"  ║ {label} Trades    {nq_r['trades']:>12} {qqq_r['trades']:>12} {nq_r['trades']-qqq_r['trades']:>+10}  ║")
        print(f"  ║ {label} 5R+       {nq_r['big5']:>12} {qqq_r['big5']:>12} {nq_r['big5']-qqq_r['big5']:>+10}  ║")
        if label == "IS":
            print(f"  ╠═══════════════════════════════════════════════════════════════╣")

    print(f"  ╚═══════════════════════════════════════════════════════════════╝")

    # Prop firm check
    print(f"\n  ═══ PROP FIRM CHECK (Topstep 50K) ═══")
    dd_ok = "✅ PASS" if ro_nq["max_dd"] <= 2000 else "❌ FAIL"
    pf_ok = "✅ PASS" if ro_nq["net_pf"] > 1.0 else "❌ FAIL"
    days_to_target = round(3000 / ro_nq["daily_pnl"], 0) if ro_nq["daily_pnl"] > 0 else 999
    print(f"  MaxDD: ${ro_nq['max_dd']:.0f} / $2,000 limit → {dd_ok}")
    print(f"  OOS PF: {ro_nq['net_pf']:.3f} → {pf_ok}")
    print(f"  Days to $3,000 target: {days_to_target:.0f} trading days")
    print(f"  Safety margin: ${2000 - ro_nq['max_dd']:.0f}")

    # Full 4-year run
    print(f"\n[4] Running FULL 4-year NQ (2022-2026)...")
    nq_full = nq[nq.index >= "2022-01-01"]
    r_full = run_nq(nq_full, use_qqq_proxy=False)
    print(f"  4Y: PF={r_full['net_pf']:.3f} $/d={r_full['daily_pnl']:+.1f} DD=${r_full['max_dd']:.0f} "
          f"N={r_full['trades']} 5R+={r_full['big5']}")
    print(f"  4Y yearly:")
    for yr, yd in sorted(r_full["yearly"].items()):
        print(f"    {yr}: PF={yd['pf']:.3f} PnL=${yd['pnl']:+,.0f} N={yd['n']}")


if __name__ == "__main__":
    main()
