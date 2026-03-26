"""
Realistic MNQ Backtest — Real NQ data, dynamic sizing, proper IS/OOS.

Fixes from audit:
  1. IS = 2022-01 to 2023-12 (2Y), OOS = 2024-01 to 2026-03 (2Y+)
     → OOS is FORWARD, no snooping
  2. Dynamic position sizing: fixed $ risk per trade, calculate contracts
     - Risk per trade = account_risk_pct × equity
     - Contracts = floor(risk_$ / (stop_distance_nq × MNQ_PER_POINT))
     - Capped by Topstep max (10 MNQ) and min (1 MNQ)
  3. Only test gate_tighten=0.0 (proven) vs v8 baseline
  4. Trailing equity tracking for prop firm DD check
"""
from __future__ import annotations
import functools, datetime as dt
import numpy as np, pandas as pd

print = functools.partial(print, flush=True)

NQ_PATH = "data/barchart_nq/NQ_1min_continuous_RTH.csv"
MNQ_PER_POINT = 2.0
COMM_RT = 2.46; SPREAD = 0.50; STOP_SLIP = 1.00; BE_SLIP = 1.00

# Topstep 50K constraints
STARTING_EQUITY = 50_000
MAX_TRAILING_DD = 2_000
PROFIT_TARGET = 3_000
MAX_CONTRACTS = 10   # Topstep 50K max MNQ


def resample(df, m):
    if m <= 1: return df
    return df.resample(f"{m}min").agg(
        {"Open": "first", "High": "max", "Low": "min",
         "Close": "last", "Volume": "sum"}).dropna()


def add_ind(df, s):
    df = df.copy()
    df["ema_f"] = df["Close"].ewm(span=s["ema_fast"], adjust=False).mean()
    df["ema_s"] = df["Close"].ewm(span=s["ema_slow"], adjust=False).mean()
    tr = np.maximum(df["High"] - df["Low"],
                    np.maximum((df["High"] - df["Close"].shift(1)).abs(),
                               (df["Low"] - df["Close"].shift(1)).abs()))
    df["atr"] = tr.rolling(s["atr_period"]).mean()
    return df


def run_realistic(df_1min, s, risk_per_trade=None, risk_pct=None):
    """
    Run with dynamic position sizing.
    risk_per_trade: fixed $ risk (e.g., $100)
    risk_pct: % of current equity (e.g., 0.01 = 1%)
    If both None, uses fixed n_contracts from params.
    """
    df = resample(df_1min, s["tf_minutes"])
    df = add_ind(df, s)
    H = df["High"].values; L = df["Low"].values; C = df["Close"].values
    ef = df["ema_f"].values; es = df["ema_s"].values; atr = df["atr"].values
    T = df.index.time; D = df.index.date; n = len(df)
    tf = max(1, s["tf_minutes"])
    mh = max(20, s["max_hold_bars"] // tf)
    cb = max(5, s["chand_bars"] // tf)
    gb = max(1, s["gate_bars"] // tf) if s["gate_bars"] > 0 else 0

    equity = STARTING_EQUITY
    peak_equity = equity
    max_trail_dd = 0.0
    cum_pnl = 0.0

    trades = []; bar = max(s["ema_slow"], s["atr_period"]) + 5
    dlr = 0.0; cd = None; sk = 0
    target_hit = False; target_day = None; busted = False

    while bar < n - mh - 5:
        a = atr[bar]
        if np.isnan(a) or a <= 0 or np.isnan(ef[bar]) or np.isnan(es[bar]):
            bar += 1; continue
        if T[bar] >= s["no_entry_after"]: bar += 1; continue
        d = D[bar]
        if cd != d: cd = d; dlr = 0.0
        if dlr >= s["daily_loss_r"]: bar += 1; continue

        # Check if busted (trailing DD exceeded)
        trail_dd = peak_equity - equity
        if trail_dd >= MAX_TRAILING_DD:
            busted = True; break

        c = C[bar]
        if c > ef[bar] and ef[bar] > es[bar]: tr = 1
        elif c < ef[bar] and ef[bar] < es[bar]: tr = -1
        else: bar += 1; continue

        tol = a * s["touch_tol"]
        if tr == 1: touch = L[bar] <= ef[bar] + tol and L[bar] >= ef[bar] - a * s["touch_below_max"]
        else: touch = H[bar] >= ef[bar] - tol and H[bar] <= ef[bar] + a * s["touch_below_max"]
        if not touch: bar += 1; continue
        if sk > 0: sk -= 1; bar += 1; continue

        entry = C[bar]
        stop = L[bar] - s["stop_buffer"] * a if tr == 1 else H[bar] + s["stop_buffer"] * a
        rp = abs(entry - stop)  # risk in NQ points
        if rp <= 0: bar += 1; continue

        # ─── DYNAMIC POSITION SIZING ───
        risk_per_contract = rp * MNQ_PER_POINT  # $ risk per MNQ contract

        if risk_per_trade is not None:
            nc = max(1, min(MAX_CONTRACTS, int(risk_per_trade / risk_per_contract)))
        elif risk_pct is not None:
            target_risk = equity * risk_pct
            nc = max(1, min(MAX_CONTRACTS, int(target_risk / risk_per_contract)))
        else:
            nc = s["n_contracts"]

        rm = rp * MNQ_PER_POINT * nc
        ec = COMM_RT * nc / 2 + SPREAD

        eb = bar; rs = stop; bt = False; mfe = 0.0; r = 0.0; endb = bar; ex = "timeout"
        for k in range(1, mh + 1):
            bi = eb + k
            if bi >= n: break
            h = H[bi]; l = L[bi]; ca = atr[bi] if not np.isnan(atr[bi]) else a
            if tr == 1: mfe = max(mfe, (h - entry) / rp)
            else: mfe = max(mfe, (entry - l) / rp)
            if T[bi] >= s["force_close_at"]:
                r = (C[bi] - entry) / rp * tr; endb = bi; ex = "close"; break
            if gb > 0 and k == gb and not bt:
                if mfe < s["gate_mfe"]:
                    ns = entry + s["gate_tighten"] * rp * tr
                    if tr == 1: rs = max(rs, ns)
                    else: rs = min(rs, ns)
            st = (tr == 1 and l <= rs) or (tr == -1 and h >= rs)
            if st:
                r = (rs - entry) / rp * tr; endb = bi
                if bt:
                    ref = entry + s["be_stop_r"] * rp * tr
                    ex = "be" if abs(rs - ref) < 0.05 * rp else "trail"
                else: ex = "stop"
                break
            if not bt and s["be_trigger_r"] > 0:
                tp = entry + s["be_trigger_r"] * rp * tr
                if (tr == 1 and h >= tp) or (tr == -1 and l <= tp):
                    bt = True; bl = entry + s["be_stop_r"] * rp * tr
                    if tr == 1: rs = max(rs, bl)
                    else: rs = min(rs, bl)
            if bt and k >= cb:
                skk = max(1, k - cb + 1)
                hv = [H[eb + kk] for kk in range(skk, k) if eb + kk < n]
                lv = [L[eb + kk] for kk in range(skk, k) if eb + kk < n]
                if hv and lv:
                    if tr == 1: rs = max(rs, max(hv) - s["chand_mult"] * ca)
                    else: rs = min(rs, min(lv) + s["chand_mult"] * ca)
        else:
            r = (C[min(eb + mh, n-1)] - entry) / rp * tr; endb = min(eb + mh, n-1)

        raw = r * rm
        xc = COMM_RT * nc / 2
        xs = STOP_SLIP if ex in ("stop", "trail") else 0
        bs = BE_SLIP if ex == "be" else 0
        net = raw - (ec + xc + xs + bs)

        equity += net; cum_pnl += net
        peak_equity = max(peak_equity, equity)
        trail_dd = peak_equity - equity
        max_trail_dd = max(max_trail_dd, trail_dd)

        # Check profit target
        if not target_hit and cum_pnl >= PROFIT_TARGET:
            target_hit = True; target_day = d

        trades.append({
            "pnl": net, "r": r, "ex": ex, "nc": nc,
            "risk_per_c": round(risk_per_contract, 2),
            "risk_total": round(rm, 2),
            "equity": round(equity, 0),
            "trail_dd": round(trail_dd, 0),
            "date": str(d), "year": d.year,
        })
        if r < 0: dlr += abs(r)
        if r > 0: sk = s.get("skip_after_win", 0)
        bar = endb + 1

    tdf = pd.DataFrame(trades) if trades else pd.DataFrame()
    if len(tdf) == 0:
        return {"pf": 0, "pnl": 0, "dd": 0, "n": 0}

    gw = tdf.loc[tdf["pnl"] > 0, "pnl"].sum()
    gl = abs(tdf.loc[tdf["pnl"] <= 0, "pnl"].sum())
    days = len(tdf["date"].unique())

    # Yearly breakdown
    yearly = {}
    for yr, grp in tdf.groupby("year"):
        yw = grp.loc[grp["pnl"] > 0, "pnl"].sum()
        yl = abs(grp.loc[grp["pnl"] <= 0, "pnl"].sum())
        yearly[yr] = {
            "pf": round(yw / yl, 3) if yl > 0 else 0,
            "pnl": round(grp["pnl"].sum(), 0),
            "n": len(grp),
            "avg_nc": round(grp["nc"].mean(), 1),
            "avg_risk": round(grp["risk_total"].mean(), 1),
        }

    return {
        "pf": round(gw / gl if gl > 0 else 0, 3),
        "pnl": round(cum_pnl, 0),
        "dd": round(max_trail_dd, 0),
        "n": len(tdf),
        "dpnl": round(cum_pnl / max(days, 1), 1),
        "b5": int((tdf["r"] >= 5).sum()),
        "avg_nc": round(tdf["nc"].mean(), 1),
        "avg_risk": round(tdf["risk_total"].mean(), 1),
        "max_nc": int(tdf["nc"].max()),
        "target_hit": target_hit,
        "target_day": str(target_day) if target_day else "N/A",
        "busted": busted,
        "final_equity": round(equity, 0),
        "yearly": yearly,
        "tdf": tdf,
    }


def main():
    print("=" * 80)
    print("REALISTIC MNQ BACKTEST — Dynamic Sizing, Topstep 50K")
    print("=" * 80)

    nq = pd.read_csv(NQ_PATH, parse_dates=["Time"], index_col="Time")
    nq.index.name = "timestamp"
    nq.index = nq.index + pd.Timedelta(hours=1)

    # IS: 2022-2023 (2Y), OOS: 2024-2026 (2Y+ forward walk)
    nq_is = nq[(nq.index >= "2022-01-01") & (nq.index < "2024-01-01")]
    nq_oos = nq[nq.index >= "2024-01-01"]
    print(f"  IS:  {nq_is.index[0].date()} → {nq_is.index[-1].date()} ({len(nq_is):,} bars)")
    print(f"  OOS: {nq_oos.index[0].date()} → {nq_oos.index[-1].date()} ({len(nq_oos):,} bars)")

    # Show typical risk per contract
    print(f"\n  ─── NQ ATR analysis ───")
    df3 = resample(nq_oos, 3)
    df3 = add_ind(df3, {"ema_fast": 20, "ema_slow": 50, "atr_period": 14})
    atr_vals = df3["atr"].dropna()
    print(f"  3-min ATR: mean={atr_vals.mean():.1f} median={atr_vals.median():.1f} "
          f"P10={atr_vals.quantile(0.1):.1f} P90={atr_vals.quantile(0.9):.1f}")
    stop_dist = atr_vals * 0.4
    risk_per_c = stop_dist * MNQ_PER_POINT
    print(f"  Stop distance: mean={stop_dist.mean():.1f} pts → ${risk_per_c.mean():.1f}/contract")
    print(f"  With $100 risk: {100/risk_per_c.mean():.1f} contracts avg")
    print(f"  With $150 risk: {150/risk_per_c.mean():.1f} contracts avg")
    print(f"  With $200 risk: {200/risk_per_c.mean():.1f} contracts avg")

    # Strategy configs
    V8 = {
        "tf_minutes": 3, "ema_fast": 20, "ema_slow": 50, "atr_period": 14,
        "touch_tol": 0.15, "touch_below_max": 0.5, "no_entry_after": dt.time(14, 0),
        "stop_buffer": 0.4, "gate_bars": 3, "gate_mfe": 0.2, "gate_tighten": -0.1,
        "be_trigger_r": 0.25, "be_stop_r": 0.15, "chand_bars": 25, "chand_mult": 0.3,
        "max_hold_bars": 180, "force_close_at": dt.time(15, 58),
        "daily_loss_r": 2.0, "skip_after_win": 1, "n_contracts": 2,
    }
    V8_GATE0 = {**V8, "gate_tighten": 0.0}  # only the one proven change

    # ═══ 1. Fixed risk per trade — sweep ═══
    print(f"\n{'='*80}")
    print("1. FIXED $ RISK PER TRADE (dynamic contracts)")
    print(f"{'='*80}")

    configs = [
        ("V8 gate=-0.1", V8),
        ("V8 gate=0.0", V8_GATE0),
    ]

    for risk_dollar in [50, 100, 150, 200, 300]:
        print(f"\n  ─── Risk = ${risk_dollar}/trade ───")
        print(f"  {'Config':>20} {'IS_PF':>7} {'IS_DD':>7} {'IS_$/d':>8} {'IS_nc':>6} "
              f"{'OOS_PF':>7} {'OOS_DD':>7} {'OOS_$/d':>8} {'OOS_nc':>6} {'5R+':>4} {'Target':>10}")

        for name, s in configs:
            ri = run_realistic(nq_is, s, risk_per_trade=risk_dollar)
            ro = run_realistic(nq_oos, s, risk_per_trade=risk_dollar)
            target_str = f"Day {ro['target_day'][:10]}" if ro["target_hit"] else "N/A"
            bust_str = " BUST!" if ro["busted"] else ""
            print(f"  {name:>20} {ri['pf']:>7.3f} {ri['dd']:>7.0f} {ri['dpnl']:>+8.1f} {ri['avg_nc']:>6.1f} "
                  f"{ro['pf']:>7.3f} {ro['dd']:>7.0f} {ro['dpnl']:>+8.1f} {ro['avg_nc']:>6.1f} {ro['b5']:>4} {target_str:>10}{bust_str}")

    # ═══ 2. % of equity risk ═══
    print(f"\n{'='*80}")
    print("2. % OF EQUITY RISK (compounds with wins)")
    print(f"{'='*80}")

    for risk_pct in [0.002, 0.003, 0.005, 0.01]:
        print(f"\n  ─── Risk = {risk_pct*100:.1f}% of equity ───")
        print(f"  {'Config':>20} {'IS_PF':>7} {'IS_DD':>7} {'IS_$/d':>8} {'IS_nc':>6} "
              f"{'OOS_PF':>7} {'OOS_DD':>7} {'OOS_$/d':>8} {'OOS_nc':>6} {'Final$':>8}")

        for name, s in configs:
            ri = run_realistic(nq_is, s, risk_pct=risk_pct)
            ro = run_realistic(nq_oos, s, risk_pct=risk_pct)
            bust_str = " BUST!" if ro["busted"] else ""
            print(f"  {name:>20} {ri['pf']:>7.3f} {ri['dd']:>7.0f} {ri['dpnl']:>+8.1f} {ri['avg_nc']:>6.1f} "
                  f"{ro['pf']:>7.3f} {ro['dd']:>7.0f} {ro['dpnl']:>+8.1f} {ro['avg_nc']:>6.1f} ${ro['final_equity']:>7,.0f}{bust_str}")

    # ═══ 3. Best config detail ═══
    print(f"\n{'='*80}")
    print("3. DETAILED RESULTS — Best realistic config")
    print(f"{'='*80}")

    # Run with $150 risk (should give ~5-8 contracts on average)
    for name, s in configs:
        ro = run_realistic(nq_oos, s, risk_per_trade=150)
        print(f"\n  [{name}] $150 risk/trade, OOS (2024-2026)")
        print(f"    PF={ro['pf']:.3f} PnL=${ro['pnl']:+,.0f} DD=${ro['dd']:.0f} N={ro['n']}")
        print(f"    $/day={ro['dpnl']:+.1f} 5R+={ro['b5']} Avg NC={ro['avg_nc']:.1f} Max NC={ro['max_nc']}")
        print(f"    Final equity: ${ro['final_equity']:,.0f}")
        print(f"    Target hit: {ro['target_day'] if ro['target_hit'] else 'N/A'}")
        print(f"    Busted: {ro['busted']}")
        print(f"\n    Yearly:")
        print(f"    {'Year':>6} {'PF':>7} {'PnL':>10} {'N':>6} {'Avg NC':>7} {'Avg Risk$':>9}")
        for yr, yd in sorted(ro["yearly"].items()):
            print(f"    {yr:>6} {yd['pf']:>7.3f} {yd['pnl']:>+10,.0f} {yd['n']:>6} {yd['avg_nc']:>7.1f} ${yd['avg_risk']:>8.1f}")

    # ═══ 4. Topstep 50K simulation ═══
    print(f"\n{'='*80}")
    print("4. TOPSTEP 50K SIMULATION (with trailing DD check)")
    print(f"{'='*80}")

    for name, s in configs:
        for risk_d in [100, 150, 200]:
            ro = run_realistic(nq_oos, s, risk_per_trade=risk_d)
            dd_pct = ro["dd"] / MAX_TRAILING_DD * 100
            days_to_target = "N/A"
            if ro["target_hit"]:
                # Count trading days to target
                tdf = ro["tdf"]
                cum = tdf["pnl"].cumsum()
                target_idx = (cum >= PROFIT_TARGET).idxmax()
                days_to_target = len(tdf.loc[:target_idx, "date"].unique())

            status = "✅" if (not ro["busted"] and ro["target_hit"]) else "❌ BUST" if ro["busted"] else "⏳"
            print(f"  {name:>16} ${risk_d:>3} risk | PF={ro['pf']:.3f} DD=${ro['dd']:>5.0f} "
                  f"({dd_pct:>4.0f}% of limit) | NC={ro['avg_nc']:.1f} | "
                  f"Target: {days_to_target} days | {status}")


def add_ind(df, s):
    df = df.copy()
    df["ema_f"] = df["Close"].ewm(span=s["ema_fast"], adjust=False).mean()
    df["ema_s"] = df["Close"].ewm(span=s["ema_slow"], adjust=False).mean()
    tr = np.maximum(df["High"] - df["Low"],
                    np.maximum((df["High"] - df["Close"].shift(1)).abs(),
                               (df["Low"] - df["Close"].shift(1)).abs()))
    df["atr"] = tr.rolling(s["atr_period"]).mean()
    return df


def resample(df, m):
    if m <= 1: return df
    return df.resample(f"{m}min").agg(
        {"Open": "first", "High": "max", "Low": "min",
         "Close": "last", "Volume": "sum"}).dropna()


if __name__ == "__main__":
    main()
