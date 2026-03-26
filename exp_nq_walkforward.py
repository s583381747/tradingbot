"""
Rolling Walk-Forward — NQ real data.

Method: Train on 12 months, test on next 6 months, roll by 6 months.
No parameter re-optimization — same params throughout (anchored walk-forward).
This tests if the strategy is ROBUST, not if we can keep re-fitting.

Also: combine QQQ IS+OOS for 4Y walk-forward on different data source.
"""
from __future__ import annotations
import functools, datetime as dt
import numpy as np, pandas as pd

print = functools.partial(print, flush=True)

NQ_PATH = "data/barchart_nq/NQ_1min_continuous_RTH.csv"
QQQ_IS = "data/QQQ_1Min_Polygon_2y_clean.csv"
QQQ_OOS = "data/QQQ_1Min_Barchart_2y_2022-2024_clean.csv"

MNQ_PER_POINT = 2.0
COMM_RT = 2.46; SPREAD = 0.50; STOP_SLIP = 1.00; BE_SLIP = 1.00

V8 = {
    "tf_minutes": 3, "ema_fast": 20, "ema_slow": 50, "atr_period": 14,
    "touch_tol": 0.15, "touch_below_max": 0.5, "no_entry_after": dt.time(14, 0),
    "stop_buffer": 0.4, "gate_bars": 3, "gate_mfe": 0.2, "gate_tighten": -0.1,
    "be_trigger_r": 0.25, "be_stop_r": 0.15, "chand_bars": 25, "chand_mult": 0.3,
    "max_hold_bars": 180, "force_close_at": dt.time(15, 58),
    "daily_loss_r": 2.0, "skip_after_win": 1, "n_contracts": 2,
}
V8G0 = {**V8, "gate_tighten": 0.0}


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


def run_period(df_1min, s, use_qqq=False):
    """Run strategy on a period. Returns (pf, pnl, dd, n, b5, daily_pnl)."""
    df = resample(df_1min, s["tf_minutes"])
    df = add_ind(df, s)
    H = df["High"].values; L = df["Low"].values; C = df["Close"].values
    ef = df["ema_f"].values; es = df["ema_s"].values; atr = df["atr"].values
    T = df.index.time; D = df.index.date; n = len(df)
    tf = max(1, s["tf_minutes"])
    mh = max(20, s["max_hold_bars"] // tf)
    cb = max(5, s["chand_bars"] // tf)
    gb = max(1, s["gate_bars"] // tf) if s["gate_bars"] > 0 else 0
    nc = s["n_contracts"]
    p2m = 40 * MNQ_PER_POINT if use_qqq else MNQ_PER_POINT

    cum = 0.0; peak = 0.0; mdd = 0.0
    trades = []; bar = max(s["ema_slow"], s["atr_period"]) + 5
    dlr = 0.0; cd = None; sk = 0

    while bar < n - mh - 5:
        a = atr[bar]
        if np.isnan(a) or a <= 0 or np.isnan(ef[bar]) or np.isnan(es[bar]):
            bar += 1; continue
        if T[bar] >= s["no_entry_after"]: bar += 1; continue
        d = D[bar]
        if cd != d: cd = d; dlr = 0.0
        if dlr >= s["daily_loss_r"]: bar += 1; continue
        c = C[bar]
        if c > ef[bar] and ef[bar] > es[bar]: tr = 1
        elif c < ef[bar] and ef[bar] < es[bar]: tr = -1
        else: bar += 1; continue
        tol = a * s["touch_tol"]
        if tr == 1: touch = L[bar] <= ef[bar] + tol and L[bar] >= ef[bar] - a * s["touch_below_max"]
        else: touch = H[bar] >= ef[bar] - tol and H[bar] <= ef[bar] + a * s["touch_below_max"]
        if not touch: bar += 1; continue
        if sk > 0: sk -= 1; bar += 1; continue

        # Dynamic sizing: $150 risk per trade
        entry = C[bar]
        stop = L[bar] - s["stop_buffer"] * a if tr == 1 else H[bar] + s["stop_buffer"] * a
        rp = abs(entry - stop)
        if rp <= 0: bar += 1; continue
        risk_per_c = rp * p2m
        nc_dyn = max(1, min(10, int(150 / risk_per_c))) if not use_qqq else nc
        rm = rp * p2m * nc_dyn
        ec = COMM_RT * nc_dyn / 2 + SPREAD

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
        xc = COMM_RT * nc_dyn / 2
        xs = STOP_SLIP if ex in ("stop", "trail") else 0
        bs = BE_SLIP if ex == "be" else 0
        net = raw - (ec + xc + xs + bs)
        cum += net; peak = max(peak, cum); mdd = max(mdd, peak - cum)
        trades.append({"pnl": net, "r": r, "nc": nc_dyn})
        if r < 0: dlr += abs(r)
        if r > 0: sk = s.get("skip_after_win", 0)
        bar = endb + 1

    if not trades:
        return {"pf": 0, "pnl": 0, "dd": 0, "n": 0, "b5": 0, "dpnl": 0, "avg_nc": 0}
    tdf = pd.DataFrame(trades)
    gw = tdf.loc[tdf["pnl"] > 0, "pnl"].sum()
    gl = abs(tdf.loc[tdf["pnl"] <= 0, "pnl"].sum())
    days = len(set(D))
    return {
        "pf": round(gw / gl if gl > 0 else 0, 3),
        "pnl": round(cum, 0), "dd": round(mdd, 0),
        "n": len(tdf), "b5": int((tdf["r"] >= 5).sum()),
        "dpnl": round(cum / max(days, 1), 1),
        "avg_nc": round(tdf["nc"].mean(), 1),
    }


def main():
    print("=" * 85)
    print("ROLLING WALK-FORWARD — NQ + QQQ")
    print("=" * 85)

    # ═══ NQ Walk-Forward ═══
    nq = pd.read_csv(NQ_PATH, parse_dates=["Time"], index_col="Time")
    nq.index.name = "timestamp"
    nq.index = nq.index + pd.Timedelta(hours=1)
    nq = nq[nq.index >= "2022-01-01"]
    print(f"\n  NQ data: {nq.index[0].date()} → {nq.index[-1].date()}")

    # Rolling windows: 6-month test periods
    starts = pd.date_range("2022-07-01", "2025-07-01", freq="6MS")

    for name, s in [("V8 gate=-0.1", V8), ("V8 gate=0.0", V8G0)]:
        print(f"\n{'='*85}")
        print(f"  NQ WALK-FORWARD: {name} ($150 risk, dynamic sizing)")
        print(f"{'='*85}")
        print(f"  {'Period':>22} {'PF':>7} {'PnL':>9} {'DD':>7} {'N':>5} {'$/d':>8} {'NC':>5} {'5R+':>4} {'Win':>4}")

        total_pnl = 0; total_n = 0; total_b5 = 0
        all_pfs = []; losing_periods = 0

        for start in starts:
            end = start + pd.DateOffset(months=6)
            if end > nq.index[-1] + pd.Timedelta(days=30):
                break
            period_data = nq[(nq.index >= start) & (nq.index < end)]
            if len(period_data) < 1000:
                continue

            r = run_period(period_data, s)
            total_pnl += r["pnl"]
            total_n += r["n"]
            total_b5 += r["b5"]
            all_pfs.append(r["pf"])
            win = "✅" if r["pnl"] > 0 else "❌"
            if r["pnl"] <= 0: losing_periods += 1

            period_str = f"{start.strftime('%Y-%m')} → {(end - pd.Timedelta(days=1)).strftime('%Y-%m')}"
            print(f"  {period_str:>22} {r['pf']:>7.3f} {r['pnl']:>+9,.0f} {r['dd']:>7.0f} {r['n']:>5} {r['dpnl']:>+8.1f} {r['avg_nc']:>5.1f} {r['b5']:>4} {win}")

        print(f"  {'─'*85}")
        print(f"  {'TOTAL':>22} {'':>7} {total_pnl:>+9,.0f} {'':>7} {total_n:>5} {'':>8} {'':>5} {total_b5:>4}")
        print(f"  Periods: {len(all_pfs)} | Profitable: {len(all_pfs)-losing_periods}/{len(all_pfs)} "
              f"| PF range: {min(all_pfs):.3f} — {max(all_pfs):.3f} | Median PF: {np.median(all_pfs):.3f}")

    # ═══ QQQ Walk-Forward (4Y, cross-dataset validation) ═══
    print(f"\n{'='*85}")
    print("  QQQ 4Y WALK-FORWARD (cross-dataset, QQQ×40 proxy, fixed 2 MNQ)")
    print(f"{'='*85}")

    qqq_oos = pd.read_csv(QQQ_OOS, index_col="timestamp", parse_dates=True)
    qqq_is = pd.read_csv(QQQ_IS, index_col="timestamp", parse_dates=True)
    qqq_all = pd.concat([qqq_oos, qqq_is]).sort_index()
    qqq_all = qqq_all[~qqq_all.index.duplicated(keep='first')]
    print(f"  QQQ combined: {qqq_all.index[0].date()} → {qqq_all.index[-1].date()} ({len(qqq_all):,} bars)")

    starts_qqq = pd.date_range("2022-07-01", "2025-07-01", freq="6MS")

    for name, s in [("V8 gate=-0.1", V8), ("V8 gate=0.0", V8G0)]:
        print(f"\n  {name} (QQQ×40)")
        print(f"  {'Period':>22} {'PF':>7} {'PnL':>9} {'DD':>7} {'N':>5} {'$/d':>8} {'5R+':>4} {'Win':>4}")

        total_pnl = 0; all_pfs = []; losing = 0

        for start in starts_qqq:
            end = start + pd.DateOffset(months=6)
            if end > qqq_all.index[-1] + pd.Timedelta(days=30):
                break
            period_data = qqq_all[(qqq_all.index >= start) & (qqq_all.index < end)]
            if len(period_data) < 1000:
                continue

            r = run_period(period_data, s, use_qqq=True)
            total_pnl += r["pnl"]
            all_pfs.append(r["pf"])
            win = "✅" if r["pnl"] > 0 else "❌"
            if r["pnl"] <= 0: losing += 1

            period_str = f"{start.strftime('%Y-%m')} → {(end - pd.Timedelta(days=1)).strftime('%Y-%m')}"
            print(f"  {period_str:>22} {r['pf']:>7.3f} {r['pnl']:>+9,.0f} {r['dd']:>7.0f} {r['n']:>5} {r['dpnl']:>+8.1f} {r['b5']:>4} {win}")

        if all_pfs:
            print(f"  {'─'*80}")
            print(f"  Total PnL: ${total_pnl:+,.0f} | Profitable: {len(all_pfs)-losing}/{len(all_pfs)} "
                  f"| PF: {min(all_pfs):.3f} — {max(all_pfs):.3f} | Median: {np.median(all_pfs):.3f}")

    # ═══ SUMMARY ═══
    print(f"\n{'='*85}")
    print("SUMMARY")
    print(f"{'='*85}")
    print(f"\n  Available data: NQ 4.3Y (2022-01 to 2026-03) + QQQ 4Y (2022-03 to 2026-03)")
    print(f"  For 10Y walk-forward: need NQ data from ~2016 onwards")
    print(f"  → Barchart has NQ 1-min back to 2010+ (downloadable with membership)")
    print(f"  → Action: download NQ continuous 2016-2022 from Barchart to extend to 10Y")


if __name__ == "__main__":
    main()
