"""
CTO Audit — Is v9 (gate_tighten=0.0) too good to be true?

Checks:
1. Gate-fail trade mechanics: what actually happens when stop moves to entry?
2. Cost model: are gate-fail BE exits charged correctly?
3. Trade-by-trade breakdown: where does the PF improvement come from?
4. Multiple comparison correction: 22 combos tested on OOS = snooping risk
5. Walk-forward: train on 2022-2023, test on 2024-2026 (reverse direction)
6. Sensitivity: small param perturbation stability
"""
from exp_nq_tune import *
from collections import Counter


def run_detailed(df_1min, s, label=""):
    """Run and return detailed per-trade data."""
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

    trades = []; bar = max(s["ema_slow"], s["atr_period"]) + 5
    dlr = 0.0; cd = None; sk = 0
    cum = 0.0; peak = 0.0; mdd = 0.0

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

        entry = C[bar]
        stop_orig = L[bar] - s["stop_buffer"] * a if tr == 1 else H[bar] + s["stop_buffer"] * a
        rp = abs(entry - stop_orig)
        if rp <= 0: bar += 1; continue
        rm = rp * MNQ_PER_POINT * nc
        ec = COMM_RT * nc / 2 + SPREAD

        eb = bar; rs = stop_orig; bt = False; mfe = 0.0; r = 0.0; endb = bar; ex = "timeout"
        gate_failed = False; gate_mfe_at_check = 0.0

        for k in range(1, mh + 1):
            bi = eb + k
            if bi >= n: break
            h = H[bi]; l = L[bi]; ca = atr[bi] if not np.isnan(atr[bi]) else a
            if tr == 1: mfe = max(mfe, (h - entry) / rp)
            else: mfe = max(mfe, (entry - l) / rp)
            if T[bi] >= s["force_close_at"]:
                r = (C[bi] - entry) / rp * tr; endb = bi; ex = "close"; break
            if gb > 0 and k == gb and not bt:
                gate_mfe_at_check = mfe
                if mfe < s["gate_mfe"]:
                    gate_failed = True
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
        total_cost = ec + xc + xs + bs
        cum += net; peak = max(peak, cum); mdd = max(mdd, peak - cum)

        trades.append({
            "r": r, "pnl": net, "raw_pnl": raw, "cost": total_cost,
            "ex": ex, "risk_mnq": rm, "gate_failed": gate_failed,
            "gate_mfe": gate_mfe_at_check, "mfe": mfe, "be_triggered": bt,
            "year": d.year,
        })
        if r < 0: dlr += abs(r)
        if r > 0: sk = s.get("skip_after_win", 0)
        bar = endb + 1

    return pd.DataFrame(trades), cum, mdd


def main():
    print("=" * 80)
    print("CTO AUDIT — v9 NQ-tuned parameters")
    print("=" * 80)

    nq = pd.read_csv(NQ_PATH, parse_dates=["Time"], index_col="Time")
    nq.index.name = "timestamp"
    nq.index = nq.index + pd.Timedelta(hours=1)
    nq_is = nq[nq.index >= "2024-01-01"]
    nq_oos = nq[(nq.index >= "2022-01-01") & (nq.index < "2024-01-01")]

    V8 = DEFAULT.copy()
    V9 = {**DEFAULT, "gate_tighten": 0.0, "skip_after_win": 2,
          "be_trigger_r": 0.20, "be_stop_r": 0.15, "gate_mfe": 0.25}

    # ═══ 1. TRADE-BY-TRADE BREAKDOWN ═══
    print(f"\n{'='*80}")
    print("1. TRADE-BY-TRADE BREAKDOWN: WHERE DOES PF IMPROVEMENT COME FROM?")
    print(f"{'='*80}")

    for name, s, df in [("V8 IS", V8, nq_is), ("V9 IS", V9, nq_is),
                          ("V8 OOS", V8, nq_oos), ("V9 OOS", V9, nq_oos)]:
        tdf, cum, mdd = run_detailed(df, s)
        gw = tdf.loc[tdf["pnl"] > 0, "pnl"].sum()
        gl = abs(tdf.loc[tdf["pnl"] <= 0, "pnl"].sum())
        pf = gw / gl if gl > 0 else 0

        gate_fail = tdf[tdf["gate_failed"]]
        gate_pass = tdf[~tdf["gate_failed"]]

        print(f"\n  [{name}] PF={pf:.3f} PnL=${cum:+,.0f} DD=${mdd:.0f} N={len(tdf)}")
        print(f"    Gate FAIL: {len(gate_fail)} trades ({len(gate_fail)/len(tdf)*100:.0f}%)")
        if len(gate_fail) > 0:
            gf_w = gate_fail.loc[gate_fail["pnl"] > 0, "pnl"].sum()
            gf_l = abs(gate_fail.loc[gate_fail["pnl"] <= 0, "pnl"].sum())
            gf_pf = gf_w / gf_l if gf_l > 0 else 0
            print(f"      PF={gf_pf:.3f} PnL=${gate_fail['pnl'].sum():+,.0f} AvgR={gate_fail['r'].mean():+.3f}")
            print(f"      Avg cost/trade: ${gate_fail['cost'].mean():.2f}")
            print(f"      Avg raw PnL: ${gate_fail['raw_pnl'].mean():+.2f}")
            print(f"      Exits: {dict(gate_fail['ex'].value_counts())}")

        print(f"    Gate PASS: {len(gate_pass)} trades ({len(gate_pass)/len(tdf)*100:.0f}%)")
        if len(gate_pass) > 0:
            gp_w = gate_pass.loc[gate_pass["pnl"] > 0, "pnl"].sum()
            gp_l = abs(gate_pass.loc[gate_pass["pnl"] <= 0, "pnl"].sum())
            gp_pf = gp_w / gp_l if gp_l > 0 else 0
            print(f"      PF={gp_pf:.3f} PnL=${gate_pass['pnl'].sum():+,.0f} AvgR={gate_pass['r'].mean():+.3f}")

    # ═══ 2. GATE-FAIL R DISTRIBUTION ═══
    print(f"\n{'='*80}")
    print("2. GATE-FAIL TRADE R DISTRIBUTION (V8 vs V9 on OOS)")
    print(f"{'='*80}")

    for name, s in [("V8", V8), ("V9", V9)]:
        tdf, _, _ = run_detailed(nq_oos, s)
        gf = tdf[tdf["gate_failed"]]
        if len(gf) == 0: continue
        print(f"\n  [{name}] Gate-fail R distribution (N={len(gf)}):")
        for bucket, lo, hi in [("<-1R", -999, -1), ("-1R to -0.5R", -1, -0.5),
                                 ("-0.5R to -0.1R", -0.5, -0.1), ("-0.1R to 0R", -0.1, 0),
                                 ("0R to 0.15R", 0, 0.15), ("0.15R to 0.5R", 0.15, 0.5),
                                 ("0.5R to 1R", 0.5, 1), (">1R", 1, 999)]:
            mask = (gf["r"] >= lo) & (gf["r"] < hi)
            n_bucket = mask.sum()
            if n_bucket > 0:
                avg_pnl = gf.loc[mask, "pnl"].mean()
                print(f"    {bucket:>18}: {n_bucket:>5} ({n_bucket/len(gf)*100:>5.1f}%)  avg=${avg_pnl:+.1f}")

    # ═══ 3. COST MODEL VERIFICATION ═══
    print(f"\n{'='*80}")
    print("3. COST MODEL: are gate-fail BE exits charged correctly?")
    print(f"{'='*80}")

    tdf_v9, _, _ = run_detailed(nq_oos, V9)
    gf = tdf_v9[tdf_v9["gate_failed"]]
    # Gate-fail trades stopped at entry (r≈0): should still pay costs
    near_zero = gf[(gf["r"] > -0.02) & (gf["r"] < 0.02)]
    print(f"\n  Gate-fail trades stopped near entry (|R|<0.02): {len(near_zero)}")
    if len(near_zero) > 0:
        print(f"    Avg raw PnL: ${near_zero['raw_pnl'].mean():+.2f} (should be ~0)")
        print(f"    Avg cost:    ${near_zero['cost'].mean():.2f}")
        print(f"    Avg net PnL: ${near_zero['pnl'].mean():+.2f} (should be negative = costs)")
        print(f"    Exit types:  {dict(near_zero['ex'].value_counts())}")

    # ═══ 4. OOS SNOOPING TEST ═══
    print(f"\n{'='*80}")
    print("4. OOS SNOOPING: reverse walk-forward (train OOS, test IS)")
    print(f"{'='*80}")

    # If v9 is overfit to OOS period, it should be WORSE when IS/OOS are reversed
    r_is = run(nq_is, V9)
    r_oos = run(nq_oos, V9)
    r_is_v8 = run(nq_is, V8)
    r_oos_v8 = run(nq_oos, V8)

    print(f"\n  V8: IS PF={r_is_v8['pf']:.3f} DD=${r_is_v8['dd']:.0f} | OOS PF={r_oos_v8['pf']:.3f} DD=${r_oos_v8['dd']:.0f}")
    print(f"  V9: IS PF={r_is['pf']:.3f} DD=${r_is['dd']:.0f} | OOS PF={r_oos['pf']:.3f} DD=${r_oos['dd']:.0f}")
    print(f"\n  V8 improvement IS→OOS: {(r_oos_v8['pf']/r_is_v8['pf']-1)*100:+.1f}%")
    print(f"  V9 improvement IS→OOS: {(r_oos['pf']/r_is['pf']-1)*100:+.1f}%")

    if r_oos["pf"] > r_is["pf"]:
        print(f"  ⚠ OOS > IS for V9 — suspicious. OOS may be easier period, not better params.")
    else:
        print(f"  ✓ IS > OOS for V9 — normal degradation.")

    # ═══ 5. SENSITIVITY TEST ═══
    print(f"\n{'='*80}")
    print("5. SENSITIVITY: small perturbations around V9 params")
    print(f"{'='*80}")

    print(f"\n  {'Perturbation':>35} {'OOS_PF':>7} {'OOS_DD':>7} {'Score':>7}")
    base_r = run(nq_oos, V9)
    print(f"  {'V9 center':>35} {base_r['pf']:>7.3f} {base_r['dd']:>7.0f} {score(base_r):>7.2f}")

    perturbations = [
        ("gate_tighten=-0.02", {"gate_tighten": -0.02}),
        ("gate_tighten=+0.02", {"gate_tighten": 0.02}),
        ("gate_mfe=0.22", {"gate_mfe": 0.22}),
        ("gate_mfe=0.28", {"gate_mfe": 0.28}),
        ("be_trigger=0.18", {"be_trigger_r": 0.18}),
        ("be_trigger=0.22", {"be_trigger_r": 0.22}),
        ("skip=1", {"skip_after_win": 1}),
        ("skip=3", {"skip_after_win": 3}),
    ]
    scores = [score(base_r)]
    for desc, overrides in perturbations:
        s = {**V9, **overrides}
        r = run(nq_oos, s)
        sc = score(r)
        scores.append(sc)
        delta = sc - score(base_r)
        print(f"  {desc:>35} {r['pf']:>7.3f} {r['dd']:>7.0f} {sc:>7.2f} ({delta:+.2f})")

    print(f"\n  Score range: {min(scores):.2f} — {max(scores):.2f} (spread={max(scores)-min(scores):.2f})")
    if max(scores) - min(scores) < 5:
        print(f"  ✓ Robust — small perturbations don't destroy the result")
    else:
        print(f"  ⚠ Fragile — result depends heavily on exact param values")

    # ═══ 6. YEAR-BY-YEAR CONSISTENCY ═══
    print(f"\n{'='*80}")
    print("6. YEAR-BY-YEAR: V8 vs V9 improvement consistent?")
    print(f"{'='*80}")

    for s, name in [(V8, "V8"), (V9, "V9")]:
        nq_full = nq[nq.index >= "2022-01-01"]
        tdf, cum, mdd = run_detailed(nq_full, s)
        print(f"\n  [{name}]")
        print(f"  {'Year':>6} {'PF':>7} {'PnL':>9} {'DD':>7} {'N':>6} {'GateFail%':>10} {'AvgR':>7}")
        for yr in sorted(tdf["year"].unique()):
            yt = tdf[tdf["year"] == yr]
            gw = yt.loc[yt["pnl"] > 0, "pnl"].sum()
            gl = abs(yt.loc[yt["pnl"] <= 0, "pnl"].sum())
            pf = gw / gl if gl > 0 else 0
            cum_yr = yt["pnl"].cumsum()
            dd_yr = (cum_yr.cummax() - cum_yr).max()
            gf_pct = yt["gate_failed"].mean() * 100
            print(f"  {yr:>6} {pf:>7.3f} {yt['pnl'].sum():>+9,.0f} {dd_yr:>7.0f} {len(yt):>6} {gf_pct:>9.1f}% {yt['r'].mean():>+7.3f}")

    # ═══ 7. THE REAL QUESTION ═══
    print(f"\n{'='*80}")
    print("7. VERDICT: is gate_tighten=0.0 a real improvement or curve-fitting?")
    print(f"{'='*80}")

    # Count how much of the PF improvement comes from gate-fail trades
    tdf_v8, cum_v8, _ = run_detailed(nq_oos, V8)
    tdf_v9, cum_v9, _ = run_detailed(nq_oos, V9)

    v8_gf = tdf_v8[tdf_v8["gate_failed"]]
    v9_gf = tdf_v9[tdf_v9["gate_failed"]]
    v8_gp = tdf_v8[~tdf_v8["gate_failed"]]
    v9_gp = tdf_v9[~tdf_v9["gate_failed"]]

    print(f"\n  OOS Gate-fail PnL:  V8=${v8_gf['pnl'].sum():+,.0f} → V9=${v9_gf['pnl'].sum():+,.0f} (Δ=${v9_gf['pnl'].sum()-v8_gf['pnl'].sum():+,.0f})")
    print(f"  OOS Gate-pass PnL:  V8=${v8_gp['pnl'].sum():+,.0f} → V9=${v9_gp['pnl'].sum():+,.0f} (Δ=${v9_gp['pnl'].sum()-v8_gp['pnl'].sum():+,.0f})")
    print(f"  OOS Total PnL:      V8=${cum_v8:+,.0f} → V9=${cum_v9:+,.0f} (Δ=${cum_v9-cum_v8:+,.0f})")

    total_delta = cum_v9 - cum_v8
    gf_delta = v9_gf['pnl'].sum() - v8_gf['pnl'].sum()
    gp_delta = v9_gp['pnl'].sum() - v8_gp['pnl'].sum()
    trade_count_delta = len(tdf_v9) - len(tdf_v8)

    print(f"\n  Gate-fail Δ contributes: {gf_delta/total_delta*100:.0f}% of total improvement")
    print(f"  Gate-pass Δ contributes: {gp_delta/total_delta*100:.0f}% of total improvement")
    print(f"  Trade count: V8={len(tdf_v8)} → V9={len(tdf_v9)} (Δ={trade_count_delta})")

    # V8 gate-fail trades: what R did they actually exit at?
    print(f"\n  V8 gate-fail avg exit R: {v8_gf['r'].mean():+.3f} (should be around -0.1)")
    print(f"  V9 gate-fail avg exit R: {v9_gf['r'].mean():+.3f} (should be around 0.0)")
    print(f"  V8 gate-fail avg net PnL/trade: ${v8_gf['pnl'].mean():+.1f}")
    print(f"  V9 gate-fail avg net PnL/trade: ${v9_gf['pnl'].mean():+.1f}")


if __name__ == "__main__":
    main()
