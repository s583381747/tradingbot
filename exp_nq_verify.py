"""Verify top NQ combos on full 4-year data with yearly breakdown."""
from exp_nq_tune import *


def run_yearly(nq, s):
    """Run on full data, return yearly stats."""
    from collections import defaultdict
    df = resample(nq, s["tf_minutes"])
    df = add_ind(df, s)
    H = df["High"].values; L = df["Low"].values; C = df["Close"].values
    ef = df["ema_f"].values; es = df["ema_s"].values; atr = df["atr"].values
    T = df.index.time; D = df.index.date; n = len(df)
    tf = max(1, s["tf_minutes"])
    mh = max(20, s["max_hold_bars"] // tf)
    cb = max(5, s["chand_bars"] // tf)
    gb = max(1, s["gate_bars"] // tf) if s["gate_bars"] > 0 else 0
    nc = s["n_contracts"]

    cum = 0.0; peak = 0.0; mdd = 0.0
    yearly_trades = defaultdict(list)
    bar = max(s["ema_slow"], s["atr_period"]) + 5
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

        entry = C[bar]
        stop = L[bar] - s["stop_buffer"] * a if tr == 1 else H[bar] + s["stop_buffer"] * a
        rp = abs(entry - stop)
        if rp <= 0: bar += 1; continue
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
        cum += net; peak = max(peak, cum); mdd = max(mdd, peak - cum)
        yr = d.year
        yearly_trades[yr].append({"pnl": net, "r": r, "ex": ex})
        if r < 0: dlr += abs(r)
        if r > 0: sk = s.get("skip_after_win", 0)
        bar = endb + 1

    # Yearly stats
    yearly = {}
    for yr, trades in sorted(yearly_trades.items()):
        tdf = pd.DataFrame(trades)
        gw = tdf.loc[tdf["pnl"] > 0, "pnl"].sum()
        gl = abs(tdf.loc[tdf["pnl"] <= 0, "pnl"].sum())
        cum_yr = tdf["pnl"].cumsum()
        dd_yr = (cum_yr.cummax() - cum_yr).max()
        yearly[yr] = {
            "pf": round(gw / gl, 3) if gl > 0 else 0,
            "pnl": round(tdf["pnl"].sum(), 0),
            "dd": round(dd_yr, 0),
            "n": len(tdf),
            "b5": int((tdf["r"] >= 5).sum()),
            "wr": round((tdf["r"] > 0).mean() * 100, 1),
        }

    all_trades = [t for trades in yearly_trades.values() for t in trades]
    tdf = pd.DataFrame(all_trades)
    gw = tdf.loc[tdf["pnl"] > 0, "pnl"].sum()
    gl = abs(tdf.loc[tdf["pnl"] <= 0, "pnl"].sum())
    return {
        "pf": round(gw / gl, 3) if gl > 0 else 0,
        "pnl": round(cum, 0), "dd": round(mdd, 0),
        "n": len(tdf), "b5": int((tdf["r"] >= 5).sum()),
        "yearly": yearly,
    }


def main():
    print("Loading NQ data...")
    nq = pd.read_csv(NQ_PATH, parse_dates=["Time"], index_col="Time")
    nq.index.name = "timestamp"
    nq.index = nq.index + pd.Timedelta(hours=1)
    nq_full = nq[nq.index >= "2022-01-01"]

    configs = {
        "V8 BASELINE": DEFAULT,
        "R: gate0.0+skip2+be0.20+mfe0.25": {**DEFAULT, "gate_tighten": 0.0, "skip_after_win": 2,
            "be_trigger_r": 0.20, "be_stop_r": 0.15, "gate_mfe": 0.25},
        "L: gate0.0+skip0+be0.20": {**DEFAULT, "gate_tighten": 0.0, "skip_after_win": 0,
            "be_trigger_r": 0.20, "be_stop_r": 0.15},
        "M: gate0.0+skip2+be0.20": {**DEFAULT, "gate_tighten": 0.0, "skip_after_win": 2,
            "be_trigger_r": 0.20, "be_stop_r": 0.15},
        "I: gate0.0+be0.20": {**DEFAULT, "gate_tighten": 0.0, "be_trigger_r": 0.20, "be_stop_r": 0.15},
    }

    for name, s in configs.items():
        print(f"\n{'='*70}")
        print(f"  {name}")
        print(f"{'='*70}")
        r = run_yearly(nq_full, s)
        print(f"  4Y: PF={r['pf']:.3f} PnL=${r['pnl']:+,.0f} DD=${r['dd']:.0f} N={r['n']} 5R+={r['b5']}")
        print(f"\n  {'Year':>6} {'PF':>7} {'PnL':>10} {'DD':>7} {'N':>6} {'5R+':>5} {'WR%':>6}")
        all_profitable = True
        for yr, yd in sorted(r["yearly"].items()):
            if yd["pnl"] <= 0: all_profitable = False
            print(f"  {yr:>6} {yd['pf']:>7.3f} {yd['pnl']:>+10,.0f} {yd['dd']:>7.0f} {yd['n']:>6} {yd['b5']:>5} {yd['wr']:>5.1f}%")
        print(f"  All years profitable: {'✅' if all_profitable else '❌'}")
        print(f"  Topstep 50K DD: ${r['dd']:.0f} / $2,000 → {'✅' if r['dd'] <= 2000 else '❌'}")
        days = sum(yd["n"] for yd in r["yearly"].values()) / 5.5  # approx trading days
        print(f"  Daily avg: ${r['pnl']/days:+.1f}")


if __name__ == "__main__":
    main()
