"""
NQ Parameter Tuning — Sweep key params on real NQ data.
Outputs: IS PF, OOS PF, IS DD, OOS DD, score, 5R+ for each config.
"""
from __future__ import annotations
import functools, datetime as dt, sys, json
import numpy as np, pandas as pd

print = functools.partial(print, flush=True)

NQ_PATH = "data/barchart_nq/NQ_1min_continuous_RTH.csv"
MNQ_PER_POINT = 2.0
COMM_RT = 2.46; SPREAD = 0.50; STOP_SLIP = 1.00; BE_SLIP = 1.00

DEFAULT = {
    "tf_minutes": 3, "ema_fast": 20, "ema_slow": 50, "atr_period": 14,
    "touch_tol": 0.15, "touch_below_max": 0.5, "no_entry_after": dt.time(14, 0),
    "stop_buffer": 0.4, "gate_bars": 3, "gate_mfe": 0.2, "gate_tighten": -0.1,
    "be_trigger_r": 0.25, "be_stop_r": 0.15, "chand_bars": 25, "chand_mult": 0.3,
    "max_hold_bars": 180, "force_close_at": dt.time(15, 58),
    "daily_loss_r": 2.0, "skip_after_win": 1, "n_contracts": 2,
}


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


def run(df_1min, s):
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

    cum = 0.0; peak = 0.0; mdd = 0.0
    trades = []; bar = max(s["ema_slow"], s["atr_period"]) + 5
    dlr = 0.0; cd = None; sk = 0

    while bar < n - mh - 5:
        a = atr[bar]
        if np.isnan(a) or a <= 0 or np.isnan(ef[bar]) or np.isnan(es[bar]):
            bar += 1; continue
        if T[bar] >= s["no_entry_after"]:
            bar += 1; continue
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
            h = H[bi]; l = L[bi]
            ca = atr[bi] if not np.isnan(atr[bi]) else a
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
        trades.append({"pnl": net, "r": r, "ex": ex})
        if r < 0: dlr += abs(r)
        if r > 0: sk = s.get("skip_after_win", 0)
        bar = endb + 1

    if not trades: return {"pf": 0, "dd": 0, "pnl": 0, "n": 0, "dpnl": 0, "b5": 0}
    tdf = pd.DataFrame(trades)
    gw = tdf.loc[tdf["pnl"] > 0, "pnl"].sum()
    gl = abs(tdf.loc[tdf["pnl"] <= 0, "pnl"].sum())
    days = len(set(D))
    return {
        "pf": round(gw / gl, 3) if gl > 0 else 0,
        "dd": round(mdd, 0), "pnl": round(cum, 0),
        "n": len(tdf), "dpnl": round(cum / max(days, 1), 1),
        "b5": int((tdf["r"] >= 5).sum()),
    }


def score(r):
    return round(r["pf"] * 10 - r["dd"] / 100, 2)


def main():
    print("Loading NQ data...")
    nq = pd.read_csv(NQ_PATH, parse_dates=["Time"], index_col="Time")
    nq.index.name = "timestamp"
    nq.index = nq.index + pd.Timedelta(hours=1)
    nq_is = nq[nq.index >= "2024-01-01"]
    nq_oos = nq[(nq.index >= "2022-01-01") & (nq.index < "2024-01-01")]

    # Baseline
    ri = run(nq_is, DEFAULT); ro = run(nq_oos, DEFAULT)
    bs = score(ro)
    print(f"\nBASELINE: IS PF={ri['pf']:.3f} $/d={ri['dpnl']:+.1f} DD=${ri['dd']:.0f} | "
          f"OOS PF={ro['pf']:.3f} $/d={ro['dpnl']:+.1f} DD=${ro['dd']:.0f} | Score={bs}")

    best_score = bs; best_cfg = "baseline"

    # ═══ Sweep 1: stop_buffer ═══
    print(f"\n{'='*70}\nSWEEP 1: stop_buffer\n{'='*70}")
    print(f"  {'Val':>6} {'IS_PF':>7} {'IS_DD':>7} {'OOS_PF':>7} {'OOS_DD':>7} {'OOS_$/d':>8} {'Score':>7} {'5R+':>4}")
    for v in [0.2, 0.3, 0.4, 0.5, 0.6, 0.8]:
        s = {**DEFAULT, "stop_buffer": v}
        ri = run(nq_is, s); ro = run(nq_oos, s); sc = score(ro)
        tag = " ★" if sc > best_score else ""
        if sc > best_score: best_score = sc; best_cfg = f"stop_buffer={v}"
        print(f"  {v:>6.1f} {ri['pf']:>7.3f} {ri['dd']:>7.0f} {ro['pf']:>7.3f} {ro['dd']:>7.0f} {ro['dpnl']:>+8.1f} {sc:>7.2f} {ro['b5']:>4}{tag}")

    # ═══ Sweep 2: gate_tighten ═══
    print(f"\n{'='*70}\nSWEEP 2: gate_tighten\n{'='*70}")
    print(f"  {'Val':>6} {'IS_PF':>7} {'IS_DD':>7} {'OOS_PF':>7} {'OOS_DD':>7} {'OOS_$/d':>8} {'Score':>7} {'5R+':>4}")
    for v in [-0.3, -0.2, -0.15, -0.1, -0.05, 0.0]:
        s = {**DEFAULT, "gate_tighten": v}
        ri = run(nq_is, s); ro = run(nq_oos, s); sc = score(ro)
        tag = " ★" if sc > best_score else ""
        if sc > best_score: best_score = sc; best_cfg = f"gate_tighten={v}"
        print(f"  {v:>6.2f} {ri['pf']:>7.3f} {ri['dd']:>7.0f} {ro['pf']:>7.3f} {ro['dd']:>7.0f} {ro['dpnl']:>+8.1f} {sc:>7.2f} {ro['b5']:>4}{tag}")

    # ═══ Sweep 3: be_trigger_r / be_stop_r ═══
    print(f"\n{'='*70}\nSWEEP 3: be_trigger_r / be_stop_r\n{'='*70}")
    print(f"  {'Trig':>5} {'Stop':>5} {'IS_PF':>7} {'IS_DD':>7} {'OOS_PF':>7} {'OOS_DD':>7} {'OOS_$/d':>8} {'Score':>7} {'5R+':>4}")
    for trig in [0.2, 0.25, 0.3, 0.35, 0.4, 0.5]:
        for stp in [0.05, 0.1, 0.15, 0.2]:
            if stp >= trig: continue
            s = {**DEFAULT, "be_trigger_r": trig, "be_stop_r": stp}
            ri = run(nq_is, s); ro = run(nq_oos, s); sc = score(ro)
            tag = " ★" if sc > best_score else ""
            if sc > best_score: best_score = sc; best_cfg = f"be_trigger={trig}/stop={stp}"
            print(f"  {trig:>5.2f} {stp:>5.2f} {ri['pf']:>7.3f} {ri['dd']:>7.0f} {ro['pf']:>7.3f} {ro['dd']:>7.0f} {ro['dpnl']:>+8.1f} {sc:>7.2f} {ro['b5']:>4}{tag}")

    # ═══ Sweep 4: chand_bars / chand_mult ═══
    print(f"\n{'='*70}\nSWEEP 4: chand_bars / chand_mult\n{'='*70}")
    print(f"  {'Bars':>5} {'Mult':>5} {'IS_PF':>7} {'IS_DD':>7} {'OOS_PF':>7} {'OOS_DD':>7} {'OOS_$/d':>8} {'Score':>7} {'5R+':>4}")
    for bars in [15, 20, 25, 30, 40]:
        for mult in [0.2, 0.3, 0.4, 0.5]:
            s = {**DEFAULT, "chand_bars": bars, "chand_mult": mult}
            ri = run(nq_is, s); ro = run(nq_oos, s); sc = score(ro)
            tag = " ★" if sc > best_score else ""
            if sc > best_score: best_score = sc; best_cfg = f"chand={bars}/{mult}"
            print(f"  {bars:>5} {mult:>5.1f} {ri['pf']:>7.3f} {ri['dd']:>7.0f} {ro['pf']:>7.3f} {ro['dd']:>7.0f} {ro['dpnl']:>+8.1f} {sc:>7.2f} {ro['b5']:>4}{tag}")

    # ═══ Sweep 5: gate_mfe ═══
    print(f"\n{'='*70}\nSWEEP 5: gate_mfe\n{'='*70}")
    print(f"  {'Val':>6} {'IS_PF':>7} {'IS_DD':>7} {'OOS_PF':>7} {'OOS_DD':>7} {'OOS_$/d':>8} {'Score':>7} {'5R+':>4}")
    for v in [0.1, 0.15, 0.2, 0.25, 0.3, 0.4]:
        s = {**DEFAULT, "gate_mfe": v}
        ri = run(nq_is, s); ro = run(nq_oos, s); sc = score(ro)
        tag = " ★" if sc > best_score else ""
        if sc > best_score: best_score = sc; best_cfg = f"gate_mfe={v}"
        print(f"  {v:>6.2f} {ri['pf']:>7.3f} {ri['dd']:>7.0f} {ro['pf']:>7.3f} {ro['dd']:>7.0f} {ro['dpnl']:>+8.1f} {sc:>7.2f} {ro['b5']:>4}{tag}")

    # ═══ Sweep 6: daily_loss_r ═══
    print(f"\n{'='*70}\nSWEEP 6: daily_loss_r\n{'='*70}")
    print(f"  {'Val':>6} {'IS_PF':>7} {'IS_DD':>7} {'OOS_PF':>7} {'OOS_DD':>7} {'OOS_$/d':>8} {'Score':>7} {'5R+':>4}")
    for v in [1.0, 1.5, 2.0, 2.5, 3.0, 5.0]:
        s = {**DEFAULT, "daily_loss_r": v}
        ri = run(nq_is, s); ro = run(nq_oos, s); sc = score(ro)
        tag = " ★" if sc > best_score else ""
        if sc > best_score: best_score = sc; best_cfg = f"daily_loss_r={v}"
        print(f"  {v:>6.1f} {ri['pf']:>7.3f} {ri['dd']:>7.0f} {ro['pf']:>7.3f} {ro['dd']:>7.0f} {ro['dpnl']:>+8.1f} {sc:>7.2f} {ro['b5']:>4}{tag}")

    # ═══ Sweep 7: no_entry_after ═══
    print(f"\n{'='*70}\nSWEEP 7: no_entry_after\n{'='*70}")
    print(f"  {'Time':>8} {'IS_PF':>7} {'IS_DD':>7} {'OOS_PF':>7} {'OOS_DD':>7} {'OOS_$/d':>8} {'Score':>7} {'5R+':>4}")
    for h, m in [(12, 0), (12, 30), (13, 0), (13, 30), (14, 0), (14, 30), (15, 0)]:
        s = {**DEFAULT, "no_entry_after": dt.time(h, m)}
        ri = run(nq_is, s); ro = run(nq_oos, s); sc = score(ro)
        tag = " ★" if sc > best_score else ""
        if sc > best_score: best_score = sc; best_cfg = f"no_entry_after={h}:{m:02d}"
        print(f"  {h:02d}:{m:02d}    {ri['pf']:>7.3f} {ri['dd']:>7.0f} {ro['pf']:>7.3f} {ro['dd']:>7.0f} {ro['dpnl']:>+8.1f} {sc:>7.2f} {ro['b5']:>4}{tag}")

    # ═══ Sweep 8: skip_after_win ═══
    print(f"\n{'='*70}\nSWEEP 8: skip_after_win\n{'='*70}")
    print(f"  {'Val':>6} {'IS_PF':>7} {'IS_DD':>7} {'OOS_PF':>7} {'OOS_DD':>7} {'OOS_$/d':>8} {'Score':>7} {'5R+':>4}")
    for v in [0, 1, 2, 3]:
        s = {**DEFAULT, "skip_after_win": v}
        ri = run(nq_is, s); ro = run(nq_oos, s); sc = score(ro)
        tag = " ★" if sc > best_score else ""
        if sc > best_score: best_score = sc; best_cfg = f"skip_after_win={v}"
        print(f"  {v:>6} {ri['pf']:>7.3f} {ri['dd']:>7.0f} {ro['pf']:>7.3f} {ro['dd']:>7.0f} {ro['dpnl']:>+8.1f} {sc:>7.2f} {ro['b5']:>4}{tag}")

    # ═══ Sweep 9: tf_minutes ═══
    print(f"\n{'='*70}\nSWEEP 9: tf_minutes\n{'='*70}")
    print(f"  {'TF':>4} {'IS_PF':>7} {'IS_DD':>7} {'OOS_PF':>7} {'OOS_DD':>7} {'OOS_$/d':>8} {'Score':>7} {'5R+':>4} {'N':>6}")
    for v in [1, 2, 3, 5, 10]:
        s = {**DEFAULT, "tf_minutes": v}
        ri = run(nq_is, s); ro = run(nq_oos, s); sc = score(ro)
        tag = " ★" if sc > best_score else ""
        if sc > best_score: best_score = sc; best_cfg = f"tf_minutes={v}"
        print(f"  {v:>4} {ri['pf']:>7.3f} {ri['dd']:>7.0f} {ro['pf']:>7.3f} {ro['dd']:>7.0f} {ro['dpnl']:>+8.1f} {sc:>7.2f} {ro['b5']:>4} {ro['n']:>6}{tag}")

    print(f"\n{'='*70}")
    print(f"BEST SINGLE-PARAM: {best_cfg} → score={best_score:.2f} (baseline={bs:.2f})")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
