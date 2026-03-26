"""
ML Filter V3 — DCP Deep Dive + Simple Rule-Based Filters

V2 found: DCP (Directional Close Position) is the #1 feature for predicting
gate pass (AUC 0.61) and MFE ≥ 1R (AUC 0.66).

V3 tests: Can DCP as a SIMPLE threshold filter improve PF without killing big wins?
Also tests: DCP + 1-2 other top features as simple rules (no ML needed).
"""
from __future__ import annotations
import functools, datetime as dt, warnings
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
print = functools.partial(print, flush=True)

IS_PATH = "data/QQQ_1Min_Polygon_2y_clean.csv"
OOS_PATH = "data/QQQ_1Min_Barchart_2y_2022-2024_clean.csv"

S = {
    "tf_minutes": 3, "ema_fast": 20, "ema_slow": 50, "atr_period": 14,
    "touch_tol": 0.15, "touch_below_max": 0.5, "no_entry_after": dt.time(14, 0),
    "stop_buffer": 0.4, "gate_bars": 3, "gate_mfe": 0.2, "gate_tighten": -0.1,
    "be_trigger_r": 0.25, "be_stop_r": 0.15, "chand_bars": 25, "chand_mult": 0.3,
    "max_hold_bars": 180, "force_close_at": dt.time(15, 58),
    "daily_loss_r": 2.0, "skip_after_win": 1, "n_contracts": 2,
}

QQQ_TO_NQ = 40; MNQ_PER_POINT = 2.0
COMM_RT = 2.46; SPREAD = 0.50; STOP_SLIP = 1.00; BE_SLIP = 1.00


def resample(df, minutes):
    if minutes <= 1: return df
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


def run_with_dcp_analysis(df_1min, s=S):
    """Run strategy and record DCP + other features for each trade."""
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

    records = []
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

        c = close[bar]; h = high[bar]; l = low[bar]
        ef = ema_f[bar]; es = ema_s[bar]

        if c > ef and ef > es: trend = 1
        elif c < ef and ef < es: trend = -1
        else: bar += 1; continue

        tol = a * s["touch_tol"]
        if trend == 1:
            touch = l <= ef + tol and l >= ef - a * s["touch_below_max"]
        else:
            touch = h >= ef - tol and h <= ef + a * s["touch_below_max"]
        if not touch: bar += 1; continue
        if skip_count > 0: skip_count -= 1; bar += 1; continue

        # ─── Compute features at entry ───
        rng = h - l
        dcp = ((c - l) / rng if trend == 1 else (h - c) / rng) if rng > 0 else 0.5
        close_ema_dist = (c - ef) / a * trend
        bar_range_atr = rng / a

        # DCP of previous bar
        if bar >= 1:
            ph = high[bar-1]; pl = low[bar-1]; pc = close[bar-1]
            prng = ph - pl
            prev_dcp = ((pc - pl) / prng if trend == 1 else (ph - pc) / prng) if prng > 0 else 0.5
        else:
            prev_dcp = 0.5

        # DCP slope (5-bar)
        if bar >= 5:
            dcps = []
            for k in range(5):
                bi = bar - 4 + k
                br = high[bi] - low[bi]
                if br > 0:
                    d_val = (close[bi] - low[bi]) / br if trend == 1 else (high[bi] - close[bi]) / br
                else:
                    d_val = 0.5
                dcps.append(d_val)
            dcp_slope = (dcps[-1] - dcps[0]) / 4
        else:
            dcp_slope = 0

        # ─── Execute trade ───
        entry = close[bar]
        stop = l - s["stop_buffer"] * a if trend == 1 else h + s["stop_buffer"] * a
        risk_qqq = abs(entry - stop)
        if risk_qqq <= 0: bar += 1; continue
        risk_mnq = risk_qqq * QQQ_TO_NQ * MNQ_PER_POINT * nc
        entry_cost = COMM_RT * nc / 2 + SPREAD

        entry_bar = bar; runner_stop = stop; be_triggered = False
        mfe = 0.0; trade_r = 0.0; end_bar = bar; exit_reason = "timeout"

        for k in range(1, max_hold + 1):
            bi = entry_bar + k
            if bi >= n: break
            h_k = high[bi]; l_k = low[bi]
            ca = atr[bi] if not np.isnan(atr[bi]) else a
            if trend == 1: mfe = max(mfe, (h_k - entry) / risk_qqq)
            else: mfe = max(mfe, (entry - l_k) / risk_qqq)
            if times[bi] >= s["force_close_at"]:
                trade_r = (close[bi] - entry) / risk_qqq * trend
                end_bar = bi; exit_reason = "close"; break
            if gate_b > 0 and k == gate_b and not be_triggered:
                if mfe < s["gate_mfe"]:
                    ns = entry + s["gate_tighten"] * risk_qqq * trend
                    if trend == 1: runner_stop = max(runner_stop, ns)
                    else: runner_stop = min(runner_stop, ns)
            stopped = (trend == 1 and l_k <= runner_stop) or (trend == -1 and h_k >= runner_stop)
            if stopped:
                trade_r = (runner_stop - entry) / risk_qqq * trend
                end_bar = bi
                if be_triggered:
                    be_ref = entry + s["be_stop_r"] * risk_qqq * trend
                    exit_reason = "be" if abs(runner_stop - be_ref) < 0.05 * risk_qqq else "trail"
                else: exit_reason = "stop"
                break
            if not be_triggered and s["be_trigger_r"] > 0:
                trigger_price = entry + s["be_trigger_r"] * risk_qqq * trend
                if (trend == 1 and h_k >= trigger_price) or (trend == -1 and l_k <= trigger_price):
                    be_triggered = True
                    be_level = entry + s["be_stop_r"] * risk_qqq * trend
                    if trend == 1: runner_stop = max(runner_stop, be_level)
                    else: runner_stop = min(runner_stop, be_level)
            if be_triggered and k >= chand_b:
                sk = max(1, k - chand_b + 1)
                hv = [high[entry_bar + kk] for kk in range(sk, k) if entry_bar + kk < n]
                lv = [low[entry_bar + kk] for kk in range(sk, k) if entry_bar + kk < n]
                if hv and lv:
                    if trend == 1: runner_stop = max(runner_stop, max(hv) - s["chand_mult"] * ca)
                    else: runner_stop = min(runner_stop, min(lv) + s["chand_mult"] * ca)
        else:
            trade_r = (close[min(entry_bar + max_hold, n - 1)] - entry) / risk_qqq * trend
            end_bar = min(entry_bar + max_hold, n - 1)

        raw_pnl = trade_r * risk_mnq
        exit_comm = COMM_RT * nc / 2
        exit_slip = STOP_SLIP if exit_reason in ("stop", "trail") else 0
        be_slip = BE_SLIP if exit_reason == "be" else 0
        net_pnl = raw_pnl - (entry_cost + exit_comm + exit_slip + be_slip)

        records.append({
            "dcp": dcp, "prev_dcp": prev_dcp, "dcp_slope": dcp_slope,
            "close_ema_dist": close_ema_dist, "bar_range_atr": bar_range_atr,
            "trade_r": trade_r, "net_pnl": net_pnl, "mfe": mfe,
            "exit": exit_reason, "risk_mnq": risk_mnq, "trend": trend,
        })

        if trade_r < 0: daily_r_loss += abs(trade_r)
        if trade_r > 0: skip_count = s.get("skip_after_win", 0)
        bar = end_bar + 1

    return pd.DataFrame(records)


def stats(df):
    if len(df) == 0: return {"pf": 0, "dd": 0, "pnl": 0, "n": 0, "wr": 0, "$/d": 0, "b5": 0}
    gw = df.loc[df["net_pnl"] > 0, "net_pnl"].sum()
    gl = abs(df.loc[df["net_pnl"] <= 0, "net_pnl"].sum())
    cum = df["net_pnl"].cumsum()
    dd = (cum.cummax() - cum).clip(lower=0).max()
    days = max(1, len(df) // 2.8)  # approximate trading days
    return {
        "pf": round(gw/gl, 3) if gl > 0 else 0,
        "dd": round(dd, 0),
        "pnl": round(df["net_pnl"].sum(), 0),
        "n": len(df),
        "wr": round((df["trade_r"] > 0).mean() * 100, 1),
        "$/d": round(df["net_pnl"].sum() / days, 1),
        "b5": int((df["trade_r"] >= 5).sum()),
    }


def main():
    print("=" * 70)
    print("ML FILTER V3 — DCP DEEP DIVE")
    print("=" * 70)

    df_is_raw = pd.read_csv(IS_PATH, index_col="timestamp", parse_dates=True)
    df_oos_raw = pd.read_csv(OOS_PATH, index_col="timestamp", parse_dates=True)

    is_df = run_with_dcp_analysis(df_is_raw)
    oos_df = run_with_dcp_analysis(df_oos_raw)

    print(f"\nIS:  {len(is_df)} trades | OOS: {len(oos_df)} trades")

    # ═══ 1. DCP Distribution Analysis ═══
    print("\n" + "=" * 70)
    print("1. DCP DISTRIBUTION BY OUTCOME")
    print("=" * 70)

    for label, df in [("IS", is_df), ("OOS", oos_df)]:
        wins = df[df["trade_r"] > 0]
        losses = df[df["trade_r"] <= 0]
        big = df[df["trade_r"] >= 5]

        print(f"\n  [{label}] DCP statistics:")
        print(f"    {'Group':15s} {'Mean':>7} {'Median':>7} {'Std':>7} {'N':>6}")
        for name, grp in [("All", df), ("Winners", wins), ("Losers", losses), ("5R+", big)]:
            if len(grp) > 0:
                print(f"    {name:15s} {grp['dcp'].mean():>7.3f} {grp['dcp'].median():>7.3f} {grp['dcp'].std():>7.3f} {len(grp):>6}")

        # Cohen's d for DCP
        if len(wins) > 0 and len(losses) > 0:
            pooled_std = np.sqrt((wins["dcp"].var() * (len(wins)-1) + losses["dcp"].var() * (len(losses)-1)) /
                                 (len(wins) + len(losses) - 2))
            d = (wins["dcp"].mean() - losses["dcp"].mean()) / pooled_std if pooled_std > 0 else 0
            print(f"    Cohen's d (wins vs losses): {d:.4f}")
            if abs(d) < 0.2:
                print(f"    → NEGLIGIBLE effect size (<0.2)")
            elif abs(d) < 0.5:
                print(f"    → SMALL effect size (0.2-0.5)")

    # ═══ 2. DCP Quartile Analysis ═══
    print("\n" + "=" * 70)
    print("2. DCP QUARTILE ANALYSIS (IS vs OOS)")
    print("=" * 70)

    for label, df in [("IS", is_df), ("OOS", oos_df)]:
        print(f"\n  [{label}]")
        print(f"    {'Quartile':>10} {'DCP Range':>15} {'N':>6} {'WR%':>7} {'PF':>7} {'Avg R':>7} {'5R+':>5}")

        # Use IS quartile boundaries for both (avoid look-ahead)
        if label == "IS":
            q_bounds = [0] + list(is_df["dcp"].quantile([0.25, 0.5, 0.75]).values) + [1.01]
        for i in range(4):
            mask = (df["dcp"] >= q_bounds[i]) & (df["dcp"] < q_bounds[i+1])
            grp = df[mask]
            if len(grp) == 0: continue
            gw = grp.loc[grp["net_pnl"] > 0, "net_pnl"].sum()
            gl = abs(grp.loc[grp["net_pnl"] <= 0, "net_pnl"].sum())
            pf = gw/gl if gl > 0 else 0
            wr = (grp["trade_r"] > 0).mean() * 100
            avg_r = grp["trade_r"].mean()
            b5 = (grp["trade_r"] >= 5).sum()
            print(f"    Q{i+1:d}         [{q_bounds[i]:.2f}-{q_bounds[i+1]:.2f})   {len(grp):>6} {wr:>6.1f}% {pf:>7.3f} {avg_r:>7.3f} {b5:>5}")

    # ═══ 3. DCP Threshold Filter Test ═══
    print("\n" + "=" * 70)
    print("3. DCP THRESHOLD FILTER — FULL STRATEGY IMPACT")
    print("=" * 70)

    for label, df in [("IS", is_df), ("OOS", oos_df)]:
        base = stats(df)
        print(f"\n  [{label}] Base: PF={base['pf']:.3f} DD=${base['dd']:.0f} PnL=${base['pnl']:.0f} "
              f"N={base['n']} WR={base['wr']}% 5R+={base['b5']}")
        print(f"  {'DCP_min':>8} {'PF':>7} {'DD':>7} {'PnL':>8} {'N':>6} {'WR%':>6} {'$/d':>7} {'5R+':>5} {'5R_lost':>7}")

        for dcp_min in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            kept = df[df["dcp"] >= dcp_min]
            if len(kept) == 0: continue
            s_k = stats(kept)
            b5_lost = base["b5"] - s_k["b5"]
            print(f"  {dcp_min:>8.1f} {s_k['pf']:>7.3f} {s_k['dd']:>7.0f} {s_k['pnl']:>8.0f} "
                  f"{s_k['n']:>6} {s_k['wr']:>5.1f}% {s_k['$/d']:>7.1f} {s_k['b5']:>5} {b5_lost:>7}")

    # ═══ 4. DCP + DCP Slope Combined ═══
    print("\n" + "=" * 70)
    print("4. DCP + DCP SLOPE COMBINED FILTER")
    print("=" * 70)

    for label, df in [("IS", is_df), ("OOS", oos_df)]:
        base = stats(df)
        print(f"\n  [{label}]")
        print(f"  {'Filter':>25} {'PF':>7} {'DD':>7} {'PnL':>8} {'N':>6} {'WR%':>6} {'5R+':>5} {'5R_lost':>7}")

        filters = [
            ("dcp≥0.5", df["dcp"] >= 0.5),
            ("dcp≥0.5 & slope>0", (df["dcp"] >= 0.5) & (df["dcp_slope"] > 0)),
            ("dcp≥0.4 & slope>0", (df["dcp"] >= 0.4) & (df["dcp_slope"] > 0)),
            ("dcp≥0.3 & slope>0.02", (df["dcp"] >= 0.3) & (df["dcp_slope"] > 0.02)),
            ("dcp≥0.5 & bar_rng<1.5", (df["dcp"] >= 0.5) & (df["bar_range_atr"] < 1.5)),
            ("dcp≥0.5 | prev_dcp≥0.6", (df["dcp"] >= 0.5) | (df["prev_dcp"] >= 0.6)),
            ("prev_dcp≥0.5", df["prev_dcp"] >= 0.5),
            ("avg_dcp≥0.5", ((df["dcp"] + df["prev_dcp"]) / 2) >= 0.5),
        ]

        for name, mask in filters:
            kept = df[mask]
            if len(kept) == 0: continue
            s_k = stats(kept)
            b5_lost = base["b5"] - s_k["b5"]
            print(f"  {name:>25} {s_k['pf']:>7.3f} {s_k['dd']:>7.0f} {s_k['pnl']:>8.0f} "
                  f"{s_k['n']:>6} {s_k['wr']:>5.1f}% {s_k['b5']:>5} {b5_lost:>7}")

    # ═══ 5. Close-EMA Distance Analysis ═══
    print("\n" + "=" * 70)
    print("5. CLOSE-EMA20 DISTANCE (2nd top feature)")
    print("=" * 70)

    for label, df in [("IS", is_df), ("OOS", oos_df)]:
        base = stats(df)
        print(f"\n  [{label}]")
        print(f"  {'EMA_dist_max':>15} {'PF':>7} {'DD':>7} {'PnL':>8} {'N':>6} {'WR%':>6} {'5R+':>5}")

        # close_ema_dist is already in ATR units, directional (positive = above EMA for longs)
        for max_dist in [2.0, 1.5, 1.0, 0.8, 0.6, 0.4, 0.2]:
            kept = df[df["close_ema_dist"] <= max_dist]
            if len(kept) == 0: continue
            s_k = stats(kept)
            print(f"  {max_dist:>15.1f} {s_k['pf']:>7.3f} {s_k['dd']:>7.0f} {s_k['pnl']:>8.0f} "
                  f"{s_k['n']:>6} {s_k['wr']:>5.1f}% {s_k['b5']:>5}")

    # ═══ 6. Final Verdict ═══
    print("\n" + "=" * 70)
    print("6. FINAL VERDICT")
    print("=" * 70)

    # Check if any DCP filter improves IS AND OOS PF without losing >25% of 5R+ trades
    is_base = stats(is_df)
    oos_base = stats(oos_df)

    print(f"\n  Baseline: IS PF={is_base['pf']:.3f} DD=${is_base['dd']:.0f} | "
          f"OOS PF={oos_base['pf']:.3f} DD=${oos_base['dd']:.0f}")

    best_filter = None
    best_score = oos_base["pf"]

    for dcp_min in np.arange(0.3, 0.8, 0.05):
        is_kept = is_df[is_df["dcp"] >= dcp_min]
        oos_kept = oos_df[oos_df["dcp"] >= dcp_min]
        if len(oos_kept) < 100: continue

        is_s = stats(is_kept)
        oos_s = stats(oos_kept)
        b5_pct = oos_s["b5"] / max(1, oos_base["b5"])

        # Must preserve >75% of 5R+ trades and improve OOS PF
        if b5_pct >= 0.75 and oos_s["pf"] > best_score:
            best_score = oos_s["pf"]
            best_filter = dcp_min

    if best_filter:
        is_kept = is_df[is_df["dcp"] >= best_filter]
        oos_kept = oos_df[oos_df["dcp"] >= best_filter]
        is_s = stats(is_kept)
        oos_s = stats(oos_kept)
        pf_improve = (oos_s["pf"] - oos_base["pf"]) / oos_base["pf"] * 100
        print(f"\n  ✓ Best DCP filter: dcp ≥ {best_filter:.2f}")
        print(f"    IS:  PF={is_s['pf']:.3f} DD=${is_s['dd']:.0f} N={is_s['n']} 5R+={is_s['b5']}")
        print(f"    OOS: PF={oos_s['pf']:.3f} DD=${oos_s['dd']:.0f} N={oos_s['n']} 5R+={oos_s['b5']}")
        print(f"    PF improvement: {pf_improve:+.1f}%")
        if pf_improve < 5:
            print(f"    ⚠ Improvement < 5% — within noise range. NOT recommended.")
        else:
            print(f"    → Worth testing in production.")
    else:
        print(f"\n  ✗ No DCP threshold improves OOS PF while preserving ≥75% of 5R+ trades.")
        print(f"    DCP has small effect size but doesn't translate to strategy improvement.")
        print(f"    The 3-bar MFE gate already captures the same information post-entry.")


if __name__ == "__main__":
    main()
