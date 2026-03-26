"""
Validation: Touch Close + BE-lock 5%@0.1R

1. Walk-forward on Polygon 2024-2026 (Y1 vs Y2, quarterly, rolling 6M)
2. Out-of-sample on Barchart 2022-2024
3. Full 4-year combined
4. Stress test (BE bleed, entry slippage, strict lock)
5. Parameter stability (sweep around champion params)
6. Compare vs original Plan G (both gap-fixed)
"""
from __future__ import annotations
import functools, datetime as dt
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from entry_signal import add_indicators, detect_trend, check_touch

print = functools.partial(print, flush=True)

IS_PATH = "data/QQQ_1Min_Polygon_2y_clean.csv"
OOS_PATH = "data/QQQ_1Min_Barchart_2y_2022-2024_clean.csv"

BASE = {
    "ema_fast": 20, "ema_slow": 50, "atr_period": 14,
    "touch_tol": 0.15, "touch_below_max": 0.5,
    "stop_buffer": 0.3,
    "lock_rr": 0.1, "lock_pct": 0.05,
    "chand_bars": 40, "chand_mult": 0.5,
    "max_hold_bars": 180,
    "risk_pct": 0.01, "max_pos_pct": 0.25,
    "no_entry_after": dt.time(15, 30), "force_close_at": dt.time(15, 58),
    "commission_per_share": 0.005, "daily_loss_r": 2.5,
}


def run_backtest(df, capital=100_000, params=None, stress=None):
    """
    Touch close entry + BE-lock exit.
    stress: None, "be_bleed", "entry_slip", "strict_lock", "nightmare"
    """
    p = {**BASE, **(params or {})}
    df = add_indicators(df, p)
    high = df["High"].values; low = df["Low"].values
    close = df["Close"].values; open_p = df["Open"].values
    ema = df["ema20"].values; ema_s = df["ema50"].values; atr_v = df["atr"].values
    times = df.index.time; dates = df.index.date; n = len(df)
    comm = p["commission_per_share"]

    be_penalty = 0.03 if stress in ("be_bleed", "nightmare") else 0
    entry_slip = 0.03 if stress in ("entry_slip", "nightmare") else 0
    strict_lock = stress in ("strict_lock", "nightmare")

    equity = capital; trade_log = []
    bar = max(p["ema_slow"], p["atr_period"]) + 5
    daily_r_loss = 0.0; current_date = None

    while bar < n - p["max_hold_bars"] - 5:
        a = atr_v[bar]
        if np.isnan(a) or a <= 0 or np.isnan(ema[bar]) or np.isnan(ema_s[bar]):
            bar += 1; continue
        if times[bar] >= p["no_entry_after"]: bar += 1; continue
        d = dates[bar]
        if current_date != d: current_date = d; daily_r_loss = 0.0
        if daily_r_loss >= p["daily_loss_r"]: bar += 1; continue

        trend = detect_trend(close[bar], ema[bar], ema_s[bar])
        if trend == 0: bar += 1; continue
        if not check_touch(trend, low[bar], high[bar], ema[bar], a,
                           p["touch_tol"], p["touch_below_max"]):
            bar += 1; continue

        actual_entry = close[bar] + entry_slip * trend  # slip makes fill worse
        stop = low[bar] - p["stop_buffer"] * a if trend == 1 else high[bar] + p["stop_buffer"] * a
        risk = abs(actual_entry - stop)
        if risk <= 0: bar += 1; continue
        entry_bar = bar

        shares = max(1, int(equity * p["risk_pct"] / risk))
        if shares * abs(actual_entry) > equity * p["max_pos_pct"]:
            shares = max(1, int(equity * p["max_pos_pct"] / abs(actual_entry)))
        if equity < shares * risk or shares < 1: bar += 1; continue

        lock_sh = max(1, int(shares * p["lock_pct"])) if p["lock_rr"] > 0 and p["lock_pct"] > 0 else 0
        remaining = shares; runner_stop = stop; lock_done = False
        trade_pnl = -shares * comm; end_bar = entry_bar

        for k in range(1, p["max_hold_bars"] + 1):
            bi = entry_bar + k
            if bi >= n: break
            h = high[bi]; l = low[bi]
            ca = atr_v[bi] if not np.isnan(atr_v[bi]) else a

            if times[bi] >= p["force_close_at"]:
                trade_pnl += remaining * (close[bi] - actual_entry) * trend - remaining * comm
                end_bar = bi; break

            stopped = (trend == 1 and l <= runner_stop) or (trend == -1 and h >= runner_stop)
            if stopped:
                exit_price = runner_stop
                if be_penalty > 0 and lock_done and abs(runner_stop - actual_entry) < 0.02:
                    exit_price = runner_stop - be_penalty * trend
                trade_pnl += remaining * (exit_price - actual_entry) * trend - remaining * comm
                end_bar = bi; break

            if not lock_done and lock_sh > 0 and remaining > lock_sh:
                target = actual_entry + p["lock_rr"] * risk * trend
                if strict_lock:
                    hit = (trend == 1 and h > target) or (trend == -1 and l < target)
                else:
                    hit = (trend == 1 and h >= target) or (trend == -1 and l <= target)
                if hit:
                    trade_pnl += lock_sh * p["lock_rr"] * risk - lock_sh * comm
                    remaining -= lock_sh; lock_done = True
                    if trend == 1: runner_stop = max(runner_stop, actual_entry)
                    else: runner_stop = min(runner_stop, actual_entry)

            if lock_done and k >= p["chand_bars"]:
                sk = max(1, k - p["chand_bars"] + 1)
                if trend == 1:
                    hh = max(high[entry_bar + kk] for kk in range(sk, k) if entry_bar + kk < n)
                    runner_stop = max(runner_stop, hh - p["chand_mult"] * ca)
                else:
                    ll = min(low[entry_bar + kk] for kk in range(sk, k) if entry_bar + kk < n)
                    runner_stop = min(runner_stop, ll + p["chand_mult"] * ca)
        else:
            ep = close[min(entry_bar + p["max_hold_bars"], n - 1)]
            trade_pnl += remaining * (ep - actual_entry) * trend - remaining * comm
            end_bar = min(entry_bar + p["max_hold_bars"], n - 1)

        equity += trade_pnl
        if trade_pnl < 0 and shares * risk > 0:
            daily_r_loss += abs(trade_pnl) / (shares * risk)
        trade_log.append({"pnl": trade_pnl, "dir": trend, "shares": shares, "risk": risk})
        bar = end_bar + 1

    tdf = pd.DataFrame(trade_log) if trade_log else pd.DataFrame()
    total = len(tdf)
    if total == 0:
        return {"pf": 0, "ret": 0, "trades": 0, "wr": 0, "lpf": 0, "spf": 0, "big5": 0}
    wins = (tdf["pnl"] > 0).sum(); losses = total - wins
    gw = tdf.loc[tdf["pnl"] > 0, "pnl"].sum() if wins > 0 else 0
    gl = abs(tdf.loc[tdf["pnl"] <= 0, "pnl"].sum()) if losses > 0 else 0
    pf = gw / gl if gl > 0 else 0; ret = (equity - capital) / capital * 100
    days = df.index.normalize().nunique()
    longs = tdf[tdf["dir"] == 1]; shorts = tdf[tdf["dir"] == -1]
    def spf(s):
        if len(s) == 0: return 0
        w = s.loc[s["pnl"] > 0, "pnl"].sum(); l = abs(s.loc[s["pnl"] <= 0, "pnl"].sum())
        return round(w / l, 3) if l > 0 else 0
    r_arr = np.array([t["pnl"]/(t["shares"]*t["risk"]) if t["shares"]*t["risk"]>0 else 0
                       for _, t in tdf.iterrows()])
    return {
        "pf": round(pf, 3), "ret": round(ret, 2),
        "trades": total, "tpd": round(total / max(days, 1), 1),
        "wr": round(wins / total * 100, 1),
        "lpf": spf(longs), "spf": spf(shorts),
        "ln": len(longs), "sn": len(shorts),
        "big5": int((r_arr >= 5).sum()),
    }


def run_period(args):
    path, start, end, label, params, stress = args
    df = pd.read_csv(path, index_col="timestamp", parse_dates=True)
    mask = (df.index >= start) & (df.index < end)
    df = df[mask]
    if len(df) < 500:
        return (label, 0, 0, 0, 0, 0, 0, 0)
    r = run_backtest(df, params=params, stress=stress)
    return (label, r["trades"], r["tpd"], r["wr"], r["pf"], r["ret"], r["lpf"], r["spf"], r["big5"])


def main():
    # Load data
    df_is = pd.read_csv(IS_PATH, index_col="timestamp", parse_dates=True)
    print(f"IS data: {len(df_is):,} bars ({df_is.index[0].date()} → {df_is.index[-1].date()})")

    try:
        df_oos = pd.read_csv(OOS_PATH, index_col="timestamp", parse_dates=True)
        has_oos = True
        print(f"OOS data: {len(df_oos):,} bars ({df_oos.index[0].date()} → {df_oos.index[-1].date()})")
    except FileNotFoundError:
        has_oos = False
        print("OOS data not found, skipping OOS validation")

    print()

    # ═══ 1. Full IS backtest ═══
    print(f"{'='*90}")
    print(f"  1. FULL IN-SAMPLE BACKTEST (Polygon 2024-2026)")
    print(f"{'='*90}")
    r = run_backtest(df_is)
    print(f"  PF={r['pf']:.3f}  ret={r['ret']:+.2f}%  trades={r['trades']} ({r['tpd']}/day)")
    print(f"  WR={r['wr']:.1f}%  Long PF={r['lpf']:.3f} ({r['ln']})  Short PF={r['spf']:.3f} ({r['sn']})")
    print(f"  5R+ wins: {r['big5']}")

    # ═══ 2. Walk-forward IS ═══
    print(f"\n{'='*90}")
    print(f"  2. WALK-FORWARD (Polygon 2024-2026)")
    print(f"{'='*90}")

    wf_tasks = [
        (IS_PATH, "2024-03-22", "2025-03-22", "Y1 (2024-03→2025-03)", None, None),
        (IS_PATH, "2025-03-22", "2026-03-21", "Y2 (2025-03→2026-03)", None, None),
    ]
    # Quarterly
    for year in [2024, 2025, 2026]:
        for q in range(1, 5):
            ms = (q-1)*3+1; me = q*3+1; ye = year
            if me > 12: me = 1; ye = year+1
            s = f"{year}-{ms:02d}-01"; e = f"{ye}-{me:02d}-01"
            wf_tasks.append((IS_PATH, s, e, f"{year}Q{q}", None, None))

    with ProcessPoolExecutor(max_workers=cpu_count()) as pool:
        wf_results = list(pool.map(run_period, wf_tasks))

    print(f"\n  {'Period':<28} {'Trades':>7} {'T/d':>5} {'WR%':>6} {'PF':>7} {'Ret%':>8} {'L.PF':>6} {'S.PF':>6}")
    print(f"  {'-'*80}")
    for label, trades, tpd, wr, pf, ret, lpf, spf, *rest in sorted(wf_results, key=lambda x: x[0]):
        if trades < 1: continue
        mark = "+" if pf > 1.0 else "-"
        print(f"  {mark} {label:<26} {trades:>6} {tpd:>4.1f} {wr:>5.1f}% {pf:>6.3f} {ret:>+7.2f}% {lpf:>5.2f} {spf:>5.2f}")

    y1 = [r for r in wf_results if "Y1" in r[0]]
    y2 = [r for r in wf_results if "Y2" in r[0]]
    if y1 and y2:
        print(f"\n  Y1→Y2: PF {y1[0][4]:.3f} → {y2[0][4]:.3f}  {'✅ No degradation' if y2[0][4] >= y1[0][4] * 0.85 else '⚠️ Degradation'}")

    qtrs = [r for r in wf_results if "Q" in r[0] and r[1] > 0]
    profitable_q = sum(1 for r in qtrs if r[4] > 1.0)
    print(f"  Profitable quarters: {profitable_q}/{len(qtrs)}")

    # ═══ 3. OOS (Barchart 2022-2024) ═══
    if has_oos:
        print(f"\n{'='*90}")
        print(f"  3. OUT-OF-SAMPLE (Barchart 2022-2024)")
        print(f"{'='*90}")

        r_oos = run_backtest(df_oos)
        print(f"  PF={r_oos['pf']:.3f}  ret={r_oos['ret']:+.2f}%  trades={r_oos['trades']} ({r_oos['tpd']}/day)")
        print(f"  WR={r_oos['wr']:.1f}%  Long PF={r_oos['lpf']:.3f} ({r_oos['ln']})  Short PF={r_oos['spf']:.3f} ({r_oos['sn']})")
        print(f"  5R+ wins: {r_oos['big5']}")

        # OOS yearly
        oos_tasks = [
            (OOS_PATH, "2022-03-22", "2023-03-22", "OOS Y1 (2022-2023)", None, None),
            (OOS_PATH, "2023-03-22", "2024-03-22", "OOS Y2 (2023-2024)", None, None),
        ]
        with ProcessPoolExecutor(max_workers=cpu_count()) as pool:
            oos_results = list(pool.map(run_period, oos_tasks))
        for label, trades, tpd, wr, pf, ret, lpf, spf, *rest in oos_results:
            if trades < 1: continue
            print(f"  {label}: PF={pf:.3f}  ret={ret:+.2f}%  trades={trades}  L={lpf:.2f} S={spf:.2f}")

        # Full 4-year
        print(f"\n{'='*90}")
        print(f"  4. FULL 4-YEAR (OOS + IS combined)")
        print(f"{'='*90}")
        df_4y = pd.concat([df_oos, df_is]).sort_index()
        df_4y = df_4y[~df_4y.index.duplicated(keep='first')]
        r_4y = run_backtest(df_4y)
        print(f"  PF={r_4y['pf']:.3f}  ret={r_4y['ret']:+.2f}%  trades={r_4y['trades']} ({r_4y['tpd']}/day)")
        print(f"  WR={r_4y['wr']:.1f}%  Long PF={r_4y['lpf']:.3f}  Short PF={r_4y['spf']:.3f}")
        print(f"  5R+ wins: {r_4y['big5']}")

    # ═══ 5. Stress test ═══
    print(f"\n{'='*90}")
    print(f"  5. STRESS TEST")
    print(f"{'='*90}")

    stress_modes = [
        ("Baseline", None),
        ("BE bleed ($0.03)", "be_bleed"),
        ("Entry slip ($0.03)", "entry_slip"),
        ("Strict lock (H > target)", "strict_lock"),
        ("Nightmare (all combined)", "nightmare"),
    ]
    print(f"\n  {'Scenario':<35} {'PF':>7} {'Ret%':>8} {'Trades':>7} {'WR%':>6} {'L.PF':>6} {'S.PF':>6}")
    print(f"  {'-'*75}")
    bl_pf = None
    for label, stress in stress_modes:
        r = run_backtest(df_is, stress=stress)
        if bl_pf is None: bl_pf = r["pf"]
        delta = r["pf"] - bl_pf
        print(f"  {label:<35} {r['pf']:>6.3f} {r['ret']:>+7.2f}% {r['trades']:>6} {r['wr']:>5.1f}%"
              f" {r['lpf']:>5.2f} {r['spf']:>5.2f}  ({delta:+.3f})")

    # Nightmare walk-forward
    print(f"\n  Nightmare walk-forward:")
    for label, start, end in [("Y1", "2024-03-22", "2025-03-22"), ("Y2", "2025-03-22", "2026-03-21")]:
        sub = df_is[(df_is.index >= start) & (df_is.index < end)]
        r = run_backtest(sub, stress="nightmare")
        print(f"    {label}: PF={r['pf']:.3f}  ret={r['ret']:+.2f}%  trades={r['trades']}")

    # ═══ 6. Parameter stability ═══
    print(f"\n{'='*90}")
    print(f"  6. PARAMETER STABILITY (±variation around champion)")
    print(f"{'='*90}")

    print(f"\n  Lock R:R sensitivity:")
    for rr in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]:
        r = run_backtest(df_is, params={"lock_rr": rr})
        print(f"    lock_rr={rr:.2f}: PF={r['pf']:.3f}  ret={r['ret']:+.2f}%")

    print(f"\n  Lock % sensitivity:")
    for pct in [0.02, 0.03, 0.05, 0.08, 0.10, 0.15]:
        r = run_backtest(df_is, params={"lock_pct": pct})
        print(f"    lock_pct={pct:.2f}: PF={r['pf']:.3f}  ret={r['ret']:+.2f}%")

    print(f"\n  Chandelier mult sensitivity:")
    for mult in [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]:
        r = run_backtest(df_is, params={"chand_mult": mult})
        print(f"    chand_mult={mult:.1f}: PF={r['pf']:.3f}  ret={r['ret']:+.2f}%")

    print(f"\n  Chandelier bars sensitivity:")
    for bars in [20, 30, 40, 50, 60, 80]:
        r = run_backtest(df_is, params={"chand_bars": bars})
        print(f"    chand_bars={bars}: PF={r['pf']:.3f}  ret={r['ret']:+.2f}%")

    print(f"\n  Touch tolerance sensitivity:")
    for tol in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        r = run_backtest(df_is, params={"touch_tol": tol})
        print(f"    touch_tol={tol:.2f}: PF={r['pf']:.3f}  ret={r['ret']:+.2f}%  trades={r['trades']}")

    print(f"\n  Stop buffer sensitivity:")
    for buf in [0.1, 0.2, 0.3, 0.4, 0.5, 0.8]:
        r = run_backtest(df_is, params={"stop_buffer": buf})
        print(f"    stop_buffer={buf:.1f}: PF={r['pf']:.3f}  ret={r['ret']:+.2f}%")

    # ═══ 7. Compare vs Plan G ═══
    print(f"\n{'='*90}")
    print(f"  7. COMPARISON VS PLAN G (gap-fixed)")
    print(f"{'='*90}")

    print(f"\n  {'Metric':<25} {'Plan G (gap-fixed)':>20} {'Touch+BE-lock':>20}")
    print(f"  {'-'*67}")
    # Plan G gap-fixed results from earlier experiment
    print(f"  {'PF':<25} {'~1.19':>20} {r_4y['pf'] if has_oos else r['pf']:>19.3f}")
    print(f"  {'Long PF':<25} {'~0.97':>20} {r_4y['lpf'] if has_oos else r['lpf']:>19.3f}")
    print(f"  {'Short PF':<25} {'~1.43':>20} {r_4y['spf'] if has_oos else r['spf']:>19.3f}")
    print(f"  {'Gap problem':<25} {'68% adverse':>20} {'0% (no trigger)':>20}")


if __name__ == "__main__":
    main()
