"""
Touch Close Entry + Multi-Exit Mega Sweep

Entry: touch bar close (no filter)
Stop: touch_low - 0.3×ATR

Systematic sweep of exit architectures:
  - 1-lock, 2-lock, 3-lock, 4-lock
  - Various R:R targets and split ratios
  - With/without chandelier trail on runner
  - With/without BE move
  - Trailing vs fixed TP on final portion
  - Aggressive early lock vs patient runner
  - Pyramid exits (increasing R targets)
  - Reverse pyramid (big early lock, small runner)
"""
from __future__ import annotations
import functools, datetime as dt
import numpy as np, pandas as pd
from entry_signal import add_indicators, detect_trend, check_touch

print = functools.partial(print, flush=True)
DATA_PATH = "data/QQQ_1Min_Polygon_2y_clean.csv"

BASE = {
    "ema_fast": 20, "ema_slow": 50, "atr_period": 14,
    "touch_tol": 0.15, "touch_below_max": 0.5,
    "stop_buffer": 0.3,
    "risk_pct": 0.01, "max_pos_pct": 0.25,
    "no_entry_after": dt.time(15, 30), "force_close_at": dt.time(15, 58),
    "commission_per_share": 0.005, "daily_loss_r": 2.5,
    "max_hold_bars": 180,
}


def run(df, capital=100_000, exit_cfg=None):
    """
    exit_cfg:
      locks: list of (pct, rr) tuples, e.g. [(0.30, 0.3), (0.30, 1.0), (0.40, 3.0)]
             pct = fraction of ORIGINAL position, rr = R:R target
             last lock can have rr=0 meaning "runner, no fixed TP"
      be_after: which lock # triggers BE move (1=after first, 0=never)
      chand_bars: chandelier lookback for runner (0=off)
      chand_mult: chandelier ATR mult
      chand_after: start chandelier after lock # (1=after first lock)
      trail_atr: simple ATR trail mult for runner (0=off)
      trail_after: start ATR trail after lock #
    """
    cfg = {
        "locks": [(0.05, 0.1)],  # default: BE-lock champion
        "be_after": 1,
        "chand_bars": 40, "chand_mult": 0.5,
        "chand_after": 1,
        "trail_atr": 0, "trail_after": 1,
    }
    if exit_cfg:
        cfg.update(exit_cfg)

    p = BASE.copy()
    df = add_indicators(df, p)
    high = df["High"].values; low = df["Low"].values
    close = df["Close"].values
    ema = df["ema20"].values; ema_s = df["ema50"].values; atr_v = df["atr"].values
    times = df.index.time; dates = df.index.date; n = len(df)
    comm = p["commission_per_share"]

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

        actual_entry = close[bar]
        stop = low[bar] - p["stop_buffer"] * a if trend == 1 else high[bar] + p["stop_buffer"] * a
        risk = abs(actual_entry - stop)
        if risk <= 0: bar += 1; continue
        entry_bar = bar

        shares = max(1, int(equity * p["risk_pct"] / risk))
        if shares * abs(actual_entry) > equity * p["max_pos_pct"]:
            shares = max(1, int(equity * p["max_pos_pct"] / abs(actual_entry)))
        if equity < shares * risk or shares < 1: bar += 1; continue

        # Setup locks
        locks = cfg["locks"]
        lock_shares = []
        for i, (pct, rr) in enumerate(locks):
            if rr > 0:  # has a target
                sh = max(1, int(shares * pct))
                lock_shares.append({"sh": sh, "rr": rr, "done": False})
            # rr=0 means this portion is the runner (no fixed TP)

        remaining = shares
        runner_stop = stop
        trade_pnl = -shares * comm
        end_bar = entry_bar
        locks_done = 0
        be_moved = False

        for k in range(1, p["max_hold_bars"] + 1):
            bi = entry_bar + k
            if bi >= n: break
            h = high[bi]; l = low[bi]
            ca = atr_v[bi] if not np.isnan(atr_v[bi]) else a

            # Force close
            if times[bi] >= p["force_close_at"]:
                trade_pnl += remaining * (close[bi] - actual_entry) * trend - remaining * comm
                end_bar = bi; break

            # Stop check
            stopped = (trend == 1 and l <= runner_stop) or (trend == -1 and h >= runner_stop)
            if stopped:
                trade_pnl += remaining * (runner_stop - actual_entry) * trend - remaining * comm
                end_bar = bi; break

            # Check locks in order
            for lk in lock_shares:
                if lk["done"] or remaining <= lk["sh"]:
                    continue
                target = actual_entry + lk["rr"] * risk * trend
                hit = (trend == 1 and h >= target) or (trend == -1 and l <= target)
                if hit:
                    trade_pnl += lk["sh"] * lk["rr"] * risk - lk["sh"] * comm
                    remaining -= lk["sh"]
                    lk["done"] = True
                    locks_done += 1

                    # BE move
                    if not be_moved and cfg["be_after"] > 0 and locks_done >= cfg["be_after"]:
                        if trend == 1: runner_stop = max(runner_stop, actual_entry)
                        else: runner_stop = min(runner_stop, actual_entry)
                        be_moved = True
                    break  # only one lock per bar

            # Chandelier trail
            if cfg["chand_bars"] > 0 and locks_done >= cfg["chand_after"] and k >= cfg["chand_bars"]:
                sk = max(1, k - cfg["chand_bars"] + 1)
                if trend == 1:
                    hh = max(high[entry_bar + kk] for kk in range(sk, k) if entry_bar + kk < n)
                    runner_stop = max(runner_stop, hh - cfg["chand_mult"] * ca)
                else:
                    ll = min(low[entry_bar + kk] for kk in range(sk, k) if entry_bar + kk < n)
                    runner_stop = min(runner_stop, ll + cfg["chand_mult"] * ca)

            # ATR trail
            if cfg["trail_atr"] > 0 and locks_done >= cfg["trail_after"]:
                if trend == 1:
                    runner_stop = max(runner_stop, h - cfg["trail_atr"] * ca)
                else:
                    runner_stop = min(runner_stop, l + cfg["trail_atr"] * ca)

            if remaining <= 0:
                end_bar = bi; break
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
        return {"pf": 0, "ret": 0, "trades": 0, "tpd": 0, "wr": 0,
                "lpf": 0, "spf": 0, "big5": 0, "avg_win": 0, "avg_loss": 0}
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
        "big5": int((r_arr >= 5).sum()),
        "avg_win": round(gw / wins, 2) if wins > 0 else 0,
        "avg_loss": round(gl / losses, 2) if losses > 0 else 0,
    }


def fmt(label, r, bl_pf=None):
    d = f"({r['pf']-bl_pf:+.3f})" if bl_pf is not None else ""
    return (f"  {label:<56} {r['pf']:>6.3f} {d:>8} {r['ret']:>+7.2f}%"
            f" {r['trades']:>5} {r['wr']:>5.1f}%"
            f" {r['lpf']:>5.2f} {r['spf']:>5.2f} {r['big5']:>4}")


def main():
    df = pd.read_csv(DATA_PATH, index_col="timestamp", parse_dates=True)
    print(f"Data: {len(df):,} bars, {df.index.normalize().nunique()} days\n")

    bl = run(df)  # champion baseline
    hdr = (f"  {'Config':<56} {'PF':>6} {'vs BL':>8} {'Ret%':>8}"
           f" {'Trd':>5} {'WR%':>6} {'L.PF':>6} {'S.PF':>6} {'5R+':>5}")
    sep = f"  {'-'*110}"

    all_results = []

    def test(label, cfg):
        r = run(df, exit_cfg=cfg)
        r["label"] = label
        all_results.append(r)
        return r

    # ═══ 1. Baseline references ═══
    print(f"{'='*115}")
    print(f"  1. BASELINES")
    print(f"{'='*115}")
    print(hdr); print(sep)
    all_results.append({**bl, "label": "★ Champion: BE-lock 5%@0.1R + chand 40/0.5"})
    print(fmt("★ Champion: BE-lock 5%@0.1R + chand 40/0.5", bl))

    # ═══ 2. 3-3-4 split with various R:R combos ═══
    print(f"\n{'='*115}")
    print(f"  2. THREE-LOCK 30/30/40 SPLIT (various R:R targets)")
    print(f"{'='*115}")
    print(hdr); print(sep)

    targets_334 = [
        # (label, [(pct, rr), ...], be_after, chand_after)
        ("334: 0.2R/0.5R/runner",      [(0.30,0.2),(0.30,0.5)], 1, 2),
        ("334: 0.3R/1R/runner",         [(0.30,0.3),(0.30,1.0)], 1, 2),
        ("334: 0.3R/1.5R/runner",       [(0.30,0.3),(0.30,1.5)], 1, 2),
        ("334: 0.3R/2R/runner",         [(0.30,0.3),(0.30,2.0)], 1, 2),
        ("334: 0.5R/1R/runner",         [(0.30,0.5),(0.30,1.0)], 1, 2),
        ("334: 0.5R/1.5R/runner",       [(0.30,0.5),(0.30,1.5)], 1, 2),
        ("334: 0.5R/2R/runner",         [(0.30,0.5),(0.30,2.0)], 1, 2),
        ("334: 0.5R/3R/runner",         [(0.30,0.5),(0.30,3.0)], 1, 2),
        ("334: 1R/2R/runner",           [(0.30,1.0),(0.30,2.0)], 1, 2),
        ("334: 1R/3R/runner",           [(0.30,1.0),(0.30,3.0)], 1, 2),
        ("334: 1R/5R/runner",           [(0.30,1.0),(0.30,5.0)], 1, 2),
        ("334: 0.1R/0.3R/runner",       [(0.30,0.1),(0.30,0.3)], 1, 2),
        ("334: 0.1R/0.5R/runner",       [(0.30,0.1),(0.30,0.5)], 1, 2),
        ("334: 0.1R/1R/runner",         [(0.30,0.1),(0.30,1.0)], 1, 2),
        ("334: 0.2R/1R/runner",         [(0.30,0.2),(0.30,1.0)], 1, 2),
    ]
    for label, locks, be, ca in targets_334:
        r = test(label, {"locks": locks, "be_after": be, "chand_after": ca})
        print(fmt(label, r, bl["pf"]))

    # ═══ 3. Different split ratios ═══
    print(f"\n{'='*115}")
    print(f"  3. SPLIT RATIO VARIATIONS (fixed R:R = 0.3R/1R/runner)")
    print(f"{'='*115}")
    print(hdr); print(sep)

    splits = [
        ("10/10/80: 0.3R/1R/runner", [(0.10,0.3),(0.10,1.0)]),
        ("10/20/70: 0.3R/1R/runner", [(0.10,0.3),(0.20,1.0)]),
        ("20/20/60: 0.3R/1R/runner", [(0.20,0.3),(0.20,1.0)]),
        ("20/30/50: 0.3R/1R/runner", [(0.20,0.3),(0.30,1.0)]),
        ("30/30/40: 0.3R/1R/runner", [(0.30,0.3),(0.30,1.0)]),
        ("40/30/30: 0.3R/1R/runner", [(0.40,0.3),(0.30,1.0)]),
        ("50/25/25: 0.3R/1R/runner", [(0.50,0.3),(0.25,1.0)]),
        ("5/5/90: 0.3R/1R/runner",   [(0.05,0.3),(0.05,1.0)]),
        ("5/10/85: 0.1R/0.5R/runner",[(0.05,0.1),(0.10,0.5)]),
        ("5/15/80: 0.1R/1R/runner",  [(0.05,0.1),(0.15,1.0)]),
    ]
    for label, locks in splits:
        r = test(label, {"locks": locks, "be_after": 1, "chand_after": 2})
        print(fmt(label, r, bl["pf"]))

    # ═══ 4. BE timing experiments ═══
    print(f"\n{'='*115}")
    print(f"  4. BREAKEVEN TIMING (334: 0.3R/1R/runner)")
    print(f"{'='*115}")
    print(hdr); print(sep)

    locks_std = [(0.30, 0.3), (0.30, 1.0)]
    for be in [0, 1, 2]:
        label = f"BE after lock #{be}" if be > 0 else "No BE move"
        r = test(label, {"locks": locks_std, "be_after": be, "chand_after": 2})
        print(fmt(label, r, bl["pf"]))

    # ═══ 5. Chandelier timing & params ═══
    print(f"\n{'='*115}")
    print(f"  5. CHANDELIER VARIATIONS (334: 0.3R/1R/runner)")
    print(f"{'='*115}")
    print(hdr); print(sep)

    for ca in [1, 2]:
        for cb in [20, 30, 40, 60]:
            for cm in [0.3, 0.5, 0.8]:
                label = f"Chand {cb}/{cm} after lock#{ca}"
                r = test(label, {"locks": locks_std, "be_after": 1,
                                 "chand_bars": cb, "chand_mult": cm, "chand_after": ca})
                print(fmt(label, r, bl["pf"]))

    # ═══ 6. ATR trail instead of chandelier ═══
    print(f"\n{'='*115}")
    print(f"  6. ATR TRAIL INSTEAD OF CHANDELIER (334: 0.3R/1R/runner)")
    print(f"{'='*115}")
    print(hdr); print(sep)

    for ta in [0.3, 0.5, 0.8, 1.0, 1.5, 2.0]:
        r = test(f"ATR trail ×{ta} after lock#1",
                 {"locks": locks_std, "be_after": 1, "chand_bars": 0,
                  "trail_atr": ta, "trail_after": 1})
        print(fmt(f"ATR trail ×{ta} after lock#1", r, bl["pf"]))

    # ═══ 7. 4-lock configurations ═══
    print(f"\n{'='*115}")
    print(f"  7. FOUR-LOCK CONFIGURATIONS")
    print(f"{'='*115}")
    print(hdr); print(sep)

    four_locks = [
        ("25/25/25/25: 0.2/0.5/1/runner",   [(0.25,0.2),(0.25,0.5),(0.25,1.0)]),
        ("25/25/25/25: 0.3/1/2/runner",      [(0.25,0.3),(0.25,1.0),(0.25,2.0)]),
        ("25/25/25/25: 0.5/1/3/runner",      [(0.25,0.5),(0.25,1.0),(0.25,3.0)]),
        ("10/20/30/40: 0.1/0.5/1.5/runner",  [(0.10,0.1),(0.20,0.5),(0.30,1.5)]),
        ("10/15/25/50: 0.1/0.3/1/runner",    [(0.10,0.1),(0.15,0.3),(0.25,1.0)]),
        ("5/10/15/70: 0.1/0.3/1/runner",     [(0.05,0.1),(0.10,0.3),(0.15,1.0)]),
        ("5/5/10/80: 0.1/0.5/1.5/runner",    [(0.05,0.1),(0.05,0.5),(0.10,1.5)]),
    ]
    for label, locks in four_locks:
        r = test(label, {"locks": locks, "be_after": 1, "chand_after": len(locks)})
        print(fmt(label, r, bl["pf"]))

    # ═══ 8. Aggressive early exit (reverse pyramid) ═══
    print(f"\n{'='*115}")
    print(f"  8. AGGRESSIVE EARLY EXIT (big first lock, small runner)")
    print(f"{'='*115}")
    print(hdr); print(sep)

    aggressive = [
        ("60/20/20: 0.2R/0.5R/runner",  [(0.60,0.2),(0.20,0.5)]),
        ("60/20/20: 0.3R/1R/runner",     [(0.60,0.3),(0.20,1.0)]),
        ("70/15/15: 0.3R/1R/runner",     [(0.70,0.3),(0.15,1.0)]),
        ("50/30/20: 0.2R/0.5R/runner",   [(0.50,0.2),(0.30,0.5)]),
        ("80/10/10: 0.2R/1R/runner",     [(0.80,0.2),(0.10,1.0)]),
    ]
    for label, locks in aggressive:
        r = test(label, {"locks": locks, "be_after": 1, "chand_after": 2})
        print(fmt(label, r, bl["pf"]))

    # ═══ 9. Patient runner (tiny locks, max runner) ═══
    print(f"\n{'='*115}")
    print(f"  9. PATIENT RUNNER (tiny locks, max runner)")
    print(f"{'='*115}")
    print(hdr); print(sep)

    patient = [
        ("3/3/94: 0.1R/0.3R/runner",    [(0.03,0.1),(0.03,0.3)]),
        ("5/5/90: 0.1R/0.5R/runner",    [(0.05,0.1),(0.05,0.5)]),
        ("5/5/90: 0.1R/1R/runner",       [(0.05,0.1),(0.05,1.0)]),
        ("2/3/95: 0.1R/0.3R/runner",    [(0.02,0.1),(0.03,0.3)]),
        ("5/10/85: 0.05R/0.2R/runner",  [(0.05,0.05),(0.10,0.2)]),
        ("3/3/3/91: 0.1/0.3/0.5/run",   [(0.03,0.1),(0.03,0.3),(0.03,0.5)]),
    ]
    for label, locks in patient:
        r = test(label, {"locks": locks, "be_after": 1, "chand_after": len(locks)})
        print(fmt(label, r, bl["pf"]))

    # ═══ 10. All-fixed (no runner) ═══
    print(f"\n{'='*115}")
    print(f"  10. ALL-FIXED (close everything at target, no runner)")
    print(f"{'='*115}")
    print(hdr); print(sep)

    all_fixed = [
        ("334 all-fixed: 0.3/1/2R",     [(0.30,0.3),(0.30,1.0),(0.40,2.0)]),
        ("334 all-fixed: 0.5/1/3R",     [(0.30,0.5),(0.30,1.0),(0.40,3.0)]),
        ("334 all-fixed: 0.3/1/5R",     [(0.30,0.3),(0.30,1.0),(0.40,5.0)]),
        ("334 all-fixed: 1/2/5R",       [(0.30,1.0),(0.30,2.0),(0.40,5.0)]),
        ("50/50 all-fixed: 0.5/2R",     [(0.50,0.5),(0.50,2.0)]),
        ("50/50 all-fixed: 1/3R",       [(0.50,1.0),(0.50,3.0)]),
        ("100% fixed: 1R",              [(1.00,1.0)]),
        ("100% fixed: 2R",              [(1.00,2.0)]),
        ("100% fixed: 3R",              [(1.00,3.0)]),
    ]
    for label, locks in all_fixed:
        r = test(label, {"locks": locks, "be_after": 1, "chand_bars": 0})
        print(fmt(label, r, bl["pf"]))

    # ═══ FINAL RANKING ═══
    print(f"\n{'='*115}")
    print(f"  FINAL RANKING — TOP 25 BY PROFIT FACTOR")
    print(f"{'='*115}")

    all_results.sort(key=lambda x: x["pf"], reverse=True)
    print(f"\n  {'#':>3} {'Config':<56} {'PF':>6} {'Ret%':>8} {'Trd':>5} {'WR%':>6}"
          f" {'L.PF':>6} {'S.PF':>6} {'5R+':>5}")
    print(f"  {'-'*100}")
    for i, r in enumerate(all_results[:25]):
        marker = " ★" if r["pf"] >= bl["pf"] else ""
        print(f"  {i+1:>3} {r['label']:<56} {r['pf']:>6.3f} {r['ret']:>+7.2f}%"
              f" {r['trades']:>5} {r['wr']:>5.1f}%"
              f" {r['lpf']:>5.2f} {r['spf']:>5.2f} {r['big5']:>4}{marker}")

    # Return vs PF frontier
    print(f"\n{'='*115}")
    print(f"  TOP 10 BY RETURN")
    print(f"{'='*115}")
    by_ret = sorted(all_results, key=lambda x: x["ret"], reverse=True)
    print(f"\n  {'#':>3} {'Config':<56} {'PF':>6} {'Ret%':>8} {'5R+':>5}")
    print(f"  {'-'*80}")
    for i, r in enumerate(by_ret[:10]):
        print(f"  {i+1:>3} {r['label']:<56} {r['pf']:>6.3f} {r['ret']:>+7.2f}% {r['big5']:>4}")


if __name__ == "__main__":
    main()
