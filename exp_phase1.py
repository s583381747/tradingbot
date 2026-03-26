"""
Phase 1: Single-variable exit experiments.

New baseline: 14:00 cutoff + skip 1 after win (PF=1.624, MaxDD=15.25R)

1A: BE alternatives (10 experiments)
1B: Chandelier optimization (10 experiments)
1C: Lock reassessment (depends on 1A)
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
    "lock_rr": 0.1, "lock_pct": 0.05,
    "chand_bars": 40, "chand_mult": 0.5,
    "max_hold_bars": 180,
    "risk_pct": 0.01, "max_pos_pct": 0.25,
    "no_entry_after": dt.time(14, 0),  # 0D baked in
    "force_close_at": dt.time(15, 58),
    "commission_per_share": 0.005, "daily_loss_r": 2.5,
    "skip_after_win": 1,  # 0D baked in
}


def run(df, capital=100_000, cfg=None):
    """
    Configurable exit engine.

    cfg keys:
      # BE mechanism
      be_type: "exact"|"offset"|"mfe"|"time"|"progressive"|"none"
      be_offset: float  # R offset from entry (positive = in profit, negative = allow loss)
      be_mfe_trigger: float  # MFE in R before BE activates
      be_time_trigger: int  # bars before BE activates
      be_prog_step: float  # R per step
      be_prog_interval: int  # bars between steps

      # Chandelier
      chand_start: int  # bar to start chandelier
      chand_mult_start: float  # initial mult
      chand_mult_end: float  # final mult (for progressive)
      chand_tighten_bars: int  # bars to go from start to end mult (0 = fixed)

      # Lock
      lock_rr: float
      lock_pct: float

      # Labels
      complexity: str
    """
    c = {
        "be_type": "exact", "be_offset": 0.0,
        "be_mfe_trigger": 0, "be_time_trigger": 0,
        "be_prog_step": 0.1, "be_prog_interval": 5,
        "chand_start": 40, "chand_mult_start": 0.5, "chand_mult_end": 0.5,
        "chand_tighten_bars": 0,
        "lock_rr": 0.1, "lock_pct": 0.05,
        "complexity": "simple",
    }
    if cfg:
        c.update(cfg)

    p = BASE.copy()
    df = add_indicators(df, p)
    high = df["High"].values; low = df["Low"].values
    close = df["Close"].values
    ema = df["ema20"].values; ema_s = df["ema50"].values; atr_v = df["atr"].values
    times = df.index.time; dates = df.index.date; n = len(df)
    comm = p["commission_per_share"]

    equity = capital; trades = []
    bar = max(p["ema_slow"], p["atr_period"]) + 5
    daily_r_loss = 0.0; current_date = None
    skip_count = 0

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

        if skip_count > 0:
            skip_count -= 1; bar += 1; continue

        actual_entry = close[bar]
        stop = low[bar] - p["stop_buffer"] * a if trend == 1 else high[bar] + p["stop_buffer"] * a
        risk = abs(actual_entry - stop)
        if risk <= 0: bar += 1; continue
        entry_bar = bar

        shares = max(1, int(equity * p["risk_pct"] / risk))
        if shares * abs(actual_entry) > equity * p["max_pos_pct"]:
            shares = max(1, int(equity * p["max_pos_pct"] / abs(actual_entry)))
        if equity < shares * risk or shares < 1: bar += 1; continue

        # Lock setup
        use_lock = c["lock_rr"] > 0 and c["lock_pct"] > 0
        lock_sh = max(1, int(shares * c["lock_pct"])) if use_lock else 0
        remaining = shares; runner_stop = stop; lock_done = False
        be_activated = False
        trade_pnl = -shares * comm; end_bar = entry_bar
        exit_reason = "timeout"; bars_held = 0
        mfe_r = 0.0  # track MFE for mfe-based BE
        max_dd_r = 0.0
        # Progressive BE state
        prog_stop = stop  # current progressive stop level

        for k in range(1, p["max_hold_bars"] + 1):
            bi = entry_bar + k
            if bi >= n: break
            h = high[bi]; l = low[bi]
            ca = atr_v[bi] if not np.isnan(atr_v[bi]) else a

            # Track MFE
            if trend == 1:
                cur_fav = (h - actual_entry) / risk
                cur_adv = (actual_entry - l) / risk
            else:
                cur_fav = (actual_entry - l) / risk
                cur_adv = (h - actual_entry) / risk
            mfe_r = max(mfe_r, cur_fav)
            max_dd_r = max(max_dd_r, cur_adv)

            # Force close
            if times[bi] >= p["force_close_at"]:
                trade_pnl += remaining * (close[bi] - actual_entry) * trend - remaining * comm
                end_bar = bi; exit_reason = "session_close"; bars_held = k; break

            # ─── BE activation logic ───
            if not be_activated and (lock_done or not use_lock):
                activate = False
                if c["be_type"] == "exact":
                    activate = lock_done  # activate immediately after lock
                elif c["be_type"] == "offset":
                    activate = lock_done
                elif c["be_type"] == "mfe":
                    activate = mfe_r >= c["be_mfe_trigger"]
                elif c["be_type"] == "time":
                    activate = k >= c["be_time_trigger"]
                elif c["be_type"] == "progressive":
                    activate = lock_done  # progressive starts after lock
                elif c["be_type"] == "none":
                    activate = False  # never move to BE

                if activate:
                    be_activated = True
                    if c["be_type"] == "exact":
                        be_level = actual_entry
                    elif c["be_type"] == "offset":
                        be_level = actual_entry + c["be_offset"] * risk * trend
                    elif c["be_type"] in ("mfe", "time"):
                        be_level = actual_entry
                    elif c["be_type"] == "progressive":
                        be_level = stop  # start from initial stop, will move up
                        prog_stop = stop
                    else:
                        be_level = actual_entry

                    if c["be_type"] != "progressive":
                        if trend == 1:
                            runner_stop = max(runner_stop, be_level)
                        else:
                            runner_stop = min(runner_stop, be_level)

            # Progressive BE: move stop up by step every N bars
            if be_activated and c["be_type"] == "progressive":
                target_level = stop + c["be_prog_step"] * risk * (k // c["be_prog_interval"]) * trend
                # Cap at entry price (don't go above BE for progressive)
                if trend == 1:
                    target_level = min(target_level, actual_entry)
                    runner_stop = max(runner_stop, target_level)
                else:
                    target_level = max(target_level, actual_entry)
                    runner_stop = min(runner_stop, target_level)

            # Stop check
            stopped = (trend == 1 and l <= runner_stop) or (trend == -1 and h >= runner_stop)
            if stopped:
                trade_pnl += remaining * (runner_stop - actual_entry) * trend - remaining * comm
                end_bar = bi; bars_held = k
                if be_activated and abs(runner_stop - actual_entry) < 0.02 * risk:
                    exit_reason = "be_stop"
                elif be_activated:
                    exit_reason = "trail_stop"
                else:
                    exit_reason = "initial_stop"
                break

            # Lock
            if use_lock and not lock_done and remaining > lock_sh:
                target = actual_entry + c["lock_rr"] * risk * trend
                if (trend == 1 and h >= target) or (trend == -1 and l <= target):
                    trade_pnl += lock_sh * c["lock_rr"] * risk - lock_sh * comm
                    remaining -= lock_sh; lock_done = True

            # Chandelier trail
            chand_mult = c["chand_mult_start"]
            if c["chand_tighten_bars"] > 0 and k > c["chand_start"]:
                progress = min(1.0, (k - c["chand_start"]) / c["chand_tighten_bars"])
                chand_mult = c["chand_mult_start"] + (c["chand_mult_end"] - c["chand_mult_start"]) * progress

            if k >= c["chand_start"] and k >= 2 and (lock_done or not use_lock or c["be_type"] == "none"):
                sk = max(1, k - 40 + 1)  # always use 40-bar lookback window
                hh_vals = [high[entry_bar + kk] for kk in range(sk, k) if entry_bar + kk < n]
                ll_vals = [low[entry_bar + kk] for kk in range(sk, k) if entry_bar + kk < n]
                if hh_vals and ll_vals:
                    if trend == 1:
                        new_stop = max(hh_vals) - chand_mult * ca
                        runner_stop = max(runner_stop, new_stop)
                    else:
                        new_stop = min(ll_vals) + chand_mult * ca
                        runner_stop = min(runner_stop, new_stop)
        else:
            ep = close[min(entry_bar + p["max_hold_bars"], n - 1)]
            trade_pnl += remaining * (ep - actual_entry) * trend - remaining * comm
            end_bar = min(entry_bar + p["max_hold_bars"], n - 1)
            bars_held = p["max_hold_bars"]

        equity += trade_pnl
        r_mult = trade_pnl / (shares * risk) if shares * risk > 0 else 0
        if trade_pnl < 0 and shares * risk > 0:
            daily_r_loss += abs(r_mult)

        if r_mult > 0:
            skip_count = p.get("skip_after_win", 0)

        trades.append({
            "pnl": trade_pnl, "r": r_mult, "dir": trend,
            "exit": exit_reason, "bars_held": bars_held,
            "shares": shares, "risk": risk, "max_dd_r": max_dd_r,
        })
        bar = end_bar + 1

    tdf = pd.DataFrame(trades) if trades else pd.DataFrame()
    total = len(tdf)
    if total == 0:
        return {"pf": 0, "ret": 0, "trades": 0, "tpd": 0, "wr": 0,
                "lpf": 0, "spf": 0, "big5": 0, "max_dd": 0,
                "init_n": 0, "be_n": 0, "trail_n": 0, "trail_avg_r": 0}
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

    trail = tdf[tdf["exit"] == "trail_stop"]
    return {
        "pf": round(pf, 3), "ret": round(ret, 2),
        "trades": total, "tpd": round(total / max(days, 1), 1),
        "wr": round(wins / total * 100, 1),
        "lpf": spf(longs), "spf": spf(shorts),
        "big5": int((r_arr >= 5).sum()),
        "max_dd": round(tdf["max_dd_r"].max(), 2),
        "init_n": int((tdf["exit"] == "initial_stop").sum()),
        "be_n": int((tdf["exit"] == "be_stop").sum()),
        "trail_n": len(trail),
        "trail_avg_r": round(trail["r"].mean(), 2) if len(trail) > 0 else 0,
    }


def fmt(label, r, bl, cmplx="simple"):
    d = r["pf"] - bl["pf"]
    return (f"  {label:<48} {r['pf']:>6.3f} ({d:>+.3f})"
            f" {r['ret']:>+7.2f}% {r['trades']:>5} {r['wr']:>5.1f}%"
            f" {r['lpf']:>5.2f} {r['spf']:>5.2f} {r['big5']:>4}"
            f" {r['trail_n']:>4}({r['trail_avg_r']:>+.1f}R)"
            f" {r['max_dd']:>6.1f}R  [{cmplx}]")


def main():
    df = pd.read_csv(DATA_PATH, index_col="timestamp", parse_dates=True)
    print(f"Data: {len(df):,} bars\n")

    bl = run(df)  # new baseline with time filters
    hdr = (f"  {'Config':<48} {'PF':>6} {'ΔPF':>7}"
           f" {'Ret%':>8} {'Trd':>5} {'WR%':>6}"
           f" {'L.PF':>6} {'S.PF':>6} {'5R+':>5}"
           f" {'Trail':>11} {'MaxDD':>7} {'Cmplx':>9}")
    sep = f"  {'-'*130}"

    print(f"New baseline (14:00 cutoff + skip1): PF={bl['pf']:.3f}, ret={bl['ret']:+.2f}%, "
          f"5R+={bl['big5']}, trail={bl['trail_n']}@{bl['trail_avg_r']:+.1f}R, maxDD={bl['max_dd']:.1f}R\n")

    # ═══════════════════════════════════════════════════════════════
    # 1A: BE ALTERNATIVES
    # ═══════════════════════════════════════════════════════════════
    print(f"{'='*140}")
    print(f"  1A. BE ALTERNATIVES")
    print(f"{'='*140}")
    print(hdr); print(sep)
    print(fmt("★ Baseline (exact BE after lock)", bl, bl))

    configs_1a = [
        ("1A-1: No BE (stop stays at initial)", {"be_type": "none"}, "simple"),
        ("1A-2: BE - 0.3R", {"be_type": "offset", "be_offset": -0.3}, "simple"),
        ("1A-3: BE - 0.5R", {"be_type": "offset", "be_offset": -0.5}, "simple"),
        ("1A-4: BE + 0.1R (micro profit)", {"be_type": "offset", "be_offset": 0.1}, "simple"),
        ("1A-5: BE + 0.2R", {"be_type": "offset", "be_offset": 0.2}, "simple"),
        ("1A-6: MFE-based BE (after 1R)", {"be_type": "mfe", "be_mfe_trigger": 1.0}, "medium"),
        ("1A-7: MFE-based BE (after 2R)", {"be_type": "mfe", "be_mfe_trigger": 2.0}, "medium"),
        ("1A-8: Time-based BE (10 bars)", {"be_type": "time", "be_time_trigger": 10}, "simple"),
        ("1A-9: Time-based BE (20 bars)", {"be_type": "time", "be_time_trigger": 20}, "simple"),
        ("1A-10: Progressive (0.1R every 5 bars)", {"be_type": "progressive", "be_prog_step": 0.1, "be_prog_interval": 5}, "medium"),
    ]

    results_1a = [("★ Baseline", bl, "simple")]
    for label, cfg, cmplx in configs_1a:
        r = run(df, cfg=cfg)
        print(fmt(label, r, bl, cmplx))
        results_1a.append((label, r, cmplx))

    # ═══════════════════════════════════════════════════════════════
    # 1B: CHANDELIER OPTIMIZATION
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*140}")
    print(f"  1B. CHANDELIER OPTIMIZATION")
    print(f"{'='*140}")
    print(hdr); print(sep)
    print(fmt("★ Baseline (chand 40/0.5)", bl, bl))

    configs_1b = [
        ("1B-1: Immediate start, mult=2.0", {"chand_start": 1, "chand_mult_start": 2.0, "chand_mult_end": 2.0}, "simple"),
        ("1B-2: Immediate start, mult=1.5", {"chand_start": 1, "chand_mult_start": 1.5, "chand_mult_end": 1.5}, "simple"),
        ("1B-3: Immediate start, mult=1.0", {"chand_start": 1, "chand_mult_start": 1.0, "chand_mult_end": 1.0}, "simple"),
        ("1B-4: Bar 10, mult=1.0", {"chand_start": 10, "chand_mult_start": 1.0, "chand_mult_end": 1.0}, "simple"),
        ("1B-5: Bar 20, mult=0.8", {"chand_start": 20, "chand_mult_start": 0.8, "chand_mult_end": 0.8}, "simple"),
        ("1B-6: Bar 20, mult=0.5", {"chand_start": 20, "chand_mult_start": 0.5, "chand_mult_end": 0.5}, "simple"),
        ("1B-7: Progressive 2.0→0.5 over 60 bars", {"chand_start": 1, "chand_mult_start": 2.0, "chand_mult_end": 0.5, "chand_tighten_bars": 60}, "medium"),
        ("1B-8: Progressive 1.5→0.5 over 40 bars", {"chand_start": 1, "chand_mult_start": 1.5, "chand_mult_end": 0.5, "chand_tighten_bars": 40}, "medium"),
        ("1B-9: Bar 10, progressive 1.5→0.3 / 50b", {"chand_start": 10, "chand_mult_start": 1.5, "chand_mult_end": 0.3, "chand_tighten_bars": 50}, "medium"),
        ("1B-10: Bar 5, progressive 2.0→0.3 / 60b", {"chand_start": 5, "chand_mult_start": 2.0, "chand_mult_end": 0.3, "chand_tighten_bars": 60}, "medium"),
    ]

    results_1b = [("★ Baseline", bl, "simple")]
    for label, cfg, cmplx in configs_1b:
        r = run(df, cfg=cfg)
        print(fmt(label, r, bl, cmplx))
        results_1b.append((label, r, cmplx))

    # ═══════════════════════════════════════════════════════════════
    # 1C: LOCK REASSESSMENT
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*140}")
    print(f"  1C. LOCK REASSESSMENT")
    print(f"{'='*140}")
    print(hdr); print(sep)

    configs_1c = [
        ("1C-1: No lock, no BE (pure chandelier)", {"lock_rr": 0, "lock_pct": 0, "be_type": "none"}, "simple"),
        ("1C-2: No lock, MFE-based BE at 1R", {"lock_rr": 0, "lock_pct": 0, "be_type": "mfe", "be_mfe_trigger": 1.0}, "medium"),
        ("1C-3: No lock, time-based BE at 20 bars", {"lock_rr": 0, "lock_pct": 0, "be_type": "time", "be_time_trigger": 20}, "simple"),
        ("1C-4: No lock, progressive stop", {"lock_rr": 0, "lock_pct": 0, "be_type": "progressive", "be_prog_step": 0.1, "be_prog_interval": 5}, "medium"),
    ]

    for label, cfg, cmplx in configs_1c:
        r = run(df, cfg=cfg)
        print(fmt(label, r, bl, cmplx))

    # ═══════════════════════════════════════════════════════════════
    # TOP RESULTS
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*140}")
    print(f"  PHASE 1 TOP RESULTS")
    print(f"{'='*140}")

    all_r = results_1a + results_1b
    all_r.sort(key=lambda x: x[1]["pf"], reverse=True)

    print(f"\n  {'#':>2} {'Config':<48} {'PF':>6} {'Ret%':>8} {'Trail':>5} {'AvgR':>6} {'5R+':>5}"
          f" {'L.PF':>6} {'S.PF':>6} {'MaxDD':>7} {'Cmplx':>8}")
    print(f"  {'-'*110}")
    for i, (label, r, cmplx) in enumerate(all_r[:15]):
        marker = " ★" if r["pf"] >= bl["pf"] else ""
        print(f"  {i+1:>2} {label:<48} {r['pf']:>6.3f} {r['ret']:>+7.2f}%"
              f" {r['trail_n']:>4} {r['trail_avg_r']:>+5.1f}R {r['big5']:>4}"
              f" {r['lpf']:>5.2f} {r['spf']:>5.2f} {r['max_dd']:>6.1f}R  {cmplx}{marker}")

    # Best from each category
    best_1a = max(results_1a[1:], key=lambda x: x[1]["pf"])
    best_1b = max(results_1b[1:], key=lambda x: x[1]["pf"])
    print(f"\n  Best 1A: {best_1a[0]} → PF={best_1a[1]['pf']:.3f}")
    print(f"  Best 1B: {best_1b[0]} → PF={best_1b[1]['pf']:.3f}")
    print(f"\n  → These two go to Phase 2 joint optimization with baseline as third option.")


if __name__ == "__main__":
    main()
