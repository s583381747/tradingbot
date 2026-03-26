"""
Touch Close Entry + Exit Optimization

Entry: touch bar close (mode C from rethink experiment)
  - Trend: Close > EMA20 > EMA50 (long) or reverse (short)
  - Touch: wick within 0.15×ATR of EMA20
  - Enter at close of touch bar
  - Stop: touch_low - 0.3×ATR (long)

Exit variants to test:
  1. Current: 20% lock at 0.3R + 80% chandelier 40/0.5
  2. Lock R:R sweep: 0.2R, 0.3R, 0.5R, 0.8R, 1.0R
  3. Lock % sweep: 10%, 20%, 30%, 40%, 50%
  4. Chandelier mult sweep: 0.3, 0.5, 0.8, 1.0, 1.5, 2.0
  5. Chandelier bars sweep: 20, 30, 40, 60, 80
  6. No lock, pure chandelier
  7. No chandelier, pure fixed R:R (1R, 1.5R, 2R, 3R, 5R)
  8. Trailing ATR stop (no chandelier HH logic)
  9. Time-based exit: force close after N bars
  10. Multi-lock: 3 locks at different R levels
  11. Breakeven-only (lock at tiny profit, trail rest)
  12. Best combos
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
    Touch close entry + configurable exit.

    exit_cfg keys:
      lock_rr:      R:R for lock take-profit (0 = no lock)
      lock_pct:     fraction of position to lock (0.20 = 20%)
      lock2_rr:     second lock R:R (0 = none)
      lock2_pct:    second lock fraction
      lock3_rr:     third lock R:R
      lock3_pct:    third lock fraction
      chand_bars:   chandelier lookback (0 = no chandelier)
      chand_mult:   chandelier ATR multiplier
      trail_atr:    simple ATR trailing stop multiplier (0 = off)
      fixed_tp:     fixed take-profit in R multiples (0 = no fixed TP)
      max_bars:     force exit after N bars (180 default)
      be_after_lock: move stop to BE after first lock (True/False)
    """
    cfg = {
        "lock_rr": 0.3, "lock_pct": 0.20,
        "lock2_rr": 0, "lock2_pct": 0,
        "lock3_rr": 0, "lock3_pct": 0,
        "chand_bars": 40, "chand_mult": 0.5,
        "trail_atr": 0,
        "fixed_tp": 0,
        "max_bars": 180,
        "be_after_lock": True,
    }
    if exit_cfg:
        cfg.update(exit_cfg)

    p = BASE.copy()
    df = add_indicators(df, p)
    high = df["High"].values; low = df["Low"].values
    close = df["Close"].values; open_p = df["Open"].values
    ema = df["ema20"].values; ema_s = df["ema50"].values; atr_v = df["atr"].values
    times = df.index.time; dates = df.index.date; n = len(df)
    comm = p["commission_per_share"]

    equity = capital; trade_log = []
    bar = max(p["ema_slow"], p["atr_period"]) + 5
    daily_r_loss = 0.0; current_date = None

    while bar < n - cfg["max_bars"] - 5:
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

        # Entry at touch bar close
        actual_entry = close[bar]
        stop = low[bar] - p["stop_buffer"] * a if trend == 1 else high[bar] + p["stop_buffer"] * a
        risk = abs(actual_entry - stop)
        if risk <= 0: bar += 1; continue

        entry_bar = bar

        # Size
        shares = max(1, int(equity * p["risk_pct"] / risk))
        if shares * actual_entry > equity * p["max_pos_pct"]:
            shares = max(1, int(equity * p["max_pos_pct"] / actual_entry))
        if equity < shares * risk or shares < 1: bar += 1; continue

        # ─── Exit management ───
        remaining = shares
        trade_pnl = -shares * comm
        runner_stop = stop
        end_bar = entry_bar

        # Lock state
        lock1_done = False
        lock2_done = False
        lock3_done = False
        lock1_sh = max(1, int(shares * cfg["lock_pct"])) if cfg["lock_rr"] > 0 and cfg["lock_pct"] > 0 else 0
        lock2_sh = max(1, int(shares * cfg["lock2_pct"])) if cfg["lock2_rr"] > 0 and cfg["lock2_pct"] > 0 else 0
        lock3_sh = max(1, int(shares * cfg["lock3_pct"])) if cfg["lock3_rr"] > 0 and cfg["lock3_pct"] > 0 else 0

        any_lock_done = False

        for k in range(1, cfg["max_bars"] + 1):
            bi = entry_bar + k
            if bi >= n: break
            h = high[bi]; l = low[bi]
            ca = atr_v[bi] if not np.isnan(atr_v[bi]) else a

            # Force close
            if times[bi] >= p["force_close_at"]:
                trade_pnl += remaining * (close[bi] - actual_entry) * trend - remaining * comm
                end_bar = bi; break

            # Fixed TP
            if cfg["fixed_tp"] > 0 and not any_lock_done:
                tp_price = actual_entry + cfg["fixed_tp"] * risk * trend
                if (trend == 1 and h >= tp_price) or (trend == -1 and l <= tp_price):
                    trade_pnl += remaining * cfg["fixed_tp"] * risk - remaining * comm
                    end_bar = bi; break

            # Stop check
            stopped = (trend == 1 and l <= runner_stop) or (trend == -1 and h >= runner_stop)
            if stopped:
                trade_pnl += remaining * (runner_stop - actual_entry) * trend - remaining * comm
                end_bar = bi; break

            # Lock 1
            if not lock1_done and lock1_sh > 0 and remaining > lock1_sh:
                target = actual_entry + cfg["lock_rr"] * risk * trend
                hit = (trend == 1 and h >= target) or (trend == -1 and l <= target)
                if hit:
                    trade_pnl += lock1_sh * cfg["lock_rr"] * risk - lock1_sh * comm
                    remaining -= lock1_sh; lock1_done = True; any_lock_done = True
                    if cfg["be_after_lock"]:
                        if trend == 1: runner_stop = max(runner_stop, actual_entry)
                        else: runner_stop = min(runner_stop, actual_entry)

            # Lock 2
            if lock1_done and not lock2_done and lock2_sh > 0 and remaining > lock2_sh:
                target2 = actual_entry + cfg["lock2_rr"] * risk * trend
                hit2 = (trend == 1 and h >= target2) or (trend == -1 and l <= target2)
                if hit2:
                    trade_pnl += lock2_sh * cfg["lock2_rr"] * risk - lock2_sh * comm
                    remaining -= lock2_sh; lock2_done = True

            # Lock 3
            if lock2_done and not lock3_done and lock3_sh > 0 and remaining > lock3_sh:
                target3 = actual_entry + cfg["lock3_rr"] * risk * trend
                hit3 = (trend == 1 and h >= target3) or (trend == -1 and l <= target3)
                if hit3:
                    trade_pnl += lock3_sh * cfg["lock3_rr"] * risk - lock3_sh * comm
                    remaining -= lock3_sh; lock3_done = True

            # Chandelier trail
            if cfg["chand_bars"] > 0 and any_lock_done and k >= cfg["chand_bars"]:
                sk = max(1, k - cfg["chand_bars"] + 1)
                if trend == 1:
                    hh = max(high[entry_bar + kk] for kk in range(sk, k) if entry_bar + kk < n)
                    runner_stop = max(runner_stop, hh - cfg["chand_mult"] * ca)
                else:
                    ll = min(low[entry_bar + kk] for kk in range(sk, k) if entry_bar + kk < n)
                    runner_stop = min(runner_stop, ll + cfg["chand_mult"] * ca)

            # Simple ATR trail (alternative to chandelier)
            if cfg["trail_atr"] > 0 and any_lock_done:
                if trend == 1:
                    atr_stop = h - cfg["trail_atr"] * ca
                    runner_stop = max(runner_stop, atr_stop)
                else:
                    atr_stop = l + cfg["trail_atr"] * ca
                    runner_stop = min(runner_stop, atr_stop)

        else:
            ep = close[min(entry_bar + cfg["max_bars"], n - 1)]
            trade_pnl += remaining * (ep - actual_entry) * trend - remaining * comm
            end_bar = min(entry_bar + cfg["max_bars"], n - 1)

        equity += trade_pnl
        if trade_pnl < 0 and shares * risk > 0:
            daily_r_loss += abs(trade_pnl) / (shares * risk)
        trade_log.append({"pnl": trade_pnl, "dir": trend, "shares": shares,
                          "risk": risk, "lock": any_lock_done})
        bar = end_bar + 1

    # Stats
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
        "ln": len(longs), "sn": len(shorts),
        "big5": int((r_arr >= 5).sum()),
        "avg_win": round(gw / wins, 2) if wins > 0 else 0,
        "avg_loss": round(gl / losses, 2) if losses > 0 else 0,
    }


def fmt(label, r, bl_pf=None):
    delta = f" ({r['pf']-bl_pf:+.3f})" if bl_pf else ""
    return (f"  {label:<52} {r['pf']:>6.3f}{delta:>8} {r['ret']:>+7.2f}%"
            f" {r['trades']:>5} {r['tpd']:>4.1f} {r['wr']:>5.1f}%"
            f" {r['avg_win']:>6.2f} {r['avg_loss']:>6.2f}"
            f" {r['lpf']:>5.2f} {r['spf']:>5.2f} {r['big5']:>4}")


def main():
    df = pd.read_csv(DATA_PATH, index_col="timestamp", parse_dates=True)
    print(f"Data: {len(df):,} bars, {df.index.normalize().nunique()} days\n")

    bl = run(df)  # baseline: current exit params
    hdr = (f"  {'Config':<52} {'PF':>6} {'vs BL':>8} {'Ret%':>8}"
           f" {'Trd':>5} {'T/d':>5} {'WR%':>6}"
           f" {'AvgW':>7} {'AvgL':>7} {'L.PF':>6} {'S.PF':>6} {'5R+':>5}")
    sep = f"  {'-'*125}"

    # ═══ 1. Lock R:R sweep ═══
    print(f"{'='*130}")
    print(f"  1. LOCK R:R SWEEP (lock_pct=20%, chandelier 40/0.5)")
    print(f"{'='*130}")
    print(hdr); print(sep)
    print(fmt("Baseline (lock 0.3R)", bl))
    for rr in [0.1, 0.2, 0.4, 0.5, 0.8, 1.0, 1.5, 2.0]:
        r = run(df, exit_cfg={"lock_rr": rr})
        print(fmt(f"Lock {rr}R", r, bl["pf"]))
    r = run(df, exit_cfg={"lock_rr": 0, "lock_pct": 0})
    print(fmt("No lock (pure chandelier)", r, bl["pf"]))

    # ═══ 2. Lock % sweep ═══
    print(f"\n{'='*130}")
    print(f"  2. LOCK PERCENTAGE SWEEP (lock_rr=0.3R, chandelier 40/0.5)")
    print(f"{'='*130}")
    print(hdr); print(sep)
    for pct in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        r = run(df, exit_cfg={"lock_pct": pct})
        print(fmt(f"Lock {pct*100:.0f}%", r, bl["pf"]))

    # ═══ 3. Chandelier mult sweep ═══
    print(f"\n{'='*130}")
    print(f"  3. CHANDELIER MULTIPLIER SWEEP (lock 0.3R/20%, bars=40)")
    print(f"{'='*130}")
    print(hdr); print(sep)
    for mult in [0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]:
        r = run(df, exit_cfg={"chand_mult": mult})
        print(fmt(f"Chandelier ×{mult}", r, bl["pf"]))

    # ═══ 4. Chandelier bars sweep ═══
    print(f"\n{'='*130}")
    print(f"  4. CHANDELIER BARS SWEEP (lock 0.3R/20%, mult=0.5)")
    print(f"{'='*130}")
    print(hdr); print(sep)
    for bars in [10, 15, 20, 30, 40, 60, 80, 100]:
        r = run(df, exit_cfg={"chand_bars": bars})
        print(fmt(f"Chandelier {bars} bars", r, bl["pf"]))

    # ═══ 5. Fixed TP (no lock/chandelier) ═══
    print(f"\n{'='*130}")
    print(f"  5. FIXED TAKE-PROFIT (no lock, no chandelier)")
    print(f"{'='*130}")
    print(hdr); print(sep)
    for tp in [0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 10.0]:
        r = run(df, exit_cfg={"lock_rr": 0, "lock_pct": 0, "chand_bars": 0,
                               "fixed_tp": tp})
        print(fmt(f"Fixed TP {tp}R", r, bl["pf"]))

    # ═══ 6. Simple ATR trail (no chandelier) ═══
    print(f"\n{'='*130}")
    print(f"  6. SIMPLE ATR TRAILING STOP (lock 0.3R/20%, no chandelier)")
    print(f"{'='*130}")
    print(hdr); print(sep)
    for mult in [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]:
        r = run(df, exit_cfg={"chand_bars": 0, "trail_atr": mult})
        print(fmt(f"ATR trail ×{mult}", r, bl["pf"]))

    # ═══ 7. Time exit sweep ═══
    print(f"\n{'='*130}")
    print(f"  7. MAX HOLD TIME SWEEP (lock 0.3R/20%, chandelier 40/0.5)")
    print(f"{'='*130}")
    print(hdr); print(sep)
    for bars in [15, 30, 45, 60, 90, 120, 180, 300]:
        r = run(df, exit_cfg={"max_bars": bars})
        print(fmt(f"Max {bars} bars ({bars} min)", r, bl["pf"]))

    # ═══ 8. Multi-lock ═══
    print(f"\n{'='*130}")
    print(f"  8. MULTI-LOCK CONFIGURATIONS")
    print(f"{'='*130}")
    print(hdr); print(sep)
    configs = [
        ("2-lock: 20%@0.3R + 30%@1R",
         {"lock_rr": 0.3, "lock_pct": 0.20, "lock2_rr": 1.0, "lock2_pct": 0.30}),
        ("2-lock: 20%@0.3R + 30%@2R",
         {"lock_rr": 0.3, "lock_pct": 0.20, "lock2_rr": 2.0, "lock2_pct": 0.30}),
        ("2-lock: 20%@0.5R + 30%@1.5R",
         {"lock_rr": 0.5, "lock_pct": 0.20, "lock2_rr": 1.5, "lock2_pct": 0.30}),
        ("3-lock: 15%@0.3R + 15%@1R + 20%@2R",
         {"lock_rr": 0.3, "lock_pct": 0.15, "lock2_rr": 1.0, "lock2_pct": 0.15,
          "lock3_rr": 2.0, "lock3_pct": 0.20}),
        ("3-lock: 10%@0.2R + 20%@0.5R + 20%@1.5R",
         {"lock_rr": 0.2, "lock_pct": 0.10, "lock2_rr": 0.5, "lock2_pct": 0.20,
          "lock3_rr": 1.5, "lock3_pct": 0.20}),
    ]
    for label, cfg in configs:
        r = run(df, exit_cfg=cfg)
        print(fmt(label, r, bl["pf"]))

    # ═══ 9. BE-only (lock tiny, trail rest) ═══
    print(f"\n{'='*130}")
    print(f"  9. BREAKEVEN-ONLY VARIANTS")
    print(f"{'='*130}")
    print(hdr); print(sep)
    for rr in [0.1, 0.15, 0.2]:
        for pct in [0.05, 0.10]:
            r = run(df, exit_cfg={"lock_rr": rr, "lock_pct": pct})
            print(fmt(f"BE-lock {pct*100:.0f}%@{rr}R → trail rest", r, bl["pf"]))

    # ═══ 10. Best combos ═══
    print(f"\n{'='*130}")
    print(f"  10. BEST COMBOS")
    print(f"{'='*130}")
    print(hdr); print(sep)
    combos = [
        ("Lock0.3R/20% + chand 40/0.3",
         {"lock_rr": 0.3, "chand_mult": 0.3}),
        ("Lock0.3R/20% + chand 20/0.5",
         {"lock_rr": 0.3, "chand_bars": 20}),
        ("Lock0.3R/20% + chand 20/0.3",
         {"lock_rr": 0.3, "chand_bars": 20, "chand_mult": 0.3}),
        ("Lock0.5R/15% + chand 30/0.5",
         {"lock_rr": 0.5, "lock_pct": 0.15, "chand_bars": 30}),
        ("Lock0.3R/10% + chand 20/0.5",
         {"lock_rr": 0.3, "lock_pct": 0.10, "chand_bars": 20}),
        ("Lock0.2R/10% + chand 15/0.3",
         {"lock_rr": 0.2, "lock_pct": 0.10, "chand_bars": 15, "chand_mult": 0.3}),
        ("Lock0.3R/20% + ATR trail 0.5",
         {"lock_rr": 0.3, "chand_bars": 0, "trail_atr": 0.5}),
        ("Lock0.3R/20% + ATR trail 0.8",
         {"lock_rr": 0.3, "chand_bars": 0, "trail_atr": 0.8}),
        ("Lock0.5R/20% + ATR trail 0.5",
         {"lock_rr": 0.5, "lock_pct": 0.20, "chand_bars": 0, "trail_atr": 0.5}),
        ("Lock0.3R/15% + chand 30/0.3 + max120",
         {"lock_rr": 0.3, "lock_pct": 0.15, "chand_bars": 30, "chand_mult": 0.3, "max_bars": 120}),
        ("No lock + chand 20/0.3",
         {"lock_rr": 0, "lock_pct": 0, "chand_bars": 20, "chand_mult": 0.3}),
        ("No lock + ATR trail 0.5",
         {"lock_rr": 0, "lock_pct": 0, "chand_bars": 0, "trail_atr": 0.5}),
        ("2-lock 20%@0.3+30%@1R + chand 30/0.3",
         {"lock_rr": 0.3, "lock_pct": 0.20, "lock2_rr": 1.0, "lock2_pct": 0.30,
          "chand_bars": 30, "chand_mult": 0.3}),
        ("2-lock 15%@0.3+20%@1R + chand 20/0.5",
         {"lock_rr": 0.3, "lock_pct": 0.15, "lock2_rr": 1.0, "lock2_pct": 0.20,
          "chand_bars": 20}),
    ]
    for label, cfg in combos:
        r = run(df, exit_cfg=cfg)
        print(fmt(label, r, bl["pf"]))

    # ═══ FINAL RANKING ═══
    print(f"\n{'='*130}")
    print(f"  COLLECTING ALL RESULTS FOR FINAL RANKING...")
    print(f"{'='*130}")

    # Re-run top candidates to collect into one ranking
    all_configs = [
        ("Baseline: lock0.3R/20% + chand40/0.5", {}),
        ("Lock0.5R/20% + chand40/0.5", {"lock_rr": 0.5}),
        ("Lock1.0R/20% + chand40/0.5", {"lock_rr": 1.0}),
        ("No lock + chand40/0.5", {"lock_rr": 0, "lock_pct": 0}),
        ("Lock0.3R/20% + chand40/0.3", {"chand_mult": 0.3}),
        ("Lock0.3R/20% + chand20/0.5", {"chand_bars": 20}),
        ("Lock0.3R/20% + chand20/0.3", {"chand_bars": 20, "chand_mult": 0.3}),
        ("Lock0.3R/20% + ATR trail 0.5", {"chand_bars": 0, "trail_atr": 0.5}),
        ("Lock0.3R/10% + chand20/0.5", {"lock_rr": 0.3, "lock_pct": 0.10, "chand_bars": 20}),
        ("Fixed TP 2R", {"lock_rr": 0, "lock_pct": 0, "chand_bars": 0, "fixed_tp": 2.0}),
        ("Fixed TP 3R", {"lock_rr": 0, "lock_pct": 0, "chand_bars": 0, "fixed_tp": 3.0}),
        ("2-lock 20%@0.3+30%@1R + chand30/0.3",
         {"lock_rr": 0.3, "lock_pct": 0.20, "lock2_rr": 1.0, "lock2_pct": 0.30,
          "chand_bars": 30, "chand_mult": 0.3}),
    ]

    final = []
    for label, cfg in all_configs:
        r = run(df, exit_cfg=cfg)
        r["label"] = label
        final.append(r)

    final.sort(key=lambda x: x["pf"], reverse=True)
    print(f"\n  {'#':>2} {'Config':<52} {'PF':>6} {'Ret%':>8} {'Trd':>5} {'WR%':>6}"
          f" {'L.PF':>6} {'S.PF':>6} {'5R+':>5}")
    print(f"  {'-'*100}")
    for i, r in enumerate(final):
        print(f"  {i+1:>2} {r['label']:<52} {r['pf']:>6.3f} {r['ret']:>+7.2f}%"
              f" {r['trades']:>5} {r['wr']:>5.1f}%"
              f" {r['lpf']:>5.2f} {r['spf']:>5.2f} {r['big5']:>4}")


if __name__ == "__main__":
    main()
