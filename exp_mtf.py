"""
Multi-Timeframe Filter + Early Exit Tighten

Stack:
  1. 5min trend confirmation (higher TF must agree)
  2. 1min touch close entry
  3. 3-bar MFE gate: if MFE < 0.3R → tighten stop to -0.3R
  4. BE-lock 5%@0.1R + Chandelier 40/0.5
  5. 14:00 cutoff + skip-after-win

5min trend: construct 5-min bars from 1-min data, compute EMA20/EMA50 on 5min.
Only enter 1min touch when 5min trend agrees.

Test on IS and OOS simultaneously.
"""
from __future__ import annotations
import functools, datetime as dt
import numpy as np, pandas as pd
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
    "no_entry_after": dt.time(14, 0),
    "force_close_at": dt.time(15, 58),
    "commission_per_share": 0.005, "daily_loss_r": 2.5,
    "skip_after_win": 1,
}


def build_htf_trend(df, tf_minutes=5):
    """Build higher timeframe trend array aligned to 1-min bars.

    Resample to tf_minutes, compute EMA20/50, then map back to 1-min index.
    """
    # Resample OHLCV
    rule = f"{tf_minutes}min"  # Use 'min' instead of 'T' for modern pandas
    htf = df.resample(rule).agg({"Open": "first", "High": "max", "Low": "min",
                                  "Close": "last", "Volume": "sum"}).dropna()
    htf["ema20"] = htf["Close"].ewm(span=20, adjust=False).mean()
    htf["ema50"] = htf["Close"].ewm(span=50, adjust=False).mean()
    htf["trend"] = 0
    htf.loc[(htf["Close"] > htf["ema20"]) & (htf["ema20"] > htf["ema50"]), "trend"] = 1
    htf.loc[(htf["Close"] < htf["ema20"]) & (htf["ema20"] < htf["ema50"]), "trend"] = -1

    # Map back: for each 1-min bar, use the LAST COMPLETED htf bar's trend
    # (no look-ahead: use shift(1) on htf, then forward-fill to 1min)
    htf_shifted = htf["trend"].shift(1)  # previous completed bar
    htf_1min = htf_shifted.reindex(df.index, method="ffill").fillna(0).astype(int).values
    return htf_1min


def run(df, capital=100_000, cfg=None):
    """
    cfg:
      htf_minutes: int (0 = no HTF filter, 5/15 = use HTF)
      gate_bars: int (0 = no early exit gate)
      gate_mfe: float
      gate_tighten_to: float
    """
    c = {"htf_minutes": 0, "gate_bars": 0, "gate_mfe": 0, "gate_tighten_to": 0}
    if cfg: c.update(cfg)

    p = BASE.copy()
    df = add_indicators(df, p)
    high = df["High"].values; low = df["Low"].values; close = df["Close"].values
    ema = df["ema20"].values; ema_s = df["ema50"].values; atr_v = df["atr"].values
    times = df.index.time; dates = df.index.date; n = len(df)
    comm = p["commission_per_share"]

    # Build HTF trend
    htf_trend = np.zeros(n, dtype=int)
    if c["htf_minutes"] > 0:
        htf_trend = build_htf_trend(df, c["htf_minutes"])

    equity = capital; peak_eq = capital; max_dd_pct = 0
    trades = []; bar = max(p["ema_slow"], p["atr_period"]) + 5
    daily_r_loss = 0.0; current_date = None; skip_count = 0
    htf_filtered = 0; gate_tightened = 0

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

        # HTF filter
        if c["htf_minutes"] > 0 and htf_trend[bar] != trend:
            htf_filtered += 1; bar += 1; continue

        if not check_touch(trend, low[bar], high[bar], ema[bar], a,
                           p["touch_tol"], p["touch_below_max"]):
            bar += 1; continue
        if skip_count > 0: skip_count -= 1; bar += 1; continue

        actual_entry = close[bar]
        stop = low[bar] - p["stop_buffer"] * a if trend == 1 else high[bar] + p["stop_buffer"] * a
        risk = abs(actual_entry - stop)
        if risk <= 0: bar += 1; continue
        entry_bar = bar

        shares = max(1, int(equity * p["risk_pct"] / risk))
        if shares * abs(actual_entry) > equity * p["max_pos_pct"]:
            shares = max(1, int(equity * p["max_pos_pct"] / abs(actual_entry)))
        if equity < shares * risk or shares < 1: bar += 1; continue

        lock_sh = max(1, int(shares * p["lock_pct"]))
        remaining = shares; runner_stop = stop; lock_done = False
        trade_pnl = -shares * comm; end_bar = entry_bar
        exit_reason = "timeout"; mfe_so_far = 0

        for k in range(1, p["max_hold_bars"] + 1):
            bi = entry_bar + k
            if bi >= n: break
            h = high[bi]; l = low[bi]
            ca = atr_v[bi] if not np.isnan(atr_v[bi]) else a

            # Track MFE
            if trend == 1: mfe_so_far = max(mfe_so_far, (h - actual_entry) / risk)
            else: mfe_so_far = max(mfe_so_far, (actual_entry - l) / risk)

            if times[bi] >= p["force_close_at"]:
                trade_pnl += remaining * (close[bi] - actual_entry) * trend - remaining * comm
                end_bar = bi; exit_reason = "session_close"; break

            # Early exit gate
            if c["gate_bars"] > 0 and k == c["gate_bars"] and not lock_done:
                if mfe_so_far < c["gate_mfe"]:
                    gate_tightened += 1
                    new_stop = actual_entry + c["gate_tighten_to"] * risk * trend
                    if trend == 1: runner_stop = max(runner_stop, new_stop)
                    else: runner_stop = min(runner_stop, new_stop)

            # Stop
            stopped = (trend == 1 and l <= runner_stop) or (trend == -1 and h >= runner_stop)
            if stopped:
                trade_pnl += remaining * (runner_stop - actual_entry) * trend - remaining * comm
                end_bar = bi
                exit_reason = "be_stop" if lock_done and abs(runner_stop - actual_entry) < 0.02 else \
                              ("trail_stop" if lock_done else "initial_stop")
                break

            # Lock
            if not lock_done and remaining > lock_sh:
                target = actual_entry + p["lock_rr"] * risk * trend
                if (trend == 1 and h >= target) or (trend == -1 and l <= target):
                    trade_pnl += lock_sh * p["lock_rr"] * risk - lock_sh * comm
                    remaining -= lock_sh; lock_done = True
                    if trend == 1: runner_stop = max(runner_stop, actual_entry)
                    else: runner_stop = min(runner_stop, actual_entry)

            # Chandelier
            if lock_done and k >= p["chand_bars"]:
                sk = max(1, k - p["chand_bars"] + 1)
                hv = [high[entry_bar+kk] for kk in range(sk, k) if entry_bar+kk < n]
                lv = [low[entry_bar+kk] for kk in range(sk, k) if entry_bar+kk < n]
                if hv and lv:
                    if trend == 1: runner_stop = max(runner_stop, max(hv) - p["chand_mult"] * ca)
                    else: runner_stop = min(runner_stop, min(lv) + p["chand_mult"] * ca)
        else:
            ep = close[min(entry_bar + p["max_hold_bars"], n - 1)]
            trade_pnl += remaining * (ep - actual_entry) * trend - remaining * comm
            end_bar = min(entry_bar + p["max_hold_bars"], n - 1)

        equity += trade_pnl
        peak_eq = max(peak_eq, equity)
        dd = (peak_eq - equity) / peak_eq * 100
        max_dd_pct = max(max_dd_pct, dd)

        r_mult = trade_pnl / (shares * risk) if shares * risk > 0 else 0
        if trade_pnl < 0 and shares * risk > 0: daily_r_loss += abs(r_mult)
        if r_mult > 0: skip_count = 1
        trades.append({"r": r_mult, "exit": exit_reason, "dir": trend})
        bar = end_bar + 1

    tdf = pd.DataFrame(trades) if trades else pd.DataFrame()
    total = len(tdf)
    if total == 0: return {"pf":0,"ret":0,"trades":0,"big5":0,"lpf":0,"spf":0,
                           "tpd":0,"htf_skip":0,"gate_tight":0,"max_dd":0,"equity":capital}
    gw = tdf.loc[tdf["r"]>0,"r"].sum(); gl = abs(tdf.loc[tdf["r"]<=0,"r"].sum())
    pf = gw/gl if gl>0 else 0; ret = (equity-capital)/capital*100
    r_arr = tdf["r"].values; days = df.index.normalize().nunique()
    longs = tdf[tdf["dir"]==1]; shorts = tdf[tdf["dir"]==-1]
    def spf(s):
        if len(s)==0: return 0
        w=s.loc[s["r"]>0,"r"].sum(); l=abs(s.loc[s["r"]<=0,"r"].sum())
        return round(w/l,3) if l>0 else 0
    trail = tdf[tdf["exit"]=="trail_stop"]
    return {"pf":round(pf,3),"ret":round(ret,2),"trades":total,
            "big5":int((r_arr>=5).sum()),"lpf":spf(longs),"spf":spf(shorts),
            "tpd":round(total/max(days,1),1),"htf_skip":htf_filtered,
            "gate_tight":gate_tightened,"max_dd":round(max_dd_pct,2),
            "equity":round(equity,0),
            "trail_n":len(trail),
            "trail_avg_r":round(trail["r"].mean(),2) if len(trail)>0 else 0}


def fmt(label, r, bl_pf):
    d = r["pf"] - bl_pf
    return (f"  {label:<50} {r['pf']:>6.3f}({d:>+.3f})"
            f" {r['ret']:>+7.2f}% ${r['equity']:>10,.0f} {r['trades']:>5} {r['big5']:>4}"
            f" {r['lpf']:>5.2f} {r['spf']:>5.2f}"
            f" {r['trail_n']:>4}@{r['trail_avg_r']:>+.1f}R"
            f" DD={r['max_dd']:.2f}%")


def main():
    df_is = pd.read_csv(IS_PATH, index_col="timestamp", parse_dates=True)
    df_oos = pd.read_csv(OOS_PATH, index_col="timestamp", parse_dates=True)
    print(f"IS: {len(df_is):,} bars | OOS: {len(df_oos):,} bars\n")

    hdr = (f"  {'Config':<50} {'PF':>10} {'Ret%':>8} {'$Equity':>11} {'Trd':>5} {'5R+':>4}"
           f" {'L.PF':>6} {'S.PF':>6} {'Trail':>11} {'MaxDD':>10}")
    sep = f"  {'-'*130}"

    # ═══ Build up incrementally ═══
    print(f"{'='*135}")
    print(f"  BUILDING THE STACK — Each layer adds to previous")
    print(f"{'='*135}")
    print(f"\n  === IN-SAMPLE (Polygon 2024-2026, $100K start) ===")
    print(hdr); print(sep)

    layers = [
        ("Layer 0: Touch close only",                      {}),
        ("Layer 1: + 14:00 cutoff + skip-after-win",       {}),  # already in BASE
        ("Layer 2: + 3b MFE gate → tighten -0.3R",        {"gate_bars":3,"gate_mfe":0.3,"gate_tighten_to":-0.3}),
        ("Layer 3: + 5min HTF filter",                     {"htf_minutes":5,"gate_bars":3,"gate_mfe":0.3,"gate_tighten_to":-0.3}),
        ("Layer 3b: + 15min HTF filter",                   {"htf_minutes":15,"gate_bars":3,"gate_mfe":0.3,"gate_tighten_to":-0.3}),
    ]

    bl_pf = 0
    for label, cfg in layers:
        r = run(df_is, cfg=cfg)
        if bl_pf == 0: bl_pf = r["pf"]
        print(fmt(label, r, bl_pf))

    # ═══ HTF timeframe sweep ═══
    print(f"\n  === HTF TIMEFRAME SWEEP (with 3b gate) ===")
    print(hdr); print(sep)
    for tf in [0, 3, 5, 7, 10, 15, 20, 30]:
        cfg = {"htf_minutes": tf, "gate_bars": 3, "gate_mfe": 0.3, "gate_tighten_to": -0.3}
        if tf == 0: cfg = {"gate_bars": 3, "gate_mfe": 0.3, "gate_tighten_to": -0.3}
        r = run(df_is, cfg=cfg)
        print(fmt(f"HTF={tf}min" if tf > 0 else "No HTF", r, bl_pf))

    # ═══ Best configs: IS + OOS side by side ═══
    print(f"\n{'='*135}")
    print(f"  IS vs OOS COMPARISON")
    print(f"{'='*135}")

    best_configs = [
        ("Baseline (no extras)",                   {}),
        ("+ 3b gate only",                         {"gate_bars":3,"gate_mfe":0.3,"gate_tighten_to":-0.3}),
        ("+ 5min HTF only",                        {"htf_minutes":5}),
        ("+ 5min HTF + 3b gate",                   {"htf_minutes":5,"gate_bars":3,"gate_mfe":0.3,"gate_tighten_to":-0.3}),
        ("+ 15min HTF + 3b gate",                  {"htf_minutes":15,"gate_bars":3,"gate_mfe":0.3,"gate_tighten_to":-0.3}),
        ("+ 10min HTF + 3b gate",                  {"htf_minutes":10,"gate_bars":3,"gate_mfe":0.3,"gate_tighten_to":-0.3}),
    ]

    print(f"\n  {'Config':<40} {'IS PF':>7} {'IS Ret':>8} {'IS $':>10} {'IS 5R+':>7}"
          f" {'OOS PF':>8} {'OOS Ret':>9} {'OOS $':>10} {'OOS 5R+':>8} {'Holds':>6}")
    print(f"  {'-'*120}")

    for label, cfg in best_configs:
        r_is = run(df_is, cfg=cfg)
        r_oos = run(df_oos, cfg=cfg)
        holds = "✅" if r_oos["pf"] > 1.5 and r_is["pf"] > 1.5 else ("⚠️" if r_oos["pf"] > 1.3 else "❌")
        print(f"  {label:<40} {r_is['pf']:>6.3f} {r_is['ret']:>+7.2f}% ${r_is['equity']:>9,.0f} {r_is['big5']:>6}"
              f" {r_oos['pf']:>7.3f} {r_oos['ret']:>+8.2f}% ${r_oos['equity']:>9,.0f} {r_oos['big5']:>7}  {holds}")

    # ═══ Full 4-year combined ═══
    print(f"\n{'='*135}")
    print(f"  4-YEAR COMBINED (OOS 2022-2024 + IS 2024-2026)")
    print(f"{'='*135}")

    df_4y = pd.concat([df_oos, df_is]).sort_index()
    df_4y = df_4y[~df_4y.index.duplicated(keep='first')]

    for label, cfg in best_configs:
        r = run(df_4y, cfg=cfg)
        annual = ((r["equity"] / 100_000) ** (1/4) - 1) * 100
        print(f"  {label:<40} PF={r['pf']:.3f}  4Y-ret={r['ret']:+.2f}%  ${r['equity']:>10,.0f}"
              f"  annual≈{annual:+.1f}%  5R+={r['big5']}  DD={r['max_dd']:.2f}%"
              f"  L={r['lpf']:.2f} S={r['spf']:.2f}")


if __name__ == "__main__":
    main()
