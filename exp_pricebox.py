"""
Price Box Chop Detection — identify consolidation range from price action.

Core idea: if price stays within a tight range for N bars, it's chop.
When price breaks out, trend starts. Only trade after breakout.

Methods:
  A. N-bar range box: last N bars' range < X×ATR → chop
  B. Opening range breakout (ORB): first 30/60 min defines box, trade after break
  C. Donchian breakout: only trade if recent bar made new N-bar high/low
  D. Dynamic box: track when price enters a tight range, flag as chop until breakout
  E. EMA20-relative range: if price oscillates within X×ATR of EMA20, it's chop
  F. Consecutive narrow bars: N bars in a row with range < X×ATR

All filters are bar-level, real-time, zero look-ahead.
Applied as entry condition: only take touch signals when NOT in chop.
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


def run_with_box(df, capital=100_000, box_cfg=None):
    """
    box_cfg keys:
      method: "range_box" | "orb" | "donchian" | "dynamic" | "ema_range" | "narrow_bars" | "none"

      # range_box: last N bars range / ATR < threshold → chop
      rb_bars: int (lookback)
      rb_threshold: float (range/ATR below this = chop)

      # orb: opening range breakout
      orb_bars: int (first N minutes define the range)

      # donchian: need recent N-bar high/low to trade
      don_bars: int (lookback for channel)
      don_recency: int (high/low must be within last M bars)

      # dynamic: detect box formation, require breakout
      dyn_bars: int (min bars to form a box)
      dyn_threshold: float (range/ATR to qualify as box)
      dyn_break_atr: float (breakout must exceed box edge by X×ATR)

      # ema_range: price oscillating around EMA20
      er_bars: int
      er_threshold: float (max distance from EMA in ATR multiples)

      # narrow_bars: consecutive bars with small range
      nb_count: int (need N consecutive narrow bars)
      nb_threshold: float (bar range / ATR < this = narrow)
      nb_break_mult: float (breakout bar range must be > X × avg narrow range)
    """
    cfg = {"method": "none"}
    if box_cfg: cfg.update(box_cfg)

    p = BASE.copy()
    df = add_indicators(df, p)
    high = df["High"].values; low = df["Low"].values
    close = df["Close"].values
    ema = df["ema20"].values; ema_s = df["ema50"].values; atr_v = df["atr"].values
    times = df.index.time; dates = df.index.date; n = len(df)
    comm = p["commission_per_share"]

    # Precompute box signals
    in_chop = np.zeros(n, dtype=bool)
    method = cfg["method"]

    if method == "range_box":
        rb = cfg.get("rb_bars", 20)
        th = cfg.get("rb_threshold", 3.0)
        for i in range(rb, n):
            a = atr_v[i]
            if a <= 0 or np.isnan(a): continue
            rng = max(high[i-rb:i+1]) - min(low[i-rb:i+1])
            in_chop[i] = (rng / a) < th

    elif method == "orb":
        ob = cfg.get("orb_bars", 30)
        # For each day, first ob bars define the range
        current_date = None; box_high = 0; box_low = 0; or_set = False; bar_in_day = 0
        for i in range(n):
            d = dates[i]
            if d != current_date:
                current_date = d; bar_in_day = 0; or_set = False
                box_high = high[i]; box_low = low[i]
            bar_in_day += 1
            if bar_in_day <= ob:
                box_high = max(box_high, high[i])
                box_low = min(box_low, low[i])
                in_chop[i] = True  # still forming range
            else:
                if not or_set:
                    or_set = True
                # After ORB period: chop if price still inside box
                if close[i] <= box_high and close[i] >= box_low:
                    in_chop[i] = True
                else:
                    in_chop[i] = False

    elif method == "donchian":
        db = cfg.get("don_bars", 20)
        dr = cfg.get("don_recency", 5)
        for i in range(db, n):
            hh = max(high[i-db:i+1])
            ll = min(low[i-db:i+1])
            # Was the high/low set within last dr bars?
            recent_hh = max(high[max(i-dr,0):i+1])
            recent_ll = min(low[max(i-dr,0):i+1])
            near_top = (hh - recent_hh) < atr_v[i] * 0.2 if atr_v[i] > 0 else False
            near_bot = (recent_ll - ll) < atr_v[i] * 0.2 if atr_v[i] > 0 else False
            in_chop[i] = not (near_top or near_bot)

    elif method == "dynamic":
        db = cfg.get("dyn_bars", 15)
        th = cfg.get("dyn_threshold", 2.5)
        brk = cfg.get("dyn_break_atr", 0.3)
        box_active = False; box_h = 0; box_l = 0; box_count = 0
        for i in range(db, n):
            a = atr_v[i]
            if a <= 0 or np.isnan(a): in_chop[i] = False; continue

            # Check if last db bars form a box
            rng = max(high[i-db:i+1]) - min(low[i-db:i+1])
            if rng / a < th:
                box_active = True
                box_h = max(high[i-db:i+1])
                box_l = min(low[i-db:i+1])

            if box_active:
                # Break out?
                if close[i] > box_h + brk * a or close[i] < box_l - brk * a:
                    box_active = False
                    in_chop[i] = False
                else:
                    in_chop[i] = True
            else:
                in_chop[i] = False

    elif method == "ema_range":
        eb = cfg.get("er_bars", 20)
        eth = cfg.get("er_threshold", 1.5)
        for i in range(eb, n):
            a = atr_v[i]
            if a <= 0 or np.isnan(a): continue
            # Max distance from EMA20 in last N bars
            max_dist = max(abs(close[j] - ema[j]) for j in range(i-eb, i+1))
            in_chop[i] = (max_dist / a) < eth

    elif method == "narrow_bars":
        nb = cfg.get("nb_count", 5)
        nth = cfg.get("nb_threshold", 0.5)
        brk = cfg.get("nb_break_mult", 2.0)
        narrow_streak = 0
        avg_narrow_range = 0
        for i in range(1, n):
            a = atr_v[i]
            if a <= 0 or np.isnan(a): narrow_streak = 0; continue
            bar_range = high[i] - low[i]
            is_narrow = (bar_range / a) < nth
            if is_narrow:
                narrow_streak += 1
                avg_narrow_range = (avg_narrow_range * (narrow_streak - 1) + bar_range) / narrow_streak
            else:
                if narrow_streak >= nb and bar_range > avg_narrow_range * brk:
                    # Breakout bar — NOT chop
                    in_chop[i] = False
                    narrow_streak = 0
                    continue
                narrow_streak = 0

            in_chop[i] = narrow_streak >= nb

    # Run backtest with chop filter
    equity = capital; trades = []
    bar = max(p["ema_slow"], p["atr_period"]) + 50
    daily_r_loss = 0.0; current_date = None; skip_count = 0
    chop_skipped = 0

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

        # Box filter
        if method != "none" and in_chop[bar]:
            chop_skipped += 1; bar += 1; continue

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
                trade_pnl += remaining * (runner_stop - actual_entry) * trend - remaining * comm
                end_bar = bi; break
            if not lock_done and remaining > lock_sh:
                target = actual_entry + p["lock_rr"] * risk * trend
                if (trend == 1 and h >= target) or (trend == -1 and l <= target):
                    trade_pnl += lock_sh * p["lock_rr"] * risk - lock_sh * comm
                    remaining -= lock_sh; lock_done = True
                    if trend == 1: runner_stop = max(runner_stop, actual_entry)
                    else: runner_stop = min(runner_stop, actual_entry)
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
        r_mult = trade_pnl / (shares * risk) if shares * risk > 0 else 0
        if trade_pnl < 0 and shares * risk > 0: daily_r_loss += abs(r_mult)
        if r_mult > 0: skip_count = 1

        trades.append({"pnl": trade_pnl, "r": r_mult, "dir": trend,
                        "shares": shares, "risk": risk})
        bar = end_bar + 1

    tdf = pd.DataFrame(trades) if trades else pd.DataFrame()
    total = len(tdf)
    if total == 0:
        return {"pf": 0, "ret": 0, "trades": 0, "big5": 0, "lpf": 0, "spf": 0,
                "tpd": 0, "skipped": chop_skipped}
    gw = tdf.loc[tdf["r"]>0,"r"].sum() if (tdf["r"]>0).any() else 0
    gl = abs(tdf.loc[tdf["r"]<=0,"r"].sum()) if (tdf["r"]<=0).any() else 0
    pf = gw/gl if gl > 0 else 0; ret = (equity - capital) / capital * 100
    r_arr = tdf["r"].values
    days = df.index.normalize().nunique()
    longs = tdf[tdf["dir"]==1]; shorts = tdf[tdf["dir"]==-1]
    def spf(s):
        if len(s)==0: return 0
        w = s.loc[s["pnl"]>0,"pnl"].sum(); l = abs(s.loc[s["pnl"]<=0,"pnl"].sum())
        return round(w/l,3) if l>0 else 0
    return {"pf": round(pf,3), "ret": round(ret,2), "trades": total,
            "big5": int((r_arr>=5).sum()), "lpf": spf(longs), "spf": spf(shorts),
            "tpd": round(total/max(days,1),1), "skipped": chop_skipped}


def fmt(label, r, bl_pf):
    d = r["pf"] - bl_pf
    return (f"  {label:<55} {r['pf']:>6.3f} ({d:>+.3f})"
            f" {r['ret']:>+7.2f}% {r['trades']:>5} {r['big5']:>4}"
            f" {r['lpf']:>5.2f} {r['spf']:>5.2f} {r['skipped']:>6}")


def run_suite(df, label_prefix=""):
    bl = run_with_box(df, box_cfg={"method": "none"})
    hdr = (f"  {'Config':<55} {'PF':>6} {'ΔPF':>7}"
           f" {'Ret%':>8} {'Trd':>5} {'5R+':>5}"
           f" {'L.PF':>6} {'S.PF':>6} {'Skip':>6}")
    sep = f"  {'-'*105}"

    print(f"\n  Baseline: PF={bl['pf']:.3f}, ret={bl['ret']:+.2f}%, trades={bl['trades']}, 5R+={bl['big5']}")
    print(hdr); print(sep)

    configs = [
        # A: Range box
        ("A1: Range box 10b, <2.0 ATR",   {"method": "range_box", "rb_bars": 10, "rb_threshold": 2.0}),
        ("A2: Range box 15b, <2.5 ATR",   {"method": "range_box", "rb_bars": 15, "rb_threshold": 2.5}),
        ("A3: Range box 20b, <3.0 ATR",   {"method": "range_box", "rb_bars": 20, "rb_threshold": 3.0}),
        ("A4: Range box 20b, <2.5 ATR",   {"method": "range_box", "rb_bars": 20, "rb_threshold": 2.5}),
        ("A5: Range box 30b, <3.0 ATR",   {"method": "range_box", "rb_bars": 30, "rb_threshold": 3.0}),
        ("A6: Range box 30b, <4.0 ATR",   {"method": "range_box", "rb_bars": 30, "rb_threshold": 4.0}),
        ("A7: Range box 50b, <4.0 ATR",   {"method": "range_box", "rb_bars": 50, "rb_threshold": 4.0}),
        ("A8: Range box 50b, <5.0 ATR",   {"method": "range_box", "rb_bars": 50, "rb_threshold": 5.0}),

        # B: Opening range breakout
        ("B1: ORB 15 min",                {"method": "orb", "orb_bars": 15}),
        ("B2: ORB 30 min",                {"method": "orb", "orb_bars": 30}),
        ("B3: ORB 45 min",                {"method": "orb", "orb_bars": 45}),
        ("B4: ORB 60 min",                {"method": "orb", "orb_bars": 60}),

        # C: Donchian
        ("C1: Donchian 20b, recent 3",    {"method": "donchian", "don_bars": 20, "don_recency": 3}),
        ("C2: Donchian 20b, recent 5",    {"method": "donchian", "don_bars": 20, "don_recency": 5}),
        ("C3: Donchian 30b, recent 5",    {"method": "donchian", "don_bars": 30, "don_recency": 5}),
        ("C4: Donchian 50b, recent 5",    {"method": "donchian", "don_bars": 50, "don_recency": 5}),

        # D: Dynamic box
        ("D1: Dynamic 10b, <2.0 ATR, break 0.2",  {"method": "dynamic", "dyn_bars": 10, "dyn_threshold": 2.0, "dyn_break_atr": 0.2}),
        ("D2: Dynamic 15b, <2.5 ATR, break 0.3",  {"method": "dynamic", "dyn_bars": 15, "dyn_threshold": 2.5, "dyn_break_atr": 0.3}),
        ("D3: Dynamic 20b, <3.0 ATR, break 0.3",  {"method": "dynamic", "dyn_bars": 20, "dyn_threshold": 3.0, "dyn_break_atr": 0.3}),
        ("D4: Dynamic 15b, <2.0 ATR, break 0.5",  {"method": "dynamic", "dyn_bars": 15, "dyn_threshold": 2.0, "dyn_break_atr": 0.5}),
        ("D5: Dynamic 20b, <2.5 ATR, break 0.5",  {"method": "dynamic", "dyn_bars": 20, "dyn_threshold": 2.5, "dyn_break_atr": 0.5}),

        # E: EMA range
        ("E1: EMA range 10b, <1.0 ATR",   {"method": "ema_range", "er_bars": 10, "er_threshold": 1.0}),
        ("E2: EMA range 15b, <1.0 ATR",   {"method": "ema_range", "er_bars": 15, "er_threshold": 1.0}),
        ("E3: EMA range 20b, <1.5 ATR",   {"method": "ema_range", "er_bars": 20, "er_threshold": 1.5}),
        ("E4: EMA range 15b, <1.5 ATR",   {"method": "ema_range", "er_bars": 15, "er_threshold": 1.5}),
        ("E5: EMA range 10b, <0.8 ATR",   {"method": "ema_range", "er_bars": 10, "er_threshold": 0.8}),

        # F: Narrow bars
        ("F1: 3 narrow bars (<0.4 ATR), break 2x",  {"method": "narrow_bars", "nb_count": 3, "nb_threshold": 0.4, "nb_break_mult": 2.0}),
        ("F2: 5 narrow bars (<0.5 ATR), break 2x",  {"method": "narrow_bars", "nb_count": 5, "nb_threshold": 0.5, "nb_break_mult": 2.0}),
        ("F3: 3 narrow bars (<0.3 ATR), break 1.5x", {"method": "narrow_bars", "nb_count": 3, "nb_threshold": 0.3, "nb_break_mult": 1.5}),
        ("F4: 5 narrow bars (<0.4 ATR), break 1.5x", {"method": "narrow_bars", "nb_count": 5, "nb_threshold": 0.4, "nb_break_mult": 1.5}),
    ]

    results = []
    for label, cfg in configs:
        r = run_with_box(df, box_cfg=cfg)
        print(fmt(label_prefix + label, r, bl["pf"]))
        results.append((label, r))

    return bl, results


def main():
    df_is = pd.read_csv(IS_PATH, index_col="timestamp", parse_dates=True)
    df_oos = pd.read_csv(OOS_PATH, index_col="timestamp", parse_dates=True)

    # ═══ IS ═══
    print(f"{'='*110}")
    print(f"  IN-SAMPLE (Polygon 2024-2026)")
    print(f"{'='*110}")
    bl_is, res_is = run_suite(df_is)

    # Top 10
    res_is.sort(key=lambda x: x[1]["pf"], reverse=True)
    print(f"\n  TOP 10 IN-SAMPLE:")
    print(f"  {'#':>2} {'Config':<55} {'PF':>6} {'Ret%':>8} {'5R+':>5} {'L.PF':>6} {'S.PF':>6}")
    print(f"  {'-'*90}")
    for i, (label, r) in enumerate(res_is[:10]):
        print(f"  {i+1:>2} {label:<55} {r['pf']:>6.3f} {r['ret']:>+7.2f}% {r['big5']:>4} {r['lpf']:>5.2f} {r['spf']:>5.2f}")

    # ═══ OOS validation on top 5 ═══
    print(f"\n{'='*110}")
    print(f"  OOS VALIDATION (Barchart 2022-2024) — Top IS configs")
    print(f"{'='*110}")

    # Reconstruct configs for top 5
    all_configs = {label: cfg for label, cfg in [
        ("A1: Range box 10b, <2.0 ATR",   {"method": "range_box", "rb_bars": 10, "rb_threshold": 2.0}),
        ("A2: Range box 15b, <2.5 ATR",   {"method": "range_box", "rb_bars": 15, "rb_threshold": 2.5}),
        ("A3: Range box 20b, <3.0 ATR",   {"method": "range_box", "rb_bars": 20, "rb_threshold": 3.0}),
        ("A4: Range box 20b, <2.5 ATR",   {"method": "range_box", "rb_bars": 20, "rb_threshold": 2.5}),
        ("A5: Range box 30b, <3.0 ATR",   {"method": "range_box", "rb_bars": 30, "rb_threshold": 3.0}),
        ("A6: Range box 30b, <4.0 ATR",   {"method": "range_box", "rb_bars": 30, "rb_threshold": 4.0}),
        ("A7: Range box 50b, <4.0 ATR",   {"method": "range_box", "rb_bars": 50, "rb_threshold": 4.0}),
        ("A8: Range box 50b, <5.0 ATR",   {"method": "range_box", "rb_bars": 50, "rb_threshold": 5.0}),
        ("B1: ORB 15 min",                {"method": "orb", "orb_bars": 15}),
        ("B2: ORB 30 min",                {"method": "orb", "orb_bars": 30}),
        ("B3: ORB 45 min",                {"method": "orb", "orb_bars": 45}),
        ("B4: ORB 60 min",                {"method": "orb", "orb_bars": 60}),
        ("C1: Donchian 20b, recent 3",    {"method": "donchian", "don_bars": 20, "don_recency": 3}),
        ("C2: Donchian 20b, recent 5",    {"method": "donchian", "don_bars": 20, "don_recency": 5}),
        ("C3: Donchian 30b, recent 5",    {"method": "donchian", "don_bars": 30, "don_recency": 5}),
        ("C4: Donchian 50b, recent 5",    {"method": "donchian", "don_bars": 50, "don_recency": 5}),
        ("D1: Dynamic 10b, <2.0 ATR, break 0.2",  {"method": "dynamic", "dyn_bars": 10, "dyn_threshold": 2.0, "dyn_break_atr": 0.2}),
        ("D2: Dynamic 15b, <2.5 ATR, break 0.3",  {"method": "dynamic", "dyn_bars": 15, "dyn_threshold": 2.5, "dyn_break_atr": 0.3}),
        ("D3: Dynamic 20b, <3.0 ATR, break 0.3",  {"method": "dynamic", "dyn_bars": 20, "dyn_threshold": 3.0, "dyn_break_atr": 0.3}),
        ("D4: Dynamic 15b, <2.0 ATR, break 0.5",  {"method": "dynamic", "dyn_bars": 15, "dyn_threshold": 2.0, "dyn_break_atr": 0.5}),
        ("D5: Dynamic 20b, <2.5 ATR, break 0.5",  {"method": "dynamic", "dyn_bars": 20, "dyn_threshold": 2.5, "dyn_break_atr": 0.5}),
        ("E1: EMA range 10b, <1.0 ATR",   {"method": "ema_range", "er_bars": 10, "er_threshold": 1.0}),
        ("E2: EMA range 15b, <1.0 ATR",   {"method": "ema_range", "er_bars": 15, "er_threshold": 1.0}),
        ("E3: EMA range 20b, <1.5 ATR",   {"method": "ema_range", "er_bars": 20, "er_threshold": 1.5}),
        ("E4: EMA range 15b, <1.5 ATR",   {"method": "ema_range", "er_bars": 15, "er_threshold": 1.5}),
        ("E5: EMA range 10b, <0.8 ATR",   {"method": "ema_range", "er_bars": 10, "er_threshold": 0.8}),
        ("F1: 3 narrow bars (<0.4 ATR), break 2x",  {"method": "narrow_bars", "nb_count": 3, "nb_threshold": 0.4, "nb_break_mult": 2.0}),
        ("F2: 5 narrow bars (<0.5 ATR), break 2x",  {"method": "narrow_bars", "nb_count": 5, "nb_threshold": 0.5, "nb_break_mult": 2.0}),
        ("F3: 3 narrow bars (<0.3 ATR), break 1.5x", {"method": "narrow_bars", "nb_count": 3, "nb_threshold": 0.3, "nb_break_mult": 1.5}),
        ("F4: 5 narrow bars (<0.4 ATR), break 1.5x", {"method": "narrow_bars", "nb_count": 5, "nb_threshold": 0.4, "nb_break_mult": 1.5}),
    ]}

    bl_oos = run_with_box(df_oos, box_cfg={"method": "none"})
    print(f"\n  OOS Baseline: PF={bl_oos['pf']:.3f}, ret={bl_oos['ret']:+.2f}%, trades={bl_oos['trades']}")

    print(f"\n  {'Config':<55} {'IS PF':>7} {'OOS PF':>8} {'ΔPF':>6} {'OOS 5R+':>8} {'OOS Ret':>9}")
    print(f"  {'-'*95}")

    for label, r_is in res_is[:10]:
        if label in all_configs:
            r_oos = run_with_box(df_oos, box_cfg=all_configs[label])
            delta = r_oos["pf"] - r_is["pf"]
            holds = "✅" if r_oos["pf"] > bl_oos["pf"] else "❌"
            print(f"  {label:<55} {r_is['pf']:>6.3f} {r_oos['pf']:>7.3f} {delta:>+5.3f}"
                  f" {r_oos['big5']:>7} {r_oos['ret']:>+8.2f}% {holds}")


if __name__ == "__main__":
    main()
