"""
Phase 3: Trade-During Management — Early Exit for Non-Starters

Core idea: "If it's going to work, price moves fast. If it doesn't move, get out cheap."

Instead of waiting for full -1R stop loss, exit early if price doesn't confirm
within the first N bars. This turns 23.7% of -1R losses into -0.2R to -0.5R losses.

Step 1: ANALYZE — compare early behavior of winners vs losers
  - How fast do winners move in the first 3/5/10 bars?
  - Is there a measurable difference?

Step 2: EXPERIMENT — various early exit rules
  A. Time + MFE gate: if MFE < X×R after N bars → exit at market
  B. Time + MFE gate: if MFE < X×R after N bars → move stop to tighter level
  C. New high gate: if no new high in N bars → exit
  D. Favorable close gate: if close < entry for N consecutive bars → exit
  E. Speed gate: if MFE/bars < threshold → exit
  F. Hybrid: combine time + MFE + close direction
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


def analyze_early_behavior(df):
    """Analyze first N bars behavior for winners vs losers."""
    p = BASE.copy()
    df = add_indicators(df, p)
    high = df["High"].values; low = df["Low"].values; close = df["Close"].values
    ema = df["ema20"].values; ema_s = df["ema50"].values; atr_v = df["atr"].values
    times = df.index.time; dates = df.index.date; n = len(df)
    comm = p["commission_per_share"]

    trades = []
    bar = max(p["ema_slow"], p["atr_period"]) + 5
    daily_r_loss = 0.0; current_date = None; skip_count = 0
    equity = 100_000

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

        # Track early behavior: MFE and MAE at bars 1,2,3,5,10
        early = {}
        for look in [1, 2, 3, 5, 10]:
            bi = entry_bar + look
            if bi >= n: break
            if trend == 1:
                mfe = max((high[entry_bar + j] - actual_entry) / risk for j in range(1, look + 1) if entry_bar + j < n)
                mae = max((actual_entry - low[entry_bar + j]) / risk for j in range(1, look + 1) if entry_bar + j < n)
                close_vs_entry = (close[bi] - actual_entry) / risk
            else:
                mfe = max((actual_entry - low[entry_bar + j]) / risk for j in range(1, look + 1) if entry_bar + j < n)
                mae = max((high[entry_bar + j] - actual_entry) / risk for j in range(1, look + 1) if entry_bar + j < n)
                close_vs_entry = (actual_entry - close[bi]) / risk
            early[f"mfe_{look}"] = mfe
            early[f"mae_{look}"] = mae
            early[f"close_{look}"] = close_vs_entry

            # New high count in first N bars
            new_highs = 0
            for j in range(1, look + 1):
                bj = entry_bar + j
                if bj >= n: break
                if trend == 1 and high[bj] > max(high[entry_bar + k] for k in range(max(0, j-1) + 0, j) if entry_bar + k < n):
                    # Simplified: is this bar's high > previous bar's high?
                    if j == 1 or high[bj] > high[bj - 1]:
                        new_highs += 1
                elif trend == -1 and low[bj] < low[bj - 1] if j > 0 else True:
                    new_highs += 1
            early[f"new_highs_{look}"] = new_highs

        # Consecutive favorable closes
        consec_fav = 0
        for j in range(1, 11):
            bj = entry_bar + j
            if bj >= n: break
            if trend == 1 and close[bj] > close[bj - 1]:
                consec_fav += 1
            elif trend == -1 and close[bj] < close[bj - 1]:
                consec_fav += 1
            else:
                break
        early["consec_favorable"] = consec_fav

        # Run full trade for exit classification
        lock_sh = max(1, int(shares * p["lock_pct"]))
        remaining = shares; runner_stop = stop; lock_done = False
        trade_pnl = -shares * comm; end_bar = entry_bar; exit_reason = "timeout"

        for k in range(1, p["max_hold_bars"] + 1):
            bi = entry_bar + k
            if bi >= n: break
            h = high[bi]; l = low[bi]
            ca = atr_v[bi] if not np.isnan(atr_v[bi]) else a
            if times[bi] >= p["force_close_at"]:
                trade_pnl += remaining * (close[bi] - actual_entry) * trend - remaining * comm
                end_bar = bi; exit_reason = "session_close"; break
            stopped = (trend == 1 and l <= runner_stop) or (trend == -1 and h >= runner_stop)
            if stopped:
                trade_pnl += remaining * (runner_stop - actual_entry) * trend - remaining * comm
                end_bar = bi
                exit_reason = "be_stop" if lock_done and abs(runner_stop - actual_entry) < 0.02 else ("trail_stop" if lock_done else "initial_stop")
                break
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

        trades.append({"r": r_mult, "exit": exit_reason, **early})
        bar = end_bar + 1

    return pd.DataFrame(trades)


def run_with_early_exit(df, capital=100_000, ee_cfg=None):
    """
    ee_cfg:
      gate_bars: int — check after N bars
      gate_mfe: float — minimum MFE in R to continue
      gate_action: "exit" | "tighten" — what to do if gate fails
      gate_tighten_to: float — if tighten, move stop to entry - X*R (negative = below entry)
      require_new_high: bool — must have made new high in gate_bars
      require_fav_close: int — need N consecutive favorable closes
    """
    cfg = {"gate_bars": 0, "gate_mfe": 0, "gate_action": "exit",
           "gate_tighten_to": 0, "require_new_high": False,
           "require_fav_close": 0}
    if ee_cfg: cfg.update(ee_cfg)

    p = BASE.copy()
    df = add_indicators(df, p)
    high = df["High"].values; low = df["Low"].values; close = df["Close"].values
    ema = df["ema20"].values; ema_s = df["ema50"].values; atr_v = df["atr"].values
    times = df.index.time; dates = df.index.date; n = len(df)
    comm = p["commission_per_share"]

    equity = capital; trades = []
    bar = max(p["ema_slow"], p["atr_period"]) + 5
    daily_r_loss = 0.0; current_date = None; skip_count = 0
    early_exits = 0

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
        trade_pnl = -shares * comm; end_bar = entry_bar; exit_reason = "timeout"
        gate_checked = False; gate_passed = True
        mfe_so_far = 0.0

        for k in range(1, p["max_hold_bars"] + 1):
            bi = entry_bar + k
            if bi >= n: break
            h = high[bi]; l = low[bi]
            ca = atr_v[bi] if not np.isnan(atr_v[bi]) else a

            # Track MFE
            if trend == 1:
                mfe_so_far = max(mfe_so_far, (h - actual_entry) / risk)
            else:
                mfe_so_far = max(mfe_so_far, (actual_entry - l) / risk)

            if times[bi] >= p["force_close_at"]:
                trade_pnl += remaining * (close[bi] - actual_entry) * trend - remaining * comm
                end_bar = bi; exit_reason = "session_close"; break

            # ─── EARLY EXIT GATE ───
            if cfg["gate_bars"] > 0 and k == cfg["gate_bars"] and not gate_checked and not lock_done:
                gate_checked = True
                failed = False

                # MFE gate
                if cfg["gate_mfe"] > 0 and mfe_so_far < cfg["gate_mfe"]:
                    failed = True

                # New high gate
                if cfg["require_new_high"]:
                    made_new = False
                    for j in range(2, k + 1):
                        bj = entry_bar + j
                        if bj >= n: break
                        if trend == 1 and high[bj] > high[bj - 1]: made_new = True; break
                        if trend == -1 and low[bj] < low[bj - 1]: made_new = True; break
                    if not made_new: failed = True

                # Favorable close gate
                if cfg["require_fav_close"] > 0:
                    fav_count = 0
                    for j in range(1, k + 1):
                        bj = entry_bar + j
                        if bj >= n: break
                        if trend == 1 and close[bj] > close[bj - 1]: fav_count += 1
                        elif trend == -1 and close[bj] < close[bj - 1]: fav_count += 1
                    if fav_count < cfg["require_fav_close"]: failed = True

                if failed:
                    gate_passed = False
                    if cfg["gate_action"] == "exit":
                        # Exit at current close
                        trade_pnl += remaining * (close[bi] - actual_entry) * trend - remaining * comm
                        end_bar = bi; exit_reason = "early_exit"; early_exits += 1; break
                    elif cfg["gate_action"] == "tighten":
                        # Move stop closer
                        new_stop = actual_entry + cfg["gate_tighten_to"] * risk * trend
                        if trend == 1: runner_stop = max(runner_stop, new_stop)
                        else: runner_stop = min(runner_stop, new_stop)

            # Stop check
            stopped = (trend == 1 and l <= runner_stop) or (trend == -1 and h >= runner_stop)
            if stopped:
                trade_pnl += remaining * (runner_stop - actual_entry) * trend - remaining * comm
                end_bar = bi
                exit_reason = "be_stop" if lock_done and abs(runner_stop - actual_entry) < 0.02 else ("trail_stop" if lock_done else "initial_stop")
                break

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
        trades.append({"r": r_mult, "exit": exit_reason, "dir": trend})
        bar = end_bar + 1

    tdf = pd.DataFrame(trades) if trades else pd.DataFrame()
    total = len(tdf)
    if total == 0: return {"pf":0,"ret":0,"trades":0,"big5":0,"lpf":0,"spf":0,"early":0}
    gw = tdf.loc[tdf["r"]>0,"r"].sum(); gl = abs(tdf.loc[tdf["r"]<=0,"r"].sum())
    pf = gw/gl if gl>0 else 0; ret = (equity-capital)/capital*100
    r_arr = tdf["r"].values; days = df.index.normalize().nunique()
    longs = tdf[tdf["dir"]==1]; shorts = tdf[tdf["dir"]==-1]
    def spf(s):
        if len(s)==0: return 0
        w=s.loc[s["r"]>0,"r"].sum(); l=abs(s.loc[s["r"]<=0,"r"].sum())
        return round(w/l,3) if l>0 else 0

    # Exit breakdown
    ee = tdf[tdf["exit"]=="early_exit"]
    init = tdf[tdf["exit"]=="initial_stop"]
    be = tdf[tdf["exit"]=="be_stop"]
    trail = tdf[tdf["exit"]=="trail_stop"]

    return {"pf":round(pf,3),"ret":round(ret,2),"trades":total,
            "big5":int((r_arr>=5).sum()),"lpf":spf(longs),"spf":spf(shorts),
            "early_n":len(ee),"early_avg_r":round(ee["r"].mean(),3) if len(ee)>0 else 0,
            "init_n":len(init),"be_n":len(be),"trail_n":len(trail),
            "trail_avg_r":round(trail["r"].mean(),2) if len(trail)>0 else 0,
            "tpd":round(total/max(days,1),1)}


def main():
    df_is = pd.read_csv(IS_PATH, index_col="timestamp", parse_dates=True)
    df_oos = pd.read_csv(OOS_PATH, index_col="timestamp", parse_dates=True)

    # ═══ STEP 1: ANALYSIS ═══
    print(f"{'='*100}")
    print(f"  STEP 1: EARLY BEHAVIOR — Winners vs Losers in first N bars")
    print(f"{'='*100}\n")

    tdf = analyze_early_behavior(df_is)
    trail = tdf[tdf["exit"] == "trail_stop"]
    init = tdf[tdf["exit"] == "initial_stop"]
    be = tdf[tdf["exit"] == "be_stop"]
    big5 = tdf[tdf["r"] >= 5]

    print(f"  {'Metric':<20} {'Trail(win)':>10} {'InitStop':>10} {'BE stop':>10} {'5R+':>10} {'Cohen d':>9}")
    print(f"  {'-'*72}")

    for col in ["mfe_1","mfe_2","mfe_3","mfe_5","mfe_10",
                "mae_1","mae_2","mae_3","mae_5","mae_10",
                "close_1","close_2","close_3","close_5","close_10",
                "new_highs_3","new_highs_5","new_highs_10",
                "consec_favorable"]:
        if col not in tdf.columns: continue
        tw = trail[col].mean(); ti = init[col].mean(); tb = be[col].mean(); t5 = big5[col].mean()
        pooled = np.sqrt((trail[col].std()**2 + init[col].std()**2) / 2)
        d_init = abs(tw - ti) / pooled if pooled > 0 else 0
        sig = "★★★" if d_init > 0.4 else ("★★" if d_init > 0.25 else ("★" if d_init > 0.15 else ""))
        print(f"  {col:<20} {tw:>10.3f} {ti:>10.3f} {tb:>10.3f} {t5:>10.3f} {d_init:>8.3f}  {sig}")

    # ═══ STEP 2: EXPERIMENTS ═══
    print(f"\n{'='*100}")
    print(f"  STEP 2: EARLY EXIT EXPERIMENTS — IS + OOS")
    print(f"{'='*100}")

    bl_is = run_with_early_exit(df_is)
    bl_oos = run_with_early_exit(df_oos)
    print(f"\n  Baseline IS:  PF={bl_is['pf']:.3f} ret={bl_is['ret']:+.2f}% trades={bl_is['trades']} 5R+={bl_is['big5']} trail={bl_is['trail_n']}@{bl_is['trail_avg_r']}R")
    print(f"  Baseline OOS: PF={bl_oos['pf']:.3f} ret={bl_oos['ret']:+.2f}% trades={bl_oos['trades']} 5R+={bl_oos['big5']}\n")

    hdr = (f"  {'Config':<45} {'IS PF':>7} {'Ret%':>7} {'Trd':>5} {'5R+':>4}"
           f" {'Early':>5} {'EarlyR':>7} {'Init':>5} {'Trail':>5} {'TrR':>5}"
           f" {'OOS PF':>8} {'O.5R+':>6} {'Holds':>5}")
    print(hdr)
    print(f"  {'-'*130}")

    configs = [
        # A: MFE gate — exit if not profitable enough
        ("3b MFE>0.1R else exit",    {"gate_bars":3, "gate_mfe":0.1, "gate_action":"exit"}),
        ("3b MFE>0.2R else exit",    {"gate_bars":3, "gate_mfe":0.2, "gate_action":"exit"}),
        ("3b MFE>0.3R else exit",    {"gate_bars":3, "gate_mfe":0.3, "gate_action":"exit"}),
        ("3b MFE>0.5R else exit",    {"gate_bars":3, "gate_mfe":0.5, "gate_action":"exit"}),
        ("5b MFE>0.2R else exit",    {"gate_bars":5, "gate_mfe":0.2, "gate_action":"exit"}),
        ("5b MFE>0.3R else exit",    {"gate_bars":5, "gate_mfe":0.3, "gate_action":"exit"}),
        ("5b MFE>0.5R else exit",    {"gate_bars":5, "gate_mfe":0.5, "gate_action":"exit"}),
        ("5b MFE>1.0R else exit",    {"gate_bars":5, "gate_mfe":1.0, "gate_action":"exit"}),
        ("10b MFE>0.5R else exit",   {"gate_bars":10,"gate_mfe":0.5, "gate_action":"exit"}),
        ("10b MFE>1.0R else exit",   {"gate_bars":10,"gate_mfe":1.0, "gate_action":"exit"}),

        # B: MFE gate — tighten stop instead of exit
        ("3b MFE>0.2R else→-0.3R",  {"gate_bars":3, "gate_mfe":0.2, "gate_action":"tighten","gate_tighten_to":-0.3}),
        ("3b MFE>0.2R else→-0.5R",  {"gate_bars":3, "gate_mfe":0.2, "gate_action":"tighten","gate_tighten_to":-0.5}),
        ("3b MFE>0.3R else→-0.3R",  {"gate_bars":3, "gate_mfe":0.3, "gate_action":"tighten","gate_tighten_to":-0.3}),
        ("5b MFE>0.3R else→-0.3R",  {"gate_bars":5, "gate_mfe":0.3, "gate_action":"tighten","gate_tighten_to":-0.3}),
        ("5b MFE>0.3R else→-0.5R",  {"gate_bars":5, "gate_mfe":0.3, "gate_action":"tighten","gate_tighten_to":-0.5}),
        ("5b MFE>0.5R else→-0.3R",  {"gate_bars":5, "gate_mfe":0.5, "gate_action":"tighten","gate_tighten_to":-0.3}),
        ("5b MFE>0.5R else→-0.5R",  {"gate_bars":5, "gate_mfe":0.5, "gate_action":"tighten","gate_tighten_to":-0.5}),
        ("10b MFE>0.5R else→-0.5R", {"gate_bars":10,"gate_mfe":0.5, "gate_action":"tighten","gate_tighten_to":-0.5}),
        ("10b MFE>1.0R else→-0.5R", {"gate_bars":10,"gate_mfe":1.0, "gate_action":"tighten","gate_tighten_to":-0.5}),

        # C: New high gate
        ("3b need new high else exit",  {"gate_bars":3, "require_new_high":True, "gate_action":"exit"}),
        ("5b need new high else exit",  {"gate_bars":5, "require_new_high":True, "gate_action":"exit"}),
        ("3b need new high else→-0.3R", {"gate_bars":3, "require_new_high":True, "gate_action":"tighten","gate_tighten_to":-0.3}),
        ("5b need new high else→-0.5R", {"gate_bars":5, "require_new_high":True, "gate_action":"tighten","gate_tighten_to":-0.5}),

        # D: Favorable close gate
        ("need 1 fav close in 3b else exit", {"gate_bars":3, "require_fav_close":1, "gate_action":"exit"}),
        ("need 2 fav close in 3b else exit", {"gate_bars":3, "require_fav_close":2, "gate_action":"exit"}),
        ("need 2 fav close in 5b else exit", {"gate_bars":5, "require_fav_close":2, "gate_action":"exit"}),
        ("need 3 fav close in 5b else exit", {"gate_bars":5, "require_fav_close":3, "gate_action":"exit"}),

        # E: Combined
        ("3b MFE>0.2R + new high else exit",    {"gate_bars":3, "gate_mfe":0.2, "require_new_high":True, "gate_action":"exit"}),
        ("5b MFE>0.3R + new high else exit",    {"gate_bars":5, "gate_mfe":0.3, "require_new_high":True, "gate_action":"exit"}),
        ("5b MFE>0.3R + new high else→-0.3R",   {"gate_bars":5, "gate_mfe":0.3, "require_new_high":True, "gate_action":"tighten","gate_tighten_to":-0.3}),
        ("3b MFE>0.2R + 1fav else→-0.3R",       {"gate_bars":3, "gate_mfe":0.2, "require_fav_close":1, "gate_action":"tighten","gate_tighten_to":-0.3}),
    ]

    results = []
    for label, cfg in configs:
        r_is = run_with_early_exit(df_is, ee_cfg=cfg)
        r_oos = run_with_early_exit(df_oos, ee_cfg=cfg)
        holds = "✅" if r_oos["pf"] > bl_oos["pf"] and r_is["pf"] > bl_is["pf"] else "❌"
        results.append((label, r_is, r_oos))
        print(f"  {label:<45} {r_is['pf']:>6.3f} {r_is['ret']:>+6.2f}% {r_is['trades']:>4} {r_is['big5']:>3}"
              f" {r_is['early_n']:>4} {r_is['early_avg_r']:>+6.3f} {r_is['init_n']:>4} {r_is['trail_n']:>4} {r_is['trail_avg_r']:>+4.1f}"
              f" {r_oos['pf']:>7.3f} {r_oos['big5']:>5}  {holds}")

    # ═══ TOP RESULTS ═══
    print(f"\n{'='*100}")
    print(f"  TOP 10 (IS PF) + OOS check")
    print(f"{'='*100}")
    results.sort(key=lambda x: x[1]["pf"], reverse=True)
    for i, (label, r_is, r_oos) in enumerate(results[:10]):
        holds = "✅" if r_oos["pf"] > bl_oos["pf"] and r_is["pf"] > bl_is["pf"] else "❌"
        print(f"  {i+1:>2}. {label:<45} IS={r_is['pf']:.3f} OOS={r_oos['pf']:.3f}"
              f"  IS_5R+={r_is['big5']} trail={r_is['trail_n']}  {holds}")


if __name__ == "__main__":
    main()
