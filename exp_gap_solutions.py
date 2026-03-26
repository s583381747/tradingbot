"""
Experiment: Gap Solutions — find ways to keep edge while reducing gap impact.

All modes use gap-fixed entry (actual_entry = max(sig, open)) as baseline truth.

Approaches:
  1. Larger offsets ($0.05 → $0.10, $0.15, $0.20, ATR-based)
  2. Delay filter: skip bar+1 triggers, only enter on bar+2/3 (where gap ≈ 0)
  3. Gap cap: skip trade if gap > threshold
  4. Adjusted risk: recalculate risk using actual_entry (not sig)
  5. Bounce strength filter: require close > high + X (stronger bounce = less gap?)
  6. Time filter: exclude high-gap hours (9:xx, 10:xx)
  7. Hybrid: offset scales with recent gap average
  8. Open confirmation: after bounce, wait 1 bar, enter at NEXT bar open if > sig
"""
from __future__ import annotations
import functools, datetime as dt
import numpy as np, pandas as pd
from entry_signal import add_indicators, detect_trend, check_touch, check_bounce, calc_signal_line

print = functools.partial(print, flush=True)
DATA_PATH = "data/QQQ_1Min_Polygon_2y_clean.csv"

PARAMS = {
    "ema_fast": 20, "ema_slow": 50, "atr_period": 14,
    "touch_tol": 0.15, "touch_below_max": 0.5,
    "stop_buffer": 0.3,
    "lock1_rr": 0.3, "lock1_pct": 0.20,
    "chandelier_bars": 40, "chandelier_mult": 0.5,
    "max_hold_bars": 180,
    "risk_pct": 0.01, "max_pos_pct": 0.25,
    "no_entry_after": dt.time(15, 30), "force_close_at": dt.time(15, 58),
    "commission_per_share": 0.005, "daily_loss_r": 2.5,
}


def run_backtest(df, capital=100_000, config=None):
    """
    config dict keys:
      offset:       signal offset $ (default 0.05)
      offset_atr:   signal offset as ATR multiple (overrides offset if set)
      min_delay:    minimum trigger bar delay after bounce (1=next bar, 2=skip first)
      max_gap:      skip trade if gap > this $ amount
      max_gap_r:    skip trade if gap > this fraction of risk
      adjust_risk:  True = recalculate risk with actual_entry
      min_bounce:   minimum bounce strength (close - touch_high) in $ or ATR mult
      min_bounce_atr: minimum bounce strength as ATR multiple
      no_hours:     list of hours to exclude (e.g. [9, 10])
      enter_at_open: True = enter at bar+2 open price directly (market order after bounce)
    """
    cfg = config or {}
    p = PARAMS.copy()
    df = add_indicators(df, p)
    high = df["High"].values; low = df["Low"].values
    close = df["Close"].values; open_p = df["Open"].values
    ema = df["ema20"].values; ema_s = df["ema50"].values
    atr_v = df["atr"].values
    times = df.index.time; dates = df.index.date; hours = df.index.hour; n = len(df)

    comm = p["commission_per_share"]; max_fwd = p["max_hold_bars"]
    no_hours = set(cfg.get("no_hours", []))

    equity = capital; trade_log = []
    bar = max(p["ema_slow"], p["atr_period"]) + 5
    daily_r_loss = 0.0; current_date = None
    signals = 0; skipped_gap = 0; skipped_hour = 0; skipped_delay = 0; skipped_bounce = 0

    while bar < n - max_fwd - 5:
        a = atr_v[bar]
        if np.isnan(a) or a <= 0 or np.isnan(ema[bar]) or np.isnan(ema_s[bar]):
            bar += 1; continue
        if times[bar] >= p["no_entry_after"]: bar += 1; continue

        d_date = dates[bar]
        if current_date != d_date: current_date = d_date; daily_r_loss = 0.0
        if daily_r_loss >= p["daily_loss_r"]: bar += 1; continue

        trend = detect_trend(close[bar], ema[bar], ema_s[bar])
        if trend == 0: bar += 1; continue
        if not check_touch(trend, low[bar], high[bar], ema[bar], a,
                           p["touch_tol"], p["touch_below_max"]):
            bar += 1; continue

        # Hour filter
        if hours[bar] in no_hours:
            skipped_hour += 1; bar += 1; continue

        bb = bar + 1
        if bb >= n: bar += 1; continue
        if not check_bounce(trend, close[bb], high[bar], low[bar]):
            bar += 1; continue

        # Bounce strength filter
        min_bounce_atr = cfg.get("min_bounce_atr", 0)
        if min_bounce_atr > 0:
            if trend == 1:
                bounce_strength = close[bb] - high[bar]
            else:
                bounce_strength = low[bar] - close[bb]
            if bounce_strength < min_bounce_atr * a:
                skipped_bounce += 1; bar += 1; continue

        # Signal line
        offset = cfg.get("offset", 0.05)
        offset_atr = cfg.get("offset_atr", None)
        if offset_atr is not None:
            offset = offset_atr * a

        if trend == 1:
            sig = high[bar] + offset
            stop = low[bar] - p["stop_buffer"] * a
        else:
            sig = low[bar] - offset
            stop = high[bar] + p["stop_buffer"] * a
        risk = abs(sig - stop)
        if risk <= 0: bar += 1; continue

        signals += 1

        # Enter at open mode: skip trigger, enter at bar+2 open
        if cfg.get("enter_at_open", False):
            entry_bar_candidate = bb + 1
            if entry_bar_candidate >= n: bar += 1; continue
            if times[entry_bar_candidate] >= p["force_close_at"]: bar += 1; continue
            actual_entry = open_p[entry_bar_candidate]
            # Check direction still valid
            if trend == 1 and actual_entry < stop: bar += 1; continue
            if trend == -1 and actual_entry > stop: bar += 1; continue
            trigger_bar = entry_bar_candidate
            actual_risk = abs(actual_entry - stop)
            if actual_risk <= 0: bar = trigger_bar + 1; continue
        else:
            # Normal trigger logic
            min_delay = cfg.get("min_delay", 1)
            trigger_bar = -1
            for j in range(min_delay, p.get("signal_valid_bars", 3) + min_delay):
                cb = bb + j
                if cb >= n: break
                if times[cb] >= p["force_close_at"]: break
                if trend == 1 and high[cb] >= sig: trigger_bar = cb; break
                if trend == -1 and low[cb] <= sig: trigger_bar = cb; break

            if trigger_bar < 0:
                if min_delay > 1: skipped_delay += 1
                bar += 1; continue

            # Gap-fixed entry
            if trend == 1:
                actual_entry = max(sig, open_p[trigger_bar])
            else:
                actual_entry = min(sig, open_p[trigger_bar])

            gap = abs(actual_entry - sig)

            # Gap cap filter
            max_gap = cfg.get("max_gap", None)
            max_gap_r = cfg.get("max_gap_r", None)
            if max_gap is not None and gap > max_gap:
                skipped_gap += 1; bar = trigger_bar + 1; continue
            if max_gap_r is not None and risk > 0 and gap / risk > max_gap_r:
                skipped_gap += 1; bar = trigger_bar + 1; continue

            # Risk calculation
            if cfg.get("adjust_risk", False):
                actual_risk = abs(actual_entry - stop)
            else:
                actual_risk = risk

            if actual_risk <= 0: bar = trigger_bar + 1; continue

        # Entry bar stop check
        if trend == 1 and low[trigger_bar] <= stop:
            sh = max(1, int(equity * p["risk_pct"] / actual_risk))
            if sh * actual_entry > equity * p["max_pos_pct"]:
                sh = max(1, int(equity * p["max_pos_pct"] / actual_entry))
            loss = sh * (stop - actual_entry) - sh * comm * 2
            equity += loss
            if sh * actual_risk > 0: daily_r_loss += abs(loss) / (sh * actual_risk)
            trade_log.append({"pnl": loss, "dir": trend, "shares": sh, "risk": actual_risk, "lock": False})
            bar = trigger_bar + 1; continue
        if trend == -1 and high[trigger_bar] >= stop:
            sh = max(1, int(equity * p["risk_pct"] / actual_risk))
            if sh * actual_entry > equity * p["max_pos_pct"]:
                sh = max(1, int(equity * p["max_pos_pct"] / actual_entry))
            loss = sh * (stop - actual_entry) * trend - sh * comm * 2
            equity += loss
            if sh * actual_risk > 0: daily_r_loss += abs(loss) / (sh * actual_risk)
            trade_log.append({"pnl": loss, "dir": trend, "shares": sh, "risk": actual_risk, "lock": False})
            bar = trigger_bar + 1; continue

        # Position sizing
        shares = max(1, int(equity * p["risk_pct"] / actual_risk))
        if shares * actual_entry > equity * p["max_pos_pct"]:
            shares = max(1, int(equity * p["max_pos_pct"] / actual_entry))
        if equity < shares * actual_risk or shares < 1: bar += 1; continue

        # Execute trade (standard exit logic)
        lock_sh = max(1, int(shares * p["lock1_pct"]))
        runner_stop = stop; lock_done = False
        trade_pnl = -shares * comm; remaining = shares; end_bar = trigger_bar

        for k in range(1, max_fwd + 1):
            bi = trigger_bar + k
            if bi >= n: break
            h = high[bi]; l = low[bi]
            ca = atr_v[bi] if not np.isnan(atr_v[bi]) else a

            if times[bi] >= p["force_close_at"]:
                trade_pnl += remaining * (close[bi] - actual_entry) * trend - remaining * comm
                end_bar = bi; break

            stopped = (trend == 1 and l <= runner_stop) or (trend == -1 and h >= runner_stop)
            lh = False
            if not lock_done:
                lh = (trend == 1 and h >= actual_entry + p["lock1_rr"] * actual_risk) or \
                     (trend == -1 and l <= actual_entry - p["lock1_rr"] * actual_risk)
            if stopped and lh and not lock_done:
                trade_pnl += remaining * (runner_stop - actual_entry) * trend - remaining * comm
                end_bar = bi; break
            if stopped:
                trade_pnl += remaining * (runner_stop - actual_entry) * trend - remaining * comm
                end_bar = bi; break
            if lh and not lock_done:
                trade_pnl += lock_sh * p["lock1_rr"] * actual_risk - lock_sh * comm
                remaining -= lock_sh; lock_done = True
                if trend == 1: runner_stop = max(runner_stop, actual_entry)
                else: runner_stop = min(runner_stop, actual_entry)
            if lock_done and k >= p["chandelier_bars"]:
                sk = max(1, k - p["chandelier_bars"] + 1)
                if trend == 1:
                    hh = max(high[trigger_bar + kk] for kk in range(sk, k) if trigger_bar + kk < n)
                    runner_stop = max(runner_stop, hh - p["chandelier_mult"] * ca)
                else:
                    ll = min(low[trigger_bar + kk] for kk in range(sk, k) if trigger_bar + kk < n)
                    runner_stop = min(runner_stop, ll + p["chandelier_mult"] * ca)
        else:
            ep = close[min(trigger_bar + max_fwd, n - 1)]
            trade_pnl += remaining * (ep - actual_entry) * trend - remaining * comm
            end_bar = min(trigger_bar + max_fwd, n - 1)

        equity += trade_pnl
        if trade_pnl < 0 and shares * actual_risk > 0:
            daily_r_loss += abs(trade_pnl) / (shares * actual_risk)
        trade_log.append({"pnl": trade_pnl, "dir": trend, "shares": shares,
                          "risk": actual_risk, "lock": lock_done})
        bar = end_bar + 1

    # Summary
    tdf = pd.DataFrame(trade_log) if trade_log else pd.DataFrame()
    total = len(tdf)
    wins = (tdf["pnl"] > 0).sum() if total > 0 else 0
    losses = total - wins
    gw = tdf.loc[tdf["pnl"] > 0, "pnl"].sum() if wins > 0 else 0
    gl = abs(tdf.loc[tdf["pnl"] <= 0, "pnl"].sum()) if losses > 0 else 0
    pf = gw / gl if gl > 0 else 0
    ret = (equity - capital) / capital * 100
    days = df.index.normalize().nunique()

    # Long/short split
    longs = tdf[tdf["dir"] == 1] if total > 0 else pd.DataFrame()
    shorts = tdf[tdf["dir"] == -1] if total > 0 else pd.DataFrame()
    def spf(sub):
        if len(sub) == 0: return 0
        w = sub.loc[sub["pnl"] > 0, "pnl"].sum()
        l = abs(sub.loc[sub["pnl"] <= 0, "pnl"].sum())
        return round(w / l, 3) if l > 0 else 0
    lpf = spf(longs); spf_v = spf(shorts)

    # R distribution
    r_arr = np.array([t["pnl"]/(t["shares"]*t["risk"]) if t["shares"]*t["risk"]>0 else 0
                       for _, t in tdf.iterrows()]) if total > 0 else np.array([0])
    big5 = int((r_arr >= 5).sum())

    return {
        "pf": round(pf, 3), "ret": round(ret, 2),
        "trades": total, "tpd": round(total / max(days, 1), 1),
        "wr": round(wins / total * 100, 1) if total > 0 else 0,
        "lpf": lpf, "spf": spf_v,
        "ln": len(longs), "sn": len(shorts),
        "big5": big5,
        "signals": signals, "skipped_gap": skipped_gap,
        "skipped_hour": skipped_hour, "skipped_delay": skipped_delay,
        "skipped_bounce": skipped_bounce,
    }


def main():
    df = pd.read_csv(DATA_PATH, index_col="timestamp", parse_dates=True)
    print(f"Data: {len(df):,} bars, {df.index.normalize().nunique()} days\n")

    experiments = [
        # Baseline
        ("BASELINE: current gap-fixed",           {}),

        # === Approach 1: Larger offsets ===
        ("Offset $0.10",                           {"offset": 0.10}),
        ("Offset $0.15",                           {"offset": 0.15}),
        ("Offset $0.20",                           {"offset": 0.20}),
        ("Offset $0.30",                           {"offset": 0.30}),
        ("Offset 0.1×ATR",                         {"offset_atr": 0.10}),
        ("Offset 0.15×ATR",                        {"offset_atr": 0.15}),
        ("Offset 0.2×ATR",                         {"offset_atr": 0.20}),
        ("Offset 0.3×ATR",                         {"offset_atr": 0.30}),

        # === Approach 2: Delay filter ===
        ("Skip bar+1 (only bar+2,3)",              {"min_delay": 2}),
        ("Skip bar+1,2 (only bar+3,4)",            {"min_delay": 3}),

        # === Approach 3: Gap cap ===
        ("Gap cap $0.05",                          {"max_gap": 0.05}),
        ("Gap cap $0.10",                          {"max_gap": 0.10}),
        ("Gap cap $0.15",                          {"max_gap": 0.15}),
        ("Gap cap 20% of R",                       {"max_gap_r": 0.20}),
        ("Gap cap 30% of R",                       {"max_gap_r": 0.30}),
        ("Gap cap 50% of R",                       {"max_gap_r": 0.50}),

        # === Approach 4: Adjust risk to actual entry ===
        ("Adjust risk to actual entry",            {"adjust_risk": True}),
        ("Adjust risk + gap cap 50%R",             {"adjust_risk": True, "max_gap_r": 0.50}),

        # === Approach 5: Bounce strength ===
        ("Bounce > 0.05×ATR",                      {"min_bounce_atr": 0.05}),
        ("Bounce > 0.10×ATR",                      {"min_bounce_atr": 0.10}),
        ("Bounce > 0.15×ATR",                      {"min_bounce_atr": 0.15}),
        ("Bounce > 0.20×ATR",                      {"min_bounce_atr": 0.20}),

        # === Approach 6: Time filter ===
        ("No 9:xx",                                {"no_hours": [9]}),
        ("No 9-10:xx",                             {"no_hours": [9, 10]}),
        ("Only 11-14:xx",                          {"no_hours": [9, 10, 15]}),

        # === Approach 7: Enter at open (market order) ===
        ("Enter at bar+2 open (mkt order)",        {"enter_at_open": True}),

        # === Combos ===
        ("Offset 0.15ATR + gap cap 50%R",          {"offset_atr": 0.15, "max_gap_r": 0.50}),
        ("Offset 0.2ATR + adjust risk",            {"offset_atr": 0.20, "adjust_risk": True}),
        ("Offset $0.15 + no 9:xx",                 {"offset": 0.15, "no_hours": [9]}),
        ("Offset $0.15 + bounce>0.1ATR",           {"offset": 0.15, "min_bounce_atr": 0.10}),
        ("Offset 0.15ATR + bounce>0.1ATR",         {"offset_atr": 0.15, "min_bounce_atr": 0.10}),
        ("Offset 0.2ATR + bounce>0.1ATR + no9",    {"offset_atr": 0.20, "min_bounce_atr": 0.10, "no_hours": [9]}),
        ("Gap cap 30%R + adjust risk",             {"max_gap_r": 0.30, "adjust_risk": True}),
        ("Gap cap 50%R + bounce>0.1ATR",           {"max_gap_r": 0.50, "min_bounce_atr": 0.10}),
        ("Offset $0.20 + gap cap 50%R + adj risk", {"offset": 0.20, "max_gap_r": 0.50, "adjust_risk": True}),
        ("Full: 0.15ATR+bounce0.1+gapcap50%+adj",  {"offset_atr": 0.15, "min_bounce_atr": 0.10, "max_gap_r": 0.50, "adjust_risk": True}),
    ]

    results = []
    for label, cfg in experiments:
        r = run_backtest(df, config=cfg)
        r["label"] = label
        results.append(r)
        print(f"  {label:<46} PF={r['pf']:.3f}  ret={r['ret']:+6.2f}%  trades={r['trades']:>5}  L={r['lpf']:.2f} S={r['spf']:.2f}")

    # ═══ Sorted by PF ═══
    print(f"\n{'='*110}")
    print(f"  TOP 15 BY PROFIT FACTOR")
    print(f"{'='*110}")
    print(f"\n  {'#':>2} {'Config':<48} {'PF':>6} {'Ret%':>8} {'Trades':>7} {'T/d':>5} {'WR%':>6}"
          f" {'L.PF':>6} {'S.PF':>6} {'5R+':>5}")
    print(f"  {'-'*105}")

    by_pf = sorted(results, key=lambda x: x["pf"], reverse=True)
    for i, r in enumerate(by_pf[:15]):
        marker = " ★" if r["pf"] > results[0]["pf"] else ""
        print(f"  {i+1:>2} {r['label']:<48} {r['pf']:>6.3f} {r['ret']:>+7.2f}%"
              f" {r['trades']:>6} {r['tpd']:>4.1f} {r['wr']:>5.1f}%"
              f" {r['lpf']:>5.2f} {r['spf']:>5.2f} {r['big5']:>4}{marker}")

    # ═══ Sorted by return ═══
    print(f"\n{'='*110}")
    print(f"  TOP 15 BY RETURN")
    print(f"{'='*110}")
    print(f"\n  {'#':>2} {'Config':<48} {'PF':>6} {'Ret%':>8} {'Trades':>7} {'T/d':>5}"
          f" {'L.PF':>6} {'S.PF':>6} {'5R+':>5}")
    print(f"  {'-'*100}")

    by_ret = sorted(results, key=lambda x: x["ret"], reverse=True)
    for i, r in enumerate(by_ret[:15]):
        marker = " ★" if r["ret"] > results[0]["ret"] else ""
        print(f"  {i+1:>2} {r['label']:<48} {r['pf']:>6.3f} {r['ret']:>+7.2f}%"
              f" {r['trades']:>6} {r['tpd']:>4.1f}"
              f" {r['lpf']:>5.2f} {r['spf']:>5.2f} {r['big5']:>4}{marker}")

    # ═══ Approach comparison ═══
    print(f"\n{'='*110}")
    print(f"  APPROACH COMPARISON SUMMARY")
    print(f"{'='*110}")

    baseline = results[0]
    approaches = {
        "Offset scaling": [r for r in results if "Offset" in r["label"] and "+" not in r["label"] and "BASELINE" not in r["label"]],
        "Delay filter": [r for r in results if "Skip" in r["label"]],
        "Gap cap": [r for r in results if "Gap cap" in r["label"] and "+" not in r["label"]],
        "Risk adjust": [r for r in results if r["label"] == "Adjust risk to actual entry"],
        "Bounce filter": [r for r in results if "Bounce" in r["label"] and "+" not in r["label"]],
        "Time filter": [r for r in results if ("No 9" in r["label"] or "Only" in r["label"]) and "+" not in r["label"]],
        "Market order": [r for r in results if "mkt" in r["label"]],
        "Combos": [r for r in results if "+" in r["label"] or "Full" in r["label"]],
    }

    print(f"\n  Baseline: PF={baseline['pf']:.3f}  ret={baseline['ret']:+.2f}%  trades={baseline['trades']}\n")

    for approach, rlist in approaches.items():
        if not rlist: continue
        best = max(rlist, key=lambda x: x["pf"])
        print(f"  {approach:<20} best PF={best['pf']:.3f} ({best['pf']-baseline['pf']:+.3f})"
              f"  ret={best['ret']:+.2f}%  trades={best['trades']}"
              f"  [{best['label']}]")

    # ═══ Final verdict ═══
    print(f"\n{'='*110}")
    print(f"  VERDICT")
    print(f"{'='*110}")

    overall_best = max(results[1:], key=lambda x: x["pf"])
    print(f"\n  Baseline (current, gap-fixed):  PF={baseline['pf']:.3f}  ret={baseline['ret']:+.2f}%")
    print(f"  Best single approach:           PF={overall_best['pf']:.3f}  ret={overall_best['ret']:+.2f}%")
    print(f"  Config: {overall_best['label']}")
    print(f"  Improvement: PF {(overall_best['pf']-baseline['pf'])/baseline['pf']*100:+.1f}%"
          f"  ret {overall_best['ret']-baseline['ret']:+.2f}pp")
    print(f"  Trades: {overall_best['trades']} ({overall_best['tpd']}/day) vs {baseline['trades']} ({baseline['tpd']}/day)")
    print(f"  Long PF: {overall_best['lpf']:.3f}  Short PF: {overall_best['spf']:.3f}")


if __name__ == "__main__":
    main()
