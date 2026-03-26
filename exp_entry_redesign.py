"""
Experiment: Entry Redesign — test EMA20 touch → break high entry.

Variants:
  A. Current:     touch → bounce → sig = touch_high + $0.05
  B. No offset:   touch → bounce → sig = touch_high (remove $0.05)
  C. No bounce:   touch → sig = touch_high (remove bounce + offset)
  D. No bounce+:  touch → sig = touch_high + $0.02 (tiny offset for spread)

All use gap-fixed entry: actual_entry = max(sig, open[trigger_bar]) for long.
This way we compare the REAL performance, not the inflated backtest number.
"""
from __future__ import annotations
import functools, datetime as dt
import numpy as np, pandas as pd
from entry_signal import add_indicators, detect_trend, check_touch

print = functools.partial(print, flush=True)
DATA_PATH = "data/QQQ_1Min_Polygon_2y_clean.csv"

BASE_PARAMS = {
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


def run_backtest(df, capital=100_000, mode="current", gap_fix=True):
    """
    Modes:
      current       — touch → bounce → sig = touch_high + $0.05
      no_offset     — touch → bounce → sig = touch_high
      no_bounce     — touch → sig = touch_high (skip bounce)
      no_bounce_02  — touch → sig = touch_high + $0.02
      no_bounce_neg — touch → sig = touch_high - $0.02 (limit-like, inside bar)
    """
    p = BASE_PARAMS.copy()
    df = add_indicators(df, p)
    high = df["High"].values
    low = df["Low"].values
    close = df["Close"].values
    open_p = df["Open"].values
    ema = df["ema20"].values
    ema_s = df["ema50"].values
    atr_v = df["atr"].values
    times = df.index.time
    dates = df.index.date
    n = len(df)

    comm = p["commission_per_share"]
    max_fwd = p["max_hold_bars"]
    valid_bars = 3 if "bounce" not in mode or mode == "current" or mode == "no_offset" else 5

    equity = capital
    trade_log = []
    bar = max(p["ema_slow"], p["atr_period"]) + 5
    daily_r_loss = 0.0
    current_date = None

    # Stats
    triggers = 0
    gap_total = 0.0
    gap_adverse_count = 0

    while bar < n - max_fwd - 5:
        a = atr_v[bar]
        if np.isnan(a) or a <= 0 or np.isnan(ema[bar]) or np.isnan(ema_s[bar]):
            bar += 1; continue
        if times[bar] >= p["no_entry_after"]:
            bar += 1; continue

        d_date = dates[bar]
        if current_date != d_date:
            current_date = d_date; daily_r_loss = 0.0
        if daily_r_loss >= p["daily_loss_r"]:
            bar += 1; continue

        trend = detect_trend(close[bar], ema[bar], ema_s[bar])
        if trend == 0: bar += 1; continue

        if not check_touch(trend, low[bar], high[bar], ema[bar], a,
                           p["touch_tol"], p["touch_below_max"]):
            bar += 1; continue

        # ─── Mode-specific: bounce check + signal line ───
        if mode == "current":
            bb = bar + 1
            if bb >= n: bar += 1; continue
            if trend == 1 and close[bb] <= high[bar]: bar += 1; continue
            if trend == -1 and close[bb] >= low[bar]: bar += 1; continue
            if trend == 1:
                sig = high[bar] + 0.05
                stop = low[bar] - p["stop_buffer"] * a
            else:
                sig = low[bar] - 0.05
                stop = high[bar] + p["stop_buffer"] * a
            scan_start = bb + 1
            scan_valid = 3

        elif mode == "no_offset":
            bb = bar + 1
            if bb >= n: bar += 1; continue
            if trend == 1 and close[bb] <= high[bar]: bar += 1; continue
            if trend == -1 and close[bb] >= low[bar]: bar += 1; continue
            if trend == 1:
                sig = high[bar]  # no offset
                stop = low[bar] - p["stop_buffer"] * a
            else:
                sig = low[bar]
                stop = high[bar] + p["stop_buffer"] * a
            scan_start = bb + 1
            scan_valid = 3

        elif mode == "no_bounce":
            # Skip bounce, signal = touch high
            if trend == 1:
                sig = high[bar]
                stop = low[bar] - p["stop_buffer"] * a
            else:
                sig = low[bar]
                stop = high[bar] + p["stop_buffer"] * a
            scan_start = bar + 1
            scan_valid = 5

        elif mode == "no_bounce_02":
            if trend == 1:
                sig = high[bar] + 0.02
                stop = low[bar] - p["stop_buffer"] * a
            else:
                sig = low[bar] - 0.02
                stop = high[bar] + p["stop_buffer"] * a
            scan_start = bar + 1
            scan_valid = 5

        elif mode == "no_bounce_neg":
            # Signal INSIDE the touch bar (like buying on pullback)
            if trend == 1:
                sig = high[bar] - 0.02
                stop = low[bar] - p["stop_buffer"] * a
            else:
                sig = low[bar] + 0.02
                stop = high[bar] + p["stop_buffer"] * a
            scan_start = bar + 1
            scan_valid = 5

        else:
            raise ValueError(mode)

        risk = abs(sig - stop)
        if risk <= 0: bar += 1; continue

        # ─── Signal trigger ───
        trigger_bar = -1
        for j in range(scan_valid):
            cb = scan_start + j
            if cb >= n: break
            if times[cb] >= p["force_close_at"]: break
            if trend == 1 and high[cb] >= sig:
                trigger_bar = cb; break
            if trend == -1 and low[cb] <= sig:
                trigger_bar = cb; break

        if trigger_bar < 0:
            bar += 1; continue

        triggers += 1

        # ─── Gap-fixed entry ───
        if gap_fix:
            if trend == 1:
                actual_entry = max(sig, open_p[trigger_bar])
            else:
                actual_entry = min(sig, open_p[trigger_bar])
        else:
            actual_entry = sig

        gap = abs(actual_entry - sig)
        if gap > 0.005:
            gap_adverse_count += 1
        gap_total += gap

        actual_risk = abs(actual_entry - stop)
        if actual_risk <= 0: bar = trigger_bar + 1; continue

        # ─── Entry bar stop check ───
        if trend == 1 and low[trigger_bar] <= stop:
            shares_eb = max(1, int(equity * p["risk_pct"] / actual_risk))
            if shares_eb * actual_entry > equity * p["max_pos_pct"]:
                shares_eb = max(1, int(equity * p["max_pos_pct"] / actual_entry))
            loss = shares_eb * (stop - actual_entry) - shares_eb * comm * 2
            equity += loss
            if shares_eb * actual_risk > 0:
                daily_r_loss += abs(loss) / (shares_eb * actual_risk)
            trade_log.append({"pnl": loss, "dir": trend, "exit": "entry_stop",
                              "lock": False, "shares": shares_eb, "risk": actual_risk})
            bar = trigger_bar + 1; continue

        if trend == -1 and high[trigger_bar] >= stop:
            shares_eb = max(1, int(equity * p["risk_pct"] / actual_risk))
            if shares_eb * actual_entry > equity * p["max_pos_pct"]:
                shares_eb = max(1, int(equity * p["max_pos_pct"] / actual_entry))
            loss = shares_eb * (stop - actual_entry) * trend - shares_eb * comm * 2
            equity += loss
            if shares_eb * actual_risk > 0:
                daily_r_loss += abs(loss) / (shares_eb * actual_risk)
            trade_log.append({"pnl": loss, "dir": trend, "exit": "entry_stop",
                              "lock": False, "shares": shares_eb, "risk": actual_risk})
            bar = trigger_bar + 1; continue

        # ─── Position sizing ───
        shares = max(1, int(equity * p["risk_pct"] / actual_risk))
        if shares * actual_entry > equity * p["max_pos_pct"]:
            shares = max(1, int(equity * p["max_pos_pct"] / actual_entry))
        if equity < shares * actual_risk or shares < 1:
            bar += 1; continue

        # ─── Execute trade ───
        lock_shares = max(1, int(shares * p["lock1_pct"]))
        runner_stop = stop
        lock_done = False
        trade_pnl = -shares * comm
        remaining = shares
        end_bar = trigger_bar
        exit_reason = "timeout"

        for k in range(1, max_fwd + 1):
            bi = trigger_bar + k
            if bi >= n: break
            h = high[bi]; l = low[bi]
            cur_atr = atr_v[bi] if not np.isnan(atr_v[bi]) else a

            if times[bi] >= p["force_close_at"]:
                trade_pnl += remaining * (close[bi] - actual_entry) * trend - remaining * comm
                end_bar = bi; exit_reason = "session_close"; break

            stopped = (trend == 1 and l <= runner_stop) or (trend == -1 and h >= runner_stop)
            lock_hit = False
            if not lock_done:
                lock_hit = (trend == 1 and h >= actual_entry + p["lock1_rr"] * actual_risk) or \
                           (trend == -1 and l <= actual_entry - p["lock1_rr"] * actual_risk)

            if stopped and lock_hit and not lock_done:
                trade_pnl += remaining * (runner_stop - actual_entry) * trend - remaining * comm
                end_bar = bi; exit_reason = "stop"; break

            if stopped:
                trade_pnl += remaining * (runner_stop - actual_entry) * trend - remaining * comm
                end_bar = bi; exit_reason = "stop" if not lock_done else "trail_stop"; break

            if lock_hit and not lock_done:
                trade_pnl += lock_shares * p["lock1_rr"] * actual_risk - lock_shares * comm
                remaining -= lock_shares; lock_done = True
                if trend == 1: runner_stop = max(runner_stop, actual_entry)
                else: runner_stop = min(runner_stop, actual_entry)

            if lock_done and k >= p["chandelier_bars"]:
                sk = max(1, k - p["chandelier_bars"] + 1)
                if trend == 1:
                    hh = max(high[trigger_bar + kk] for kk in range(sk, k) if trigger_bar + kk < n)
                    runner_stop = max(runner_stop, hh - p["chandelier_mult"] * cur_atr)
                else:
                    ll = min(low[trigger_bar + kk] for kk in range(sk, k) if trigger_bar + kk < n)
                    runner_stop = min(runner_stop, ll + p["chandelier_mult"] * cur_atr)
        else:
            ep = close[min(trigger_bar + max_fwd, n - 1)]
            trade_pnl += remaining * (ep - actual_entry) * trend - remaining * comm
            end_bar = min(trigger_bar + max_fwd, n - 1)

        equity += trade_pnl
        if trade_pnl < 0 and shares * actual_risk > 0:
            daily_r_loss += abs(trade_pnl) / (shares * actual_risk)

        trade_log.append({"pnl": trade_pnl, "dir": trend, "exit": exit_reason,
                          "lock": lock_done, "shares": shares, "risk": actual_risk})
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

    # Long vs short
    longs = tdf[tdf["dir"] == 1] if total > 0 else pd.DataFrame()
    shorts = tdf[tdf["dir"] == -1] if total > 0 else pd.DataFrame()

    def sub_pf(sub):
        if len(sub) == 0: return 0, 0, 0
        w = sub.loc[sub["pnl"] > 0, "pnl"].sum()
        l = abs(sub.loc[sub["pnl"] <= 0, "pnl"].sum())
        return round(w / l, 3) if l > 0 else 0, len(sub), round((sub["pnl"] > 0).mean() * 100, 1)

    lpf, ln, lwr = sub_pf(longs)
    spf, sn, swr = sub_pf(shorts)

    # R distribution
    r_mults = []
    if total > 0:
        for _, t in tdf.iterrows():
            rm = t["pnl"] / (t["shares"] * t["risk"]) if t["shares"] * t["risk"] > 0 else 0
            r_mults.append(rm)
    r_arr = np.array(r_mults) if r_mults else np.array([0])
    big5 = (r_arr >= 5).sum()
    big5r = r_arr[r_arr >= 5].sum() if big5 > 0 else 0

    return {
        "mode": mode, "gap_fix": gap_fix,
        "pf": round(pf, 3), "ret": round(ret, 2),
        "trades": total, "tpd": round(total / max(days, 1), 1),
        "wr": round(wins / total * 100, 1) if total > 0 else 0,
        "wins": wins, "losses": losses,
        "avg_win": round(gw / wins, 2) if wins > 0 else 0,
        "avg_loss": round(gl / losses, 2) if losses > 0 else 0,
        "triggers": triggers,
        "gap_adverse_pct": round(gap_adverse_count / triggers * 100, 1) if triggers > 0 else 0,
        "gap_avg": round(gap_total / triggers, 4) if triggers > 0 else 0,
        "long_pf": lpf, "long_n": ln, "long_wr": lwr,
        "short_pf": spf, "short_n": sn, "short_wr": swr,
        "big5": big5, "big5r": round(big5r, 1),
    }


def main():
    df = pd.read_csv(DATA_PATH, index_col="timestamp", parse_dates=True)
    print(f"Data: {len(df):,} bars, {df.index.normalize().nunique()} days\n")

    modes = [
        ("current",       "A. Current (bounce + $0.05 offset)"),
        ("no_offset",     "B. Bounce + no offset (sig=high)"),
        ("no_bounce",     "C. No bounce, sig=touch_high"),
        ("no_bounce_02",  "D. No bounce, sig=touch_high+$0.02"),
        ("no_bounce_neg", "E. No bounce, sig=touch_high-$0.02"),
    ]

    # ═══ Part 1: All with gap fix (real performance) ═══
    print(f"{'='*100}")
    print(f"  ENTRY REDESIGN — all with gap-fixed entry (real performance)")
    print(f"{'='*100}")
    print(f"\n  {'Mode':<42} {'PF':>6} {'Ret%':>8} {'Trades':>7} {'T/d':>5} {'WR%':>6}"
          f" {'AvgW':>7} {'AvgL':>7} {'Gap%':>6} {'5R+':>5}")
    print(f"  {'-'*100}")

    results_fixed = []
    for mode, label in modes:
        print(f"  Running {label}...", end=" ")
        r = run_backtest(df, mode=mode, gap_fix=True)
        results_fixed.append((label, r))
        print(f"PF={r['pf']:.3f}")

    print()
    for label, r in results_fixed:
        print(f"  {label:<42} {r['pf']:>6.3f} {r['ret']:>+7.2f}%"
              f" {r['trades']:>6} {r['tpd']:>4.1f} {r['wr']:>5.1f}%"
              f" {r['avg_win']:>6.2f} {r['avg_loss']:>6.2f}"
              f" {r['gap_adverse_pct']:>5.1f}% {r['big5']:>4}")

    # ═══ Part 2: Compare gap-fix vs no-gap-fix for each mode ═══
    print(f"\n{'='*100}")
    print(f"  GAP IMPACT PER MODE (how much does gap-fix reduce PF?)")
    print(f"{'='*100}")
    print(f"\n  {'Mode':<42} {'PF (no fix)':>11} {'PF (fixed)':>11} {'Delta':>8}")
    print(f"  {'-'*75}")

    for mode, label in modes:
        r_nofix = run_backtest(df, mode=mode, gap_fix=False)
        r_fix = [r for l, r in results_fixed if l == label][0]
        delta_pct = (r_fix["pf"] - r_nofix["pf"]) / r_nofix["pf"] * 100 if r_nofix["pf"] > 0 else 0
        print(f"  {label:<42} {r_nofix['pf']:>10.3f} {r_fix['pf']:>10.3f} {delta_pct:>+7.1f}%")

    # ═══ Part 3: Long vs Short breakdown ═══
    print(f"\n{'='*100}")
    print(f"  LONG vs SHORT BREAKDOWN (gap-fixed)")
    print(f"{'='*100}")
    print(f"\n  {'Mode':<42} {'L.PF':>6} {'L.N':>6} {'L.WR':>6} {'S.PF':>6} {'S.N':>6} {'S.WR':>6}")
    print(f"  {'-'*80}")

    for label, r in results_fixed:
        print(f"  {label:<42} {r['long_pf']:>6.3f} {r['long_n']:>5} {r['long_wr']:>5.1f}%"
              f" {r['short_pf']:>6.3f} {r['short_n']:>5} {r['short_wr']:>5.1f}%")

    # ═══ Part 4: Scan window sensitivity for no_bounce ═══
    print(f"\n{'='*100}")
    print(f"  SCAN WINDOW SENSITIVITY (no_bounce mode, gap-fixed)")
    print(f"{'='*100}")
    print(f"\n  {'Scan bars':>10} {'PF':>6} {'Ret%':>8} {'Trades':>7} {'Gap%':>6} {'5R+':>5}")
    print(f"  {'-'*45}")

    # Temporarily modify scan window
    for scan in [2, 3, 5, 8, 10]:
        # We need to hack the valid_bars inside the function
        # Simplest: just run with different modes' scan_valid
        r = run_backtest_scan(df, scan_bars=scan, gap_fix=True)
        print(f"  {scan:>10} {r['pf']:>6.3f} {r['ret']:>+7.2f}% {r['trades']:>6}"
              f" {r['gap_adverse_pct']:>5.1f}% {r['big5']:>4}")

    # ═══ Verdict ═══
    print(f"\n{'='*100}")
    print(f"  VERDICT")
    print(f"{'='*100}")

    best = max(results_fixed, key=lambda x: x[1]["pf"])
    curr = results_fixed[0]
    print(f"\n  Current (gap-fixed):  PF={curr[1]['pf']:.3f}  ret={curr[1]['ret']:+.2f}%")
    print(f"  Best redesign:        PF={best[1]['pf']:.3f}  ret={best[1]['ret']:+.2f}%  ({best[0]})")
    print(f"  Improvement:          {(best[1]['pf'] - curr[1]['pf']) / curr[1]['pf'] * 100:+.1f}% PF")
    if best[1]["gap_adverse_pct"] < curr[1]["gap_adverse_pct"]:
        print(f"  Gap reduction:        {curr[1]['gap_adverse_pct']:.1f}% → {best[1]['gap_adverse_pct']:.1f}% adverse gaps")


def run_backtest_scan(df, scan_bars=5, capital=100_000, gap_fix=True):
    """No-bounce mode with configurable scan window."""
    p = BASE_PARAMS.copy()
    df2 = add_indicators(df, p)
    high = df2["High"].values; low = df2["Low"].values
    close = df2["Close"].values; open_p = df2["Open"].values
    ema = df2["ema20"].values; ema_s = df2["ema50"].values
    atr_v = df2["atr"].values
    times = df2.index.time; dates = df2.index.date; n = len(df2)
    comm = p["commission_per_share"]; max_fwd = p["max_hold_bars"]

    equity = capital; trade_log = []; bar = max(p["ema_slow"], p["atr_period"]) + 5
    daily_r_loss = 0.0; current_date = None
    triggers = 0; gap_adverse = 0; gap_total = 0.0

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

        if trend == 1: sig = high[bar]; stop = low[bar] - p["stop_buffer"] * a
        else: sig = low[bar]; stop = high[bar] + p["stop_buffer"] * a
        risk = abs(sig - stop)
        if risk <= 0: bar += 1; continue

        trigger_bar = -1
        for j in range(1, scan_bars + 1):
            cb = bar + j
            if cb >= n or times[cb] >= p["force_close_at"]: break
            if trend == 1 and high[cb] >= sig: trigger_bar = cb; break
            if trend == -1 and low[cb] <= sig: trigger_bar = cb; break
        if trigger_bar < 0: bar += 1; continue

        triggers += 1
        if gap_fix:
            actual_entry = max(sig, open_p[trigger_bar]) if trend == 1 else min(sig, open_p[trigger_bar])
        else:
            actual_entry = sig
        g = abs(actual_entry - sig)
        if g > 0.005: gap_adverse += 1
        gap_total += g
        actual_risk = abs(actual_entry - stop)
        if actual_risk <= 0: bar = trigger_bar + 1; continue

        # Entry bar stop
        if trend == 1 and low[trigger_bar] <= stop:
            sh = max(1, int(equity * p["risk_pct"] / actual_risk))
            loss = sh * (stop - actual_entry) - sh * comm * 2
            equity += loss
            if sh * actual_risk > 0: daily_r_loss += abs(loss) / (sh * actual_risk)
            trade_log.append({"pnl": loss, "shares": sh, "risk": actual_risk})
            bar = trigger_bar + 1; continue
        if trend == -1 and high[trigger_bar] >= stop:
            sh = max(1, int(equity * p["risk_pct"] / actual_risk))
            loss = sh * (stop - actual_entry) * trend - sh * comm * 2
            equity += loss
            if sh * actual_risk > 0: daily_r_loss += abs(loss) / (sh * actual_risk)
            trade_log.append({"pnl": loss, "shares": sh, "risk": actual_risk})
            bar = trigger_bar + 1; continue

        shares = max(1, int(equity * p["risk_pct"] / actual_risk))
        if shares * actual_entry > equity * p["max_pos_pct"]:
            shares = max(1, int(equity * p["max_pos_pct"] / actual_entry))
        if equity < shares * actual_risk: bar += 1; continue

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
        trade_log.append({"pnl": trade_pnl, "shares": shares, "risk": actual_risk})
        bar = end_bar + 1

    tdf = pd.DataFrame(trade_log) if trade_log else pd.DataFrame()
    total = len(tdf)
    wins = (tdf["pnl"] > 0).sum() if total > 0 else 0
    gl_v = abs(tdf.loc[tdf["pnl"] <= 0, "pnl"].sum()) if total - wins > 0 else 0
    gw_v = tdf.loc[tdf["pnl"] > 0, "pnl"].sum() if wins > 0 else 0
    pf = gw_v / gl_v if gl_v > 0 else 0
    days = df2.index.normalize().nunique()
    r_arr = np.array([t["pnl"]/(t["shares"]*t["risk"]) if t["shares"]*t["risk"]>0 else 0
                       for _, t in tdf.iterrows()]) if total > 0 else np.array([0])

    return {
        "pf": round(pf, 3), "ret": round((equity - capital) / capital * 100, 2),
        "trades": total, "gap_adverse_pct": round(gap_adverse / triggers * 100, 1) if triggers > 0 else 0,
        "big5": int((r_arr >= 5).sum()),
    }


if __name__ == "__main__":
    main()
