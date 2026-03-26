"""
Entry Mechanism Rethink — fundamentally different entry approaches.

Key insight: the current "touch → bounce → signal trigger" structure
has a structural gap problem. The bounce confirmation guarantees the
next bar opens above sig. We need to enter EARLIER or DIFFERENTLY.

New approaches:
  1. Touch bar close    — enter at close of touch bar itself (earliest possible)
  2. Bounce bar open    — enter at open of bounce bar (before confirmation)
  3. Bounce bar close   — enter at close of bounce bar (after confirmation, no trigger)
  4. Limit at EMA20     — place limit buy at EMA20 during trending market
  5. Touch + pattern    — touch bar must be bullish (close > open) or hammer
  6. Two-touch          — require 2 consecutive touches before entry
  7. Bounce close + pullback — enter if next bar dips below bounce close
  8. Asymmetric         — different entry for long vs short (since long loses)

All use realistic entry prices (no fake fills).
"""
from __future__ import annotations
import functools, datetime as dt
import numpy as np, pandas as pd
from entry_signal import add_indicators, detect_trend, check_touch

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


def execute_trade(trend, actual_entry, stop, actual_risk, shares, entry_bar,
                  high, low, close, open_p, atr_v, times, n, p):
    """Shared exit logic. Returns (trade_pnl, end_bar, exit_reason, lock_done)."""
    comm = p["commission_per_share"]
    max_fwd = p["max_hold_bars"]
    lock_sh = max(1, int(shares * p["lock1_pct"]))
    runner_stop = stop; lock_done = False
    trade_pnl = -shares * comm; remaining = shares; end_bar = entry_bar

    for k in range(1, max_fwd + 1):
        bi = entry_bar + k
        if bi >= n: break
        h = high[bi]; l = low[bi]
        ca = atr_v[bi] if not np.isnan(atr_v[bi]) else atr_v[entry_bar]

        if times[bi] >= p["force_close_at"]:
            trade_pnl += remaining * (close[bi] - actual_entry) * trend - remaining * comm
            end_bar = bi; return trade_pnl, end_bar, "session_close", lock_done

        stopped = (trend == 1 and l <= runner_stop) or (trend == -1 and h >= runner_stop)
        lh = False
        if not lock_done:
            lh = (trend == 1 and h >= actual_entry + p["lock1_rr"] * actual_risk) or \
                 (trend == -1 and l <= actual_entry - p["lock1_rr"] * actual_risk)
        if stopped and lh and not lock_done:
            trade_pnl += remaining * (runner_stop - actual_entry) * trend - remaining * comm
            end_bar = bi; return trade_pnl, end_bar, "stop", lock_done
        if stopped:
            trade_pnl += remaining * (runner_stop - actual_entry) * trend - remaining * comm
            end_bar = bi; return trade_pnl, end_bar, "stop" if not lock_done else "trail", lock_done
        if lh and not lock_done:
            trade_pnl += lock_sh * p["lock1_rr"] * actual_risk - lock_sh * comm
            remaining -= lock_sh; lock_done = True
            if trend == 1: runner_stop = max(runner_stop, actual_entry)
            else: runner_stop = min(runner_stop, actual_entry)
        if lock_done and k >= p["chandelier_bars"]:
            sk = max(1, k - p["chandelier_bars"] + 1)
            if trend == 1:
                hh = max(high[entry_bar + kk] for kk in range(sk, k) if entry_bar + kk < n)
                runner_stop = max(runner_stop, hh - p["chandelier_mult"] * ca)
            else:
                ll = min(low[entry_bar + kk] for kk in range(sk, k) if entry_bar + kk < n)
                runner_stop = min(runner_stop, ll + p["chandelier_mult"] * ca)

    else:
        ep = close[min(entry_bar + max_fwd, n - 1)]
        trade_pnl += remaining * (ep - actual_entry) * trend - remaining * comm
        end_bar = min(entry_bar + max_fwd, n - 1)

    return trade_pnl, end_bar, "timeout", lock_done


def run_mode(df, mode, capital=100_000):
    p = PARAMS.copy()
    df = add_indicators(df, p)
    high = df["High"].values; low = df["Low"].values
    close = df["Close"].values; open_p = df["Open"].values
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

        entry_bar = -1
        actual_entry = 0
        stop = 0

        # ════════════════════════════════════════════════════
        # MODE DISPATCH
        # ════════════════════════════════════════════════════

        if mode == "baseline":
            # Current strategy with gap fix
            bb = bar + 1
            if bb >= n: bar += 1; continue
            if trend == 1 and close[bb] <= high[bar]: bar += 1; continue
            if trend == -1 and close[bb] >= low[bar]: bar += 1; continue
            sig = high[bar] + 0.05 if trend == 1 else low[bar] - 0.05
            stop = low[bar] - p["stop_buffer"] * a if trend == 1 else high[bar] + p["stop_buffer"] * a
            # Trigger
            for j in range(1, 4):
                cb = bb + j
                if cb >= n: break
                if trend == 1 and high[cb] >= sig: entry_bar = cb; break
                if trend == -1 and low[cb] <= sig: entry_bar = cb; break
            if entry_bar < 0: bar += 1; continue
            actual_entry = max(sig, open_p[entry_bar]) if trend == 1 else min(sig, open_p[entry_bar])

        elif mode == "touch_close":
            # Enter at close of touch bar — earliest possible, risk: no confirmation
            if trend == 1:
                if close[bar] <= open_p[bar]: bar += 1; continue  # must be bullish candle
                actual_entry = close[bar]
                stop = low[bar] - p["stop_buffer"] * a
            else:
                if close[bar] >= open_p[bar]: bar += 1; continue  # must be bearish
                actual_entry = close[bar]
                stop = high[bar] + p["stop_buffer"] * a
            entry_bar = bar

        elif mode == "touch_close_nofilter":
            # Enter at close of touch bar, no candle direction filter
            actual_entry = close[bar]
            stop = low[bar] - p["stop_buffer"] * a if trend == 1 else high[bar] + p["stop_buffer"] * a
            entry_bar = bar

        elif mode == "bounce_open":
            # Enter at open of bounce bar (bar after touch)
            bb = bar + 1
            if bb >= n: bar += 1; continue
            if times[bb] >= p["force_close_at"]: bar += 1; continue
            actual_entry = open_p[bb]
            stop = low[bar] - p["stop_buffer"] * a if trend == 1 else high[bar] + p["stop_buffer"] * a
            # Sanity: don't enter if open is past stop
            if trend == 1 and actual_entry <= stop: bar += 1; continue
            if trend == -1 and actual_entry >= stop: bar += 1; continue
            entry_bar = bb

        elif mode == "bounce_close":
            # Wait for bounce confirmation, enter at close of bounce bar
            bb = bar + 1
            if bb >= n: bar += 1; continue
            if trend == 1 and close[bb] <= high[bar]: bar += 1; continue
            if trend == -1 and close[bb] >= low[bar]: bar += 1; continue
            actual_entry = close[bb]
            stop = low[bar] - p["stop_buffer"] * a if trend == 1 else high[bar] + p["stop_buffer"] * a
            entry_bar = bb

        elif mode == "bounce_close_market":
            # Same as bounce_close but enter at OPEN of bar after bounce (next bar market order)
            bb = bar + 1
            if bb >= n: bar += 1; continue
            if trend == 1 and close[bb] <= high[bar]: bar += 1; continue
            if trend == -1 and close[bb] >= low[bar]: bar += 1; continue
            nb = bb + 1
            if nb >= n: bar += 1; continue
            if times[nb] >= p["force_close_at"]: bar += 1; continue
            actual_entry = open_p[nb]
            stop = low[bar] - p["stop_buffer"] * a if trend == 1 else high[bar] + p["stop_buffer"] * a
            if trend == 1 and actual_entry <= stop: bar += 1; continue
            if trend == -1 and actual_entry >= stop: bar += 1; continue
            entry_bar = nb

        elif mode == "limit_ema":
            # Place limit order at EMA20. Enter if next bars' Low <= EMA20 (long).
            target = ema[bar]
            stop = low[bar] - p["stop_buffer"] * a if trend == 1 else high[bar] + p["stop_buffer"] * a
            for j in range(1, 6):
                cb = bar + j
                if cb >= n or times[cb] >= p["force_close_at"]: break
                if trend == 1 and low[cb] <= target:
                    actual_entry = target  # limit fill at EMA
                    entry_bar = cb; break
                if trend == -1 and high[cb] >= target:
                    actual_entry = target
                    entry_bar = cb; break
            if entry_bar < 0: bar += 1; continue

        elif mode == "limit_ema_confirm":
            # Limit at EMA20 but only after touch confirms (touch bar confirms EMA area)
            # Next bars: place limit at EMA20, wait for re-touch
            target = ema[bar]
            stop = low[bar] - p["stop_buffer"] * a if trend == 1 else high[bar] + p["stop_buffer"] * a
            # Must see bounce first, then re-touch
            bb = bar + 1
            if bb >= n: bar += 1; continue
            if trend == 1 and close[bb] <= high[bar]: bar += 1; continue
            if trend == -1 and close[bb] >= low[bar]: bar += 1; continue
            # Now look for re-touch of EMA in next bars
            for j in range(2, 8):
                cb = bar + j
                if cb >= n or times[cb] >= p["force_close_at"]: break
                if trend == 1 and low[cb] <= target:
                    actual_entry = target
                    entry_bar = cb; break
                if trend == -1 and high[cb] >= target:
                    actual_entry = target
                    entry_bar = cb; break
            if entry_bar < 0: bar += 1; continue

        elif mode == "hammer":
            # Touch bar must be hammer pattern: long lower wick, small body at top
            body = abs(close[bar] - open_p[bar])
            if trend == 1:
                lower_wick = min(close[bar], open_p[bar]) - low[bar]
                upper_wick = high[bar] - max(close[bar], open_p[bar])
                if lower_wick < body * 1.5: bar += 1; continue  # need long lower wick
                if upper_wick > body * 0.5: bar += 1; continue  # small upper wick
                actual_entry = close[bar]
                stop = low[bar] - p["stop_buffer"] * a
            else:
                upper_wick = high[bar] - max(close[bar], open_p[bar])
                lower_wick = min(close[bar], open_p[bar]) - low[bar]
                if upper_wick < body * 1.5: bar += 1; continue
                if lower_wick > body * 0.5: bar += 1; continue
                actual_entry = close[bar]
                stop = high[bar] + p["stop_buffer"] * a
            entry_bar = bar

        elif mode == "two_touch":
            # Require 2 touches in last 5 bars before entry
            touch_count = 0
            for look in range(max(0, bar - 4), bar + 1):
                if np.isnan(atr_v[look]) or atr_v[look] <= 0: continue
                if check_touch(trend, low[look], high[look], ema[look], atr_v[look],
                               p["touch_tol"], p["touch_below_max"]):
                    touch_count += 1
            if touch_count < 2: bar += 1; continue
            # Bounce confirmation
            bb = bar + 1
            if bb >= n: bar += 1; continue
            if trend == 1 and close[bb] <= high[bar]: bar += 1; continue
            if trend == -1 and close[bb] >= low[bar]: bar += 1; continue
            actual_entry = close[bb]
            stop = low[bar] - p["stop_buffer"] * a if trend == 1 else high[bar] + p["stop_buffer"] * a
            entry_bar = bb

        elif mode == "pullback_after_bounce":
            # Bounce confirmed, then wait for pullback towards entry
            bb = bar + 1
            if bb >= n: bar += 1; continue
            if trend == 1 and close[bb] <= high[bar]: bar += 1; continue
            if trend == -1 and close[bb] >= low[bar]: bar += 1; continue
            # Wait for next bar to pull back to touch_high level
            target = high[bar] if trend == 1 else low[bar]
            for j in range(2, 8):
                cb = bar + j
                if cb >= n or times[cb] >= p["force_close_at"]: break
                if trend == 1 and low[cb] <= target:
                    actual_entry = target  # limit fill
                    entry_bar = cb; break
                if trend == -1 and high[cb] >= target:
                    actual_entry = target
                    entry_bar = cb; break
            if entry_bar < 0: bar += 1; continue
            stop = low[bar] - p["stop_buffer"] * a if trend == 1 else high[bar] + p["stop_buffer"] * a

        elif mode == "short_only":
            # Only trade short side (where edge exists)
            if trend != -1: bar += 1; continue
            bb = bar + 1
            if bb >= n: bar += 1; continue
            if close[bb] >= low[bar]: bar += 1; continue
            sig = low[bar] - 0.05
            stop = high[bar] + p["stop_buffer"] * a
            for j in range(1, 4):
                cb = bb + j
                if cb >= n: break
                if low[cb] <= sig: entry_bar = cb; break
            if entry_bar < 0: bar += 1; continue
            actual_entry = min(sig, open_p[entry_bar])

        else:
            raise ValueError(mode)

        if entry_bar < 0: bar += 1; continue

        actual_risk = abs(actual_entry - stop)
        if actual_risk <= 0: bar = entry_bar + 1; continue

        # Entry bar stop check
        if trend == 1 and low[entry_bar] <= stop:
            sh = max(1, int(equity * p["risk_pct"] / actual_risk))
            if sh * actual_entry > equity * p["max_pos_pct"]:
                sh = max(1, int(equity * p["max_pos_pct"] / actual_entry))
            loss = sh * (stop - actual_entry) - sh * comm * 2
            equity += loss
            if sh * actual_risk > 0: daily_r_loss += abs(loss) / (sh * actual_risk)
            trade_log.append({"pnl": loss, "dir": trend, "shares": sh, "risk": actual_risk, "lock": False})
            bar = entry_bar + 1; continue
        if trend == -1 and high[entry_bar] >= stop:
            sh = max(1, int(equity * p["risk_pct"] / actual_risk))
            if sh * actual_entry > equity * p["max_pos_pct"]:
                sh = max(1, int(equity * p["max_pos_pct"] / actual_entry))
            loss = sh * (stop - actual_entry) * trend - sh * comm * 2
            equity += loss
            if sh * actual_risk > 0: daily_r_loss += abs(loss) / (sh * actual_risk)
            trade_log.append({"pnl": loss, "dir": trend, "shares": sh, "risk": actual_risk, "lock": False})
            bar = entry_bar + 1; continue

        # Size
        shares = max(1, int(equity * p["risk_pct"] / actual_risk))
        if shares * actual_entry > equity * p["max_pos_pct"]:
            shares = max(1, int(equity * p["max_pos_pct"] / actual_entry))
        if equity < shares * actual_risk or shares < 1: bar += 1; continue

        # Execute
        trade_pnl, end_bar, exit_reason, lock_done = execute_trade(
            trend, actual_entry, stop, actual_risk, shares, entry_bar,
            high, low, close, open_p, atr_v, times, n, p)

        equity += trade_pnl
        if trade_pnl < 0 and shares * actual_risk > 0:
            daily_r_loss += abs(trade_pnl) / (shares * actual_risk)
        trade_log.append({"pnl": trade_pnl, "dir": trend, "shares": shares,
                          "risk": actual_risk, "lock": lock_done})
        bar = end_bar + 1

    # Stats
    tdf = pd.DataFrame(trade_log) if trade_log else pd.DataFrame()
    total = len(tdf)
    if total == 0:
        return {"mode": mode, "pf": 0, "ret": 0, "trades": 0, "tpd": 0, "wr": 0,
                "lpf": 0, "spf": 0, "ln": 0, "sn": 0, "big5": 0,
                "avg_win": 0, "avg_loss": 0, "lock_rate": 0}
    wins = (tdf["pnl"] > 0).sum(); losses = total - wins
    gw = tdf.loc[tdf["pnl"] > 0, "pnl"].sum() if wins > 0 else 0
    gl = abs(tdf.loc[tdf["pnl"] <= 0, "pnl"].sum()) if losses > 0 else 0
    pf = gw / gl if gl > 0 else 0
    ret = (equity - capital) / capital * 100
    days = df.index.normalize().nunique()
    longs = tdf[tdf["dir"] == 1]; shorts = tdf[tdf["dir"] == -1]
    def spf(sub):
        if len(sub) == 0: return 0
        w = sub.loc[sub["pnl"] > 0, "pnl"].sum()
        l = abs(sub.loc[sub["pnl"] <= 0, "pnl"].sum())
        return round(w / l, 3) if l > 0 else 0
    r_arr = np.array([t["pnl"]/(t["shares"]*t["risk"]) if t["shares"]*t["risk"]>0 else 0
                       for _, t in tdf.iterrows()])
    lock_rate = tdf["lock"].mean() * 100 if "lock" in tdf.columns else 0

    return {
        "mode": mode, "pf": round(pf, 3), "ret": round(ret, 2),
        "trades": total, "tpd": round(total / max(days, 1), 1),
        "wr": round(wins / total * 100, 1),
        "lpf": spf(longs), "spf": spf(shorts),
        "ln": len(longs), "sn": len(shorts),
        "big5": int((r_arr >= 5).sum()),
        "avg_win": round(gw / wins, 2) if wins > 0 else 0,
        "avg_loss": round(gl / losses, 2) if losses > 0 else 0,
        "lock_rate": round(lock_rate, 1),
    }


def main():
    df = pd.read_csv(DATA_PATH, index_col="timestamp", parse_dates=True)
    print(f"Data: {len(df):,} bars, {df.index.normalize().nunique()} days\n")

    modes = [
        ("baseline",              "A. Baseline (current + gap fix)"),
        ("touch_close",           "B. Touch bar close (bullish filter)"),
        ("touch_close_nofilter",  "C. Touch bar close (no filter)"),
        ("bounce_open",           "D. Bounce bar open (before confirm)"),
        ("bounce_close",          "E. Bounce bar close (after confirm)"),
        ("bounce_close_market",   "F. Market order bar after bounce"),
        ("limit_ema",             "G. Limit at EMA20"),
        ("limit_ema_confirm",     "H. Bounce + limit re-touch EMA20"),
        ("hammer",                "I. Hammer pattern at touch"),
        ("two_touch",             "J. Two touches + bounce close"),
        ("pullback_after_bounce", "K. Bounce + pullback to touch_high"),
        ("short_only",            "L. Short only (gap-fixed)"),
    ]

    results = []
    for mode, label in modes:
        print(f"  Running {label}...")
        r = run_mode(df, mode)
        r["label"] = label
        results.append(r)

    # ═══ Main table ═══
    print(f"\n{'='*120}")
    print(f"  ENTRY MECHANISM RETHINK — Fundamental Alternatives")
    print(f"{'='*120}")
    print(f"\n  {'Mode':<44} {'PF':>6} {'Ret%':>8} {'Trades':>7} {'T/d':>5} {'WR%':>6}"
          f" {'AvgW':>7} {'AvgL':>7} {'Lock%':>6} {'L.PF':>6} {'S.PF':>6} {'5R+':>5}")
    print(f"  {'-'*120}")

    for r in results:
        marker = " ★" if r["pf"] > results[0]["pf"] else ""
        print(f"  {r['label']:<44} {r['pf']:>6.3f} {r['ret']:>+7.2f}%"
              f" {r['trades']:>6} {r['tpd']:>4.1f} {r['wr']:>5.1f}%"
              f" {r['avg_win']:>6.2f} {r['avg_loss']:>6.2f}"
              f" {r['lock_rate']:>5.1f}%"
              f" {r['lpf']:>5.2f} {r['spf']:>5.2f} {r['big5']:>4}{marker}")

    # ═══ Sorted by PF ═══
    print(f"\n{'='*120}")
    print(f"  RANKED BY PROFIT FACTOR")
    print(f"{'='*120}")
    by_pf = sorted(results, key=lambda x: x["pf"], reverse=True)
    for i, r in enumerate(by_pf):
        delta = r["pf"] - results[0]["pf"]
        print(f"  {i+1:>2}. {r['label']:<44} PF={r['pf']:.3f} ({delta:+.3f})"
              f"  ret={r['ret']:+.2f}%  trades={r['trades']}"
              f"  L={r['lpf']:.2f} S={r['spf']:.2f}")

    # ═══ Analysis ═══
    print(f"\n{'='*120}")
    print(f"  ANALYSIS")
    print(f"{'='*120}")

    bl = results[0]
    print(f"\n  Baseline:         PF={bl['pf']:.3f}  Long={bl['lpf']:.3f}  Short={bl['spf']:.3f}")

    # Best overall
    best = max(results, key=lambda x: x["pf"])
    print(f"  Best overall:     PF={best['pf']:.3f}  [{best['label']}]")

    # Best long
    best_long = max(results, key=lambda x: x["lpf"])
    print(f"  Best long PF:     {best_long['lpf']:.3f}  [{best_long['label']}]")

    # Best short
    best_short = max(results, key=lambda x: x["spf"])
    print(f"  Best short PF:    {best_short['spf']:.3f}  [{best_short['label']}]")

    # Entry timing comparison
    print(f"\n  Entry timing spectrum (earliest → latest):")
    timing_order = ["touch_close", "touch_close_nofilter", "bounce_open",
                    "bounce_close", "bounce_close_market", "baseline"]
    for mode in timing_order:
        r = [x for x in results if x["mode"] == mode][0]
        print(f"    {r['label']:<44} PF={r['pf']:.3f}  WR={r['wr']:.1f}%  5R+={r['big5']}")


if __name__ == "__main__":
    main()
