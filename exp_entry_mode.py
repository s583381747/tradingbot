"""
Experiment: Entry Order Type Comparison

Compares three entry modes on the same strategy:
  A. stop-buy (current) — fill at sig regardless of open
  B. stop-buy-fixed    — fill at max(sig, open) for long / min(sig, open) for short
  C. retest-limit      — after trigger bar, wait for price to come back to sig
                          within N bars (limit order retest)

Uses strategy_final.py logic with only the entry mechanism changed.
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
    "signal_offset": 0.05, "stop_buffer": 0.3,
    "lock1_rr": 0.3, "lock1_pct": 0.20,
    "chandelier_bars": 40, "chandelier_mult": 0.5,
    "signal_valid_bars": 3, "max_hold_bars": 180,
    "risk_pct": 0.01, "max_pos_pct": 0.25,
    "no_entry_after": dt.time(15, 30), "force_close_at": dt.time(15, 58),
    "commission_per_share": 0.005, "daily_loss_r": 2.5,
    "retest_wait_bars": 5,  # how many bars to wait for retest
}


def run_backtest(df, capital=100_000, mode="stop_buy"):
    """
    Modes:
      stop_buy       — current: fill at sig
      stop_buy_fixed — fill at max(sig, open) for long
      retest_limit   — trigger bar confirms breakout, then wait for retest fill at sig
    """
    p = PARAMS.copy()
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

    equity = capital
    trade_log = []
    bar = max(p["ema_slow"], p["atr_period"]) + 5
    daily_r_loss = 0.0
    current_date = None

    # Stats
    triggers = 0
    fills = 0
    missed = 0
    gap_cost_total = 0.0

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

        bb = bar + 1
        if bb >= n: bar += 1; continue
        if not check_bounce(trend, close[bb], high[bar], low[bar]):
            bar += 1; continue

        sl = calc_signal_line(trend, high[bar], low[bar], a,
                              p["signal_offset"], p["stop_buffer"])
        if sl is None: bar += 1; continue
        sig, stop, risk = sl

        # ─── Signal trigger (same for all modes) ───
        trigger_bar = -1
        for j in range(1, p["signal_valid_bars"] + 1):
            cb = bb + j
            if cb >= n: break
            if trend == 1 and high[cb] >= sig:
                trigger_bar = cb; break
            if trend == -1 and low[cb] <= sig:
                trigger_bar = cb; break

        if trigger_bar < 0:
            bar += 1; continue

        triggers += 1

        # ─── Determine actual entry bar and price based on mode ───
        if mode == "stop_buy":
            entry_bar = trigger_bar
            actual_entry = sig

        elif mode == "stop_buy_fixed":
            entry_bar = trigger_bar
            if trend == 1:
                actual_entry = max(sig, open_p[trigger_bar])
            else:
                actual_entry = min(sig, open_p[trigger_bar])
            gap_cost_total += abs(actual_entry - sig)

        elif mode == "retest_limit":
            # Trigger bar confirmed breakout. Now wait for price to come BACK to sig.
            retest_bar = -1
            wait = p["retest_wait_bars"]
            for rb in range(trigger_bar + 1, min(trigger_bar + wait + 1, n)):
                if times[rb] >= p["force_close_at"]:
                    break
                # For long: price must dip back to sig (Low <= sig)
                # For short: price must rise back to sig (High >= sig)
                if trend == 1 and low[rb] <= sig:
                    retest_bar = rb; break
                if trend == -1 and high[rb] >= sig:
                    retest_bar = rb; break

            if retest_bar < 0:
                missed += 1
                bar = trigger_bar + 1; continue

            entry_bar = retest_bar
            actual_entry = sig  # limit order fills at exact price
        else:
            raise ValueError(f"Unknown mode: {mode}")

        fills += 1

        # Recalculate risk with actual entry
        actual_risk = abs(actual_entry - stop)
        if actual_risk <= 0:
            bar = entry_bar + 1; continue

        # ─── Bug #1: entry bar stop check ───
        if trend == 1 and low[entry_bar] <= stop:
            shares_eb = max(1, int(equity * p["risk_pct"] / actual_risk))
            if shares_eb * actual_entry > equity * p["max_pos_pct"]:
                shares_eb = max(1, int(equity * p["max_pos_pct"] / actual_entry))
            loss = shares_eb * (stop - actual_entry) - shares_eb * comm * 2
            equity += loss
            r_lost = abs(loss / (shares_eb * actual_risk)) if shares_eb * actual_risk > 0 else 0
            daily_r_loss += r_lost
            trade_log.append({"pnl": loss, "direction": "LONG", "exit_reason": "entry_bar_stop",
                              "entry_price": actual_entry, "lock_hit": False,
                              "shares": shares_eb, "risk": actual_risk})
            bar = entry_bar + 1; continue

        if trend == -1 and high[entry_bar] >= stop:
            shares_eb = max(1, int(equity * p["risk_pct"] / actual_risk))
            if shares_eb * actual_entry > equity * p["max_pos_pct"]:
                shares_eb = max(1, int(equity * p["max_pos_pct"] / actual_entry))
            loss = shares_eb * (stop - actual_entry) * trend - shares_eb * comm * 2
            equity += loss
            r_lost = abs(loss / (shares_eb * actual_risk)) if shares_eb * actual_risk > 0 else 0
            daily_r_loss += r_lost
            trade_log.append({"pnl": loss, "direction": "SHORT", "exit_reason": "entry_bar_stop",
                              "lock_hit": False, "entry_price": actual_entry,
                              "shares": shares_eb, "risk": actual_risk})
            bar = entry_bar + 1; continue

        # ─── Position sizing ───
        shares = max(1, int(equity * p["risk_pct"] / actual_risk))
        if shares * actual_entry > equity * p["max_pos_pct"]:
            shares = max(1, int(equity * p["max_pos_pct"] / actual_entry))
        if equity < shares * actual_risk or shares < 1:
            bar += 1; continue

        # ─── Execute trade (same exit logic for all modes) ───
        lock_shares = max(1, int(shares * p["lock1_pct"]))
        runner_stop = stop
        lock_done = False
        trade_pnl = -shares * comm
        remaining = shares
        end_bar = entry_bar
        exit_reason = "timeout"
        chand_bars = p["chandelier_bars"]
        chand_mult = p["chandelier_mult"]

        for k in range(1, max_fwd + 1):
            bi = entry_bar + k
            if bi >= n: break
            h = high[bi]; l = low[bi]
            cur_atr = atr_v[bi] if not np.isnan(atr_v[bi]) else a

            if times[bi] >= p["force_close_at"]:
                trade_pnl += remaining * (close[bi] - actual_entry) * trend - remaining * comm
                end_bar = bi; exit_reason = "session_close"; break

            stopped = (trend == 1 and l <= runner_stop) or (trend == -1 and h >= runner_stop)

            lock_hit_bar = False
            if not lock_done:
                lock_hit_bar = (trend == 1 and h >= actual_entry + p["lock1_rr"] * actual_risk) or \
                               (trend == -1 and l <= actual_entry - p["lock1_rr"] * actual_risk)

            if stopped and lock_hit_bar and not lock_done:
                trade_pnl += remaining * (runner_stop - actual_entry) * trend - remaining * comm
                end_bar = bi; exit_reason = "stop"; break

            if stopped:
                trade_pnl += remaining * (runner_stop - actual_entry) * trend - remaining * comm
                end_bar = bi; exit_reason = "stop" if not lock_done else "trail_stop"; break

            if lock_hit_bar and not lock_done:
                trade_pnl += lock_shares * p["lock1_rr"] * actual_risk - lock_shares * comm
                remaining -= lock_shares; lock_done = True
                if trend == 1: runner_stop = max(runner_stop, actual_entry)
                else: runner_stop = min(runner_stop, actual_entry)

            if lock_done and k >= chand_bars:
                sk = max(1, k - chand_bars + 1)
                if trend == 1:
                    hh = max(high[entry_bar + kk] for kk in range(sk, k) if entry_bar + kk < n)
                    runner_stop = max(runner_stop, hh - chand_mult * cur_atr)
                else:
                    ll = min(low[entry_bar + kk] for kk in range(sk, k) if entry_bar + kk < n)
                    runner_stop = min(runner_stop, ll + chand_mult * cur_atr)
        else:
            ep = close[min(entry_bar + max_fwd, n - 1)]
            trade_pnl += remaining * (ep - actual_entry) * trend - remaining * comm
            end_bar = min(entry_bar + max_fwd, n - 1)

        equity += trade_pnl
        if trade_pnl < 0:
            r_lost = abs(trade_pnl / (shares * actual_risk)) if shares * actual_risk > 0 else 0
            daily_r_loss += r_lost

        trade_log.append({
            "pnl": trade_pnl, "direction": "LONG" if trend == 1 else "SHORT",
            "exit_reason": exit_reason, "lock_hit": lock_done,
            "entry_price": actual_entry, "shares": shares, "risk": actual_risk,
        })
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

    # R-multiple distribution for big wins analysis
    r_mults = []
    if total > 0:
        for _, t in tdf.iterrows():
            rm = t["pnl"] / (t["shares"] * t["risk"]) if t["shares"] * t["risk"] > 0 else 0
            r_mults.append(rm)
    r_arr = np.array(r_mults) if r_mults else np.array([0])
    big_wins = (r_arr >= 5.0).sum()
    big_win_r = r_arr[r_arr >= 5.0].sum() if big_wins > 0 else 0

    return {
        "mode": mode, "pf": round(pf, 3), "return_pct": round(ret, 2),
        "trades": total, "wins": wins, "losses": losses,
        "win_rate": round(wins / total * 100, 1) if total > 0 else 0,
        "avg_win": round(gw / wins, 2) if wins > 0 else 0,
        "avg_loss": round(gl / losses, 2) if losses > 0 else 0,
        "triggers": triggers, "fills": fills, "missed": missed,
        "fill_rate": round(fills / triggers * 100, 1) if triggers > 0 else 0,
        "gap_cost_avg": round(gap_cost_total / fills, 4) if fills > 0 and mode == "stop_buy_fixed" else 0,
        "big_wins_5r": big_wins,
        "big_win_r_total": round(big_win_r, 1),
        "trades_per_day": round(total / max(days, 1), 1),
    }


def main():
    df = pd.read_csv(DATA_PATH, index_col="timestamp", parse_dates=True)
    print(f"Data: {len(df):,} bars, {df.index.normalize().nunique()} days\n")

    modes = ["stop_buy", "stop_buy_fixed", "retest_limit"]
    mode_names = {
        "stop_buy":       "A. Stop-buy (current)",
        "stop_buy_fixed": "B. Stop-buy (gap-fixed)",
        "retest_limit":   "C. Retest limit (wait 5 bars)",
    }

    results = []
    for m in modes:
        print(f"  Running {mode_names[m]}...")
        r = run_backtest(df, mode=m)
        results.append(r)

    # ═══ Main comparison ═══
    print(f"\n{'='*95}")
    print(f"  ENTRY MODE COMPARISON — Plan G")
    print(f"{'='*95}")
    print(f"\n  {'Mode':<30} {'PF':>6} {'Ret%':>8} {'Trades':>7} {'T/day':>6} {'WR%':>6} {'AvgW':>8} {'AvgL':>8}")
    print(f"  {'-'*90}")

    for r in results:
        print(f"  {mode_names[r['mode']]:<30} {r['pf']:>6.3f} {r['return_pct']:>+7.2f}%"
              f" {r['trades']:>6} {r['trades_per_day']:>5.1f} {r['win_rate']:>5.1f}%"
              f" {r['avg_win']:>7.2f} {r['avg_loss']:>7.2f}")

    # ═══ Fill rate analysis ═══
    print(f"\n{'='*95}")
    print(f"  FILL RATE & SIGNAL ANALYSIS")
    print(f"{'='*95}")
    print(f"\n  {'Mode':<30} {'Triggers':>9} {'Fills':>7} {'Missed':>8} {'Fill%':>7}")
    print(f"  {'-'*65}")
    for r in results:
        print(f"  {mode_names[r['mode']]:<30} {r['triggers']:>8} {r['fills']:>6}"
              f" {r['missed']:>7} {r['fill_rate']:>6.1f}%")

    # ═══ Big wins analysis ═══
    print(f"\n{'='*95}")
    print(f"  BIG WINS (5R+) ANALYSIS — tail profit impact")
    print(f"{'='*95}")
    print(f"\n  {'Mode':<30} {'5R+ count':>10} {'5R+ total R':>12}")
    print(f"  {'-'*55}")
    for r in results:
        print(f"  {mode_names[r['mode']]:<30} {r['big_wins_5r']:>9} {r['big_win_r_total']:>11.1f}R")

    # ═══ Gap cost (mode B) ═══
    b = results[1]
    if b["gap_cost_avg"] > 0:
        print(f"\n{'='*95}")
        print(f"  GAP ANALYSIS (Stop-buy fixed)")
        print(f"{'='*95}")
        print(f"  Average gap cost per fill: ${b['gap_cost_avg']:.4f}")
        print(f"  (This is how much worse the real fill is vs sig on average)")

    # ═══ Retest wait sensitivity ═══
    print(f"\n{'='*95}")
    print(f"  RETEST WAIT SENSITIVITY (how many bars to wait for retest)")
    print(f"{'='*95}")
    print(f"\n  {'Wait bars':>10} {'PF':>6} {'Ret%':>8} {'Trades':>7} {'Fill%':>7} {'5R+':>5}")
    print(f"  {'-'*50}")
    for wait in [2, 3, 5, 8, 10, 15]:
        PARAMS["retest_wait_bars"] = wait
        r = run_backtest(df, mode="retest_limit")
        print(f"  {wait:>10} {r['pf']:>6.3f} {r['return_pct']:>+7.2f}% {r['trades']:>6}"
              f" {r['fill_rate']:>6.1f}% {r['big_wins_5r']:>4}")
    PARAMS["retest_wait_bars"] = 5  # reset

    # ═══ Verdict ═══
    print(f"\n{'='*95}")
    print(f"  VERDICT")
    print(f"{'='*95}")
    a_pf = results[0]["pf"]; b_pf = results[1]["pf"]; c_pf = results[2]["pf"]
    print(f"\n  Stop-buy (current):     PF={a_pf:.3f}")
    print(f"  Stop-buy (gap-fixed):   PF={b_pf:.3f}  ({(b_pf-a_pf)/a_pf*100:+.1f}% vs current)")
    print(f"  Retest limit:           PF={c_pf:.3f}  ({(c_pf-a_pf)/a_pf*100:+.1f}% vs current)")
    print(f"\n  Retest fill rate: {results[2]['fill_rate']:.1f}% — missed {results[2]['missed']} of {results[2]['triggers']} triggers")
    print(f"  Retest 5R+ wins: {results[2]['big_wins_5r']} vs current {results[0]['big_wins_5r']}")


if __name__ == "__main__":
    main()
