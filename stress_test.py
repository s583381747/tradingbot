"""
Extreme Stress Test — Market Microstructure Reality Check

Tests three "illusions" raised by external audit:
  1. Queue Position: Lock limit order only fills on penetration (High > target, not >=)
  2. BE Bleed: All breakeven stops get penalty slippage ($0.03)
  3. Conditional Volatility: Measure entry bar ranges vs global bars

Runs 6 scenarios:
  A. Baseline (current strategy_final.py logic)
  B. Strict lock fill (High > target only)
  C. BE slippage penalty ($0.03 on all BE exits)
  D. B + C combined
  E. B + C + entry bar stop check (Bug #1 fix)
  F. Full pessimistic (B + C + Bug fixes + $0.05 extra entry slippage)
"""
from __future__ import annotations
import functools, datetime as dt, sys
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count

from entry_signal import add_indicators

print = functools.partial(print, flush=True)

DATA_PATH = "data/QQQ_1Min_Polygon_2y_clean.csv"

PARAMS = {
    "ema_fast": 20,
    "ema_slow": 50,
    "atr_period": 14,
    "touch_tol": 0.15,
    "touch_below_max": 0.5,
    "signal_offset": 0.05,
    "stop_buffer": 0.3,
    "lock1_rr": 0.3,
    "lock1_pct": 0.20,
    "chandelier_bars": 40,       # Plan F
    "chandelier_mult": 1.0,      # Plan F
    "signal_valid_bars": 3,
    "max_hold_bars": 180,
    "risk_pct": 0.01,
    "max_pos_pct": 0.25,
    "no_entry_after": dt.time(15, 30),
    "force_close_at": dt.time(15, 58),
    "commission_per_share": 0.005,
    "daily_loss_r": 2.5,
}


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    return add_indicators(df, PARAMS)


def run_backtest(df, capital=100_000, mode="baseline"):
    """
    Modes:
      baseline    — current logic (High >= target for lock fill)
      strict_lock — lock only fills on penetration (High > target)
      be_bleed    — BE stops get $0.03 penalty slippage
      combined    — strict_lock + be_bleed
      full_fix    — combined + entry bar stop check (Bug #1)
      nightmare   — full_fix + $0.05 entry slippage on signal fill
    """
    p = PARAMS.copy()
    df = prepare_data(df)
    high = df["High"].values
    low = df["Low"].values
    close = df["Close"].values
    ema = df["ema20"].values
    ema_s = df["ema50"].values
    atr_v = df["atr"].values
    times = df.index.time
    dates = df.index.date
    n = len(df)

    comm = p["commission_per_share"]
    max_fwd = p["max_hold_bars"]
    be_penalty = 0.03  # $0.03 adverse slippage on BE stops
    entry_slip = 0.05 if mode == "nightmare" else 0.0

    strict_lock = mode in ("strict_lock", "combined", "full_fix", "nightmare")
    apply_be_bleed = mode in ("be_bleed", "combined", "full_fix", "nightmare")
    fix_entry_bar = mode in ("full_fix", "nightmare")

    equity = capital
    trade_log = []
    bar = max(p["ema_slow"], p["atr_period"]) + 5

    # Daily R tracking
    daily_r_loss = 0.0
    current_date = None

    # Stats for analysis
    lock_just_touched = 0
    lock_penetrated = 0
    be_exits = 0
    entry_bar_ranges = []  # range of bars where entry happens
    global_ranges = []     # range of all bars

    while bar < n - max_fwd - 5:
        a = atr_v[bar]
        if np.isnan(a) or a <= 0 or np.isnan(ema[bar]) or np.isnan(ema_s[bar]):
            bar += 1
            continue
        if times[bar] >= p["no_entry_after"]:
            bar += 1
            continue

        # Daily R limit
        d = dates[bar]
        if current_date != d:
            current_date = d
            daily_r_loss = 0.0
        if daily_r_loss >= p["daily_loss_r"]:
            bar += 1
            continue

        # Collect global range stats
        global_ranges.append(high[bar] - low[bar])

        # Trend
        trend = 0
        if close[bar] > ema[bar] and ema[bar] > ema_s[bar]:
            trend = 1
        elif close[bar] < ema[bar] and ema[bar] < ema_s[bar]:
            trend = -1
        if trend == 0:
            bar += 1
            continue

        # Touch
        tol = a * p["touch_tol"]
        if trend == 1:
            is_touch = (low[bar] <= ema[bar] + tol) and \
                       (low[bar] >= ema[bar] - a * p["touch_below_max"])
        else:
            is_touch = (high[bar] >= ema[bar] - tol) and \
                       (high[bar] <= ema[bar] + a * p["touch_below_max"])
        if not is_touch:
            bar += 1
            continue

        # Bounce
        bb = bar + 1
        if bb >= n:
            bar += 1
            continue
        if trend == 1 and close[bb] <= high[bar]:
            bar += 1
            continue
        if trend == -1 and close[bb] >= low[bar]:
            bar += 1
            continue

        # Signal line
        if trend == 1:
            sig = high[bar] + p["signal_offset"]
            stop = low[bar] - p["stop_buffer"] * a
        else:
            sig = low[bar] - p["signal_offset"]
            stop = high[bar] + p["stop_buffer"] * a

        risk = abs(sig - stop)
        if risk <= 0:
            bar += 1
            continue

        # Signal trigger
        triggered = False
        entry_bar = -1
        for j in range(1, p["signal_valid_bars"] + 1):
            cb = bb + j
            if cb >= n:
                break
            if trend == 1 and high[cb] >= sig:
                triggered = True
                entry_bar = cb
                break
            elif trend == -1 and low[cb] <= sig:
                triggered = True
                entry_bar = cb
                break

        if not triggered:
            bar += 1
            continue

        # Apply entry slippage (nightmare mode)
        actual_sig = sig + entry_slip * trend  # worse fill

        # Record entry bar range
        entry_bar_ranges.append(high[entry_bar] - low[entry_bar])

        # Position sizing
        shares = max(1, int(equity * p["risk_pct"] / risk))
        if shares * sig > equity * p["max_pos_pct"]:
            shares = max(1, int(equity * p["max_pos_pct"] / sig))
        if equity < shares * risk or shares < 1:
            bar += 1
            continue

        # Bug #1 fix: entry bar stop check
        if fix_entry_bar:
            if trend == 1 and low[entry_bar] <= stop:
                loss = shares * (stop - actual_sig) * trend - shares * comm * 2
                equity += loss
                r_lost = abs(loss) / (shares * risk) if shares * risk > 0 else 0
                daily_r_loss += r_lost
                trade_log.append({
                    "entry_bar": entry_bar, "direction": "LONG" if trend == 1 else "SHORT",
                    "entry_price": actual_sig, "stop_price": stop, "risk": risk, "shares": shares,
                    "pnl": loss, "exit_reason": "entry_bar_stop", "lock_hit": False,
                    "r_multiple": loss / (shares * risk) if shares * risk > 0 else 0,
                })
                bar = entry_bar + 1
                continue
            elif trend == -1 and high[entry_bar] >= stop:
                loss = shares * (stop - actual_sig) * trend - shares * comm * 2
                equity += loss
                r_lost = abs(loss) / (shares * risk) if shares * risk > 0 else 0
                daily_r_loss += r_lost
                trade_log.append({
                    "entry_bar": entry_bar, "direction": "LONG" if trend == 1 else "SHORT",
                    "entry_price": actual_sig, "stop_price": stop, "risk": risk, "shares": shares,
                    "pnl": loss, "exit_reason": "entry_bar_stop", "lock_hit": False,
                    "r_multiple": loss / (shares * risk) if shares * risk > 0 else 0,
                })
                bar = entry_bar + 1
                continue

        # Execute trade
        lock_shares = max(1, int(shares * p["lock1_pct"]))
        runner_stop = stop
        lock_done = False
        trade_pnl = -shares * comm  # entry commission
        remaining = shares
        end_bar = entry_bar
        exit_reason = "timeout"
        chand_bars = p["chandelier_bars"]
        chand_mult = p["chandelier_mult"]

        for k in range(1, max_fwd + 1):
            bi = entry_bar + k
            if bi >= n:
                break
            h = high[bi]
            l = low[bi]
            cur_atr = atr_v[bi] if bi < n and not np.isnan(atr_v[bi]) else a

            # Force close
            if times[bi] >= p["force_close_at"]:
                ep = close[bi]
                trade_pnl += remaining * (ep - actual_sig) * trend - remaining * comm
                end_bar = bi
                exit_reason = "session_close"
                break

            # Stop check
            stopped = (trend == 1 and l <= runner_stop) or \
                      (trend == -1 and h >= runner_stop)
            if stopped:
                exit_price = runner_stop
                # BE bleed: if stop is at entry (breakeven), add penalty
                if apply_be_bleed and lock_done and abs(runner_stop - actual_sig) < 0.02:
                    exit_price = runner_stop - be_penalty * trend
                    be_exits += 1

                trade_pnl += remaining * (exit_price - actual_sig) * trend - remaining * comm
                end_bar = bi
                exit_reason = "stop" if not lock_done else "trail_stop"
                break

            # Lock check
            if not lock_done:
                lock_target = actual_sig + p["lock1_rr"] * risk * trend

                if strict_lock:
                    # Strict: must penetrate, not just touch
                    hit = (trend == 1 and h > lock_target) or \
                          (trend == -1 and l < lock_target)
                else:
                    hit = (trend == 1 and h >= lock_target) or \
                          (trend == -1 and l <= lock_target)

                if hit:
                    lock_penetrated += 1
                    trade_pnl += lock_shares * p["lock1_rr"] * risk - lock_shares * comm
                    remaining -= lock_shares
                    lock_done = True
                    if trend == 1:
                        runner_stop = max(runner_stop, actual_sig)
                    else:
                        runner_stop = min(runner_stop, actual_sig)
                else:
                    # Track "just touched" for stats
                    just_touch = (trend == 1 and h == lock_target) or \
                                 (trend == -1 and l == lock_target)
                    if just_touch:
                        lock_just_touched += 1

            # Chandelier trail
            if lock_done and k >= chand_bars:
                sk = max(1, k - chand_bars + 1)
                if trend == 1:
                    hh = max(high[entry_bar + kk]
                             for kk in range(sk, k) if entry_bar + kk < n)
                    new_stop = hh - chand_mult * cur_atr
                    runner_stop = max(runner_stop, new_stop)
                else:
                    ll = min(low[entry_bar + kk]
                             for kk in range(sk, k) if entry_bar + kk < n)
                    new_stop = ll + chand_mult * cur_atr
                    runner_stop = min(runner_stop, new_stop)
        else:
            ep = close[min(entry_bar + max_fwd, n - 1)]
            trade_pnl += remaining * (ep - actual_sig) * trend - remaining * comm
            end_bar = min(entry_bar + max_fwd, n - 1)

        equity += trade_pnl
        r_mult = trade_pnl / (shares * risk) if shares * risk > 0 else 0

        # Daily R loss tracking
        if trade_pnl < 0:
            daily_r_loss += abs(r_mult)

        trade_log.append({
            "entry_bar": entry_bar, "direction": "LONG" if trend == 1 else "SHORT",
            "entry_price": actual_sig, "stop_price": stop, "risk": risk, "shares": shares,
            "pnl": trade_pnl, "exit_reason": exit_reason, "lock_hit": lock_done,
            "r_multiple": r_mult,
        })

        bar = end_bar + 1

    # Summary
    trades_df = pd.DataFrame(trade_log) if trade_log else pd.DataFrame()
    total = len(trades_df)
    wins = (trades_df["pnl"] > 0).sum() if total > 0 else 0
    losses = total - wins
    gross_won = trades_df.loc[trades_df["pnl"] > 0, "pnl"].sum() if wins > 0 else 0
    gross_lost = abs(trades_df.loc[trades_df["pnl"] <= 0, "pnl"].sum()) if losses > 0 else 0
    pf = gross_won / gross_lost if gross_lost > 0 else 0
    ret = (equity - capital) / capital * 100

    return {
        "mode": mode,
        "equity": equity,
        "return_pct": round(ret, 2),
        "pf": round(pf, 3),
        "trades": total,
        "wins": wins,
        "losses": losses,
        "win_rate": round(wins / total * 100, 1) if total > 0 else 0,
        "gross_won": round(gross_won, 2),
        "gross_lost": round(gross_lost, 2),
        "trade_log": trades_df,
        "lock_just_touched": lock_just_touched,
        "lock_penetrated": lock_penetrated,
        "be_exits": be_exits,
        "entry_bar_ranges": entry_bar_ranges,
        "global_ranges": global_ranges,
    }


def run_mode(args):
    df, mode = args
    return run_backtest(df, mode=mode)


def main():
    df = pd.read_csv(DATA_PATH, index_col="timestamp", parse_dates=True)
    print(f"Data: {len(df):,} bars, {df.index.date[-1] - df.index.date[0]} range\n")

    modes = ["baseline", "strict_lock", "be_bleed", "combined", "full_fix", "nightmare"]
    mode_desc = {
        "baseline":    "A. Baseline (Plan F, 40/1.0, 2.5R limit)",
        "strict_lock": "B. Strict lock fill (High > target only)",
        "be_bleed":    "C. BE slippage penalty ($0.03/share)",
        "combined":    "D. B + C combined",
        "full_fix":    "E. D + entry bar stop check (Bug #1)",
        "nightmare":   "F. E + $0.05 entry slippage (WORST CASE)",
    }

    # Run all modes in parallel
    with ProcessPoolExecutor(max_workers=cpu_count()) as ex:
        results = list(ex.map(run_mode, [(df, m) for m in modes]))

    # ═══ Summary Table ═══
    print(f"{'='*90}")
    print(f"  EXTREME STRESS TEST — Market Microstructure Reality Check")
    print(f"{'='*90}")
    print(f"\n{'Mode':<50} {'PF':>6} {'Ret%':>8} {'Trades':>7} {'WR%':>6} {'ΔPF':>7}")
    print(f"{'-'*90}")

    baseline_pf = results[0]["pf"]
    for r in results:
        delta = r["pf"] - baseline_pf
        desc = mode_desc[r["mode"]]
        print(f"  {desc:<48} {r['pf']:>6.3f} {r['return_pct']:>+7.2f}% {r['trades']:>6} "
              f"{r['win_rate']:>5.1f}% {delta:>+6.3f}")

    # ═══ Queue Position Analysis ═══
    print(f"\n{'='*90}")
    print(f"  ILLUSION #1: Queue Position Analysis")
    print(f"{'='*90}")

    bl = results[0]
    sl = results[1]
    print(f"\n  Baseline lock fills:          {bl['lock_penetrated']}")
    print(f"  Strict lock fills (H > tgt):  {sl['lock_penetrated']}")
    print(f"  Lock 'just touched' (H==tgt): {sl['lock_just_touched']}")
    missed = bl["lock_penetrated"] - sl["lock_penetrated"]
    if bl["lock_penetrated"] > 0:
        print(f"  Missed fills on strict rule:  {missed} ({missed/bl['lock_penetrated']*100:.1f}%)")
    print(f"  PF impact: {bl['pf']:.3f} → {sl['pf']:.3f} ({sl['pf']-bl['pf']:+.3f})")

    # ═══ BE Bleed Analysis ═══
    print(f"\n{'='*90}")
    print(f"  ILLUSION #2: Breakeven Stop Bleed Analysis")
    print(f"{'='*90}")

    bb = results[2]
    print(f"\n  BE exits penalized:  {bb['be_exits']}")
    if bb["trades"] > 0:
        print(f"  BE exits as % of trades: {bb['be_exits']/bb['trades']*100:.1f}%")
    print(f"  Total penalty: ${bb['be_exits'] * 0.03:.2f} per share (estimated)")
    print(f"  PF impact: {bl['pf']:.3f} → {bb['pf']:.3f} ({bb['pf']-bl['pf']:+.3f})")

    # ═══ Conditional Volatility ═══
    print(f"\n{'='*90}")
    print(f"  ILLUSION #3: Conditional Volatility Analysis")
    print(f"{'='*90}")

    global_r = np.array(bl["global_ranges"])
    entry_r = np.array(bl["entry_bar_ranges"])

    print(f"\n  Global bar range:  median=${np.median(global_r):.4f}  mean=${np.mean(global_r):.4f}  p95=${np.percentile(global_r, 95):.4f}")
    print(f"  Entry bar range:   median=${np.median(entry_r):.4f}  mean=${np.mean(entry_r):.4f}  p95=${np.percentile(entry_r, 95):.4f}")
    print(f"  Ratio (entry/global): {np.median(entry_r)/np.median(global_r):.2f}x median, {np.mean(entry_r)/np.mean(global_r):.2f}x mean")

    # What % of entry bars have range > stop distance?
    # Stop distance ≈ candle_range + $0.05 + 0.3*ATR ≈ $0.60
    for threshold in [0.30, 0.40, 0.50, 0.60, 0.80, 1.00]:
        pct = (entry_r >= threshold).mean() * 100
        print(f"  Entry bars with range >= ${threshold:.2f}: {pct:.1f}%")

    # ═══ R-Distribution under nightmare ═══
    print(f"\n{'='*90}")
    print(f"  R-DISTRIBUTION: Nightmare Mode")
    print(f"{'='*90}")

    nm = results[-1]
    if nm["trades"] > 0:
        tr = nm["trade_log"]
        r_vals = tr["r_multiple"].values
        print(f"\n  Trades: {len(tr)}")
        print(f"  Mean R: {r_vals.mean():+.4f}")
        print(f"  Median R: {np.median(r_vals):+.4f}")

        bins = [(-999, -1.0), (-1.0, -0.5), (-0.5, 0), (0, 0.1), (0.1, 0.5),
                (0.5, 1.0), (1.0, 2.0), (2.0, 5.0), (5.0, 999)]
        labels = ["< -1R", "-1R to -0.5R", "-0.5R to 0R", "0R to 0.1R",
                  "0.1R to 0.5R", "0.5R to 1R", "1R to 2R", "2R to 5R", "5R+"]
        for (lo, hi), label in zip(bins, labels):
            mask = (r_vals >= lo) & (r_vals < hi)
            cnt = mask.sum()
            if cnt > 0:
                total_r = r_vals[mask].sum()
                print(f"  {label:<15}: {cnt:>5} ({cnt/len(tr)*100:>5.1f}%)  "
                      f"total R: {total_r:>+8.1f}")

    # ═══ Walk-forward under nightmare ═══
    print(f"\n{'='*90}")
    print(f"  WALK-FORWARD: Nightmare Mode (Y1 vs Y2)")
    print(f"{'='*90}")

    mid = "2025-03-22"
    for label, start, end in [("Y1", "2024-01-01", mid), ("Y2", mid, "2027-01-01")]:
        sub = df[(df.index >= start) & (df.index < end)]
        r = run_backtest(sub, mode="nightmare")
        print(f"  {label}: PF={r['pf']:.3f} Ret={r['return_pct']:+.2f}% "
              f"Trades={r['trades']} WR={r['win_rate']:.1f}%")

    # ═══ Final Verdict ═══
    print(f"\n{'='*90}")
    print(f"  VERDICT")
    print(f"{'='*90}")
    bl_pf = results[0]["pf"]
    nm_pf = results[-1]["pf"]
    print(f"\n  Baseline PF:   {bl_pf:.3f}")
    print(f"  Nightmare PF:  {nm_pf:.3f}")
    print(f"  Degradation:   {(nm_pf-bl_pf)/bl_pf*100:+.1f}%")
    print(f"  Still profitable: {'YES' if nm_pf > 1.0 else 'NO'}")
    if nm_pf > 1.0:
        print(f"  Nightmare scenario still has edge (PF > 1.0)")
    else:
        print(f"  ⚠️ Strategy loses money under worst-case assumptions")


if __name__ == "__main__":
    main()
