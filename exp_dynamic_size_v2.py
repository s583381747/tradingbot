"""
Dynamic Position Sizing v2 — R-Multiple Approach (Unlimited Capital)

PROBLEM: QQQ at ~$450/share with ~$0.20 risk means 1% risk on $100k wants
5000 shares = $2.25M notional. Position is ALWAYS capped by equity, not risk.
No amount of leverage hacks or max_pos_pct tweaks fixes this cleanly.

SOLUTION: Track P&L purely in R-multiples. Each trade risks exactly 1R.
No equity, no shares, no position caps. The sizing multiplier scales the
EFFECTIVE R contribution of each trade.

For each strategy:
  - Base trade outcome = R-multiple (from the backtest engine)
  - Effective R = base_R * multiplier (from the sizing strategy)
  - Total R = sum of all effective R
  - Drawdown = tracked in cumulative R units

Strategies:
  1. Fixed 1.0x (baseline)
  2. Win/loss: after win -> 0.5x, after 3+ losses -> 1.0x, after 10+ -> 1.5x
  3. Streak: linearly scale 0.5x (0 losses) to 2.0x (20+ losses)
  4. ATR-based: low ATR quintile -> 0.7x, mid -> 1.0x, high -> 1.3x
  5. Combined: win/loss + ATR
  6. Anti-martingale: after win -> 1.5x, after loss -> 0.7x
  7. Recent performance: rolling 50-trade PF > 1.5 -> 1.3x, PF < 1.0 -> 0.5x

Entry: touch bar close (NOT signal trigger).
Params: ema 20/50, atr 14, touch_tol 0.15, stop_buffer 0.3,
        lock 0.1R/5%, chand 40/0.5, no_entry_after 14:00, skip_after_win 1.
Includes 3-bar MFE gate.
"""
from __future__ import annotations
import functools, datetime as dt
import numpy as np, pandas as pd

from entry_signal import (
    add_indicators, detect_trend, check_touch,
)

print = functools.partial(print, flush=True)

IS_PATH  = "data/QQQ_1Min_Polygon_2y_clean.csv"
OOS_PATH = "data/QQQ_1Min_Barchart_2y_2022-2024_clean.csv"

BASE_PARAMS = {
    "ema_fast": 20, "ema_slow": 50, "atr_period": 14,
    "touch_tol": 0.15, "touch_below_max": 0.5,
    "stop_buffer": 0.3,
    "lock_rr": 0.1, "lock_pct": 0.05,
    "chandelier_bars": 40, "chandelier_mult": 0.5,
    "max_hold_bars": 180,
    "no_entry_after": dt.time(14, 0),
    "force_close_at": dt.time(15, 58),
    "commission_per_share": 0.005,
    "daily_loss_r": 2.5,
    "skip_after_win": 1,
    # MFE gate
    "mfe_gate_bars": 3,
    "mfe_gate_threshold": 0.3,
    "mfe_gate_tighten": 0.3,
}


# ======================================================================
# BACKTEST ENGINE — R-MULTIPLE ONLY (no equity, no shares)
# ======================================================================

def run_backtest_r(df, params=None):
    """
    Run backtest tracking only R-multiples. No equity, no position sizing.
    Each trade risks exactly 1R. Returns list of trade dicts with base_r.

    Entry: touch bar close.
    Stop: touch_low - stop_buffer * ATR (long) or touch_high + stop_buffer * ATR (short).
    Risk: |entry - stop|.
    Lock: 5% of position at 0.1R, move stop to breakeven.
    Runner: 95% Chandelier trail.
    """
    p = {**BASE_PARAMS, **(params or {})}
    df = add_indicators(df, p)
    high = df["High"].values
    low = df["Low"].values
    close = df["Close"].values
    ema = df["ema20"].values
    ema_s = df["ema50"].values
    atr_v = df["atr"].values
    times = df.index.time
    dates = df.index.date
    n = len(df)

    max_fwd = p["max_hold_bars"]
    chand_bars = p["chandelier_bars"]
    chand_mult = p["chandelier_mult"]
    lock_rr = p["lock_rr"]
    lock_pct = p["lock_pct"]
    mfe_gate_bars = p["mfe_gate_bars"]
    mfe_gate_threshold = p["mfe_gate_threshold"]
    mfe_gate_tighten = p["mfe_gate_tighten"]
    # Commission in R units: approximate as fraction of risk
    # With ~$0.20 risk/share, $0.005 commission * 2 = $0.01 = 0.05R per share
    # But since we track in R, we approximate: comm_r = 2 * comm / avg_risk
    # We'll compute this per trade: comm_r = 2 * comm_per_share / risk_per_share
    comm_per_share = p["commission_per_share"]

    trades = []
    bar = max(p["ema_slow"], p["atr_period"]) + 5
    daily_r_loss = 0.0
    current_date = None
    skip_count = 0

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

        trend = detect_trend(close[bar], ema[bar], ema_s[bar])
        if trend == 0:
            bar += 1
            continue
        if not check_touch(trend, low[bar], high[bar], ema[bar], a,
                           p["touch_tol"], p["touch_below_max"]):
            bar += 1
            continue

        # Skip after win
        if skip_count > 0:
            skip_count -= 1
            bar += 1
            continue

        # Entry at touch bar close
        entry_price = close[bar]
        stop = (low[bar] - p["stop_buffer"] * a) if trend == 1 \
               else (high[bar] + p["stop_buffer"] * a)
        risk = abs(entry_price - stop)
        if risk <= 0:
            bar += 1
            continue
        entry_bar = bar

        # Commission in R for this trade (round-trip for full position)
        comm_r = 2.0 * comm_per_share / risk

        # Lock setup: lock_pct of position at lock_rr R
        # Runner: (1 - lock_pct) of position
        runner_frac = 1.0 - lock_pct
        lock_frac = lock_pct
        runner_stop_r = (stop - entry_price) / risk * trend  # should be ~ -1.0
        lock_done = False
        trade_r = -comm_r  # entry+exit commission (in R) for full position
        end_bar = entry_bar
        exit_reason = "timeout"
        mfe_r = 0.0
        mfe_gate_applied = False

        for k in range(1, max_fwd + 1):
            bi = entry_bar + k
            if bi >= n:
                break
            h = high[bi]
            l = low[bi]
            ca = atr_v[bi] if not np.isnan(atr_v[bi]) else a

            # Track MFE in R
            if trend == 1:
                cur_fav = (h - entry_price) / risk
            else:
                cur_fav = (entry_price - l) / risk
            mfe_r = max(mfe_r, cur_fav)

            # Force close
            if times[bi] >= p["force_close_at"]:
                ep = close[bi]
                exit_r = (ep - entry_price) * trend / risk
                if lock_done:
                    trade_r += runner_frac * exit_r
                else:
                    trade_r += exit_r
                end_bar = bi
                exit_reason = "session_close"
                break

            # 3-bar MFE gate
            if k == mfe_gate_bars and not lock_done and not mfe_gate_applied:
                if mfe_r < mfe_gate_threshold:
                    # Tighten stop: new stop at -mfe_gate_tighten R from entry
                    new_stop_r = -mfe_gate_tighten
                    if new_stop_r > runner_stop_r:
                        runner_stop_r = new_stop_r
                    mfe_gate_applied = True

            # Current stop price from runner_stop_r
            if trend == 1:
                cur_stop_price = entry_price + runner_stop_r * risk
            else:
                cur_stop_price = entry_price - runner_stop_r * risk

            # Stop check
            stopped = (trend == 1 and l <= cur_stop_price) or \
                      (trend == -1 and h >= cur_stop_price)

            # Lock check
            lock_hit_this_bar = False
            if not lock_done:
                if trend == 1:
                    lock_hit_this_bar = h >= entry_price + lock_rr * risk
                else:
                    lock_hit_this_bar = l <= entry_price - lock_rr * risk

            # Bug #2: same-bar stop/lock collision — stop wins (pessimistic)
            if stopped and lock_hit_this_bar and not lock_done:
                trade_r += runner_stop_r  # full position stopped at runner_stop_r
                end_bar = bi
                exit_reason = "stop"
                break

            if stopped:
                if lock_done:
                    trade_r += runner_frac * runner_stop_r
                else:
                    trade_r += runner_stop_r
                end_bar = bi
                exit_reason = "stop" if not lock_done else "trail_stop"
                break

            # Lock: lock_pct at lock_rr
            if lock_hit_this_bar and not lock_done:
                trade_r += lock_frac * lock_rr
                lock_done = True
                # Move stop to breakeven (0R)
                runner_stop_r = max(runner_stop_r, 0.0)

            # Chandelier trail (in R units)
            if lock_done and k >= chand_bars:
                sk = max(1, k - chand_bars + 1)
                if trend == 1:
                    hh = max(high[entry_bar + kk]
                             for kk in range(sk, k) if entry_bar + kk < n)
                    new_stop_price = hh - chand_mult * ca
                    new_stop_r = (new_stop_price - entry_price) / risk
                else:
                    ll = min(low[entry_bar + kk]
                             for kk in range(sk, k) if entry_bar + kk < n)
                    new_stop_price = ll + chand_mult * ca
                    new_stop_r = (entry_price - new_stop_price) / risk
                runner_stop_r = max(runner_stop_r, new_stop_r)
        else:
            # Timeout exit
            ep = close[min(entry_bar + max_fwd, n - 1)]
            exit_r = (ep - entry_price) * trend / risk
            if lock_done:
                trade_r += runner_frac * exit_r
            else:
                trade_r += exit_r
            end_bar = min(entry_bar + max_fwd, n - 1)

        # Daily R loss tracking
        if trade_r < 0:
            daily_r_loss += abs(trade_r)

        # Skip after win
        if trade_r > 0:
            skip_count = p.get("skip_after_win", 0)

        trades.append({
            "entry_bar": entry_bar,
            "entry_time": df.index[entry_bar],
            "direction": "LONG" if trend == 1 else "SHORT",
            "base_r": trade_r,
            "atr": a,
            "risk": risk,
            "exit_reason": exit_reason,
            "lock_hit": lock_done,
            "mfe_r": mfe_r,
        })

        bar = end_bar + 1

    return trades


# ======================================================================
# SIZING MULTIPLIER STRATEGIES
# ======================================================================

def mult_fixed(trade_history, atr_val, atr_quintiles, rolling_pf):
    """Baseline: always 1.0x."""
    return 1.0


def mult_win_loss(trade_history, atr_val, atr_quintiles, rolling_pf):
    """After win -> 0.5x, after 3+ losses -> 1.0x, after 10+ losses -> 1.5x."""
    if not trade_history:
        return 1.0
    last_r = trade_history[-1]
    if last_r > 0:
        return 0.5
    # Count consecutive losses
    consec = 0
    for r in reversed(trade_history):
        if r <= 0:
            consec += 1
        else:
            break
    if consec >= 10:
        return 1.5
    if consec >= 3:
        return 1.0
    return 1.0  # 1-2 losses: stay at 1.0


def mult_streak(trade_history, atr_val, atr_quintiles, rolling_pf):
    """Linear scale: 0.5x at 0 losses (just won) to 2.0x at 20+ losses."""
    if not trade_history:
        return 1.0
    consec = 0
    for r in reversed(trade_history):
        if r <= 0:
            consec += 1
        else:
            break
    if consec == 0:
        return 0.5  # just won
    # Linear from 0.5 (0 losses) to 2.0 (20 losses)
    scale = min(consec, 20) / 20.0
    return 0.5 + scale * 1.5


def mult_atr(trade_history, atr_val, atr_quintiles, rolling_pf):
    """Low ATR quintile -> 0.7x, mid -> 1.0x, high -> 1.3x."""
    if atr_quintiles is None or len(atr_quintiles) < 4:
        return 1.0
    if np.isnan(atr_val):
        return 1.0
    q20, q40, q60, q80 = atr_quintiles
    if atr_val <= q20:
        return 0.7
    elif atr_val <= q40:
        return 0.85
    elif atr_val <= q60:
        return 1.0
    elif atr_val <= q80:
        return 1.15
    else:
        return 1.3


def mult_combined(trade_history, atr_val, atr_quintiles, rolling_pf):
    """Win/loss + ATR combined."""
    m_wl = mult_win_loss(trade_history, atr_val, atr_quintiles, rolling_pf)
    m_atr = mult_atr(trade_history, atr_val, atr_quintiles, rolling_pf)
    return m_wl * m_atr


def mult_anti_martingale(trade_history, atr_val, atr_quintiles, rolling_pf):
    """After win -> 1.5x, after loss -> 0.7x. Tests BOTH directions."""
    if not trade_history:
        return 1.0
    last_r = trade_history[-1]
    if last_r > 0:
        return 1.5
    else:
        return 0.7


def mult_recent_pf(trade_history, atr_val, atr_quintiles, rolling_pf):
    """Rolling 50-trade PF > 1.5 -> 1.3x, PF < 1.0 -> 0.5x, else 1.0x."""
    if rolling_pf is None or np.isnan(rolling_pf):
        return 1.0
    if rolling_pf > 1.5:
        return 1.3
    elif rolling_pf < 1.0:
        return 0.5
    else:
        return 1.0


STRATEGIES = {
    "1_Fixed_1.0x":         mult_fixed,
    "2_WinLoss":            mult_win_loss,
    "3_Streak":             mult_streak,
    "4_ATR":                mult_atr,
    "5_Combined_WL+ATR":    mult_combined,
    "6_AntiMartingale":     mult_anti_martingale,
    "7_RecentPF":           mult_recent_pf,
}


# ======================================================================
# APPLY SIZING STRATEGIES TO BASE R-MULTIPLES
# ======================================================================

def apply_strategy(trades, sizing_fn, atr_quintiles):
    """
    Apply a sizing multiplier to each trade's base_r.
    Returns dict with summary metrics (all in R units).
    """
    if not trades:
        return {
            "total_r": 0.0, "eff_pf": 0.0, "max_dd_r": 0.0,
            "trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0,
            "avg_mult": 1.0, "avg_win_r": 0.0, "avg_loss_r": 0.0,
        }

    trade_history = []  # base_r history for sizing decisions
    cumulative_r = 0.0
    peak_r = 0.0
    max_dd_r = 0.0
    eff_r_list = []

    # Rolling PF lookback
    pf_lookback = 50

    for t in trades:
        # Compute rolling PF from last N base R values
        if len(trade_history) >= pf_lookback:
            recent = trade_history[-pf_lookback:]
            wins_r = sum(r for r in recent if r > 0)
            losses_r = abs(sum(r for r in recent if r <= 0))
            rolling_pf = wins_r / losses_r if losses_r > 0 else 999.0
        elif len(trade_history) >= 10:
            recent = trade_history
            wins_r = sum(r for r in recent if r > 0)
            losses_r = abs(sum(r for r in recent if r <= 0))
            rolling_pf = wins_r / losses_r if losses_r > 0 else 999.0
        else:
            rolling_pf = float('nan')

        mult = sizing_fn(trade_history, t["atr"], atr_quintiles, rolling_pf)
        eff_r = t["base_r"] * mult
        eff_r_list.append(eff_r)

        cumulative_r += eff_r
        peak_r = max(peak_r, cumulative_r)
        dd = peak_r - cumulative_r
        max_dd_r = max(max_dd_r, dd)

        trade_history.append(t["base_r"])

    total_r = sum(eff_r_list)
    wins = [r for r in eff_r_list if r > 0]
    losses = [r for r in eff_r_list if r <= 0]
    gross_won = sum(wins) if wins else 0
    gross_lost = abs(sum(losses)) if losses else 0
    eff_pf = gross_won / gross_lost if gross_lost > 0 else 0

    # Compute multiplier stats
    # Re-run to get multipliers (fast)
    trade_hist2 = []
    mults = []
    for t in trades:
        if len(trade_hist2) >= pf_lookback:
            recent = trade_hist2[-pf_lookback:]
            wr = sum(r for r in recent if r > 0)
            lr = abs(sum(r for r in recent if r <= 0))
            rpf = wr / lr if lr > 0 else 999.0
        elif len(trade_hist2) >= 10:
            wr = sum(r for r in trade_hist2 if r > 0)
            lr = abs(sum(r for r in trade_hist2 if r <= 0))
            rpf = wr / lr if lr > 0 else 999.0
        else:
            rpf = float('nan')
        m = sizing_fn(trade_hist2, t["atr"], atr_quintiles, rpf)
        mults.append(m)
        trade_hist2.append(t["base_r"])

    return {
        "total_r": round(total_r, 3),
        "eff_pf": round(eff_pf, 3),
        "max_dd_r": round(max_dd_r, 3),
        "trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / len(trades) * 100, 1) if trades else 0,
        "avg_win_r": round(np.mean(wins), 4) if wins else 0,
        "avg_loss_r": round(abs(np.mean(losses)), 4) if losses else 0,
        "avg_mult": round(np.mean(mults), 3),
        "ret_per_dd": round(total_r / max_dd_r, 2) if max_dd_r > 0 else 0,
    }


# ======================================================================
# MAIN
# ======================================================================

def run_all(label, trades, atr_quintiles):
    """Run all strategies on a set of base trades. Return results DataFrame."""
    results = []
    for name, fn in STRATEGIES.items():
        r = apply_strategy(trades, fn, atr_quintiles)
        results.append({"Strategy": name, **r})
    return pd.DataFrame(results)


def print_table(rdf, label, base_trades_count):
    """Print formatted comparison table."""
    rdf = rdf.sort_values("ret_per_dd", ascending=False).reset_index(drop=True)

    print()
    print("=" * 130)
    print(f"  {label} ({base_trades_count} trades)")
    print("=" * 130)
    hdr = (f"{'Strategy':<22} {'TotalR':>8} {'EffPF':>7} {'MaxDD_R':>8} "
           f"{'Ret/DD':>7} {'WR%':>6} {'Wins':>5} {'Loss':>5} "
           f"{'AvgWinR':>8} {'AvgLossR':>9} {'AvgMult':>8}")
    print(hdr)
    print("-" * 130)

    for _, row in rdf.iterrows():
        line = (f"{row['Strategy']:<22} {row['total_r']:>+8.2f} {row['eff_pf']:>7.3f} "
                f"{row['max_dd_r']:>8.3f} {row['ret_per_dd']:>7.2f} "
                f"{row['win_rate']:>6.1f} {row['wins']:>5} {row['losses']:>5} "
                f"{row['avg_win_r']:>8.4f} {row['avg_loss_r']:>9.4f} "
                f"{row['avg_mult']:>8.3f}")
        print(line)
    print("-" * 130)

    # Baseline comparison
    baseline = rdf[rdf["Strategy"] == "1_Fixed_1.0x"].iloc[0]
    print(f"\n  Baseline (Fixed 1.0x): TotalR={baseline['total_r']:+.2f}  "
          f"PF={baseline['eff_pf']:.3f}  MaxDD={baseline['max_dd_r']:.3f}R  "
          f"Ret/DD={baseline['ret_per_dd']:.2f}")
    print()

    for _, row in rdf.iterrows():
        if row["Strategy"] == "1_Fixed_1.0x":
            continue
        r_delta = row["total_r"] - baseline["total_r"]
        dd_delta = row["max_dd_r"] - baseline["max_dd_r"]
        pf_delta = row["eff_pf"] - baseline["eff_pf"]
        retdd_delta = row["ret_per_dd"] - baseline["ret_per_dd"]

        if r_delta > 0.01 and dd_delta <= 0.001:
            verdict = "++ MORE R + LESS/SAME DD"
        elif r_delta > 0.01 and dd_delta > 0.001:
            verdict = "+- More R, more DD"
        elif r_delta < -0.01 and dd_delta < -0.001:
            verdict = "-+ Less R, less DD"
        else:
            verdict = "-- Worse or no change"

        # Check Ret/DD improvement
        if row["ret_per_dd"] > baseline["ret_per_dd"] * 1.05:
            verdict += " | Ret/DD IMPROVED"
        elif row["ret_per_dd"] < baseline["ret_per_dd"] * 0.95:
            verdict += " | Ret/DD degraded"

        print(f"  {row['Strategy']:<22}: dR={r_delta:+.2f}  dPF={pf_delta:+.3f}  "
              f"dDD={dd_delta:+.3f}R  Ret/DD={row['ret_per_dd']:.2f} "
              f"({retdd_delta:+.2f})  {verdict}")


if __name__ == "__main__":
    print("=" * 80)
    print("DYNAMIC POSITION SIZING v2 — R-MULTIPLE APPROACH")
    print("=" * 80)
    print()
    print("Each trade risks exactly 1R. No equity, no shares, no position caps.")
    print("Sizing multiplier scales the EFFECTIVE R contribution of each trade.")
    print("This isolates the pure sizing signal from equity/leverage artifacts.")
    print()
    print("Entry: touch bar close | Stop: low - 0.3*ATR")
    print("Lock: 5% at 0.1R -> BE | Runner: 95% Chandelier(40, 0.5)")
    print("MFE gate: tighten to -0.3R if MFE < 0.3R after 3 bars")
    print("Skip 1 trade after each win | No entries after 14:00")
    print()

    # Load data
    print("Loading data...")
    df_is = pd.read_csv(IS_PATH, index_col="timestamp", parse_dates=True)
    df_oos = pd.read_csv(OOS_PATH, index_col="timestamp", parse_dates=True)
    print(f"  IS:  {len(df_is):,} bars  ({df_is.index[0].date()} to {df_is.index[-1].date()})")
    print(f"  OOS: {len(df_oos):,} bars  ({df_oos.index[0].date()} to {df_oos.index[-1].date()})")
    print()

    # Run base backtest (R-multiples only)
    print("Running base backtest (IS)...")
    trades_is = run_backtest_r(df_is)
    print(f"  {len(trades_is)} trades")
    print("Running base backtest (OOS)...")
    trades_oos = run_backtest_r(df_oos)
    print(f"  {len(trades_oos)} trades")
    print()

    # Sanity check: print base R stats
    base_r_is = [t["base_r"] for t in trades_is]
    base_r_oos = [t["base_r"] for t in trades_oos]
    print("Base R-multiple distribution:")
    print(f"  IS:  mean={np.mean(base_r_is):.4f}R  median={np.median(base_r_is):.4f}R  "
          f"sum={sum(base_r_is):.2f}R  winR%={sum(1 for r in base_r_is if r>0)/len(base_r_is)*100:.1f}%")
    print(f"  OOS: mean={np.mean(base_r_oos):.4f}R  median={np.median(base_r_oos):.4f}R  "
          f"sum={sum(base_r_oos):.2f}R  winR%={sum(1 for r in base_r_oos if r>0)/len(base_r_oos)*100:.1f}%")
    print()

    # Compute ATR quintiles for ATR-based strategies
    atr_is = [t["atr"] for t in trades_is]
    atr_oos = [t["atr"] for t in trades_oos]
    atr_q_is = np.percentile(atr_is, [20, 40, 60, 80])
    atr_q_oos = np.percentile(atr_oos, [20, 40, 60, 80])
    print(f"  ATR quintiles IS:  {atr_q_is}")
    print(f"  ATR quintiles OOS: {atr_q_oos}")
    print()

    # Run all strategies
    print("=" * 80)
    print("IN-SAMPLE RESULTS")
    print("=" * 80)
    rdf_is = run_all("IS", trades_is, atr_q_is)
    print_table(rdf_is, "IN-SAMPLE", len(trades_is))

    print()
    print("=" * 80)
    print("OUT-OF-SAMPLE RESULTS")
    print("=" * 80)
    rdf_oos = run_all("OOS", trades_oos, atr_q_oos)
    print_table(rdf_oos, "OUT-OF-SAMPLE", len(trades_oos))

    # Combined view
    print()
    print()
    print("=" * 130)
    print("  COMBINED IS vs OOS COMPARISON")
    print("=" * 130)
    hdr = (f"{'Strategy':<22} {'IS_TotR':>8} {'OOS_TotR':>9} {'IS_PF':>7} {'OOS_PF':>8} "
           f"{'IS_DD_R':>8} {'OOS_DD_R':>9} {'IS_R/DD':>8} {'OOS_R/DD':>9} "
           f"{'IS_Mult':>8} {'OOS_Mult':>9}")
    print(hdr)
    print("-" * 130)

    for name in STRATEGIES.keys():
        r_is = rdf_is[rdf_is["Strategy"] == name].iloc[0]
        r_oos = rdf_oos[rdf_oos["Strategy"] == name].iloc[0]
        line = (f"{name:<22} {r_is['total_r']:>+8.2f} {r_oos['total_r']:>+9.2f} "
                f"{r_is['eff_pf']:>7.3f} {r_oos['eff_pf']:>8.3f} "
                f"{r_is['max_dd_r']:>8.3f} {r_oos['max_dd_r']:>9.3f} "
                f"{r_is['ret_per_dd']:>8.2f} {r_oos['ret_per_dd']:>9.2f} "
                f"{r_is['avg_mult']:>8.3f} {r_oos['avg_mult']:>9.3f}")
        print(line)
    print("-" * 130)

    # Final verdict
    print()
    print("=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)
    print()

    baseline_is = rdf_is[rdf_is["Strategy"] == "1_Fixed_1.0x"].iloc[0]
    baseline_oos = rdf_oos[rdf_oos["Strategy"] == "1_Fixed_1.0x"].iloc[0]

    best_retdd_oos = rdf_oos.sort_values("ret_per_dd", ascending=False).iloc[0]
    best_total_r_oos = rdf_oos.sort_values("total_r", ascending=False).iloc[0]

    print(f"  Baseline (Fixed 1.0x):")
    print(f"    IS:  TotalR={baseline_is['total_r']:+.2f}  PF={baseline_is['eff_pf']:.3f}  "
          f"MaxDD={baseline_is['max_dd_r']:.3f}R  Ret/DD={baseline_is['ret_per_dd']:.2f}")
    print(f"    OOS: TotalR={baseline_oos['total_r']:+.2f}  PF={baseline_oos['eff_pf']:.3f}  "
          f"MaxDD={baseline_oos['max_dd_r']:.3f}R  Ret/DD={baseline_oos['ret_per_dd']:.2f}")
    print()
    print(f"  Best OOS Ret/DD: {best_retdd_oos['Strategy']} "
          f"(Ret/DD={best_retdd_oos['ret_per_dd']:.2f}, "
          f"TotalR={best_retdd_oos['total_r']:+.2f})")
    print(f"  Best OOS TotalR: {best_total_r_oos['Strategy']} "
          f"(TotalR={best_total_r_oos['total_r']:+.2f}, "
          f"Ret/DD={best_total_r_oos['ret_per_dd']:.2f})")
    print()

    # Check if any strategy beats baseline on BOTH IS and OOS
    print("  Strategies beating baseline Ret/DD on BOTH IS and OOS:")
    any_winner = False
    for name in STRATEGIES.keys():
        if name == "1_Fixed_1.0x":
            continue
        r_is = rdf_is[rdf_is["Strategy"] == name].iloc[0]
        r_oos = rdf_oos[rdf_oos["Strategy"] == name].iloc[0]
        if (r_is["ret_per_dd"] > baseline_is["ret_per_dd"] and
                r_oos["ret_per_dd"] > baseline_oos["ret_per_dd"]):
            pct_is = (r_is["ret_per_dd"] / baseline_is["ret_per_dd"] - 1) * 100
            pct_oos = (r_oos["ret_per_dd"] / baseline_oos["ret_per_dd"] - 1) * 100
            print(f"    {name}: IS +{pct_is:.1f}%  OOS +{pct_oos:.1f}%")
            any_winner = True
    if not any_winner:
        print("    (none)")

    print()
    print("Done.")
