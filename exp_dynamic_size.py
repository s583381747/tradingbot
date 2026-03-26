"""
Dynamic Position Sizing Experiment

Tests 7 adaptive sizing strategies against the baseline 1% fixed risk.
Since PF is computed from R-multiples and won't change with sizing,
the primary metrics are: Return %, Final Equity, MaxDD%.

Key findings driving this experiment:
  - Win-after-win PF=0.411 (next trade after a win is likely bad)
  - After 15+ consecutive losses PF=2.397 (streaks predict mean reversion)
  - Low ATR quintile PF=1.242, High ATR PF=1.628
  - These suggest dynamic sizing could improve risk-adjusted returns

Approaches tested:
  1. Win/loss adaptive: after win reduce to 0.5%, after 5+ losses increase to 1.5%
  2. Streak-based: scale risk 0.5%→2% based on loss streak length
  3. ATR-based: low ATR→0.7%, high ATR→1.3%
  4. Equity curve: if equity < 20-trade SMA, reduce risk to 0.5%
  5. Combined: win-loss + ATR adaptive
  6. Anti-martingale: increase risk after wins (test both directions)
  7. Kelly fraction: adjust risk based on recent 100-trade WR and avg W/L ratio

Includes 3-bar MFE gate: if MFE < 0.3R after 3 bars, tighten stop to -0.3R.
"""
from __future__ import annotations
import functools, datetime as dt
import numpy as np, pandas as pd

from entry_signal import (
    add_indicators, detect_trend, check_touch,
    check_bounce, calc_signal_line, check_signal_trigger,
)

print = functools.partial(print, flush=True)

IS_PATH  = "data/QQQ_1Min_Polygon_2y_clean.csv"
OOS_PATH = "data/QQQ_1Min_Barchart_2y_2022-2024_clean.csv"

BASE_PARAMS = {
    "ema_fast": 20, "ema_slow": 50, "atr_period": 14,
    "touch_tol": 0.15, "touch_below_max": 0.5,
    "signal_offset": 0.05, "stop_buffer": 0.3,
    "lock1_rr": 0.3, "lock1_pct": 0.20,
    "chandelier_bars": 40, "chandelier_mult": 0.5,
    "signal_valid_bars": 3, "max_hold_bars": 180,
    "risk_pct": 0.01,
    # NOTE: With QQQ at ~$450, risking 1% of $100k ($1000) at $0.20 risk/share
    # requires ~5000 shares = ~$2.25M notional. The prod max_pos_pct=0.25
    # caps at ~55 shares, making ALL risk% levels produce identical shares.
    # To test dynamic sizing we need leverage. max_pos_pct=25.0 simulates
    # a typical futures/CFD margin environment (25:1). This lets risk_pct
    # be the binding constraint so different strategies produce different sizes.
    "max_pos_pct": 25.0,
    "no_entry_after": dt.time(15, 30),
    "force_close_at": dt.time(15, 58),
    "commission_per_share": 0.005,
    "daily_loss_r": 2.5,
    # MFE gate params
    "mfe_gate_bars": 3,
    "mfe_gate_threshold": 0.3,  # R-multiples
    "mfe_gate_tighten": 0.3,    # tighten stop to -0.3R
}


# ======================================================================
# SIZING STRATEGIES
# ======================================================================

def size_fixed(base_risk, trade_history, equity_history, atr_val, atr_median):
    """Baseline: fixed 1% risk."""
    return base_risk


def size_win_loss_adaptive(base_risk, trade_history, equity_history, atr_val, atr_median):
    """After a win, reduce to 0.5%. After 5+ consecutive losses, increase to 1.5%."""
    if not trade_history:
        return base_risk
    last_pnl = trade_history[-1]
    if last_pnl > 0:
        return 0.005  # 0.5% after a win
    # Count consecutive losses
    consec_losses = 0
    for pnl in reversed(trade_history):
        if pnl <= 0:
            consec_losses += 1
        else:
            break
    if consec_losses >= 5:
        return 0.015  # 1.5% after 5+ losses
    return base_risk


def size_streak_based(base_risk, trade_history, equity_history, atr_val, atr_median):
    """Scale risk from 0.5% to 2% based on current loss streak length.
    0 losses (after win) → 0.5%, 1-4 losses → linear scale, 5+ → 1.5%, 10+ → 2%."""
    if not trade_history:
        return base_risk
    consec_losses = 0
    for pnl in reversed(trade_history):
        if pnl <= 0:
            consec_losses += 1
        else:
            break
    if consec_losses == 0:
        return 0.005  # just won, reduce
    elif consec_losses < 5:
        # Linear scale: 1 loss → 0.75%, 2→1.0%, 3→1.25%, 4→1.5%
        return 0.005 + consec_losses * 0.0025
    elif consec_losses < 10:
        return 0.015
    elif consec_losses < 15:
        return 0.018
    else:
        return 0.020  # 15+ losses: PF=2.397, go big


def size_atr_based(base_risk, trade_history, equity_history, atr_val, atr_median):
    """Low ATR → 0.7% risk, high ATR → 1.3% risk."""
    if atr_median <= 0 or np.isnan(atr_val) or np.isnan(atr_median):
        return base_risk
    ratio = atr_val / atr_median
    if ratio < 0.8:
        return 0.007  # low vol
    elif ratio > 1.2:
        return 0.013  # high vol
    else:
        # Linear interpolation between 0.7% and 1.3%
        t = (ratio - 0.8) / 0.4
        return 0.007 + t * 0.006


def size_equity_curve(base_risk, trade_history, equity_history, atr_val, atr_median):
    """If equity < its 20-trade SMA, reduce risk to 0.5%."""
    if len(equity_history) < 20:
        return base_risk
    sma = np.mean(equity_history[-20:])
    if equity_history[-1] < sma:
        return 0.005  # equity below SMA, reduce risk
    return base_risk


def size_combined(base_risk, trade_history, equity_history, atr_val, atr_median):
    """Win-loss adaptive + ATR adaptive combined."""
    # Start from base
    risk = base_risk

    # Win/loss component
    if trade_history:
        last_pnl = trade_history[-1]
        if last_pnl > 0:
            risk *= 0.6  # reduce after win
        else:
            consec_losses = 0
            for pnl in reversed(trade_history):
                if pnl <= 0:
                    consec_losses += 1
                else:
                    break
            if consec_losses >= 5:
                risk *= 1.4  # increase after streak

    # ATR component
    if atr_median > 0 and not np.isnan(atr_val) and not np.isnan(atr_median):
        ratio = atr_val / atr_median
        if ratio > 1.2:
            risk *= 1.2  # high vol boost
        elif ratio < 0.8:
            risk *= 0.8  # low vol reduce

    return min(risk, 0.025)  # cap at 2.5%


def size_anti_martingale(base_risk, trade_history, equity_history, atr_val, atr_median):
    """Anti-martingale: DECREASE risk after wins (since win-after-win PF=0.411).
    After a win: 0.5%. After a loss: 1.2%. After 3+ losses: 1.5%."""
    if not trade_history:
        return base_risk
    last_pnl = trade_history[-1]
    if last_pnl > 0:
        return 0.005  # decrease after win (anti-traditional anti-martingale)
    consec_losses = 0
    for pnl in reversed(trade_history):
        if pnl <= 0:
            consec_losses += 1
        else:
            break
    if consec_losses >= 3:
        return 0.015
    return 0.012


def size_kelly(base_risk, trade_history, equity_history, atr_val, atr_median):
    """Kelly fraction: adjust risk based on recent 100-trade WR and avg W/L ratio.
    Kelly% = WR - (1-WR)/(avg_win/avg_loss). Capped between 0.3% and 2%."""
    lookback = 100
    if len(trade_history) < 20:
        return base_risk
    recent = trade_history[-lookback:]
    wins = [p for p in recent if p > 0]
    losses = [p for p in recent if p <= 0]
    if not wins or not losses:
        return base_risk
    wr = len(wins) / len(recent)
    avg_win = np.mean(wins)
    avg_loss = abs(np.mean(losses))
    if avg_loss == 0:
        return base_risk
    wl_ratio = avg_win / avg_loss
    if wl_ratio == 0:
        return base_risk
    kelly = wr - (1 - wr) / wl_ratio
    # Half-Kelly for safety
    half_kelly = kelly / 2
    # Clamp
    risk = max(0.003, min(0.020, half_kelly))
    return risk


STRATEGIES = {
    "0_Fixed_1pct":        size_fixed,
    "1_WinLoss_Adaptive":  size_win_loss_adaptive,
    "2_Streak_Based":      size_streak_based,
    "3_ATR_Based":         size_atr_based,
    "4_Equity_Curve":      size_equity_curve,
    "5_Combined_WL+ATR":   size_combined,
    "6_Anti_Martingale":   size_anti_martingale,
    "7_Kelly_Fraction":    size_kelly,
}


# ======================================================================
# BACKTEST ENGINE WITH DYNAMIC SIZING + MFE GATE
# ======================================================================

def run_backtest_dynamic(df, sizing_fn, capital=100_000, params=None,
                         compound=False):
    """
    Run full backtest with dynamic position sizing and 3-bar MFE gate.

    compound=False (default): size off initial capital — isolates sizing effect.
    compound=True: size off current equity — realistic but exponential divergence.

    Returns dict with equity curve and trade log.
    """
    p = {**BASE_PARAMS, **(params or {})}

    df = add_indicators(df, p)
    high = df["High"].values
    low = df["Low"].values
    close = df["Close"].values
    open_prices = df["Open"].values
    ema = df["ema20"].values
    ema_s = df["ema50"].values
    atr_v = df["atr"].values
    times = df.index.time
    dates = df.index.date
    n = len(df)

    # Pre-compute ATR median for ATR-based sizing
    atr_valid = atr_v[~np.isnan(atr_v)]
    atr_median = float(np.median(atr_valid)) if len(atr_valid) > 0 else 1.0

    comm = p["commission_per_share"]
    max_fwd = p["max_hold_bars"]
    mfe_gate_bars = p["mfe_gate_bars"]
    mfe_gate_threshold = p["mfe_gate_threshold"]
    mfe_gate_tighten = p["mfe_gate_tighten"]

    equity = capital
    sizing_base = capital  # for non-compounded sizing
    peak_equity = capital
    max_drawdown = 0.0
    equity_curve = []
    trade_log = []
    trade_pnl_history = []    # for sizing decisions
    equity_history = [capital]  # for equity curve sizing
    bar = max(p["ema_slow"], p["atr_period"]) + 5
    daily_loss_r = p.get("daily_loss_r", 999)
    daily_r_loss = 0.0
    current_date = None

    while bar < n - max_fwd - 5:
        a = atr_v[bar]
        if np.isnan(a) or a <= 0 or np.isnan(ema[bar]) or np.isnan(ema_s[bar]):
            bar += 1
            continue
        if times[bar] >= p["no_entry_after"]:
            bar += 1
            continue

        # Daily R limit
        d_date = dates[bar]
        if current_date != d_date:
            current_date = d_date
            daily_r_loss = 0.0
        if daily_r_loss >= daily_loss_r:
            bar += 1
            continue

        # --- Entry signal detection (shared module) ---
        trend = detect_trend(close[bar], ema[bar], ema_s[bar])
        if trend == 0:
            bar += 1
            continue

        if not check_touch(trend, low[bar], high[bar], ema[bar], a,
                           p["touch_tol"], p["touch_below_max"]):
            bar += 1
            continue

        bb = bar + 1
        if bb >= n:
            bar += 1
            continue
        if not check_bounce(trend, close[bb], high[bar], low[bar]):
            bar += 1
            continue

        sl = calc_signal_line(trend, high[bar], low[bar], a,
                              p["signal_offset"], p["stop_buffer"])
        if sl is None:
            bar += 1
            continue
        sig, stop, risk = sl

        entry_bar = check_signal_trigger(
            trend, sig, high, low, bb, n, p["signal_valid_bars"])
        if entry_bar < 0:
            bar += 1
            continue

        # --- Dynamic sizing ---
        dynamic_risk_pct = sizing_fn(
            p["risk_pct"], trade_pnl_history, equity_history, a, atr_median)
        # Use initial capital for sizing (non-compound) or current equity (compound)
        sz_eq = equity if compound else sizing_base

        # --- Bug #1 fix: entry bar stop check ---
        if trend == 1 and low[entry_bar] <= stop:
            shares_eb = max(1, int(sz_eq * dynamic_risk_pct / risk))
            if shares_eb * sig > sz_eq * p["max_pos_pct"]:
                shares_eb = max(1, int(sz_eq * p["max_pos_pct"] / sig))
            loss = shares_eb * (stop - sig) - shares_eb * comm * 2
            equity += loss
            r_lost = abs(loss / (shares_eb * risk)) if shares_eb * risk > 0 else 0
            daily_r_loss += r_lost
            trade_pnl_history.append(loss)
            equity_history.append(equity)
            equity_curve.append({"bar": entry_bar, "equity": equity, "date": dates[entry_bar]})
            trade_log.append({
                "entry_bar": entry_bar, "entry_time": df.index[entry_bar],
                "direction": "LONG", "entry_price": sig, "stop_price": stop,
                "risk": risk, "shares": shares_eb, "pnl": loss,
                "exit_reason": "entry_bar_stop", "lock_hit": False,
                "equity_after": equity, "risk_pct_used": dynamic_risk_pct,
            })
            peak_equity = max(peak_equity, equity)
            dd = (peak_equity - equity) / peak_equity * 100 if peak_equity > 0 else 0
            max_drawdown = max(max_drawdown, dd)
            bar = entry_bar + 1
            continue
        if trend == -1 and high[entry_bar] >= stop:
            shares_eb = max(1, int(sz_eq * dynamic_risk_pct / risk))
            if shares_eb * sig > sz_eq * p["max_pos_pct"]:
                shares_eb = max(1, int(sz_eq * p["max_pos_pct"] / sig))
            loss = shares_eb * (stop - sig) * trend - shares_eb * comm * 2
            equity += loss
            r_lost = abs(loss / (shares_eb * risk)) if shares_eb * risk > 0 else 0
            daily_r_loss += r_lost
            trade_pnl_history.append(loss)
            equity_history.append(equity)
            equity_curve.append({"bar": entry_bar, "equity": equity, "date": dates[entry_bar]})
            trade_log.append({
                "entry_bar": entry_bar, "entry_time": df.index[entry_bar],
                "direction": "SHORT", "entry_price": sig, "stop_price": stop,
                "risk": risk, "shares": shares_eb, "pnl": loss,
                "exit_reason": "entry_bar_stop", "lock_hit": False,
                "equity_after": equity, "risk_pct_used": dynamic_risk_pct,
            })
            peak_equity = max(peak_equity, equity)
            dd = (peak_equity - equity) / peak_equity * 100 if peak_equity > 0 else 0
            max_drawdown = max(max_drawdown, dd)
            bar = entry_bar + 1
            continue

        # --- Position sizing ---
        shares = max(1, int(sz_eq * dynamic_risk_pct / risk))
        if shares * sig > sz_eq * p["max_pos_pct"]:
            shares = max(1, int(sz_eq * p["max_pos_pct"] / sig))
        if equity < shares * risk or shares < 1:
            bar += 1
            continue

        # --- Execute trade ---
        lock_shares = max(1, int(shares * p["lock1_pct"]))
        runner_stop = stop
        lock_done = False
        trade_pnl = -shares * comm  # entry commission
        remaining = shares
        end_bar = entry_bar
        exit_reason = "timeout"
        chand_bars = p["chandelier_bars"]
        chand_mult = p["chandelier_mult"]
        mfe_gate_applied = False

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
                trade_pnl += remaining * (ep - sig) * trend - remaining * comm
                end_bar = bi
                exit_reason = "session_close"
                break

            # --- 3-bar MFE gate ---
            if k == mfe_gate_bars and not lock_done and not mfe_gate_applied:
                # Calculate MFE over first mfe_gate_bars bars
                mfe = 0.0
                for kb in range(1, mfe_gate_bars + 1):
                    bk = entry_bar + kb
                    if bk >= n:
                        break
                    if trend == 1:
                        mfe = max(mfe, (high[bk] - sig) / risk)
                    else:
                        mfe = max(mfe, (sig - low[bk]) / risk)
                if mfe < mfe_gate_threshold:
                    # Tighten stop to -0.3R from entry
                    if trend == 1:
                        new_stop = sig - mfe_gate_tighten * risk
                        runner_stop = max(runner_stop, new_stop)
                    else:
                        new_stop = sig + mfe_gate_tighten * risk
                        runner_stop = min(runner_stop, new_stop)
                    mfe_gate_applied = True

            # Stop check
            stopped = (trend == 1 and l <= runner_stop) or \
                      (trend == -1 and h >= runner_stop)

            # Bug #2 fix: same-bar stop/lock collision
            lock_hit_this_bar = False
            if not lock_done:
                lock_hit_this_bar = (trend == 1 and h >= sig + p["lock1_rr"] * risk) or \
                                    (trend == -1 and l <= sig - p["lock1_rr"] * risk)

            if stopped and lock_hit_this_bar and not lock_done:
                trade_pnl += remaining * (runner_stop - sig) * trend - remaining * comm
                end_bar = bi
                exit_reason = "stop"
                break

            if stopped:
                trade_pnl += remaining * (runner_stop - sig) * trend - remaining * comm
                end_bar = bi
                exit_reason = "stop" if not lock_done else "trail_stop"
                break

            # Lock: 20% at 0.3:1 R:R
            if lock_hit_this_bar and not lock_done:
                trade_pnl += lock_shares * p["lock1_rr"] * risk - lock_shares * comm
                remaining -= lock_shares
                lock_done = True
                if trend == 1:
                    runner_stop = max(runner_stop, sig)
                else:
                    runner_stop = min(runner_stop, sig)

            # Runner: Chandelier trail
            # Bug #3 fix: exclude current bar
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
            trade_pnl += remaining * (ep - sig) * trend - remaining * comm
            end_bar = min(entry_bar + max_fwd, n - 1)

        equity += trade_pnl
        peak_equity = max(peak_equity, equity)
        dd = (peak_equity - equity) / peak_equity * 100 if peak_equity > 0 else 0
        max_drawdown = max(max_drawdown, dd)

        if trade_pnl < 0:
            r_lost = abs(trade_pnl / (shares * risk)) if shares * risk > 0 else 0
            daily_r_loss += r_lost

        trade_pnl_history.append(trade_pnl)
        equity_history.append(equity)

        equity_curve.append({"bar": entry_bar, "equity": equity, "date": dates[entry_bar]})
        trade_log.append({
            "entry_bar": entry_bar,
            "entry_time": df.index[entry_bar],
            "direction": "LONG" if trend == 1 else "SHORT",
            "entry_price": sig,
            "stop_price": stop,
            "risk": risk,
            "shares": shares,
            "pnl": trade_pnl,
            "exit_reason": exit_reason,
            "lock_hit": lock_done,
            "equity_after": equity,
            "risk_pct_used": dynamic_risk_pct,
        })

        bar = end_bar + 1

    # --- Summary ---
    trades_df = pd.DataFrame(trade_log) if trade_log else pd.DataFrame()
    total = len(trades_df)
    wins = (trades_df["pnl"] > 0).sum() if total > 0 else 0
    losses = total - wins
    gross_won = trades_df.loc[trades_df["pnl"] > 0, "pnl"].sum() if wins > 0 else 0
    gross_lost = abs(trades_df.loc[trades_df["pnl"] <= 0, "pnl"].sum()) if losses > 0 else 0
    pf = gross_won / gross_lost if gross_lost > 0 else 0
    ret = (equity - capital) / capital * 100
    wr = wins / total * 100 if total > 0 else 0

    # Avg risk pct used
    avg_risk_pct = np.mean([t["risk_pct_used"] for t in trade_log]) * 100 if trade_log else 1.0

    return {
        "equity": equity,
        "return_pct": round(ret, 2),
        "pf": round(pf, 3),
        "trades": total,
        "wins": wins,
        "losses": losses,
        "win_rate": round(wr, 1),
        "max_drawdown": round(max_drawdown, 2),
        "avg_risk_pct": round(avg_risk_pct, 3),
        "trade_log": trades_df,
        "equity_curve": pd.DataFrame(equity_curve) if equity_curve else pd.DataFrame(),
    }


# ======================================================================
# MAIN
# ======================================================================

def run_experiment(df_is, df_oos, compound, label):
    """Run all strategies and return results DataFrame."""
    results = []
    for name, fn in STRATEGIES.items():
        print(f"  {name}...", end=" ")
        r_is  = run_backtest_dynamic(df_is, fn, compound=compound)
        r_oos = run_backtest_dynamic(df_oos, fn, compound=compound)
        results.append({
            "Strategy": name,
            "IS_Return%": r_is["return_pct"],
            "OOS_Return%": r_oos["return_pct"],
            "IS_Equity": round(r_is["equity"], 0),
            "OOS_Equity": round(r_oos["equity"], 0),
            "IS_MaxDD%": r_is["max_drawdown"],
            "OOS_MaxDD%": r_oos["max_drawdown"],
            "IS_PF": r_is["pf"],
            "OOS_PF": r_oos["pf"],
            "IS_Trades": r_is["trades"],
            "OOS_Trades": r_oos["trades"],
            "IS_WR%": r_is["win_rate"],
            "OOS_WR%": r_oos["win_rate"],
            "IS_AvgRisk%": r_is["avg_risk_pct"],
            "OOS_AvgRisk%": r_oos["avg_risk_pct"],
        })
        print(f"IS={r_is['return_pct']:+.1f}%  OOS={r_oos['return_pct']:+.1f}%  "
              f"DD(IS)={r_is['max_drawdown']:.2f}%  DD(OOS)={r_oos['max_drawdown']:.2f}%")
    return pd.DataFrame(results)


def print_table(rdf, label):
    """Print formatted comparison table."""
    rdf = rdf.sort_values("OOS_Return%", ascending=False).reset_index(drop=True)

    print()
    print("=" * 130)
    print(f"  {label}")
    print("=" * 130)
    hdr = (f"{'Strategy':<22} {'IS Ret%':>9} {'OOS Ret%':>9} {'IS Eq':>12} {'OOS Eq':>12} "
           f"{'IS DD%':>7} {'OOS DD%':>8} {'IS PF':>6} {'OOS PF':>7} "
           f"{'Trd':>5} {'AvgR%':>6}")
    print(hdr)
    print("-" * 130)

    for _, row in rdf.iterrows():
        line = (f"{row['Strategy']:<22} {row['IS_Return%']:>+9.2f} {row['OOS_Return%']:>+9.2f} "
                f"{row['IS_Equity']:>12,.0f} {row['OOS_Equity']:>12,.0f} "
                f"{row['IS_MaxDD%']:>7.2f} {row['OOS_MaxDD%']:>8.2f} "
                f"{row['IS_PF']:>6.3f} {row['OOS_PF']:>7.3f} "
                f"{row['IS_Trades']:>5} {row['IS_AvgRisk%']:>6.3f}")
        print(line)
    print("-" * 130)

    # Insights vs baseline
    baseline = rdf[rdf["Strategy"] == "0_Fixed_1pct"].iloc[0]
    print(f"\n  Baseline: IS={baseline['IS_Return%']:+.2f}%  OOS={baseline['OOS_Return%']:+.2f}%  "
          f"MaxDD: IS={baseline['IS_MaxDD%']:.2f}% OOS={baseline['OOS_MaxDD%']:.2f}%")
    print()

    for _, row in rdf.iterrows():
        if row["Strategy"] == "0_Fixed_1pct":
            continue
        is_delta = row["IS_Return%"] - baseline["IS_Return%"]
        oos_delta = row["OOS_Return%"] - baseline["OOS_Return%"]
        dd_is = row["IS_MaxDD%"] - baseline["IS_MaxDD%"]
        dd_oos = row["OOS_MaxDD%"] - baseline["OOS_MaxDD%"]

        if oos_delta > 0.01 and dd_oos <= 0.01:
            v = "++ BETTER return + LESS risk"
        elif oos_delta > 0.01 and dd_oos > 0.01:
            v = "+- Better return, MORE risk"
        elif oos_delta < -0.01 and dd_oos < -0.01:
            v = "-+ Less return, LESS risk"
        else:
            v = "-- Worse or no improvement"

        # Return/DD ratio
        rr_oos = row["OOS_Return%"] / row["OOS_MaxDD%"] if row["OOS_MaxDD%"] > 0 else 0
        rr_base = baseline["OOS_Return%"] / baseline["OOS_MaxDD%"] if baseline["OOS_MaxDD%"] > 0 else 0

        print(f"  {row['Strategy']:<22}: OOS {oos_delta:+.2f}%  DD {dd_oos:+.2f}%  "
              f"Ret/DD={rr_oos:.1f}x (base={rr_base:.1f}x)  {v}")


if __name__ == "__main__":
    print("=" * 80)
    print("DYNAMIC POSITION SIZING EXPERIMENT")
    print("=" * 80)
    print()
    print("Tests 7 adaptive sizing strategies against fixed 1% baseline.")
    print("Includes 3-bar MFE gate (tighten stop to -0.3R if MFE < 0.3R).")
    print("max_pos_pct=25x (leverage) so risk% is the binding constraint.")
    print()

    # Load data
    print("Loading data...")
    df_is = pd.read_csv(IS_PATH, index_col="timestamp", parse_dates=True)
    df_oos = pd.read_csv(OOS_PATH, index_col="timestamp", parse_dates=True)
    print(f"  IS:  {len(df_is):,} bars  ({df_is.index[0].date()} to {df_is.index[-1].date()})")
    print(f"  OOS: {len(df_oos):,} bars  ({df_oos.index[0].date()} to {df_oos.index[-1].date()})")
    print()

    # --- MODE 1: Non-compounded (size off initial capital) ---
    print("=" * 80)
    print("MODE 1: NON-COMPOUNDED (size off initial $100k)")
    print("  Isolates the sizing signal — no exponential amplification.")
    print("=" * 80)
    rdf1 = run_experiment(df_is, df_oos, compound=False, label="non-compound")
    print_table(rdf1, "NON-COMPOUNDED COMPARISON (sizing off initial capital)")

    # --- MODE 2: Compounded (size off current equity) ---
    print()
    print()
    print("=" * 80)
    print("MODE 2: COMPOUNDED (size off current equity, 25x leverage)")
    print("  Realistic growth simulation — differences get amplified.")
    print("=" * 80)
    rdf2 = run_experiment(df_is, df_oos, compound=True, label="compound")
    print_table(rdf2, "COMPOUNDED COMPARISON (sizing off current equity)")

    # --- Final Summary ---
    print()
    print()
    print("=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print()
    print("Non-compound mode isolates the pure sizing signal.")
    print("Compound mode shows real-world equity growth implications.")
    print()
    print("KEY: In non-compound mode, strategies that reduce risk after wins")
    print("(exploiting win-after-win PF=0.411 finding) should show BETTER")
    print("risk-adjusted returns even if total return is lower.")
    print()
    print("NOTE: The production strategy uses max_pos_pct=0.25 (no leverage),")
    print("which caps all positions at ~55 shares regardless of risk%.")
    print("Dynamic sizing only matters with leverage or smaller-priced instruments.")
    print()
    print("Done.")
