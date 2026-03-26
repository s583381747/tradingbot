"""
Experiment: Re-entry after BE (breakeven) exit.

KEY FINDING: 82.4% of BE exits resume to +1R within 60 bars.
This means most BE exits are "false stops" — the trend continues.

Test: After a BE exit, if price touches EMA20 again within N bars,
re-enter with the same logic.

Sweep:
  - reentry_window: 5/10/20/30 bars after BE exit to look for re-entry
  - cooldown: 1/2/3/5 bars to wait before looking for re-entry
  - max_reentries: 1/2/3 per original signal

Base: Plan G strategy with 3-bar MFE gate.
"""
from __future__ import annotations
import datetime as dt
import functools
import time
import numpy as np
import pandas as pd

from entry_signal import (
    add_indicators,
    detect_trend,
    check_touch,
    check_bounce,
    calc_signal_line,
    check_signal_trigger,
)

print = functools.partial(print, flush=True)

# ══════════════════════════════════════════════════════════════════
# BASE PARAMS (Plan G)
# ══════════════════════════════════════════════════════════════════

BASE = {
    "ema_fast": 20,
    "ema_slow": 50,
    "atr_period": 14,
    "touch_tol": 0.15,
    "touch_below_max": 0.5,
    "signal_offset": 0.05,
    "stop_buffer": 0.3,
    "lock_rr": 0.1,
    "lock_pct": 0.05,
    "chand_bars": 40,
    "chand_mult": 0.5,
    "max_hold_bars": 180,
    "risk_pct": 0.01,
    "max_pos_pct": 0.25,
    "no_entry_after": dt.time(14, 0),
    "force_close_at": dt.time(15, 58),
    "commission_per_share": 0.005,
    "daily_loss_r": 2.5,
    "skip_after_win": 1,
    "signal_valid_bars": 3,
    # MFE gate
    "mfe_gate_bars": 3,
    "mfe_gate_r": 0.3,
    "mfe_gate_tighten_r": -0.3,
    # Re-entry params (defaults: off)
    "reentry_window": 0,       # 0 = disabled
    "reentry_cooldown": 1,
    "max_reentries": 0,
}

IS_PATH = "data/QQQ_1Min_Polygon_2y_clean.csv"
OOS_PATH = "data/QQQ_1Min_Barchart_2y_2022-2024_clean.csv"


# ══════════════════════════════════════════════════════════════════
# BACKTEST ENGINE (with re-entry logic)
# ══════════════════════════════════════════════════════════════════

def run_backtest(df: pd.DataFrame, capital: float = 100_000, params: dict | None = None) -> dict:
    """
    Run Plan G backtest with optional re-entry after BE exit.

    Re-entry logic: after a BE (breakeven) exit, if within `reentry_window`
    bars price touches EMA20 again and a new entry signal forms, re-enter.
    Wait `reentry_cooldown` bars before looking. Max `max_reentries` per
    original signal chain.
    """
    p = {**BASE, **(params or {})}

    df = add_indicators(df, p)
    high = df["High"].values
    low = df["Low"].values
    close = df["Close"].values
    open_ = df["Open"].values
    ema = df["ema20"].values
    ema_s = df["ema50"].values
    atr_v = df["atr"].values
    times = df.index.time
    dates = df.index.date
    n = len(df)

    comm = p["commission_per_share"]
    max_fwd = p["max_hold_bars"]
    mfe_gate_bars = p["mfe_gate_bars"]
    mfe_gate_r = p["mfe_gate_r"]
    mfe_gate_tighten_r = p["mfe_gate_tighten_r"]

    # Re-entry params
    reentry_window = p["reentry_window"]
    reentry_cooldown = p["reentry_cooldown"]
    max_reentries = p["max_reentries"]

    equity = capital
    peak_equity = capital
    max_drawdown = 0.0
    equity_curve = []
    trade_log = []
    bar = max(p["ema_slow"], p["atr_period"]) + 5
    daily_loss_r = p.get("daily_loss_r", 999)
    daily_r_loss = 0.0
    current_date = None

    # Pending re-entry state
    # When a BE exit occurs, we store info for potential re-entry
    pending_reentry = None  # dict or None

    def _execute_trade(entry_bar_idx, trend, sig, stop, risk, is_reentry=False, chain_count=0):
        """Execute a single trade from entry_bar_idx. Returns (end_bar, trade_dict, was_be)."""
        nonlocal equity, peak_equity, max_drawdown, daily_r_loss

        # Position sizing
        shares = max(1, int(equity * p["risk_pct"] / risk))
        if shares * sig > equity * p["max_pos_pct"]:
            shares = max(1, int(equity * p["max_pos_pct"] / sig))
        if equity < shares * risk or shares < 1:
            return entry_bar_idx, None, False

        # Bug #1: entry bar stop check
        if trend == 1 and low[entry_bar_idx] <= stop:
            loss = shares * (stop - sig) - shares * comm * 2
            equity += loss
            r_lost = abs(loss / (shares * risk)) if shares * risk > 0 else 0
            daily_r_loss += r_lost
            td = {
                "entry_bar": entry_bar_idx, "entry_time": df.index[entry_bar_idx],
                "direction": "LONG", "entry_price": sig, "stop_price": stop,
                "risk": risk, "shares": shares, "pnl": loss,
                "exit_reason": "entry_bar_stop", "lock_hit": False,
                "equity_after": equity, "is_reentry": is_reentry,
                "chain_count": chain_count,
            }
            return entry_bar_idx, td, False
        if trend == -1 and high[entry_bar_idx] >= stop:
            loss = shares * (stop - sig) * trend - shares * comm * 2
            equity += loss
            r_lost = abs(loss / (shares * risk)) if shares * risk > 0 else 0
            daily_r_loss += r_lost
            td = {
                "entry_bar": entry_bar_idx, "entry_time": df.index[entry_bar_idx],
                "direction": "SHORT", "entry_price": sig, "stop_price": stop,
                "risk": risk, "shares": shares, "pnl": loss,
                "exit_reason": "entry_bar_stop", "lock_hit": False,
                "equity_after": equity, "is_reentry": is_reentry,
                "chain_count": chain_count,
            }
            return entry_bar_idx, td, False

        # Execute trade
        lock_shares = max(1, int(shares * p["lock_pct"]))
        runner_stop = stop
        lock_done = False
        trade_pnl = -shares * comm  # entry commission
        remaining = shares
        end_bar = entry_bar_idx
        exit_reason = "timeout"
        chand_bars = p["chand_bars"]
        chand_mult = p["chand_mult"]
        was_be = False

        # MFE tracking for gate
        max_favorable = 0.0

        for k in range(1, max_fwd + 1):
            bi = entry_bar_idx + k
            if bi >= n:
                break
            h = high[bi]
            l = low[bi]
            cur_atr = atr_v[bi] if bi < n and not np.isnan(atr_v[bi]) else atr_v[entry_bar_idx]

            # Force close
            if times[bi] >= p["force_close_at"]:
                ep = close[bi]
                trade_pnl += remaining * (ep - sig) * trend - remaining * comm
                end_bar = bi
                exit_reason = "session_close"
                break

            # Track MFE
            if trend == 1:
                mfe_price = h - sig
            else:
                mfe_price = sig - l
            max_favorable = max(max_favorable, mfe_price)

            # 3-bar MFE gate: if MFE < 0.3R after 3 bars, tighten stop
            if k == mfe_gate_bars and risk > 0:
                mfe_r = max_favorable / risk
                if mfe_r < mfe_gate_r:
                    # Tighten stop to -0.3R from entry
                    tight_stop = sig - mfe_gate_tighten_r * risk * trend  # note: for long, sig + (-0.3)*risk*(-1) would be wrong
                    if trend == 1:
                        tight_stop = sig + mfe_gate_tighten_r * risk  # sig - 0.3*risk (since mfe_gate_tighten_r is -0.3)
                        runner_stop = max(runner_stop, tight_stop)
                    else:
                        tight_stop = sig - mfe_gate_tighten_r * risk  # sig + 0.3*risk
                        runner_stop = min(runner_stop, tight_stop)

            # Stop check
            stopped = (trend == 1 and l <= runner_stop) or \
                      (trend == -1 and h >= runner_stop)

            # Bug #2: same-bar stop/lock collision
            lock_hit_this_bar = False
            if not lock_done:
                lock_hit_this_bar = (trend == 1 and h >= sig + p["lock_rr"] * risk) or \
                                    (trend == -1 and l <= sig - p["lock_rr"] * risk)

            if stopped and lock_hit_this_bar and not lock_done:
                trade_pnl += remaining * (runner_stop - sig) * trend - remaining * comm
                end_bar = bi
                exit_reason = "stop"
                break

            if stopped:
                trade_pnl += remaining * (runner_stop - sig) * trend - remaining * comm
                end_bar = bi
                if lock_done:
                    exit_reason = "trail_stop"
                    # Check if this is a BE exit (stop ~ entry price)
                    if abs(runner_stop - sig) < risk * 0.15:
                        was_be = True
                        exit_reason = "be_stop"
                else:
                    exit_reason = "stop"
                break

            # Lock: partial at lock_rr R:R
            if lock_hit_this_bar and not lock_done:
                trade_pnl += lock_shares * p["lock_rr"] * risk - lock_shares * comm
                remaining -= lock_shares
                lock_done = True
                if trend == 1:
                    runner_stop = max(runner_stop, sig)  # move to BE
                else:
                    runner_stop = min(runner_stop, sig)

            # Runner: Chandelier trail (Bug #3 fix: exclude current bar)
            if lock_done and k >= chand_bars:
                sk = max(1, k - chand_bars + 1)
                if trend == 1:
                    hh = max(high[entry_bar_idx + kk]
                             for kk in range(sk, k) if entry_bar_idx + kk < n)
                    new_stop = hh - chand_mult * cur_atr
                    runner_stop = max(runner_stop, new_stop)
                else:
                    ll = min(low[entry_bar_idx + kk]
                             for kk in range(sk, k) if entry_bar_idx + kk < n)
                    new_stop = ll + chand_mult * cur_atr
                    runner_stop = min(runner_stop, new_stop)
        else:
            ep = close[min(entry_bar_idx + max_fwd, n - 1)]
            trade_pnl += remaining * (ep - sig) * trend - remaining * comm
            end_bar = min(entry_bar_idx + max_fwd, n - 1)

        equity += trade_pnl
        peak_equity = max(peak_equity, equity)
        dd = (peak_equity - equity) / peak_equity * 100 if peak_equity > 0 else 0
        max_drawdown = max(max_drawdown, dd)
        if trade_pnl < 0:
            r_lost = abs(trade_pnl / (shares * risk)) if shares * risk > 0 else 0
            daily_r_loss += r_lost

        # Compute R-multiple for this trade
        r_mult = trade_pnl / (shares * risk) if shares * risk > 0 else 0

        td = {
            "entry_bar": entry_bar_idx, "entry_time": df.index[entry_bar_idx],
            "direction": "LONG" if trend == 1 else "SHORT",
            "entry_price": sig, "stop_price": stop, "risk": risk,
            "shares": shares, "pnl": trade_pnl, "r_mult": r_mult,
            "exit_reason": exit_reason, "lock_hit": lock_done,
            "equity_after": equity, "is_reentry": is_reentry,
            "chain_count": chain_count,
        }
        return end_bar, td, was_be

    def _try_reentry(start_bar, window, cooldown, orig_trend, chain_count):
        """
        Scan for a re-entry signal after a BE exit.
        Look from start_bar+cooldown to start_bar+window for an EMA20 touch.
        Returns (entry_bar, trend, sig, stop, risk) or None.
        """
        scan_start = start_bar + cooldown
        scan_end = min(start_bar + window, n - 10)
        for sb in range(scan_start, scan_end):
            a = atr_v[sb]
            if np.isnan(a) or a <= 0 or np.isnan(ema[sb]) or np.isnan(ema_s[sb]):
                continue
            if times[sb] >= p["no_entry_after"]:
                continue

            trend = detect_trend(close[sb], ema[sb], ema_s[sb])
            if trend == 0:
                continue
            # Must be same direction as original trade
            if trend != orig_trend:
                continue

            if not check_touch(trend, low[sb], high[sb], ema[sb], a,
                               p["touch_tol"], p["touch_below_max"]):
                continue

            bb = sb + 1
            if bb >= n:
                continue
            if not check_bounce(trend, close[bb], high[sb], low[sb]):
                continue

            sl = calc_signal_line(trend, high[sb], low[sb], a,
                                  p["signal_offset"], p["stop_buffer"])
            if sl is None:
                continue
            sig, stop, risk = sl

            entry_bar = check_signal_trigger(
                trend, sig, high, low, bb, n, p["signal_valid_bars"])
            if entry_bar < 0:
                continue

            return entry_bar, trend, sig, stop, risk
        return None

    # ─── Main loop ───
    while bar < n - max_fwd - 5:
        # Check pending re-entry first
        if pending_reentry is not None:
            pr = pending_reentry
            pending_reentry = None  # consume it

            if pr["chain_count"] < max_reentries and reentry_window > 0:
                result = _try_reentry(
                    pr["exit_bar"], reentry_window, reentry_cooldown,
                    pr["trend"], pr["chain_count"],
                )
                if result is not None:
                    re_entry_bar, re_trend, re_sig, re_stop, re_risk = result
                    new_chain = pr["chain_count"] + 1
                    end_bar, td, was_be = _execute_trade(
                        re_entry_bar, re_trend, re_sig, re_stop, re_risk,
                        is_reentry=True, chain_count=new_chain,
                    )
                    if td is not None:
                        equity_curve.append({"bar": re_entry_bar, "equity": equity, "date": dates[re_entry_bar]})
                        trade_log.append(td)

                        # If this re-entry also hit BE, chain another re-entry
                        if was_be and new_chain < max_reentries:
                            pending_reentry = {
                                "exit_bar": end_bar,
                                "trend": re_trend,
                                "chain_count": new_chain,
                            }
                    bar = max(bar, end_bar + 1)
                    continue

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

        # Entry signal detection
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

        # Execute the trade
        end_bar, td, was_be = _execute_trade(entry_bar, trend, sig, stop, risk,
                                              is_reentry=False, chain_count=0)
        if td is not None:
            equity_curve.append({"bar": entry_bar, "equity": equity, "date": dates[entry_bar]})
            trade_log.append(td)

            # If BE exit, set up pending re-entry
            if was_be and max_reentries > 0 and reentry_window > 0:
                pending_reentry = {
                    "exit_bar": end_bar,
                    "trend": trend,
                    "chain_count": 0,
                }

        bar = end_bar + 1

    # ─── Summary ───
    trades_df = pd.DataFrame(trade_log) if trade_log else pd.DataFrame()
    total = len(trades_df)
    wins = (trades_df["pnl"] > 0).sum() if total > 0 else 0
    losses = total - wins
    gross_won = trades_df.loc[trades_df["pnl"] > 0, "pnl"].sum() if wins > 0 else 0
    gross_lost = abs(trades_df.loc[trades_df["pnl"] <= 0, "pnl"].sum()) if losses > 0 else 0
    pf = gross_won / gross_lost if gross_lost > 0 else 0
    ret = (equity - capital) / capital * 100
    wr = wins / total * 100 if total > 0 else 0

    # Count 5R+ trades
    big_r = 0
    if total > 0 and "r_mult" in trades_df.columns:
        big_r = (trades_df["r_mult"] >= 5.0).sum()

    # Count re-entries
    reentries = 0
    if total > 0 and "is_reentry" in trades_df.columns:
        reentries = trades_df["is_reentry"].sum()

    # Count BE exits
    be_exits = 0
    if total > 0:
        be_exits = (trades_df["exit_reason"] == "be_stop").sum()

    return {
        "equity": equity,
        "return_pct": round(ret, 2),
        "pf": round(pf, 3),
        "trades": total,
        "wins": wins,
        "losses": losses,
        "win_rate": round(wr, 1),
        "gross_won": round(gross_won, 2),
        "gross_lost": round(gross_lost, 2),
        "max_drawdown": round(max_drawdown, 2),
        "big_r_5plus": big_r,
        "reentries": reentries,
        "be_exits": be_exits,
        "trade_log": trades_df,
        "equity_curve": pd.DataFrame(equity_curve) if equity_curve else pd.DataFrame(),
    }


# ══════════════════════════════════════════════════════════════════
# EXPERIMENT CONFIGS
# ══════════════════════════════════════════════════════════════════

def build_configs():
    """Build all experiment configurations."""
    configs = {}

    # Baseline: no re-entry, no MFE gate
    configs["baseline_no_mfe"] = {
        "mfe_gate_bars": 999, "mfe_gate_r": 0.0,
        "reentry_window": 0, "max_reentries": 0,
    }

    # Baseline: with MFE gate only
    configs["baseline_mfe"] = {
        "reentry_window": 0, "max_reentries": 0,
    }

    # Sweep: re-entry window × cooldown × max_reentries
    for window in [5, 10, 20, 30]:
        for cooldown in [1, 2, 3, 5]:
            for max_re in [1, 2, 3]:
                name = f"w{window}_cd{cooldown}_re{max_re}"
                configs[name] = {
                    "reentry_window": window,
                    "reentry_cooldown": cooldown,
                    "max_reentries": max_re,
                }

    # Best-guess sweet spots (larger window, small cooldown, 1-2 re-entries)
    # These overlap with the sweep but are called out for clarity
    configs["sweet_w20_cd2_re1"] = {
        "reentry_window": 20, "reentry_cooldown": 2, "max_reentries": 1,
    }
    configs["sweet_w30_cd2_re2"] = {
        "reentry_window": 30, "reentry_cooldown": 2, "max_reentries": 2,
    }

    # Without MFE gate + re-entry (to isolate re-entry effect)
    for window in [10, 20, 30]:
        configs[f"nomfe_w{window}_cd2_re1"] = {
            "mfe_gate_bars": 999, "mfe_gate_r": 0.0,
            "reentry_window": window, "reentry_cooldown": 2, "max_reentries": 1,
        }

    return configs


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()

    print("=" * 120)
    print("EXP: RE-ENTRY AFTER BREAKEVEN EXIT")
    print("=" * 120)
    print(f"Base params: lock_rr={BASE['lock_rr']}, lock_pct={BASE['lock_pct']}, "
          f"chand={BASE['chand_bars']}/{BASE['chand_mult']}")
    print(f"MFE gate: {BASE['mfe_gate_bars']} bars, {BASE['mfe_gate_r']}R threshold, "
          f"tighten to {BASE['mfe_gate_tighten_r']}R")
    print()

    # Load data
    print("Loading data...")
    df_is = pd.read_csv(IS_PATH, index_col="timestamp", parse_dates=True)
    df_oos = pd.read_csv(OOS_PATH, index_col="timestamp", parse_dates=True)
    print(f"  IS:  {len(df_is):,} bars ({df_is.index[0].date()} to {df_is.index[-1].date()})")
    print(f"  OOS: {len(df_oos):,} bars ({df_oos.index[0].date()} to {df_oos.index[-1].date()})")
    print()

    configs = build_configs()
    print(f"Running {len(configs)} configurations on IS + OOS...")
    print()

    results = []
    total = len(configs)
    for i, (name, overrides) in enumerate(configs.items()):
        params = {**BASE, **overrides}  # keep BASE defaults, override specific ones
        r_is = run_backtest(df_is, params=params)
        r_oos = run_backtest(df_oos, params=params)
        results.append({
            "name": name,
            "is_pf": r_is["pf"],
            "oos_pf": r_oos["pf"],
            "is_trades": r_is["trades"],
            "oos_trades": r_oos["trades"],
            "is_ret": r_is["return_pct"],
            "oos_ret": r_oos["return_pct"],
            "is_wr": r_is["win_rate"],
            "oos_wr": r_oos["win_rate"],
            "is_5r": r_is["big_r_5plus"],
            "oos_5r": r_oos["big_r_5plus"],
            "is_reentries": r_is["reentries"],
            "oos_reentries": r_oos["reentries"],
            "is_be": r_is["be_exits"],
            "oos_be": r_oos["be_exits"],
            "is_dd": r_is["max_drawdown"],
            "oos_dd": r_oos["max_drawdown"],
        })
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{total}] {name:<30} IS PF={r_is['pf']:.3f} OOS PF={r_oos['pf']:.3f} "
                  f"IS T={r_is['trades']} OOS T={r_oos['trades']} "
                  f"IS re={r_is['reentries']} OOS re={r_oos['reentries']}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s\n")

    # ─── Results table ───
    rdf = pd.DataFrame(results)

    # Sort by average PF (IS + OOS)
    rdf["avg_pf"] = (rdf["is_pf"] + rdf["oos_pf"]) / 2
    rdf = rdf.sort_values("avg_pf", ascending=False)

    # Print baseline first
    print("=" * 140)
    print("BASELINES")
    print("=" * 140)
    hdr = (f"{'Config':<35} {'IS PF':>7} {'OOS PF':>7} {'Avg PF':>7} "
           f"{'IS T':>5} {'OOS T':>5} {'IS Ret%':>8} {'OOS Ret%':>8} "
           f"{'IS 5R+':>6} {'OOS 5R+':>6} {'IS RE':>5} {'OOS RE':>6} "
           f"{'IS BE':>5} {'OOS BE':>6} {'IS DD%':>7} {'OOS DD%':>7}")
    print(hdr)
    print("-" * 140)

    def print_row(r):
        flag = ""
        if r["avg_pf"] > 1.6:
            flag = " ***"
        elif r["avg_pf"] > 1.5:
            flag = " **"
        elif r["avg_pf"] > 1.4:
            flag = " *"
        print(f"{r['name']:<35} {r['is_pf']:>7.3f} {r['oos_pf']:>7.3f} {r['avg_pf']:>7.3f} "
              f"{r['is_trades']:>5} {r['oos_trades']:>5} {r['is_ret']:>+8.2f} {r['oos_ret']:>+8.2f} "
              f"{r['is_5r']:>6} {r['oos_5r']:>6} {r['is_reentries']:>5} {r['oos_reentries']:>6} "
              f"{r['is_be']:>5} {r['oos_be']:>6} {r['is_dd']:>7.2f} {r['oos_dd']:>7.2f}{flag}")

    baselines = rdf[rdf["name"].str.startswith("baseline")]
    for _, r in baselines.iterrows():
        print_row(r)

    print()
    print("=" * 140)
    print("TOP 20 BY AVERAGE PF (IS+OOS)")
    print("=" * 140)
    print(hdr)
    print("-" * 140)

    for _, r in rdf.head(20).iterrows():
        print_row(r)

    # Group analysis: by window size
    print()
    print("=" * 140)
    print("BY RE-ENTRY WINDOW (averaged across cooldown/max_reentry)")
    print("=" * 140)
    sweep_rows = rdf[~rdf["name"].str.startswith("baseline") &
                      ~rdf["name"].str.startswith("sweet") &
                      ~rdf["name"].str.startswith("nomfe")]
    if len(sweep_rows) > 0:
        # Extract window from name
        def get_window(name):
            parts = name.split("_")
            for p in parts:
                if p.startswith("w"):
                    try:
                        return int(p[1:])
                    except ValueError:
                        pass
            return 0

        sweep_rows = sweep_rows.copy()
        sweep_rows["window"] = sweep_rows["name"].apply(get_window)
        for w in sorted(sweep_rows["window"].unique()):
            g = sweep_rows[sweep_rows["window"] == w]
            print(f"\n  Window={w:>2} bars ({len(g)} configs):")
            print(f"    IS PF:  {g['is_pf'].mean():.3f} (min {g['is_pf'].min():.3f}, max {g['is_pf'].max():.3f})")
            print(f"    OOS PF: {g['oos_pf'].mean():.3f} (min {g['oos_pf'].min():.3f}, max {g['oos_pf'].max():.3f})")
            print(f"    Avg PF: {g['avg_pf'].mean():.3f}")
            print(f"    Trades: IS={g['is_trades'].mean():.0f} OOS={g['oos_trades'].mean():.0f}")
            print(f"    ReEntr: IS={g['is_reentries'].mean():.1f} OOS={g['oos_reentries'].mean():.1f}")

    # Group by max_reentries
    print()
    print("=" * 140)
    print("BY MAX RE-ENTRIES (averaged across window/cooldown)")
    print("=" * 140)
    if len(sweep_rows) > 0:
        def get_max_re(name):
            parts = name.split("_")
            for p in parts:
                if p.startswith("re"):
                    try:
                        return int(p[2:])
                    except ValueError:
                        pass
            return 0

        sweep_rows["max_re"] = sweep_rows["name"].apply(get_max_re)
        for mr in sorted(sweep_rows["max_re"].unique()):
            g = sweep_rows[sweep_rows["max_re"] == mr]
            print(f"\n  max_reentries={mr} ({len(g)} configs):")
            print(f"    IS PF:  {g['is_pf'].mean():.3f} OOS PF: {g['oos_pf'].mean():.3f} Avg: {g['avg_pf'].mean():.3f}")
            print(f"    Trades: IS={g['is_trades'].mean():.0f} OOS={g['oos_trades'].mean():.0f}")

    # Group by cooldown
    print()
    print("=" * 140)
    print("BY COOLDOWN (averaged across window/max_reentry)")
    print("=" * 140)
    if len(sweep_rows) > 0:
        def get_cooldown(name):
            parts = name.split("_")
            for p in parts:
                if p.startswith("cd"):
                    try:
                        return int(p[2:])
                    except ValueError:
                        pass
            return 0

        sweep_rows["cooldown"] = sweep_rows["name"].apply(get_cooldown)
        for cd in sorted(sweep_rows["cooldown"].unique()):
            g = sweep_rows[sweep_rows["cooldown"] == cd]
            print(f"\n  cooldown={cd} ({len(g)} configs):")
            print(f"    IS PF:  {g['is_pf'].mean():.3f} OOS PF: {g['oos_pf'].mean():.3f} Avg: {g['avg_pf'].mean():.3f}")
            print(f"    Trades: IS={g['is_trades'].mean():.0f} OOS={g['oos_trades'].mean():.0f}")

    # No-MFE comparison
    print()
    print("=" * 140)
    print("MFE GATE EFFECT (with vs without, re-entry enabled)")
    print("=" * 140)
    nomfe_rows = rdf[rdf["name"].str.startswith("nomfe")]
    if len(nomfe_rows) > 0:
        print(f"\n  {'Config':<35} {'IS PF':>7} {'OOS PF':>7} {'Avg PF':>7}")
        print(f"  {'-'*60}")
        for _, r in nomfe_rows.iterrows():
            # Find matching with-MFE config
            # nomfe_w10_cd2_re1 -> w10_cd2_re1
            match_name = r["name"].replace("nomfe_", "")
            match = rdf[rdf["name"] == match_name]
            if len(match) > 0:
                m = match.iloc[0]
                print(f"  {r['name']:<35} {r['is_pf']:>7.3f} {r['oos_pf']:>7.3f} {r['avg_pf']:>7.3f}  (no MFE gate)")
                print(f"  {m['name']:<35} {m['is_pf']:>7.3f} {m['oos_pf']:>7.3f} {m['avg_pf']:>7.3f}  (with MFE gate)")
                delta = m["avg_pf"] - r["avg_pf"]
                print(f"  {'':35} delta: {delta:>+7.3f}")
                print()

    # Final verdict
    print()
    print("=" * 140)
    print("VERDICT")
    print("=" * 140)
    baseline_mfe = rdf[rdf["name"] == "baseline_mfe"]
    if len(baseline_mfe) > 0:
        bl = baseline_mfe.iloc[0]
        print(f"\n  Baseline (MFE gate, no re-entry):")
        print(f"    IS PF={bl['is_pf']:.3f}, OOS PF={bl['oos_pf']:.3f}, "
              f"Avg PF={bl['avg_pf']:.3f}")
        print(f"    IS Trades={bl['is_trades']}, OOS Trades={bl['oos_trades']}")

    best = rdf.iloc[0]
    print(f"\n  Best config: {best['name']}")
    print(f"    IS PF={best['is_pf']:.3f}, OOS PF={best['oos_pf']:.3f}, "
          f"Avg PF={best['avg_pf']:.3f}")
    print(f"    IS Trades={best['is_trades']}, OOS Trades={best['oos_trades']}")
    print(f"    IS Return={best['is_ret']:+.2f}%, OOS Return={best['oos_ret']:+.2f}%")
    print(f"    IS 5R+={best['is_5r']}, OOS 5R+={best['oos_5r']}")
    print(f"    IS ReEntries={best['is_reentries']}, OOS ReEntries={best['oos_reentries']}")

    # Improvement vs baseline
    if len(baseline_mfe) > 0:
        bl = baseline_mfe.iloc[0]
        print(f"\n  vs Baseline:")
        print(f"    IS PF delta:  {best['is_pf'] - bl['is_pf']:+.3f}")
        print(f"    OOS PF delta: {best['oos_pf'] - bl['oos_pf']:+.3f}")
        print(f"    IS trade delta: {best['is_trades'] - bl['is_trades']:+d}")
        print(f"    OOS trade delta: {best['oos_trades'] - bl['oos_trades']:+d}")

    # Recommendation
    # Check if re-entry improves both IS and OOS
    improved_both = rdf[
        (rdf["is_pf"] > bl["is_pf"] if len(baseline_mfe) > 0 else True) &
        (rdf["oos_pf"] > bl["oos_pf"] if len(baseline_mfe) > 0 else True) &
        (~rdf["name"].str.startswith("baseline"))
    ]
    if len(improved_both) > 0:
        print(f"\n  {len(improved_both)} configs improve BOTH IS and OOS PF vs baseline:")
        for _, r in improved_both.head(5).iterrows():
            print(f"    {r['name']:<30} IS {r['is_pf']:.3f} OOS {r['oos_pf']:.3f}")
    else:
        print(f"\n  WARNING: No config improves both IS and OOS PF simultaneously.")
        print(f"  Re-entry may not add value, or the BE detection threshold needs tuning.")

    print(f"\nTotal time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
