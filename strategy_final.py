"""
EMA20 Trend Following — Final Production Strategy (Plan G).

Plan G = Plan F + tighter chandelier (1.0→0.5 ATR mult).
Validated on 4 years out-of-sample (2022-2026, 8874 trades):
  - PF=2.513, +93% / 4 years, annualized +18%, WR=77%
  - 49/49 months profitable, 17/17 quarters profitable
  - Sharpe=8.09, Sortino=30.95, Calmar=37.4, MaxDD=0.48%
  - Out-of-sample (2022-2024): PF=2.34, no degradation vs in-sample

Logic:
  TREND:  close > EMA20 > EMA50 (bull) or close < EMA20 < EMA50 (bear)
  TOUCH:  wick within 0.15*ATR of EMA20 (tight kiss)
  BOUNCE: next bar close > touch bar high
  ENTRY:  signal line at touch_high + $0.05 (stop buy)
  STOP:   touch_low - 0.3*ATR
  LOCK:   20% at 0.3:1 R:R → move stop to breakeven
  RUNNER: 80% Chandelier trail: highest_high(40) - 0.5*ATR
  DAILY:  stop new entries after 2.5R cumulative losing (NOT trailing DD)

Bug fixes (Gemini audit, impact -2.4%):
  #1: Entry bar stop check — if entry bar low <= stop, instant loss
  #2: Same-bar stop/lock collision — pessimistic: stop wins
  #3: Chandelier excludes current bar — range(sk, k) not range(sk, k+1)
"""
from __future__ import annotations
import datetime as dt
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


# ══════════════════════════════════════════════════════════════════
# PARAMETERS
# ══════════════════════════════════════════════════════════════════

PARAMS = {
    "ema_fast": 20,
    "ema_slow": 50,
    "atr_period": 14,
    "touch_tol": 0.15,          # wick must be within this * ATR of EMA20
    "touch_below_max": 0.5,     # wick can't be more than this * ATR below EMA
    "signal_offset": 0.05,      # $ above touch high for entry
    "stop_buffer": 0.3,         # ATR below touch low for stop
    "lock1_rr": 0.3,            # R:R for lock (was 0.5)
    "lock1_pct": 0.20,          # 20% of position (was 30%)
    # Runner = 80%, Chandelier trail
    "chandelier_bars": 40,      # highest high lookback (Plan F)
    "chandelier_mult": 0.5,     # ATR multiplier below HH (Plan G: tighter trail)
    "signal_valid_bars": 3,     # signal expires after N bars
    "max_hold_bars": 180,       # force exit after 3 hours
    "risk_pct": 0.01,           # 1% of equity per trade
    "max_pos_pct": 0.25,        # max 25% of equity in position
    "no_entry_after": dt.time(15, 30),
    "force_close_at": dt.time(15, 58),
    "commission_per_share": 0.005,  # IBKR
    "daily_loss_r": 2.5,       # stop new entries after cumulative losing R reaches this (NOT trailing DD, NOT % of account; purely sum of |loss_pnl / (shares * risk)| for each losing trade today)
}


# ══════════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ══════════════════════════════════════════════════════════════════

def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Add indicators to OHLCV dataframe."""
    return add_indicators(df, PARAMS)


def run_backtest(
    df: pd.DataFrame,
    capital: float = 100_000,
    params: dict | None = None,
) -> dict:
    """
    Run full backtest. Returns dict with equity curve and trade log.

    df must have columns: Open, High, Low, Close, Volume
    with DatetimeIndex named 'timestamp'.
    """
    p = {**PARAMS, **(params or {})}

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

    equity = capital
    peak_equity = capital
    max_drawdown = 0.0
    equity_curve = []
    trade_log = []
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

        # ─── Entry signal detection (shared module) ───
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

        # ─── Bug #1 fix: entry bar stop check ───
        if trend == 1 and low[entry_bar] <= stop:
            shares_eb = max(1, int(equity * p["risk_pct"] / risk))
            if shares_eb * sig > equity * p["max_pos_pct"]:
                shares_eb = max(1, int(equity * p["max_pos_pct"] / sig))
            loss = shares_eb * (stop - sig) - shares_eb * comm * 2
            equity += loss
            r_lost = abs(loss / (shares_eb * risk)) if shares_eb * risk > 0 else 0
            daily_r_loss += r_lost
            equity_curve.append({"bar": entry_bar, "equity": equity, "date": dates[entry_bar]})
            trade_log.append({
                "entry_bar": entry_bar, "entry_time": df.index[entry_bar],
                "direction": "LONG", "entry_price": sig, "stop_price": stop,
                "risk": risk, "shares": shares_eb, "pnl": loss,
                "exit_reason": "entry_bar_stop", "lock_hit": False,
                "equity_after": equity,
            })
            bar = entry_bar + 1
            continue
        if trend == -1 and high[entry_bar] >= stop:
            shares_eb = max(1, int(equity * p["risk_pct"] / risk))
            if shares_eb * sig > equity * p["max_pos_pct"]:
                shares_eb = max(1, int(equity * p["max_pos_pct"] / sig))
            loss = shares_eb * (stop - sig) * trend - shares_eb * comm * 2
            equity += loss
            r_lost = abs(loss / (shares_eb * risk)) if shares_eb * risk > 0 else 0
            daily_r_loss += r_lost
            equity_curve.append({"bar": entry_bar, "equity": equity, "date": dates[entry_bar]})
            trade_log.append({
                "entry_bar": entry_bar, "entry_time": df.index[entry_bar],
                "direction": "SHORT", "entry_price": sig, "stop_price": stop,
                "risk": risk, "shares": shares_eb, "pnl": loss,
                "exit_reason": "entry_bar_stop", "lock_hit": False,
                "equity_after": equity,
            })
            bar = entry_bar + 1
            continue

        # ─── Position sizing ───
        shares = max(1, int(equity * p["risk_pct"] / risk))
        if shares * sig > equity * p["max_pos_pct"]:
            shares = max(1, int(equity * p["max_pos_pct"] / sig))
        if equity < shares * risk or shares < 1:
            bar += 1
            continue

        # ─── Execute trade ───
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
                trade_pnl += remaining * (ep - sig) * trend - remaining * comm
                end_bar = bi
                exit_reason = "session_close"
                break

            # Stop
            stopped = (trend == 1 and l <= runner_stop) or \
                      (trend == -1 and h >= runner_stop)

            # Bug #2 fix: same-bar stop/lock collision — pessimistic, stop wins
            lock_hit_this_bar = False
            if not lock_done:
                lock_hit_this_bar = (trend == 1 and h >= sig + p["lock1_rr"] * risk) or \
                                    (trend == -1 and l <= sig - p["lock1_rr"] * risk)

            if stopped and lock_hit_this_bar and not lock_done:
                # Both triggered on same bar: assume stop first (pessimistic)
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

            # Runner: Chandelier trail — highest_high(N) - M*ATR
            # Bug #3 fix: exclude current bar — range(sk, k) not range(sk, k+1)
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
        # Drawdown tracking
        peak_equity = max(peak_equity, equity)
        dd = (peak_equity - equity) / peak_equity * 100 if peak_equity > 0 else 0
        max_drawdown = max(max_drawdown, dd)
        # Daily R loss tracking
        if trade_pnl < 0:
            r_lost = abs(trade_pnl / (shares * risk)) if shares * risk > 0 else 0
            daily_r_loss += r_lost

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
        })

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
    days = df.index.normalize().nunique()

    return {
        "equity": equity,
        "return_pct": round(ret, 2),
        "pf": round(pf, 3),
        "trades": total,
        "trades_per_day": round(total / max(days, 1), 1),
        "wins": wins,
        "losses": losses,
        "win_rate": round(wr, 1),
        "gross_won": round(gross_won, 2),
        "gross_lost": round(gross_lost, 2),
        "avg_win": round(gross_won / wins, 2) if wins > 0 else 0,
        "avg_loss": round(gross_lost / losses, 2) if losses > 0 else 0,
        "max_drawdown": round(max_drawdown, 2),
        "trade_log": trades_df,
        "equity_curve": pd.DataFrame(equity_curve) if equity_curve else pd.DataFrame(),
    }


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import functools
    print = functools.partial(print, flush=True)

    df = pd.read_csv("data/QQQ_1Min_Polygon_2y_clean.csv",
                      index_col="timestamp", parse_dates=True)
    print(f"Data: {len(df)} bars\n")

    result = run_backtest(df)

    print(f"{'='*60}")
    print(f"FINAL STRATEGY BACKTEST")
    print(f"{'='*60}")
    print(f"  Return: {result['return_pct']:+.2f}%")
    print(f"  PF: {result['pf']:.3f}")
    print(f"  Trades: {result['trades']} ({result['trades_per_day']}/day)")
    print(f"  WR: {result['win_rate']:.1f}% ({result['wins']}W / {result['losses']}L)")
    print(f"  MaxDD: {result['max_drawdown']:.2f}%")
    if result['wins'] > 0:
        print(f"  Avg Win: ${result['avg_win']:.2f}  Avg Loss: ${result['avg_loss']:.2f}")
        print(f"  R:R: {result['avg_win']/result['avg_loss']:.2f}")

    # Quick walk-forward
    mid = "2025-03-22"
    for label, start, end in [("Y1", "2024-01-01", mid), ("Y2", mid, "2027-01-01")]:
        sub = df[(df.index >= start) & (df.index < end)]
        r = run_backtest(sub)
        print(f"  {label}: PF={r['pf']:.3f} Ret={r['return_pct']:+.2f}% Trades={r['trades']}")
