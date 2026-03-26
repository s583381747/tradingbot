"""
Path 1: Bounce Market Entry — enter at bounce bar close instead of stop-buy.

Hypothesis: The gap problem exists because we wait for a stop-buy trigger AFTER
the bounce already confirmed. By entering immediately at the bounce bar close
(market order), we eliminate the gap entirely.

Changes vs strategy_final.py:
  - Entry price = close[bounce_bar] (not signal line)
  - Stop = same (low[touch_bar] - 0.3*ATR for longs)
  - Risk = |entry_price - stop| (recalculated)
  - No signal line / signal trigger window — immediate entry
  - Lock/runner/chandelier: identical

Also runs the ORIGINAL strategy for side-by-side comparison.
"""
from __future__ import annotations
import functools, datetime as dt, sys, os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from entry_signal import add_indicators, detect_trend, check_touch, check_bounce

print = functools.partial(print, flush=True)

DATA_POLYGON = "data/QQQ_1Min_Polygon_2y_clean.csv"
DATA_BARCHART = "data/QQQ_1Min_Barchart_2y_2022-2024_clean.csv"

PARAMS = {
    "ema_fast": 20, "ema_slow": 50, "atr_period": 14,
    "touch_tol": 0.15, "touch_below_max": 0.5,
    "stop_buffer": 0.3,
    "lock1_rr": 0.3, "lock1_pct": 0.20,
    "chandelier_bars": 40, "chandelier_mult": 0.5,
    "max_hold_bars": 180,
    "risk_pct": 0.01, "max_pos_pct": 0.25,
    "no_entry_after": dt.time(15, 30),
    "force_close_at": dt.time(15, 58),
    "commission_per_share": 0.005,
    "daily_loss_r": 2.5,
}


def run_bounce_entry(df: pd.DataFrame, capital: float = 100_000) -> dict:
    """
    Path 1: Enter at bounce bar close (market order).
    Touch → Bounce confirmed → enter at close[bounce_bar].
    """
    p = PARAMS
    df = add_indicators(df, p)

    high = df["High"].values
    low = df["Low"].values
    close = df["Close"].values
    opn = df["Open"].values
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
    trade_log = []
    bar = max(p["ema_slow"], p["atr_period"]) + 5
    daily_r_loss = 0.0
    current_date = None

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
        if trend == 0:
            bar += 1; continue

        if not check_touch(trend, low[bar], high[bar], ema[bar], a,
                           p["touch_tol"], p["touch_below_max"]):
            bar += 1; continue

        bb = bar + 1
        if bb >= n:
            bar += 1; continue
        if not check_bounce(trend, close[bb], high[bar], low[bar]):
            bar += 1; continue

        # ─── PATH 1 CHANGE: entry at bounce bar close ───
        entry_price = close[bb]
        if trend == 1:
            stop = low[bar] - p["stop_buffer"] * a
        else:
            stop = high[bar] + p["stop_buffer"] * a
        risk = abs(entry_price - stop)
        if risk <= 0:
            bar += 1; continue

        # Verify entry makes sense (not already past stop)
        if trend == 1 and entry_price <= stop:
            bar += 1; continue
        if trend == -1 and entry_price >= stop:
            bar += 1; continue

        entry_bar = bb  # we enter on the bounce bar itself

        # ─── Position sizing (based on new risk) ───
        shares = max(1, int(equity * p["risk_pct"] / risk))
        if shares * entry_price > equity * p["max_pos_pct"]:
            shares = max(1, int(equity * p["max_pos_pct"] / entry_price))
        if equity < shares * risk or shares < 1:
            bar += 1; continue

        # ─── Execute trade (lock + chandelier, identical logic) ───
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
                ep = close[bi]
                trade_pnl += remaining * (ep - entry_price) * trend - remaining * comm
                end_bar = bi; exit_reason = "session_close"; break

            stopped = (trend == 1 and l <= runner_stop) or \
                      (trend == -1 and h >= runner_stop)

            lock_hit_bar = False
            if not lock_done:
                lock_target = entry_price + p["lock1_rr"] * risk * trend
                lock_hit_bar = (trend == 1 and h >= lock_target) or \
                               (trend == -1 and l <= lock_target)

            if stopped and lock_hit_bar and not lock_done:
                trade_pnl += remaining * (runner_stop - entry_price) * trend - remaining * comm
                end_bar = bi; exit_reason = "stop"; break

            if stopped:
                trade_pnl += remaining * (runner_stop - entry_price) * trend - remaining * comm
                end_bar = bi
                exit_reason = "stop" if not lock_done else "trail_stop"
                break

            if lock_hit_bar and not lock_done:
                trade_pnl += lock_shares * p["lock1_rr"] * risk - lock_shares * comm
                remaining -= lock_shares; lock_done = True
                if trend == 1: runner_stop = max(runner_stop, entry_price)
                else: runner_stop = min(runner_stop, entry_price)

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
            trade_pnl += remaining * (ep - entry_price) * trend - remaining * comm
            end_bar = min(entry_bar + max_fwd, n - 1)

        equity += trade_pnl
        peak_equity = max(peak_equity, equity)
        dd = (peak_equity - equity) / peak_equity * 100 if peak_equity > 0 else 0
        max_drawdown = max(max_drawdown, dd)
        if trade_pnl < 0:
            r_lost = abs(trade_pnl / (shares * risk)) if shares * risk > 0 else 0
            daily_r_loss += r_lost

        trade_log.append({
            "entry_bar": entry_bar, "entry_time": df.index[entry_bar],
            "direction": "LONG" if trend == 1 else "SHORT",
            "entry_price": entry_price, "stop_price": stop,
            "risk": risk, "shares": shares, "pnl": trade_pnl,
            "exit_reason": exit_reason, "lock_hit": lock_done,
        })
        bar = end_bar + 1

    return _summarize(trade_log, equity, capital, max_drawdown, df)


def run_original(df: pd.DataFrame, capital: float = 100_000) -> dict:
    """Original strategy_final.py logic with entry gap correction for comparison."""
    from entry_signal import calc_signal_line, check_signal_trigger
    p = PARAMS
    df = add_indicators(df, p)

    high = df["High"].values; low = df["Low"].values
    close = df["Close"].values; opn = df["Open"].values
    ema = df["ema20"].values; ema_s = df["ema50"].values
    atr_v = df["atr"].values; times = df.index.time; dates = df.index.date
    n = len(df); comm = p["commission_per_share"]; max_fwd = p["max_hold_bars"]

    equity = capital; peak_equity = capital; max_drawdown = 0.0
    trade_log = []; bar = max(p["ema_slow"], p["atr_period"]) + 5
    daily_r_loss = 0.0; current_date = None

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

        sl = calc_signal_line(trend, high[bar], low[bar], a, 0.05, p["stop_buffer"])
        if sl is None: bar += 1; continue
        sig, stop, risk = sl

        entry_bar = check_signal_trigger(trend, sig, high, low, bb, n, 3)
        if entry_bar < 0: bar += 1; continue

        # ─── CORRECTED ENTRY: use realistic fill price ───
        if trend == 1:
            entry_price = max(sig, opn[entry_bar])
        else:
            entry_price = min(sig, opn[entry_bar])
        # Recalculate risk with actual entry
        risk_actual = abs(entry_price - stop)
        if risk_actual <= 0: bar += 1; continue

        # Entry bar stop check with corrected price
        if trend == 1 and low[entry_bar] <= stop:
            shares_eb = max(1, int(equity * p["risk_pct"] / risk_actual))
            if shares_eb * entry_price > equity * p["max_pos_pct"]:
                shares_eb = max(1, int(equity * p["max_pos_pct"] / entry_price))
            loss = shares_eb * (stop - entry_price) - shares_eb * comm * 2
            equity += loss
            if shares_eb * risk_actual > 0: daily_r_loss += abs(loss) / (shares_eb * risk_actual)
            trade_log.append({"entry_bar": entry_bar, "entry_time": df.index[entry_bar],
                "direction": "LONG", "entry_price": entry_price, "stop_price": stop,
                "risk": risk_actual, "shares": shares_eb, "pnl": loss,
                "exit_reason": "entry_bar_stop", "lock_hit": False})
            bar = entry_bar + 1; continue
        if trend == -1 and high[entry_bar] >= stop:
            shares_eb = max(1, int(equity * p["risk_pct"] / risk_actual))
            if shares_eb * entry_price > equity * p["max_pos_pct"]:
                shares_eb = max(1, int(equity * p["max_pos_pct"] / entry_price))
            loss = shares_eb * (stop - entry_price) * trend - shares_eb * comm * 2
            equity += loss
            if shares_eb * risk_actual > 0: daily_r_loss += abs(loss) / (shares_eb * risk_actual)
            trade_log.append({"entry_bar": entry_bar, "entry_time": df.index[entry_bar],
                "direction": "SHORT", "entry_price": entry_price, "stop_price": stop,
                "risk": risk_actual, "shares": shares_eb, "pnl": loss,
                "exit_reason": "entry_bar_stop", "lock_hit": False})
            bar = entry_bar + 1; continue

        shares = max(1, int(equity * p["risk_pct"] / risk_actual))
        if shares * entry_price > equity * p["max_pos_pct"]:
            shares = max(1, int(equity * p["max_pos_pct"] / entry_price))
        if equity < shares * risk_actual or shares < 1: bar += 1; continue

        lock_shares = max(1, int(shares * p["lock1_pct"]))
        runner_stop = stop; lock_done = False
        trade_pnl = -shares * comm; remaining = shares
        end_bar = entry_bar; exit_reason = "timeout"

        for k in range(1, max_fwd + 1):
            bi = entry_bar + k
            if bi >= n: break
            h = high[bi]; l = low[bi]
            cur_atr = atr_v[bi] if not np.isnan(atr_v[bi]) else a

            if times[bi] >= p["force_close_at"]:
                trade_pnl += remaining * (close[bi] - entry_price) * trend - remaining * comm
                end_bar = bi; exit_reason = "session_close"; break

            stopped = (trend == 1 and l <= runner_stop) or (trend == -1 and h >= runner_stop)
            lock_hit_bar = False
            if not lock_done:
                lt = entry_price + p["lock1_rr"] * risk_actual * trend
                lock_hit_bar = (trend == 1 and h >= lt) or (trend == -1 and l <= lt)

            if stopped and lock_hit_bar and not lock_done:
                trade_pnl += remaining * (runner_stop - entry_price) * trend - remaining * comm
                end_bar = bi; exit_reason = "stop"; break
            if stopped:
                trade_pnl += remaining * (runner_stop - entry_price) * trend - remaining * comm
                end_bar = bi; exit_reason = "stop" if not lock_done else "trail_stop"; break
            if lock_hit_bar and not lock_done:
                trade_pnl += lock_shares * p["lock1_rr"] * risk_actual - lock_shares * comm
                remaining -= lock_shares; lock_done = True
                if trend == 1: runner_stop = max(runner_stop, entry_price)
                else: runner_stop = min(runner_stop, entry_price)
            if lock_done and k >= p["chandelier_bars"]:
                sk = max(1, k - p["chandelier_bars"] + 1)
                if trend == 1:
                    hh = max(high[entry_bar+kk] for kk in range(sk, k) if entry_bar+kk < n)
                    runner_stop = max(runner_stop, hh - p["chandelier_mult"] * cur_atr)
                else:
                    ll = min(low[entry_bar+kk] for kk in range(sk, k) if entry_bar+kk < n)
                    runner_stop = min(runner_stop, ll + p["chandelier_mult"] * cur_atr)
        else:
            ep = close[min(entry_bar + max_fwd, n - 1)]
            trade_pnl += remaining * (ep - entry_price) * trend - remaining * comm
            end_bar = min(entry_bar + max_fwd, n - 1)

        equity += trade_pnl
        peak_equity = max(peak_equity, equity)
        dd = (peak_equity - equity) / peak_equity * 100 if peak_equity > 0 else 0
        max_drawdown = max(max_drawdown, dd)
        if trade_pnl < 0 and shares * risk_actual > 0:
            daily_r_loss += abs(trade_pnl) / (shares * risk_actual)

        trade_log.append({"entry_bar": entry_bar, "entry_time": df.index[entry_bar],
            "direction": "LONG" if trend == 1 else "SHORT",
            "entry_price": entry_price, "stop_price": stop,
            "risk": risk_actual, "shares": shares, "pnl": trade_pnl,
            "exit_reason": exit_reason, "lock_hit": lock_done})
        bar = end_bar + 1

    return _summarize(trade_log, equity, capital, max_drawdown, df)


def _summarize(trade_log, equity, capital, max_drawdown, df):
    trades_df = pd.DataFrame(trade_log) if trade_log else pd.DataFrame()
    total = len(trades_df)
    wins = (trades_df["pnl"] > 0).sum() if total > 0 else 0
    losses = total - wins
    gw = trades_df.loc[trades_df["pnl"] > 0, "pnl"].sum() if wins > 0 else 0
    gl = abs(trades_df.loc[trades_df["pnl"] <= 0, "pnl"].sum()) if losses > 0 else 0
    pf = gw / gl if gl > 0 else 0
    ret = (equity - capital) / capital * 100
    wr = wins / total * 100 if total > 0 else 0
    days = df.index.normalize().nunique()
    return {
        "equity": equity, "return_pct": round(ret, 2), "pf": round(pf, 3),
        "trades": total, "trades_per_day": round(total / max(days, 1), 1),
        "wins": wins, "losses": losses, "win_rate": round(wr, 1),
        "gross_won": round(gw, 2), "gross_lost": round(gl, 2),
        "avg_win": round(gw / wins, 2) if wins > 0 else 0,
        "avg_loss": round(gl / losses, 2) if losses > 0 else 0,
        "max_drawdown": round(max_drawdown, 2),
        "trade_log": trades_df,
    }


def print_result(label, r):
    print(f"  {label}")
    print(f"    Return: {r['return_pct']:+.2f}%  PF: {r['pf']:.3f}  "
          f"Trades: {r['trades']} ({r['trades_per_day']}/day)")
    print(f"    WR: {r['win_rate']:.1f}% ({r['wins']}W/{r['losses']}L)  "
          f"MaxDD: {r['max_drawdown']:.2f}%")
    if r['wins'] > 0 and r['losses'] > 0:
        print(f"    AvgWin: ${r['avg_win']:.2f}  AvgLoss: ${r['avg_loss']:.2f}  "
              f"R:R: {r['avg_win']/r['avg_loss']:.2f}")
    # Long/short breakdown
    tl = r["trade_log"]
    if len(tl) > 0:
        for d in ["LONG", "SHORT"]:
            sub = tl[tl["direction"] == d]
            if len(sub) == 0: continue
            w = (sub["pnl"] > 0).sum()
            gw = sub.loc[sub["pnl"] > 0, "pnl"].sum()
            gl_d = abs(sub.loc[sub["pnl"] <= 0, "pnl"].sum())
            pf_d = gw / gl_d if gl_d > 0 else 0
            print(f"    {d:>5}: {len(sub)} trades, PF={pf_d:.3f}, "
                  f"WR={w/len(sub)*100:.1f}%, PnL=${sub['pnl'].sum():.0f}")


def main():
    os.chdir(os.path.join(os.path.dirname(__file__), "../.."))

    datasets = [
        ("Polygon 2024-2026 (IS)", DATA_POLYGON),
        ("Barchart 2022-2024 (OOS)", DATA_BARCHART),
    ]

    for ds_label, ds_path in datasets:
        if not os.path.exists(ds_path):
            print(f"SKIP: {ds_path} not found"); continue
        df = pd.read_csv(ds_path, index_col="timestamp", parse_dates=True)
        print(f"\n{'='*80}")
        print(f"  {ds_label} — {len(df):,} bars")
        print(f"{'='*80}")

        # Original with corrected entry
        print(f"\n  [BASELINE] Original + Corrected Entry (max(sig, Open)):")
        r_orig = run_original(df)
        print_result("", r_orig)

        # Path 1: Bounce entry
        print(f"\n  [PATH 1] Bounce Market Entry (enter at bounce close):")
        r_bounce = run_bounce_entry(df)
        print_result("", r_bounce)

        # Comparison
        print(f"\n  ─── Comparison ───")
        print(f"    PF:     {r_orig['pf']:.3f} → {r_bounce['pf']:.3f} "
              f"({(r_bounce['pf']-r_orig['pf'])/r_orig['pf']*100:+.1f}%)" if r_orig['pf'] > 0 else "")
        print(f"    Return: {r_orig['return_pct']:+.2f}% → {r_bounce['return_pct']:+.2f}%")
        print(f"    Trades: {r_orig['trades']} → {r_bounce['trades']}")

        # Walk-forward on Path 1
        print(f"\n  ─── Walk-Forward (Path 1) ───")
        mid = df.index[len(df)//2]
        for label, sub in [("H1", df[:mid]), ("H2", df[mid:])]:
            r = run_bounce_entry(sub)
            print(f"    {label}: PF={r['pf']:.3f} Ret={r['return_pct']:+.2f}% "
                  f"Trades={r['trades']} WR={r['win_rate']:.1f}%")


if __name__ == "__main__":
    main()
