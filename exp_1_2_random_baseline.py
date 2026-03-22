"""
Experiment 1.2: Random Entry Baseline Test

Tests whether the strategy's edge comes from entry logic or exit logic.
Replaces entry signals with random entries, keeps exact same exit rules,
and runs 500 iterations.

If random entry + same exits is also profitable, then entry logic adds no value.

Uses a pure-Python simulation engine (no backtrader) for speed.
Indicators are precomputed once with pandas; the bar loop runs in plain Python.
"""

from __future__ import annotations

import math
import random
import sys
import time

import numpy as np
import pandas as pd

# ── Constants ──────────────────────────────────────────────────
DATA_PATH = "data/QQQ_1Min_2025-09-21_2026-03-21.csv"
CASH = 100_000
COMMISSION = 0.001        # 0.1% per trade (applied on entry and exit)
NUM_ITERATIONS = 500

# Score weights (from prepare.py)
W_SHARPE = 0.30
W_PROFIT_FACTOR = 0.25
W_RETURN = 0.20
W_WIN_RATE = 0.10
W_DRAWDOWN = 0.15

# Real strategy reference values (from running prepare.py)
REAL_SCORE = 0.4558
REAL_PF = 1.21
REAL_RETURN = 0.62

# ── Strategy hyperparameters (from strategy.py) ───────────────
EMA_PERIOD = 20
EMA_SLOW_PERIOD = 50
ATR_PERIOD = 14
EMA_SLOPE_PERIOD = 5
EMA_SLOPE_THRESHOLD = 0.012

INITIAL_STOP_ATR_MULT = 2.5

CANDLE_TRAIL_AFTER_BARS = 0
CANDLE_TRAIL_OFFSET = 0.0
EMA_TRAIL_DISTANCE = 0.0
EMA_TRAIL_OFFSET = 6.0

# TP multipliers are extremely high (100x ATR, 200x ATR) -- effectively disabled
TP1_ATR_MULT = 100.0
TP2_ATR_MULT = 200.0
TP1_PCT = 0.15
TP2_PCT = 0.25

RISK_PCT = 0.01
MAX_POSITION_PCT = 0.25

LOSERS_MAX_BARS = 45

# Entry probability: ~0.4 trades/day / ~390 bars per trading day
ENTRY_PROB_PER_BAR = 0.4 / 390.0


# ── Precompute indicators ─────────────────────────────────────

def precompute_indicators(df):
    """Compute EMA, ATR, EMA slope arrays once for all iterations."""
    close = df["Close"].values.astype(np.float64)
    high = df["High"].values.astype(np.float64)
    low = df["Low"].values.astype(np.float64)
    n = len(close)

    # EMA-20
    ema = np.empty(n, dtype=np.float64)
    alpha = 2.0 / (EMA_PERIOD + 1)
    ema[0] = close[0]
    for i in range(1, n):
        ema[i] = alpha * close[i] + (1 - alpha) * ema[i - 1]

    # EMA-50
    ema_slow = np.empty(n, dtype=np.float64)
    alpha_s = 2.0 / (EMA_SLOW_PERIOD + 1)
    ema_slow[0] = close[0]
    for i in range(1, n):
        ema_slow[i] = alpha_s * close[i] + (1 - alpha_s) * ema_slow[i - 1]

    # ATR-14 (Wilder's smoothed)
    tr = np.empty(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                     abs(high[i] - close[i - 1]),
                     abs(low[i] - close[i - 1]))
    atr = np.empty(n, dtype=np.float64)
    atr[:ATR_PERIOD] = 0.0
    atr[ATR_PERIOD - 1] = np.mean(tr[:ATR_PERIOD])
    for i in range(ATR_PERIOD, n):
        atr[i] = (atr[i - 1] * (ATR_PERIOD - 1) + tr[i]) / ATR_PERIOD

    # EMA slope (normalized, % per bar)
    ema_slope = np.zeros(n, dtype=np.float64)
    for i in range(EMA_SLOPE_PERIOD, n):
        raw = (ema[i] - ema[i - EMA_SLOPE_PERIOD]) / EMA_SLOPE_PERIOD
        if close[i] > 0:
            ema_slope[i] = raw / close[i] * 100

    # Time info
    timestamps = df.index
    hours = np.array([t.hour for t in timestamps], dtype=np.int32)
    minutes = np.array([t.minute for t in timestamps], dtype=np.int32)
    dates = np.array([t.date() for t in timestamps])

    return {
        "close": close,
        "high": high,
        "low": low,
        "ema": ema,
        "ema_slow": ema_slow,
        "atr": atr,
        "ema_slope": ema_slope,
        "hours": hours,
        "minutes": minutes,
        "dates": dates,
        "n": n,
    }


# ── Pure-Python bar-by-bar simulator ──────────────────────────

def simulate(ind, seed, trend_biased):
    """
    Simulate the random-entry strategy over all bars.

    Equity model: equity = CASH + sum(realized_pnls) + unrealized_pnl.
    Each trade's PnL includes commission on both entry and exit.
    Position sizing uses current equity at time of entry.
    """
    rng = random.Random(seed)

    close = ind["close"]
    high = ind["high"]
    low = ind["low"]
    ema = ind["ema"]
    atr = ind["atr"]
    ema_slope = ind["ema_slope"]
    hours = ind["hours"]
    minutes = ind["minutes"]
    dates = ind["dates"]
    n = ind["n"]

    # Realized PnL accumulator
    realized_pnl = 0.0
    trade_pnls = []  # net PnL per closed trade

    # Drawdown tracking
    peak_equity = float(CASH)
    max_dd_pct = 0.0

    # Daily equity tracking for Sharpe
    daily_returns = []
    prev_day = None
    day_start_equity = float(CASH)

    # Position state
    in_position = False
    direction = 0         # +1 long, -1 short
    entry_price = 0.0
    entry_bar = 0
    orig_size = 0         # original entry size (for commission calc)
    remaining_size = 0    # current shares held
    stop_price = 0.0
    trail_phase = 0
    profitable_bars = 0
    entry_comm = 0.0      # commission paid at entry

    # TP state
    tp1_price = 0.0
    tp2_price = 0.0
    tp1_done = False
    tp2_done = False
    tp1_size = 0
    tp2_size = 0

    # Minimum bar index for valid indicators
    min_bar = max(EMA_SLOW_PERIOD, ATR_PERIOD, EMA_SLOPE_PERIOD) + 1

    def close_position(exit_price, shares_to_close):
        """Close position (or partial), return net PnL for the closed portion."""
        nonlocal remaining_size
        if direction == 1:
            gross = (exit_price - entry_price) * shares_to_close
        else:
            gross = (entry_price - exit_price) * shares_to_close
        # Commission: proportional share of entry comm + exit comm
        exit_comm = exit_price * shares_to_close * COMMISSION
        # Entry commission proportional to shares closed
        entry_comm_share = entry_price * shares_to_close * COMMISSION
        net = gross - entry_comm_share - exit_comm
        remaining_size -= shares_to_close
        return net

    def update_equity_tracking(equity):
        nonlocal peak_equity, max_dd_pct
        if equity > peak_equity:
            peak_equity = equity
        if peak_equity > 0:
            dd = (peak_equity - equity) / peak_equity * 100
            if dd > max_dd_pct:
                max_dd_pct = dd

    for i in range(min_bar, n):
        h = hours[i]
        mi = minutes[i]
        cur_date = dates[i]

        # Compute current equity (for daily tracking)
        if in_position:
            if direction == 1:
                unrealized = (close[i] - entry_price) * remaining_size
            else:
                unrealized = (entry_price - close[i]) * remaining_size
            equity = CASH + realized_pnl + unrealized
        else:
            equity = CASH + realized_pnl

        # Daily tracking
        if cur_date != prev_day:
            if prev_day is not None:
                day_ret = (equity - day_start_equity) / day_start_equity if day_start_equity > 0 else 0.0
                daily_returns.append(day_ret)
            prev_day = cur_date
            day_start_equity = equity

        update_equity_tracking(equity)

        atr_val = atr[i] if atr[i] > 0 else 0.01

        # Force close check (15:58 or later)
        past_force_close = (h > 15) or (h == 15 and mi >= 58)

        if past_force_close and in_position:
            net = close_position(close[i], remaining_size)
            trade_pnls.append(net)
            realized_pnl += net
            in_position = False
            direction = 0
            equity = CASH + realized_pnl
            update_equity_tracking(equity)
            continue

        if in_position:
            # ── Manage position (identical exit logic) ──
            bars_in_trade = i - entry_bar

            # 1. Time-based exit for losers
            if bars_in_trade > LOSERS_MAX_BARS:
                is_losing = (
                    (direction == 1 and close[i] < entry_price)
                    or (direction == -1 and close[i] > entry_price)
                )
                if is_losing:
                    net = close_position(close[i], remaining_size)
                    trade_pnls.append(net)
                    realized_pnl += net
                    in_position = False
                    direction = 0
                    continue

            # 2. Check stop hit
            stop_hit = False
            if direction == 1 and low[i] <= stop_price:
                stop_hit = True
            elif direction == -1 and high[i] >= stop_price:
                stop_hit = True

            if stop_hit:
                # In backtrader, the strategy calls self.close() which executes
                # at the close price of the current bar (market order).
                # The stop is a mental stop, not a broker stop-loss order.
                exit_price = close[i]
                net = close_position(exit_price, remaining_size)
                trade_pnls.append(net)
                realized_pnl += net
                in_position = False
                direction = 0
                continue

            # 3. Take profit checks (TP at 100x ATR -- effectively never)
            if not tp1_done and tp1_size > 0:
                hit = (direction == 1 and high[i] >= tp1_price) or \
                      (direction == -1 and low[i] <= tp1_price)
                if hit:
                    tp_sell = min(tp1_size, remaining_size - 1)
                    if tp_sell > 0:
                        # Partial close at close price (backtrader behavior)
                        net = close_position(close[i], tp_sell)
                        # TP partials are part of the same trade in BT
                        # We'll accumulate and record when fully closed
                        realized_pnl += net
                        tp1_done = True
                        if direction == 1:
                            stop_price = max(stop_price, entry_price)
                        else:
                            stop_price = min(stop_price, entry_price)

            if tp1_done and not tp2_done and tp2_size > 0:
                hit = (direction == 1 and high[i] >= tp2_price) or \
                      (direction == -1 and low[i] <= tp2_price)
                if hit:
                    tp_sell = min(tp2_size, remaining_size - 1)
                    if tp_sell > 0:
                        net = close_position(close[i], tp_sell)
                        realized_pnl += net
                        tp2_done = True

            if remaining_size <= 0:
                in_position = False
                direction = 0
                continue

            # 4. Count profitable bars
            if direction == 1 and close[i] > entry_price:
                profitable_bars += 1
            elif direction == -1 and close[i] < entry_price:
                profitable_bars += 1

            # 5. Update trailing stop
            if trail_phase == 0:
                if profitable_bars >= CANDLE_TRAIL_AFTER_BARS:
                    trail_phase = 1

            if trail_phase >= 1:
                dist_from_ema = abs(close[i] - ema[i])
                if dist_from_ema > EMA_TRAIL_DISTANCE * atr_val:
                    trail_phase = 2
                elif trail_phase == 2:
                    trail_phase = 1

            if direction == 1:
                ema_stop = ema[i] - EMA_TRAIL_OFFSET * atr_val
            else:
                ema_stop = ema[i] + EMA_TRAIL_OFFSET * atr_val

            new_stop = ema_stop

            if trail_phase == 1 and i >= 1:
                if direction == 1:
                    candle_stop = low[i - 1] - CANDLE_TRAIL_OFFSET
                    new_stop = max(ema_stop, candle_stop)
                else:
                    candle_stop = high[i - 1] + CANDLE_TRAIL_OFFSET
                    new_stop = min(ema_stop, candle_stop)

            if direction == 1 and new_stop > stop_price:
                stop_price = new_stop
            elif direction == -1 and new_stop < stop_price:
                stop_price = new_stop

            continue  # done managing position

        # ── Not in position: check for random entry ──
        past_entry_cutoff = (h > 15) or (h == 15 and mi >= 30)
        if past_entry_cutoff:
            continue

        # Random entry decision
        if rng.random() < ENTRY_PROB_PER_BAR:
            # Decide direction
            if trend_biased:
                slope = ema_slope[i]
                if abs(slope) < EMA_SLOPE_THRESHOLD * 0.5:
                    d = 1 if rng.random() < 0.5 else -1
                else:
                    trend_dir = 1 if slope > 0 else -1
                    d = trend_dir if rng.random() < 0.70 else -trend_dir
            else:
                d = 1 if rng.random() < 0.5 else -1

            # Calculate initial stop (ATR-based)
            ep = close[i]
            offset = atr_val * INITIAL_STOP_ATR_MULT
            if d == 1:
                sp = ep - offset
            else:
                sp = ep + offset

            # Position sizing (1% risk, based on current equity)
            cur_equity = CASH + realized_pnl
            risk_per_share = abs(ep - sp)
            if risk_per_share <= 0:
                continue
            risk_amount = cur_equity * RISK_PCT
            size_by_risk = int(risk_amount / risk_per_share)
            max_by_capital = int(cur_equity * MAX_POSITION_PCT / ep)
            sz = max(1, min(size_by_risk, max_by_capital))

            if sz <= 0 or cur_equity <= 0:
                continue

            # Enter position
            in_position = True
            direction = d
            entry_price = ep
            entry_bar = i
            orig_size = sz
            remaining_size = sz
            stop_price = sp
            trail_phase = 0
            profitable_bars = 0

            # TP levels (100x/200x ATR -- essentially disabled)
            if d == 1:
                tp1_price = ep + TP1_ATR_MULT * atr_val
                tp2_price = ep + TP2_ATR_MULT * atr_val
            else:
                tp1_price = ep - TP1_ATR_MULT * atr_val
                tp2_price = ep - TP2_ATR_MULT * atr_val
            tp1_done = False
            tp2_done = False
            tp1_size = max(1, int(sz * TP1_PCT))
            tp2_size = max(1, int(sz * TP2_PCT))
            if tp1_size + tp2_size >= sz:
                tp2_size = max(0, sz - tp1_size - 1)
                if tp1_size + tp2_size >= sz:
                    tp1_size = max(0, sz - 1)
                    tp2_size = 0

    # End: close any open position at last bar
    if in_position:
        net = close_position(close[n - 1], remaining_size)
        trade_pnls.append(net)
        realized_pnl += net

    # Final daily return
    final_equity = CASH + realized_pnl
    if prev_day is not None:
        day_ret = (final_equity - day_start_equity) / day_start_equity if day_start_equity > 0 else 0.0
        daily_returns.append(day_ret)

    # Compute metrics
    total_trades = len(trade_pnls)
    if total_trades == 0:
        return {
            "total_return": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "total_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "score": -10.0,
        }

    won = sum(1 for p in trade_pnls if p > 0)
    win_rate = won / total_trades * 100

    gross_profit = sum(p for p in trade_pnls if p > 0)
    gross_loss = abs(sum(p for p in trade_pnls if p < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

    total_return = (final_equity - CASH) / CASH * 100

    # Sharpe ratio (annualized from daily returns, rf=5%)
    if len(daily_returns) > 1:
        dr = np.array(daily_returns)
        rf_daily = 0.05 / 252
        excess = dr - rf_daily
        if np.std(excess) > 0:
            sharpe = float((np.mean(excess) / np.std(excess)) * np.sqrt(252))
        else:
            sharpe = 0.0
    else:
        sharpe = 0.0

    metrics = {
        "total_return": round(total_return, 4),
        "sharpe": round(sharpe, 4),
        "max_drawdown": round(max_dd_pct, 4),
        "total_trades": total_trades,
        "win_rate": round(win_rate, 2),
        "profit_factor": round(profit_factor, 4),
    }
    metrics["score"] = compute_score(metrics)
    return metrics


# ── Scoring (from prepare.py) ──────────────────────────────────

def compute_score(m):
    trades = m["total_trades"]
    if trades < 5:
        return -10.0

    sharpe_norm = max(0.0, min(1.0, (m["sharpe"] + 2) / 5.0))
    pf_norm = max(0.0, min(1.0, m["profit_factor"] / 3.0))
    ret_norm = max(0.0, min(1.0, (m["total_return"] + 20) / 70.0))
    wr_norm = max(0.0, min(1.0, m["win_rate"] / 100.0))
    dd_norm = max(0.0, min(1.0, 1.0 - m["max_drawdown"] / 30.0))

    raw = (
        W_SHARPE * sharpe_norm
        + W_PROFIT_FACTOR * pf_norm
        + W_RETURN * ret_norm
        + W_WIN_RATE * wr_norm
        + W_DRAWDOWN * dd_norm
    )

    if trades < 20:
        mult = 0.2
    elif trades < 50:
        mult = 0.6
    elif trades <= 500:
        mult = 1.0
    else:
        mult = 0.8

    return round(raw * mult, 6)


# ── Main ──────────────────────────────────────────────────────

def main():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH, index_col="timestamp", parse_dates=True)
    for col in ("Open", "High", "Low", "Close", "Volume"):
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
    print(f"Bars: {len(df)}")
    print(f"Date range: {df.index[0]} -> {df.index[-1]}")
    print()

    print("Precomputing indicators...")
    ind = precompute_indicators(df)
    print("Done.")
    print()

    results_5050 = []
    results_trend = []

    # ── Phase 1: 50/50 Random ──
    print(f"=== Phase 1: 50/50 Random Entry ({NUM_ITERATIONS} iterations) ===")
    t0 = time.time()
    for i in range(NUM_ITERATIONS):
        m = simulate(ind, seed=i, trend_biased=False)
        results_5050.append(m)
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            avg_time = elapsed / (i + 1)
            eta = avg_time * (NUM_ITERATIONS - i - 1)
            print(f"  [{i+1}/{NUM_ITERATIONS}] elapsed={elapsed:.1f}s  ETA={eta:.0f}s  "
                  f"last: score={m['score']:.4f} trades={m['total_trades']} "
                  f"PF={m['profit_factor']:.2f} ret={m['total_return']:+.2f}%")

    t1 = time.time()
    print(f"  Phase 1 complete in {t1 - t0:.1f}s")
    print()

    # ── Phase 2: Trend-Biased Random ──
    print(f"=== Phase 2: Trend-Biased Random Entry ({NUM_ITERATIONS} iterations) ===")
    t2 = time.time()
    for i in range(NUM_ITERATIONS):
        m = simulate(ind, seed=i + 10000, trend_biased=True)
        results_trend.append(m)
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t2
            avg_time = elapsed / (i + 1)
            eta = avg_time * (NUM_ITERATIONS - i - 1)
            print(f"  [{i+1}/{NUM_ITERATIONS}] elapsed={elapsed:.1f}s  ETA={eta:.0f}s  "
                  f"last: score={m['score']:.4f} trades={m['total_trades']} "
                  f"PF={m['profit_factor']:.2f} ret={m['total_return']:+.2f}%")

    t3 = time.time()
    print(f"  Phase 2 complete in {t3 - t2:.1f}s")
    print()

    # ── Analysis ──
    def analyze(results, label):
        scores = [r["score"] for r in results]
        pfs = [r["profit_factor"] for r in results]
        returns = [r["total_return"] for r in results]
        trades = [r["total_trades"] for r in results]
        wrs = [r["win_rate"] for r in results]

        mean_score = np.mean(scores)
        mean_pf = np.mean(pfs)
        mean_ret = np.mean(returns)
        mean_trades = np.mean(trades)
        mean_wr = np.mean(wrs)
        median_score = np.median(scores)
        median_pf = np.median(pfs)

        pf_gt1 = sum(1 for pf in pfs if pf > 1.0) / len(pfs) * 100
        score_gt_real = sum(1 for s in scores if s > REAL_SCORE) / len(scores) * 100
        pct95_score = np.percentile(scores, 95)
        pct5_score = np.percentile(scores, 5)
        std_score = np.std(scores)
        std_ret = np.std(returns)

        print(f"{label}:")
        print(f"  Mean score: {mean_score:.4f}  (real strategy: {REAL_SCORE:.4f})")
        print(f"  Median score: {median_score:.4f}")
        print(f"  Std score: {std_score:.4f}")
        print(f"  Mean PF: {mean_pf:.2f}  (real: {REAL_PF:.2f})")
        print(f"  Median PF: {median_pf:.2f}")
        print(f"  Mean return: {mean_ret:+.2f}%  (real: +{REAL_RETURN:.2f}%)")
        print(f"  Std return: {std_ret:.2f}%")
        print(f"  Mean trades/iteration: {mean_trades:.1f}")
        print(f"  Mean win rate: {mean_wr:.1f}%")
        print(f"  % iterations with PF > 1.0: {pf_gt1:.1f}%")
        print(f"  % iterations with score > {REAL_SCORE:.4f}: {score_gt_real:.1f}%")
        print(f"  5th percentile score: {pct5_score:.4f}")
        print(f"  95th percentile score: {pct95_score:.4f}")
        print()

        return {
            "mean_score": mean_score,
            "mean_pf": mean_pf,
            "mean_ret": mean_ret,
            "pf_gt1": pf_gt1,
            "score_gt_real": score_gt_real,
            "pct95_score": pct95_score,
        }

    print("=" * 60)
    print(f"=== Random Entry Baseline ({NUM_ITERATIONS} iterations) ===")
    print("=" * 60)
    print()

    stats_5050 = analyze(results_5050, "50/50 Random")
    stats_trend = analyze(results_trend, "Trend-Biased Random")

    # ── Verdict ──
    print("VERDICT:")

    # Entry edge: if <5% beat real strategy -> REAL, 5-20% -> MARGINAL, >20% -> NONEXISTENT
    best_random_beat = max(stats_5050["score_gt_real"], stats_trend["score_gt_real"])
    if best_random_beat < 5:
        entry_verdict = "REAL"
        entry_detail = f"Only {best_random_beat:.1f}% of random iterations beat the real strategy's score"
    elif best_random_beat < 20:
        entry_verdict = "MARGINAL"
        entry_detail = f"{best_random_beat:.1f}% of random iterations beat the real strategy's score"
    else:
        entry_verdict = "NONEXISTENT"
        entry_detail = f"{best_random_beat:.1f}% of random iterations beat the real strategy's score"

    # Exit edge: if random entries still profitable on average -> exits carry value
    avg_random_pf = (stats_5050["mean_pf"] + stats_trend["mean_pf"]) / 2
    avg_random_pf_gt1 = (stats_5050["pf_gt1"] + stats_trend["pf_gt1"]) / 2
    if avg_random_pf > 1.0 and avg_random_pf_gt1 > 50:
        exit_verdict = "REAL"
        exit_detail = f"Random entries still profitable (avg PF={avg_random_pf:.2f}, {avg_random_pf_gt1:.0f}% profitable)"
    elif avg_random_pf > 0.8 and avg_random_pf_gt1 > 30:
        exit_verdict = "MARGINAL"
        exit_detail = f"Random entries sometimes profitable (avg PF={avg_random_pf:.2f}, {avg_random_pf_gt1:.0f}% profitable)"
    else:
        exit_verdict = "NONEXISTENT"
        exit_detail = f"Random entries mostly unprofitable (avg PF={avg_random_pf:.2f}, {avg_random_pf_gt1:.0f}% profitable)"

    print(f"  Entry edge is {entry_verdict}: {entry_detail}")
    print(f"  Exit edge is {exit_verdict}: {exit_detail}")
    print()

    total_time = time.time() - t0
    print(f"Total runtime: {total_time:.0f}s ({total_time/60:.1f}min)")


if __name__ == "__main__":
    main()
