"""
Tournament strategy: Mean Reversion on NQ futures.

Hypothesis: When NQ price becomes statistically overextended from its
short-term mean (Z-score > 2.0) AND shows momentum exhaustion (RSI(14) < 38
or > 62) AND prints a reversal candle, it tends to snap back toward the mean.

Design (arrived at through 5 rounds of parameter optimization):
  - 5-min bars for sufficient signal frequency
  - Z-score(20) > 2.0 for statistical overextension detection
  - RSI(14) with 38/62 thresholds for exhaustion confirmation
  - Reversal candle required (close vs open direction)
  - No ADX filter (adds too few trades for marginal improvement)
  - Stop: 3.0 ATR (wide -- mean reversion needs breathing room)
  - Target: 55% of distance-to-mean reversion (min 0.5 ATR floor)
  - Max hold: 8 bars (40 min) -- quick in, quick out
  - No entries after 15:25 ET, flatten by 15:50 ET
  - Max 4 trades/day

IS/OOS validation (IS=2022-2023, OOS=2024-2026):
  - IS PF: 1.094, OOS PF: 1.086 --> decay 0.7% (essentially zero)
  - 759 total trades over 4.25Y (well above 300 minimum)

Cost model (NQ per contract):
  comm=$2.46 RT, spread=$5.00/c, stop_slip=$1.25/c
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from src.backtest_engine import load_nq, resample, COSTS, compute_stats, gap_through_fill

# ──────────────────────── Parameters ────────────────────────
INSTRUMENT = "NQ"
TF_MINUTES = 5

# Signal
ZSCORE_PERIOD = 20
ZSCORE_ENTRY = 2.0
RSI_PERIOD = 14
RSI_LONG_THRESH = 38       # RSI < this -> oversold -> long
RSI_SHORT_THRESH = 62      # RSI > this -> overbought -> short

# Risk/Exit
ATR_PERIOD = 14
STOP_ATR_MULT = 3.0        # wide stop for mean rev breathing room
TARGET_REVERT_PCT = 0.55   # target = 55% of distance back to mean
TARGET_MIN_ATR = 0.5       # minimum target = 0.5 ATR
MAX_HOLD_BARS = 8          # 40 min on 5-min bars

# Session
NO_ENTRY_AFTER = "15:25"
FORCE_CLOSE = "15:50"
MIN_ATR = 3.0
MAX_DAILY_TRADES = 4


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI -- no future leak."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add Z-score, RSI(14), ATR. All computed on closed bars."""
    df = df.copy()

    # Z-score = (price - SMA) / rolling_std
    sma = df["Close"].rolling(ZSCORE_PERIOD).mean()
    std = df["Close"].rolling(ZSCORE_PERIOD).std()
    df["zscore"] = (df["Close"] - sma) / std.replace(0, np.nan)
    df["sma"] = sma

    # RSI(14)
    df["rsi"] = compute_rsi(df["Close"], RSI_PERIOD)

    # ATR
    tr = np.maximum(
        df["High"] - df["Low"],
        np.maximum(
            (df["High"] - df["Close"].shift(1)).abs(),
            (df["Low"] - df["Close"].shift(1)).abs(),
        ),
    )
    df["atr"] = tr.rolling(ATR_PERIOD).mean()

    return df


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate mean-reversion signals. Zero look-ahead: all on shift(1).

    Long: prev Z < -2.0 AND prev RSI < 38 AND prev close > prev open (reversal)
    Short: prev Z > 2.0 AND prev RSI > 62 AND prev close < prev open (reversal)
    """
    df = df.copy()

    # All shifted by 1 to avoid look-ahead
    p1_z = df["zscore"].shift(1)
    p1_rsi = df["rsi"].shift(1)
    p1_atr = df["atr"].shift(1)
    p1_close = df["Close"].shift(1)
    p1_open = df["Open"].shift(1)
    p1_sma = df["sma"].shift(1)

    # Time filter
    entry_end = pd.Timestamp(NO_ENTRY_AFTER).time()
    time_ok = df.index.time < entry_end

    # ATR filter
    atr_ok = p1_atr >= MIN_ATR

    # Reversal candle: must close in direction of expected reversion
    rev_long = p1_close > p1_open   # bullish candle in oversold
    rev_short = p1_close < p1_open  # bearish candle in overbought

    # Long: deeply oversold + exhaustion + reversal
    long_sig = (
        (p1_z < -ZSCORE_ENTRY) &
        (p1_rsi < RSI_LONG_THRESH) &
        time_ok & atr_ok & rev_long
    )

    # Short: deeply overbought + exhaustion + reversal
    short_sig = (
        (p1_z > ZSCORE_ENTRY) &
        (p1_rsi > RSI_SHORT_THRESH) &
        time_ok & atr_ok & rev_short
    )

    df["signal"] = 0
    df.loc[long_sig, "signal"] = 1
    df.loc[short_sig, "signal"] = -1

    df["sig_atr"] = p1_atr
    df["sig_sma"] = p1_sma

    return df


def backtest(df: pd.DataFrame, instrument: str = INSTRUMENT) -> pd.DataFrame:
    """
    Bar-by-bar backtest with gap-through stop fill.

    Entry: at bar Open when signal fires.
    Exit priority: stop > target > time stop > EOD force close.
    """
    c = COSTS[instrument]
    pt_val = c["pt_val"]
    nc = 1

    trades = []
    in_trade = False
    entry_px = stop_px = target_px = 0.0
    direction = 0
    entry_time = None
    bars_held = 0
    current_date = None
    daily_trades = 0

    force_close_time = pd.Timestamp(FORCE_CLOSE).time()

    for i in range(len(df)):
        row = df.iloc[i]
        ts = df.index[i]
        bar_open = row["Open"]
        bar_high = row["High"]
        bar_low = row["Low"]
        bar_close = row["Close"]
        current_time = ts.time()

        # Reset daily counter
        if ts.date() != current_date:
            current_date = ts.date()
            daily_trades = 0

        # ── Manage open position ──
        if in_trade:
            bars_held += 1
            exit_px_final = None
            exit_type = None

            # 1. Stop loss (gap-through model)
            if direction == 1 and bar_low <= stop_px:
                exit_px_final = gap_through_fill(stop_px, bar_open, 1)
                exit_type = "stop"
            elif direction == -1 and bar_high >= stop_px:
                exit_px_final = gap_through_fill(stop_px, bar_open, -1)
                exit_type = "stop"

            # 2. Target hit
            if exit_type is None:
                if direction == 1 and bar_high >= target_px:
                    # Conservative fill: if bar opens above target, fill at open
                    exit_px_final = max(target_px, bar_open) if bar_open > target_px else target_px
                    exit_type = "target"
                elif direction == -1 and bar_low <= target_px:
                    exit_px_final = min(target_px, bar_open) if bar_open < target_px else target_px
                    exit_type = "target"

            # 3. Time stop
            if exit_type is None and bars_held >= MAX_HOLD_BARS:
                exit_px_final = bar_close
                exit_type = "time"

            # 4. EOD force close
            if exit_type is None and current_time >= force_close_time:
                exit_px_final = bar_close
                exit_type = "eod"

            if exit_px_final is not None:
                # PnL
                raw_pts = (exit_px_final - entry_px) * direction
                raw_pnl = raw_pts * pt_val * nc

                # Cost
                if exit_type == "stop":
                    cost = c["comm_rt"] * nc + c["spread"] * nc + c["slip_stop"] * nc
                else:
                    cost = c["comm_rt"] * nc + c["spread"] * nc

                net_pnl = raw_pnl - cost
                risk_pts = abs(entry_px - stop_px)
                risk_dollar = risk_pts * pt_val * nc
                r_val = net_pnl / risk_dollar if risk_dollar > 0 else 0.0

                trades.append({
                    "entry_time": entry_time,
                    "exit_time": ts,
                    "direction": direction,
                    "entry_px": entry_px,
                    "exit_px": exit_px_final,
                    "stop_px": stop_px,
                    "target_px": target_px,
                    "pnl": net_pnl,
                    "r": r_val,
                    "cost": cost,
                    "risk": risk_dollar,
                    "exit": exit_type,
                    "date": str(ts.date()),
                    "bars_held": bars_held,
                })
                in_trade = False
                continue

        # ── Check for new entry (only if flat) ──
        if not in_trade and row["signal"] != 0 and daily_trades < MAX_DAILY_TRADES:
            direction = int(row["signal"])
            entry_px = bar_open
            atr_val = row["sig_atr"]
            sma_val = row["sig_sma"]

            if np.isnan(atr_val) or np.isnan(sma_val) or atr_val <= 0:
                continue

            # Stop: wide ATR-based
            if direction == 1:
                stop_px = entry_px - STOP_ATR_MULT * atr_val
                dist_to_mean = sma_val - entry_px
                target_px = entry_px + max(
                    dist_to_mean * TARGET_REVERT_PCT,
                    atr_val * TARGET_MIN_ATR,
                )
            else:
                stop_px = entry_px + STOP_ATR_MULT * atr_val
                dist_to_mean = entry_px - sma_val
                target_px = entry_px - max(
                    dist_to_mean * TARGET_REVERT_PCT,
                    atr_val * TARGET_MIN_ATR,
                )

            entry_time = ts
            bars_held = 0
            in_trade = True
            daily_trades += 1

    return pd.DataFrame(trades)


def run_full_backtest():
    """Run IS + OOS + Full 4Y backtest and print comprehensive results."""
    print("=" * 70)
    print("MEAN REVERSION STRATEGY — NQ Futures (Tournament Entry)")
    print("Z-score(20,2.0) + RSI(14,38/62) + Reversal Candle | 5min bars")
    print("Stop 3.0 ATR | Target 55% reversion | Max hold 8 bars (40min)")
    print("=" * 70)

    # Load and prepare data
    df = load_nq(start="2022-01-01")
    df = resample(df, TF_MINUTES)
    df = add_indicators(df)
    df = generate_signals(df)

    # Define periods
    is_end = "2024-01-01"

    df_is = df[df.index < is_end]
    df_oos = df[df.index >= is_end]
    df_full = df.copy()

    results = {}
    for label, data in [("IS (2022-2023)", df_is), ("OOS (2024-2026)", df_oos), ("FULL (2022-2026)", df_full)]:
        trades = backtest(data)
        if len(trades) == 0:
            print(f"\n{label}: NO TRADES")
            results[label] = None
            continue

        stats = compute_stats(trades)
        results[label] = {"stats": stats, "trades": trades}

        print(f"\n{'─' * 60}")
        print(f"  {label}")
        print(f"{'─' * 60}")
        print(f"  Trades: {stats['n']:>6d}    PF: {stats['pf']:>6.3f}    WR: {stats['wr']:>5.1f}%")
        print(f"  Net PnL: ${stats['pnl']:>10,.0f}    $/day: ${stats['dpnl']:>8.1f}")
        print(f"  Max DD:  ${stats['dd']:>10,.0f}    DD(R): {stats['dd_r']:>6.2f}R")
        print(f"  Sharpe:  {stats['sharpe']:>6.2f}    Sortino: {stats['sortino']:>6.2f}")
        print(f"  APR:     {stats['apr']:>6.1f}%   Calmar:  {stats['calmar']:>6.2f}")
        print(f"  Avg Win: {stats['avg_win_r']:>6.3f}R  Avg Loss: {stats['avg_loss_r']:>6.3f}R")
        print(f"  5R+ trades: {stats['b5']}   MCL: {stats['mcl']}   Cost/Risk: {stats['cost_pct']:.1f}%")
        print(f"  Win Days: {stats['win_days_pct']:.0f}%   Best Day: ${stats['best_day']:,.0f}   Worst: ${stats['worst_day']:,.0f}")

        # Exit type breakdown
        if len(trades) > 0:
            exit_counts = trades["exit"].value_counts()
            print(f"  Exit breakdown: {dict(exit_counts)}")

    # Yearly breakdown
    if results.get("FULL (2022-2026)") is not None:
        full_trades = results["FULL (2022-2026)"]["trades"]
        full_trades["year"] = pd.to_datetime(full_trades["date"]).dt.year

        print(f"\n{'─' * 60}")
        print("  YEARLY BREAKDOWN")
        print(f"{'─' * 60}")
        print(f"  {'Year':<6} {'Trades':>7} {'PF':>7} {'WR':>6} {'Net PnL':>12} {'$/day':>8} {'DD':>10} {'Sharpe':>7}")

        for year in sorted(full_trades["year"].unique()):
            yt = full_trades[full_trades["year"] == year]
            ys = compute_stats(yt)
            print(f"  {year:<6} {ys['n']:>7d} {ys['pf']:>7.3f} {ys['wr']:>5.1f}% ${ys['pnl']:>10,.0f} ${ys['dpnl']:>7.1f} ${ys['dd']:>9,.0f} {ys['sharpe']:>7.2f}")

    # IS/OOS Decay
    if results.get("IS (2022-2023)") and results.get("OOS (2024-2026)"):
        is_s = results["IS (2022-2023)"]["stats"]
        oos_s = results["OOS (2024-2026)"]["stats"]
        pf_d = (is_s["pf"] - oos_s["pf"]) / is_s["pf"] * 100 if is_s["pf"] > 0 else 0
        dp_d = (is_s["dpnl"] - oos_s["dpnl"]) / abs(is_s["dpnl"]) * 100 if is_s["dpnl"] != 0 else 0
        print(f"\n  IS->OOS Decay: PF {pf_d:+.1f}%  |  $/day {dp_d:+.1f}%")

    return results


if __name__ == "__main__":
    results = run_full_backtest()
