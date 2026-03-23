"""
Shared entry signal detection — Plan F.

Single source of truth for: trend → touch → bounce → signal line.
Used by strategy_final.py, walk_forward.py, stress_test.py, live_sim.py.
"""
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════
# DEFAULT PARAMETERS
# ══════════════════════════════════════════════════════════════════

DEFAULT_PARAMS = {
    "ema_fast": 20,
    "ema_slow": 50,
    "atr_period": 14,
    "touch_tol": 0.15,
    "touch_below_max": 0.5,
    "signal_offset": 0.05,
    "stop_buffer": 0.3,
    "signal_valid_bars": 3,
    "no_entry_after": dt.time(15, 30),
}


# ══════════════════════════════════════════════════════════════════
# INDICATORS
# ══════════════════════════════════════════════════════════════════

def add_indicators(df: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    """Add EMA20, EMA50, ATR to OHLCV dataframe."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    df = df.copy()
    df["ema20"] = df["Close"].ewm(span=p["ema_fast"], adjust=False).mean()
    df["ema50"] = df["Close"].ewm(span=p["ema_slow"], adjust=False).mean()
    tr = np.maximum(
        df["High"] - df["Low"],
        np.maximum(
            (df["High"] - df["Close"].shift(1)).abs(),
            (df["Low"] - df["Close"].shift(1)).abs(),
        ),
    )
    df["atr"] = tr.rolling(p["atr_period"]).mean()
    return df


# ══════════════════════════════════════════════════════════════════
# SIGNAL DETECTION (array-based)
# ══════════════════════════════════════════════════════════════════

def detect_trend(close_i, ema_i, ema_slow_i):
    """Return 1 (bull), -1 (bear), or 0 (no trend)."""
    if close_i > ema_i and ema_i > ema_slow_i:
        return 1
    if close_i < ema_i and ema_i < ema_slow_i:
        return -1
    return 0


def check_touch(trend, low_i, high_i, ema_i, atr_i, tol=0.15, below_max=0.5):
    """Check if wick touches EMA20 within tolerance."""
    if trend == 1:
        return (low_i <= ema_i + tol * atr_i) and (low_i >= ema_i - below_max * atr_i)
    if trend == -1:
        return (high_i >= ema_i - tol * atr_i) and (high_i <= ema_i + below_max * atr_i)
    return False


def check_bounce(trend, close_next, high_touch, low_touch):
    """Check if next bar confirms bounce."""
    if trend == 1:
        return close_next > high_touch
    if trend == -1:
        return close_next < low_touch
    return False


def calc_signal_line(trend, high_touch, low_touch, atr_touch,
                     signal_offset=0.05, stop_buffer=0.3):
    """
    Calculate signal entry price, stop price, and risk.
    Returns (signal_price, stop_price, risk) or None if risk <= 0.
    """
    if trend == 1:
        sig = high_touch + signal_offset
        stop = low_touch - stop_buffer * atr_touch
    else:
        sig = low_touch - signal_offset
        stop = high_touch + stop_buffer * atr_touch
    risk = abs(sig - stop)
    if risk <= 0:
        return None
    return sig, stop, risk


def check_signal_trigger(trend, sig, high, low, bounce_bar, n, valid_bars=3):
    """
    Check if signal triggers within valid_bars after bounce.
    Returns entry_bar index or -1 if not triggered.
    """
    for j in range(1, valid_bars + 1):
        cb = bounce_bar + j
        if cb >= n:
            break
        if trend == 1 and high[cb] >= sig:
            return cb
        if trend == -1 and low[cb] <= sig:
            return cb
    return -1
