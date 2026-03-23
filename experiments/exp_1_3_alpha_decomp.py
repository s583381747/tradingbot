"""
Experiment 1.3 — Alpha Source Decomposition
============================================
Determines whether the strategy's edge comes from ENTRY signals, EXIT logic,
or both, by testing each independently.

Test A: Real entries + fixed hold periods (isolates entry signal quality)
Test B: Random entries + real exits (isolates exit system quality)
Test C: Buy-and-hold baseline over same period
Test D: Entry signal predictive power at multiple horizons
"""

from __future__ import annotations

import datetime
import random
import sys
import warnings

import backtrader as bt
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────
DATA_PATH = "data/QQQ_1Min_2025-09-21_2026-03-21.csv"
INITIAL_CASH = 100_000.0
COMMISSION = 0.001

# Import strategy hyperparameters directly
EMA_PERIOD = 20
EMA_SLOW_PERIOD = 50
ATR_PERIOD = 14
VOLUME_AVG_PERIOD = 20
RSI_PERIOD = 14
EMA_SLOPE_PERIOD = 5
EMA_SLOPE_THRESHOLD = 0.012
PULLBACK_TOUCH_MULT = 1.2
MIN_PULLBACK_BARS = 1
RSI_OVERBOUGHT = 63
RSI_OVERSOLD = 32
INITIAL_STOP_ATR_MULT = 2.5
EMA_TRAIL_OFFSET = 6.0
LOSERS_MAX_BARS = 45
NO_ENTRY_AFTER_HOUR = 15
NO_ENTRY_AFTER_MINUTE = 30
FORCE_CLOSE_HOUR = 15
FORCE_CLOSE_MINUTE = 58
OPENING_RANGE_BARS = 5
OPENING_RANGE_ATR_MULT = 1.5
MAX_DAILY_TRADES = 6
MAX_DAILY_LOSS_PCT = 0.02
MAX_SAME_LEVEL_ATTEMPTS = 2
RISK_PCT = 0.01
MAX_POSITION_PCT = 0.25


# ────────────────────────────────────────────────────────────────
# Shared indicators (copied from strategy.py)
# ────────────────────────────────────────────────────────────────

class VWAP(bt.Indicator):
    lines = ("vwap",)
    plotinfo = dict(subplot=False)

    def __init__(self):
        self._cum_vol = 0.0
        self._cum_vp = 0.0
        self._prev_date = None

    def next(self):
        dt = self.data.datetime.date(0)
        if dt != self._prev_date:
            self._cum_vol = 0.0
            self._cum_vp = 0.0
            self._prev_date = dt
        tp = (self.data.high[0] + self.data.low[0] + self.data.close[0]) / 3.0
        self._cum_vol += self.data.volume[0]
        self._cum_vp += tp * self.data.volume[0]
        if self._cum_vol > 0:
            self.lines.vwap[0] = self._cum_vp / self._cum_vol
        else:
            self.lines.vwap[0] = self.data.close[0]


class EMASlope(bt.Indicator):
    lines = ("slope",)
    params = (("ema_period", 20), ("slope_period", 5),)

    def __init__(self):
        self.ema = bt.indicators.EMA(self.data.close, period=self.p.ema_period)

    def next(self):
        if len(self) <= self.p.slope_period:
            self.lines.slope[0] = 0.0
            return
        raw = (self.ema[0] - self.ema[-self.p.slope_period]) / self.p.slope_period
        self.lines.slope[0] = raw / self.data.close[0] * 100 if self.data.close[0] > 0 else 0.0


class SwingDetector(bt.Indicator):
    lines = ("swing_high", "swing_low")
    params = (("period", 5),)

    def __init__(self):
        self.addminperiod(self.p.period)

    def next(self):
        hi = max(self.data.high.get(size=self.p.period))
        lo = min(self.data.low.get(size=self.p.period))
        self.lines.swing_high[0] = hi
        self.lines.swing_low[0] = lo


# ────────────────────────────────────────────────────────────────
# Data loader helper
# ────────────────────────────────────────────────────────────────

def load_data():
    """Load the CSV into a backtrader data feed."""
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    df = df.rename(columns={
        "timestamp": "datetime",
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume",
    })
    df = df.set_index("datetime")
    df = df[["open", "high", "low", "close", "volume"]]
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


def make_bt_feed(df):
    """Convert a pandas DataFrame to a backtrader feed."""
    feed = bt.feeds.PandasData(
        dataname=df,
        open="open", high="high", low="low",
        close="close", volume="volume",
        openinterest=-1,
    )
    return feed


# ════════════════════════════════════════════════════════════════
# TEST A: Entry Signal + Fixed Hold Period
# ════════════════════════════════════════════════════════════════

class EntryOnlyStrategy(bt.Strategy):
    """Uses real entry logic, but exits after a fixed number of bars."""

    params = (
        ("hold_bars", 30),      # fixed hold duration
        ("hold_eod", False),    # if True, hold until EOD instead of fixed bars
    )

    def __init__(self):
        self.ema = bt.indicators.EMA(self.data.close, period=EMA_PERIOD)
        self.ema_slow = bt.indicators.EMA(self.data.close, period=EMA_SLOW_PERIOD)
        self.atr = bt.indicators.ATR(self.data, period=ATR_PERIOD)
        self.ema_slope = EMASlope(self.data, ema_period=EMA_PERIOD, slope_period=EMA_SLOPE_PERIOD)
        self.rsi = bt.indicators.RSI(self.data.close, period=RSI_PERIOD)
        self.swing = SwingDetector(self.data, period=5)

        self.order = None
        self.trade_direction = 0
        self._entry_bar = 0
        self._avg_entry_price = 0.0
        self._total_size = 0

        # Daily state
        self._today = None
        self._daily_trades = 0
        self._daily_pnl = 0.0
        self._daily_start_value = 0.0
        self._same_level_attempts = 0
        self._last_entry_bar = -999

        # Pullback state
        self._state = 0
        self._pullback_extreme = 0.0
        self._pullback_dir = 0
        self._pb_bar_count = 0

        # Opening range
        self._or_high = 0.0
        self._or_low = 999999.0
        self._or_bars_counted = 0
        self._or_is_wide = False

        # Trade tracking
        self.trade_returns = []

    def _check_new_day(self):
        dt = self.data.datetime.date(0)
        if dt != self._today:
            self._today = dt
            self._daily_trades = 0
            self._daily_pnl = 0.0
            self._daily_start_value = self.broker.getvalue()
            self._same_level_attempts = 0
            self._or_high = 0.0
            self._or_low = 999999.0
            self._or_bars_counted = 0
            self._or_is_wide = False
            self._state = 0
            self._pullback_extreme = 0.0
            self._pullback_dir = 0
            self._pb_bar_count = 0

    def _current_time(self):
        return self.data.datetime.time(0)

    def _past_entry_cutoff(self):
        t = self._current_time()
        return t >= datetime.time(NO_ENTRY_AFTER_HOUR, NO_ENTRY_AFTER_MINUTE)

    def _past_force_close(self):
        t = self._current_time()
        return t >= datetime.time(FORCE_CLOSE_HOUR, FORCE_CLOSE_MINUTE)

    def _daily_limit_reached(self):
        if self._daily_trades >= MAX_DAILY_TRADES:
            return True
        current_value = self.broker.getvalue()
        if self._daily_start_value > 0:
            loss_pct = (self._daily_start_value - current_value) / self._daily_start_value
            if loss_pct >= MAX_DAILY_LOSS_PCT:
                return True
        return False

    def _update_opening_range(self):
        if self._or_bars_counted < OPENING_RANGE_BARS:
            self._or_high = max(self._or_high, self.data.high[0])
            self._or_low = min(self._or_low, self.data.low[0])
            self._or_bars_counted += 1
            if self._or_bars_counted == OPENING_RANGE_BARS:
                or_range = self._or_high - self._or_low
                atr_val = self.atr[0] if self.atr[0] > 0 else 1.0
                self._or_is_wide = or_range > atr_val * OPENING_RANGE_ATR_MULT

    def _in_opening_range_zone(self):
        if not self._or_is_wide:
            return False
        if self._or_bars_counted < OPENING_RANGE_BARS:
            return True
        return self._or_low <= self.data.close[0] <= self._or_high

    def _detect_trend(self):
        slope = self.ema_slope.slope[0]
        if abs(slope) < EMA_SLOPE_THRESHOLD:
            return 0
        direction = 1 if slope > 0 else -1
        if direction == 1 and self.ema[0] < self.ema_slow[0]:
            return 0
        if direction == -1 and self.ema[0] > self.ema_slow[0]:
            return 0
        return direction

    def _calc_size(self, entry_price, stop_price):
        risk_per_share = abs(entry_price - stop_price)
        if risk_per_share <= 0:
            return 0
        account_value = self.broker.getvalue()
        risk_amount = account_value * RISK_PCT
        size_by_risk = int(risk_amount / risk_per_share)
        max_by_capital = int(account_value * MAX_POSITION_PCT / entry_price)
        return max(1, min(size_by_risk, max_by_capital))

    def next(self):
        if self.order:
            return

        self._check_new_day()
        self._update_opening_range()

        # Force close at EOD
        if self._past_force_close() and self.position:
            self.order = self.close()
            self._record_trade()
            self.trade_direction = 0
            return

        # Manage open position with fixed hold exit
        if self.position:
            bars_held = len(self) - self._entry_bar

            if self.p.hold_eod:
                # Hold until EOD — only force close exits
                pass
            else:
                # Fixed hold period exit
                if bars_held >= self.p.hold_bars:
                    self.order = self.close()
                    self._record_trade()
                    self.trade_direction = 0
                    return
            return

        # No new entries after cutoff or limits
        if self._past_entry_cutoff():
            return
        if self._daily_limit_reached():
            return
        if self._in_opening_range_zone():
            return

        # Entry logic (identical to strategy.py)
        self._run_entry_logic()

    def _run_entry_logic(self):
        trend = self._detect_trend()
        if trend == 0:
            self._state = 0
            self._pb_bar_count = 0
            return

        if trend != self._pullback_dir:
            self._pullback_dir = trend
            self._state = 0
            self._pb_bar_count = 0
            self._pullback_extreme = 0.0

        atr_val = self.atr[0] if self.atr[0] > 0 else 999
        dist = abs(self.data.close[0] - self.ema[0])
        touch_dist = atr_val * PULLBACK_TOUCH_MULT
        in_pullback_zone = dist <= touch_dist

        if self._state == 0:
            if in_pullback_zone:
                self._state = 1
                self._pb_bar_count = 1
                if trend == 1:
                    self._pullback_extreme = self.data.low[0]
                else:
                    self._pullback_extreme = self.data.high[0]
        elif self._state == 1:
            if in_pullback_zone:
                self._pb_bar_count += 1
                if trend == 1:
                    self._pullback_extreme = min(self._pullback_extreme, self.data.low[0])
                else:
                    self._pullback_extreme = max(self._pullback_extreme, self.data.high[0])
            else:
                on_correct_side = (
                    (trend == 1 and self.data.close[0] > self.ema[0])
                    or (trend == -1 and self.data.close[0] < self.ema[0])
                )
                rsi_ok = True
                if trend == 1 and self.rsi[0] > RSI_OVERBOUGHT:
                    rsi_ok = False
                if trend == -1 and self.rsi[0] < RSI_OVERSOLD:
                    rsi_ok = False
                momentum_ok = False
                if len(self) >= 2:
                    if trend == 1 and self.data.close[0] > self.data.high[-1]:
                        momentum_ok = True
                    elif trend == -1 and self.data.close[0] < self.data.low[-1]:
                        momentum_ok = True

                if (on_correct_side and self._pb_bar_count >= MIN_PULLBACK_BARS
                        and rsi_ok and momentum_ok):
                    self._execute_entry(trend)

                self._state = 0
                self._pb_bar_count = 0
                self._pullback_extreme = 0.0

    def _execute_entry(self, direction):
        entry_price = self.data.close[0]
        atr_val = self.atr[0] if self.atr[0] > 0 else 0.01
        stop = entry_price - INITIAL_STOP_ATR_MULT * atr_val if direction == 1 else entry_price + INITIAL_STOP_ATR_MULT * atr_val
        size = self._calc_size(entry_price, stop)
        if size <= 0:
            return

        if abs(len(self) - self._last_entry_bar) < 10:
            self._same_level_attempts += 1
            if self._same_level_attempts > MAX_SAME_LEVEL_ATTEMPTS:
                return
        else:
            self._same_level_attempts = 1

        self.trade_direction = direction
        self._avg_entry_price = entry_price
        self._total_size = size
        self._entry_bar = len(self)
        self._last_entry_bar = len(self)
        self._daily_trades += 1

        if direction == 1:
            self.order = self.buy(size=size)
        else:
            self.order = self.sell(size=size)

    def _record_trade(self):
        if self._avg_entry_price > 0:
            ret = (self.data.close[0] - self._avg_entry_price) / self._avg_entry_price * 100
            if self.trade_direction == -1:
                ret = -ret
            self.trade_returns.append(ret)

    def notify_order(self, order):
        if order.status in (order.Submitted, order.Accepted):
            return
        self.order = None

    def notify_trade(self, trade):
        if trade.isclosed:
            self._daily_pnl += trade.pnlcomm


def run_test_a():
    """Test A: Entry Signal + Fixed Hold Periods"""
    print("=== Test A: Entry Signal + Fixed Hold ===")
    df = load_data()

    hold_configs = [
        (10, False, "10min"),
        (30, False, "30min"),
        (60, False, "60min"),
        (120, False, "2hr"),
        (9999, True, "EOD"),
    ]

    for hold_bars, hold_eod, label in hold_configs:
        cerebro = bt.Cerebro()
        cerebro.addstrategy(EntryOnlyStrategy, hold_bars=hold_bars, hold_eod=hold_eod)
        cerebro.adddata(make_bt_feed(df))
        cerebro.broker.setcash(INITIAL_CASH)
        cerebro.broker.setcommission(commission=COMMISSION)
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

        results = cerebro.run()
        strat = results[0]

        final_value = cerebro.broker.getvalue()
        total_pnl = final_value - INITIAL_CASH

        ta = strat.analyzers.trades.get_analysis()
        total_obj = ta.get("total", {})
        total_closed = total_obj.get("closed", 0) if isinstance(total_obj, dict) else 0
        won_obj = ta.get("won", {})
        won_total = won_obj.get("total", 0) if isinstance(won_obj, dict) else 0

        rets = strat.trade_returns
        avg_ret = np.mean(rets) if rets else 0.0
        wr = (won_total / total_closed * 100) if total_closed > 0 else 0.0

        print(f"  Hold {label:>5s}:  avg_ret={avg_ret:+.2f}%  wr={wr:.0f}%  "
              f"total_pnl=${total_pnl:+,.0f}  trades={total_closed}")
        sys.stdout.flush()

    print()
    sys.stdout.flush()


# ════════════════════════════════════════════════════════════════
# TEST B: Random Entry + Real Exit (200 iterations)
# ════════════════════════════════════════════════════════════════

class RandomEntryRealExitStrategy(bt.Strategy):
    """Random entries with the real strategy's exit logic."""

    params = (
        ("entry_probability", 0.005),  # ~1 entry per 200 bars
        ("seed", 42),
    )

    def __init__(self):
        self.ema = bt.indicators.EMA(self.data.close, period=EMA_PERIOD)
        self.ema_slow = bt.indicators.EMA(self.data.close, period=EMA_SLOW_PERIOD)
        self.atr = bt.indicators.ATR(self.data, period=ATR_PERIOD)
        self.ema_slope = EMASlope(self.data, ema_period=EMA_PERIOD, slope_period=EMA_SLOPE_PERIOD)
        self.rsi = bt.indicators.RSI(self.data.close, period=RSI_PERIOD)

        self.order = None
        self.trade_direction = 0
        self._entry_bar = 0
        self._avg_entry_price = 0.0
        self._stop_price = 0.0
        self._trail_phase = 0
        self._profitable_bars = 0

        self._today = None
        self._daily_trades = 0
        self._rng = random.Random(self.p.seed)

    def _check_new_day(self):
        dt = self.data.datetime.date(0)
        if dt != self._today:
            self._today = dt
            self._daily_trades = 0

    def _current_time(self):
        return self.data.datetime.time(0)

    def _past_entry_cutoff(self):
        return self._current_time() >= datetime.time(NO_ENTRY_AFTER_HOUR, NO_ENTRY_AFTER_MINUTE)

    def _past_force_close(self):
        return self._current_time() >= datetime.time(FORCE_CLOSE_HOUR, FORCE_CLOSE_MINUTE)

    def next(self):
        if self.order:
            return

        self._check_new_day()

        # Force close at EOD
        if self._past_force_close() and self.position:
            self.order = self.close()
            self._reset()
            return

        # Manage open position with REAL exit logic
        if self.position:
            self._manage_position()
            return

        if self._past_entry_cutoff():
            return
        if self._daily_trades >= MAX_DAILY_TRADES:
            return

        # Random entry
        if self._rng.random() < self.p.entry_probability:
            self._random_entry()

    def _random_entry(self):
        direction = 1 if self._rng.random() > 0.5 else -1
        entry_price = self.data.close[0]
        atr_val = self.atr[0] if self.atr[0] > 0 else 0.01

        if direction == 1:
            stop = entry_price - INITIAL_STOP_ATR_MULT * atr_val
        else:
            stop = entry_price + INITIAL_STOP_ATR_MULT * atr_val

        risk_per_share = abs(entry_price - stop)
        if risk_per_share <= 0:
            return
        account_value = self.broker.getvalue()
        risk_amount = account_value * RISK_PCT
        size_by_risk = int(risk_amount / risk_per_share)
        max_by_capital = int(account_value * MAX_POSITION_PCT / entry_price)
        size = max(1, min(size_by_risk, max_by_capital))

        self.trade_direction = direction
        self._avg_entry_price = entry_price
        self._entry_bar = len(self)
        self._stop_price = stop
        self._trail_phase = 0
        self._profitable_bars = 0
        self._daily_trades += 1

        if direction == 1:
            self.order = self.buy(size=size)
        else:
            self.order = self.sell(size=size)

    def _manage_position(self):
        d = self.trade_direction
        if d == 0:
            return

        bar_low = self.data.low[0]
        bar_high = self.data.high[0]
        price = self.data.close[0]

        # 1. Time-based exit for losers
        bars_in_trade = len(self) - self._entry_bar
        if bars_in_trade > LOSERS_MAX_BARS:
            is_losing = (
                (d == 1 and price < self._avg_entry_price)
                or (d == -1 and price > self._avg_entry_price)
            )
            if is_losing:
                self.order = self.close()
                self._reset()
                return

        # 2. Check stop hit
        if d == 1 and bar_low <= self._stop_price:
            self.order = self.close()
            self._reset()
            return
        if d == -1 and bar_high >= self._stop_price:
            self.order = self.close()
            self._reset()
            return

        # 3. Count profitable bars
        if d == 1 and price > self._avg_entry_price:
            self._profitable_bars += 1
        elif d == -1 and price < self._avg_entry_price:
            self._profitable_bars += 1

        # 4. Update trailing stop (EMA trail logic from strategy.py)
        atr_val = self.atr[0] if self.atr[0] > 0 else 0.01
        if d == 1:
            ema_stop = self.ema[0] - EMA_TRAIL_OFFSET * atr_val
        else:
            ema_stop = self.ema[0] + EMA_TRAIL_OFFSET * atr_val

        new_stop = ema_stop
        if d == 1 and new_stop > self._stop_price:
            self._stop_price = new_stop
        elif d == -1 and new_stop < self._stop_price:
            self._stop_price = new_stop

    def _reset(self):
        self.trade_direction = 0
        self._entry_bar = 0
        self._avg_entry_price = 0.0
        self._stop_price = 0.0
        self._trail_phase = 0
        self._profitable_bars = 0

    def notify_order(self, order):
        if order.status in (order.Submitted, order.Accepted):
            return
        self.order = None

    def notify_trade(self, trade):
        pass


def run_test_b(n_iters=200):
    """Test B: Random Entry + Real Exit (200 iterations)"""
    print(f"=== Test B: Random Entry + Real Exit ({n_iters} iters) ===")
    sys.stdout.flush()
    df = load_data()

    pfs = []
    returns = []
    wrs = []

    for i in range(n_iters):
        cerebro = bt.Cerebro()
        cerebro.addstrategy(RandomEntryRealExitStrategy, seed=i * 17 + 42)
        cerebro.adddata(make_bt_feed(df))
        cerebro.broker.setcash(INITIAL_CASH)
        cerebro.broker.setcommission(commission=COMMISSION)
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

        results = cerebro.run()
        strat = results[0]

        final_value = cerebro.broker.getvalue()
        ret = (final_value - INITIAL_CASH) / INITIAL_CASH * 100

        ta = strat.analyzers.trades.get_analysis()
        total_obj = ta.get("total", {})
        total_closed = total_obj.get("closed", 0) if isinstance(total_obj, dict) else 0
        won_obj = ta.get("won", {})
        won_total = won_obj.get("total", 0) if isinstance(won_obj, dict) else 0

        wr = (won_total / total_closed * 100) if total_closed > 0 else 0.0

        # Profit factor
        gross_profit = 0.0
        gross_loss = 0.0
        pnl_won = won_obj.get("pnl", {}) if isinstance(won_obj, dict) else {}
        if isinstance(pnl_won, dict):
            gross_profit = abs(float(pnl_won.get("total", 0.0)))
        else:
            gross_profit = abs(float(pnl_won))
        lost_obj = ta.get("lost", {})
        pnl_lost = lost_obj.get("pnl", {}) if isinstance(lost_obj, dict) else {}
        if isinstance(pnl_lost, dict):
            gross_loss = abs(float(pnl_lost.get("total", 0.0)))
        else:
            gross_loss = abs(float(pnl_lost))

        pf = gross_profit / gross_loss if gross_loss > 0 else 0.0

        pfs.append(pf)
        returns.append(ret)
        wrs.append(wr)

        if (i + 1) % 50 == 0:
            print(f"    ... {i + 1}/{n_iters} iterations complete")
            sys.stdout.flush()

    mean_pf = np.mean(pfs)
    mean_ret = np.mean(returns)
    mean_wr = np.mean(wrs)
    std_ret = np.std(returns)

    print(f"  Mean PF: {mean_pf:.2f}  Mean return: {mean_ret:+.2f}%  "
          f"Mean WR: {mean_wr:.0f}%  Std return: {std_ret:.2f}%")
    print()
    sys.stdout.flush()

    return mean_ret, mean_pf


# ════════════════════════════════════════════════════════════════
# TEST C: Buy & Hold Baseline
# ════════════════════════════════════════════════════════════════

def run_test_c():
    """Test C: Buy & Hold QQQ over data period."""
    print("=== Test C: Buy & Hold Baseline ===")
    df = load_data()

    first_close = df["close"].iloc[0]
    last_close = df["close"].iloc[-1]
    bh_return = (last_close - first_close) / first_close * 100

    print(f"  QQQ return over period: {bh_return:+.2f}%  "
          f"(${first_close:.2f} -> ${last_close:.2f})")
    print()
    return bh_return


# ════════════════════════════════════════════════════════════════
# TEST D: Entry Signal Predictive Power
# ════════════════════════════════════════════════════════════════

class SignalCollectorStrategy(bt.Strategy):
    """Collects all entry signals without trading — just records bar indices + direction."""

    def __init__(self):
        self.ema = bt.indicators.EMA(self.data.close, period=EMA_PERIOD)
        self.ema_slow = bt.indicators.EMA(self.data.close, period=EMA_SLOW_PERIOD)
        self.atr = bt.indicators.ATR(self.data, period=ATR_PERIOD)
        self.ema_slope = EMASlope(self.data, ema_period=EMA_PERIOD, slope_period=EMA_SLOPE_PERIOD)
        self.rsi = bt.indicators.RSI(self.data.close, period=RSI_PERIOD)
        self.swing = SwingDetector(self.data, period=5)

        self.order = None

        # Daily state
        self._today = None
        self._daily_trades = 0
        self._daily_start_value = 0.0
        self._same_level_attempts = 0
        self._last_entry_bar = -999

        # Pullback state
        self._state = 0
        self._pullback_extreme = 0.0
        self._pullback_dir = 0
        self._pb_bar_count = 0

        # Opening range
        self._or_high = 0.0
        self._or_low = 999999.0
        self._or_bars_counted = 0
        self._or_is_wide = False

        # Collected signals: list of (bar_index, direction, entry_price)
        self.signals = []

    def _check_new_day(self):
        dt = self.data.datetime.date(0)
        if dt != self._today:
            self._today = dt
            self._daily_trades = 0
            self._daily_start_value = self.broker.getvalue()
            self._same_level_attempts = 0
            self._or_high = 0.0
            self._or_low = 999999.0
            self._or_bars_counted = 0
            self._or_is_wide = False
            self._state = 0
            self._pullback_extreme = 0.0
            self._pullback_dir = 0
            self._pb_bar_count = 0

    def _current_time(self):
        return self.data.datetime.time(0)

    def _past_entry_cutoff(self):
        return self._current_time() >= datetime.time(NO_ENTRY_AFTER_HOUR, NO_ENTRY_AFTER_MINUTE)

    def _update_opening_range(self):
        if self._or_bars_counted < OPENING_RANGE_BARS:
            self._or_high = max(self._or_high, self.data.high[0])
            self._or_low = min(self._or_low, self.data.low[0])
            self._or_bars_counted += 1
            if self._or_bars_counted == OPENING_RANGE_BARS:
                or_range = self._or_high - self._or_low
                atr_val = self.atr[0] if self.atr[0] > 0 else 1.0
                self._or_is_wide = or_range > atr_val * OPENING_RANGE_ATR_MULT

    def _in_opening_range_zone(self):
        if not self._or_is_wide:
            return False
        if self._or_bars_counted < OPENING_RANGE_BARS:
            return True
        return self._or_low <= self.data.close[0] <= self._or_high

    def _detect_trend(self):
        slope = self.ema_slope.slope[0]
        if abs(slope) < EMA_SLOPE_THRESHOLD:
            return 0
        direction = 1 if slope > 0 else -1
        if direction == 1 and self.ema[0] < self.ema_slow[0]:
            return 0
        if direction == -1 and self.ema[0] > self.ema_slow[0]:
            return 0
        return direction

    def _daily_limit_reached(self):
        return self._daily_trades >= MAX_DAILY_TRADES

    def next(self):
        self._check_new_day()
        self._update_opening_range()

        if self._past_entry_cutoff():
            return
        if self._daily_limit_reached():
            return
        if self._in_opening_range_zone():
            return

        self._run_entry_logic()

    def _run_entry_logic(self):
        trend = self._detect_trend()
        if trend == 0:
            self._state = 0
            self._pb_bar_count = 0
            return

        if trend != self._pullback_dir:
            self._pullback_dir = trend
            self._state = 0
            self._pb_bar_count = 0
            self._pullback_extreme = 0.0

        atr_val = self.atr[0] if self.atr[0] > 0 else 999
        dist = abs(self.data.close[0] - self.ema[0])
        touch_dist = atr_val * PULLBACK_TOUCH_MULT
        in_pullback_zone = dist <= touch_dist

        if self._state == 0:
            if in_pullback_zone:
                self._state = 1
                self._pb_bar_count = 1
                if trend == 1:
                    self._pullback_extreme = self.data.low[0]
                else:
                    self._pullback_extreme = self.data.high[0]
        elif self._state == 1:
            if in_pullback_zone:
                self._pb_bar_count += 1
                if trend == 1:
                    self._pullback_extreme = min(self._pullback_extreme, self.data.low[0])
                else:
                    self._pullback_extreme = max(self._pullback_extreme, self.data.high[0])
            else:
                on_correct_side = (
                    (trend == 1 and self.data.close[0] > self.ema[0])
                    or (trend == -1 and self.data.close[0] < self.ema[0])
                )
                rsi_ok = True
                if trend == 1 and self.rsi[0] > RSI_OVERBOUGHT:
                    rsi_ok = False
                if trend == -1 and self.rsi[0] < RSI_OVERSOLD:
                    rsi_ok = False
                momentum_ok = False
                if len(self) >= 2:
                    if trend == 1 and self.data.close[0] > self.data.high[-1]:
                        momentum_ok = True
                    elif trend == -1 and self.data.close[0] < self.data.low[-1]:
                        momentum_ok = True

                if (on_correct_side and self._pb_bar_count >= MIN_PULLBACK_BARS
                        and rsi_ok and momentum_ok):
                    # Record the signal (don't actually trade)
                    if abs(len(self) - self._last_entry_bar) < 10:
                        self._same_level_attempts += 1
                        if self._same_level_attempts > MAX_SAME_LEVEL_ATTEMPTS:
                            self._state = 0
                            self._pb_bar_count = 0
                            self._pullback_extreme = 0.0
                            return
                    else:
                        self._same_level_attempts = 1

                    self.signals.append((len(self), trend, self.data.close[0]))
                    self._last_entry_bar = len(self)
                    self._daily_trades += 1

                self._state = 0
                self._pb_bar_count = 0
                self._pullback_extreme = 0.0

    def notify_order(self, order):
        if order.status in (order.Submitted, order.Accepted):
            return
        self.order = None


def run_test_d():
    """Test D: Entry Signal Predictive Power at multiple horizons."""
    print("=== Test D: Entry Signal Predictive Power ===")
    df = load_data()

    # First, collect all signals via backtrader
    cerebro = bt.Cerebro()
    cerebro.addstrategy(SignalCollectorStrategy)
    cerebro.adddata(make_bt_feed(df))
    cerebro.broker.setcash(INITIAL_CASH)

    results = cerebro.run()
    strat = results[0]
    signals = strat.signals

    # Now we need to look up future prices from the raw dataframe
    # The bar index in backtrader corresponds to position in the dataframe
    # (offset by the warm-up period). We'll use the close prices array.
    closes = df["close"].values
    n_bars = len(closes)

    horizons = [5, 10, 30, 60]

    print(f"  Total entry signals collected: {len(signals)}")
    print()
    print(f"  {'Bars ahead':>10s} | {'Avg Return':>10s} | {'Win Rate':>8s} | {'N':>5s} | Significant?")
    print(f"  {'-'*10} | {'-'*10} | {'-'*8} | {'-'*5} | {'-'*12}")

    entry_avg_rets = {}
    for horizon in horizons:
        forward_returns = []
        for bar_idx, direction, entry_price in signals:
            # bar_idx is 1-based backtrader index; map to df index
            # backtrader len(self) starts at 1 and increments each bar
            # The df index is bar_idx - 1 (0-based)
            df_idx = bar_idx - 1
            future_idx = df_idx + horizon

            if future_idx >= n_bars:
                continue

            future_price = closes[future_idx]
            if direction == 1:
                ret = (future_price - entry_price) / entry_price * 100
            else:
                ret = (entry_price - future_price) / entry_price * 100
            forward_returns.append(ret)

        if len(forward_returns) < 2:
            print(f"  {horizon:>10d} | {'N/A':>10s} | {'N/A':>8s} | {len(forward_returns):>5d} | N/A")
            entry_avg_rets[horizon] = 0.0
            continue

        arr = np.array(forward_returns)
        avg_ret = np.mean(arr)
        wr = np.sum(arr > 0) / len(arr) * 100

        # T-test: is mean significantly different from 0?
        t_stat, p_val = stats.ttest_1samp(arr, 0)
        significant = "Yes" if p_val < 0.05 else "No"

        entry_avg_rets[horizon] = avg_ret
        print(f"  {horizon:>10d} | {avg_ret:>+9.2f}% | {wr:>6.0f}%  | {len(arr):>5d} | "
              f"{significant} (p={p_val:.4f})")

    print()
    return signals, entry_avg_rets


# ════════════════════════════════════════════════════════════════
# ALPHA DECOMPOSITION SUMMARY
# ════════════════════════════════════════════════════════════════

def summarize(entry_avg_rets, random_exit_ret, random_exit_pf, bh_return):
    """Print the final decomposition summary."""
    print("=== ALPHA DECOMPOSITION SUMMARY ===")

    # Entry alpha assessment:
    # Check if entry signals at short horizons (5-10 bars) show significant positive returns
    short_horizon_rets = [entry_avg_rets.get(h, 0) for h in [5, 10]]
    med_horizon_rets = [entry_avg_rets.get(h, 0) for h in [30, 60]]
    avg_short = np.mean(short_horizon_rets) if short_horizon_rets else 0
    avg_med = np.mean(med_horizon_rets) if med_horizon_rets else 0

    if avg_short > 0.05 and avg_med > 0.05:
        entry_alpha = "STRONG"
    elif avg_short > 0.02 or avg_med > 0.02:
        entry_alpha = "WEAK"
    else:
        entry_alpha = "NONE"

    # Exit alpha assessment:
    # If random entries + real exits beat random baseline (0), exits add value
    if random_exit_pf > 1.1 and random_exit_ret > 0.5:
        exit_alpha = "STRONG"
    elif random_exit_pf > 0.95 or random_exit_ret > 0:
        exit_alpha = "WEAK"
    else:
        exit_alpha = "NONE"

    # Combined synergy: if both contribute positively
    has_synergy = entry_alpha != "NONE" and exit_alpha != "NONE"
    synergy = "YES" if has_synergy else "NO"

    print(f"  Entry alpha: {entry_alpha}")
    print(f"  Exit alpha:  {exit_alpha}")
    print(f"  Combined synergy: {synergy}")
    print()


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print()
    print("=" * 60)
    print("  EXPERIMENT 1.3 — Alpha Source Decomposition")
    print("=" * 60)
    print()

    run_test_a()

    random_exit_ret, random_exit_pf = run_test_b(n_iters=200)

    bh_return = run_test_c()

    signals, entry_avg_rets = run_test_d()

    summarize(entry_avg_rets, random_exit_ret, random_exit_pf, bh_return)
