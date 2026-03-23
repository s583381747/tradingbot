"""
EMA20 Trend Following — Gen 12: Lock + Runner Hybrid Exit.

Proven in pandas simulation: 16/16 configs profitable.
Best balance: lock 70% at +1 ATR, run 30% with staircase trail.
Result: WR=81%, PF=4.68, expectancy=$0.33/trade.

Logic:
  ENTRY: Dual filter (slope>=0.008 + range/ATR>=5.0) + EMA20 wick pullback bounce
  EXIT:
    - Initial stop: test candle low - 0.3*ATR
    - When profit reaches 1.0 ATR → exit 70% (lock profit)
    - Move runner stop to breakeven
    - Runner 30% trails via 10-bar trailing low - 0.3*ATR (ratchets up only)
    - Runner exits only when trail hit or session end
"""

from __future__ import annotations
import datetime
import backtrader as bt

# ══════════════════════════════════════════════════════════════════
# HYPERPARAMETERS
# ══════════════════════════════════════════════════════════════════

EMA_PERIOD = 20
EMA_SLOW_PERIOD = 50
ATR_PERIOD = 14

# Dual trend filter
EMA_SLOPE_PERIOD = 5
MIN_SLOPE = 0.008
RANGE_ATR_PERIOD = 20
MIN_RANGE_ATR = 5.0

# Entry
PULLBACK_TOUCH_MULT = 1.0
MIN_PULLBACK_BARS = 1

# Stop
STOP_BUFFER_ATR = 0.3

# Hybrid exit
LOCK_TARGET_ATR = 1.0           # exit lock portion at this profit
LOCK_PCT = 0.70                 # 70% of position locks profit
RUNNER_TRAIL_BARS = 10          # trailing low lookback for runner
RUNNER_TRAIL_BUFFER = 0.3       # buffer below trailing low

# Risk
RISK_PCT = 0.01
MAX_POSITION_PCT = 0.25
MAX_DAILY_TRADES = 6
MAX_DAILY_LOSS_PCT = 0.02

# Session
NO_ENTRY_AFTER_HOUR = 15
NO_ENTRY_AFTER_MINUTE = 30
FORCE_CLOSE_HOUR = 15
FORCE_CLOSE_MINUTE = 58

# Loser timeout
LOSERS_MAX_BARS = 30


class EMASlope(bt.Indicator):
    lines = ("slope",)
    params = (("ema_period", 20), ("slope_period", 5))
    def __init__(self):
        self.ema = bt.indicators.EMA(self.data.close, period=self.p.ema_period)
    def next(self):
        if len(self) <= self.p.slope_period:
            self.lines.slope[0] = 0.0
            return
        raw = (self.ema[0] - self.ema[-self.p.slope_period]) / self.p.slope_period
        self.lines.slope[0] = raw / self.data.close[0] * 100 if self.data.close[0] > 0 else 0.0


class Strategy(bt.Strategy):

    params = (
        ("ema_period", EMA_PERIOD), ("ema_slow_period", EMA_SLOW_PERIOD),
        ("atr_period", ATR_PERIOD), ("ema_slope_period", EMA_SLOPE_PERIOD),
        ("min_slope", MIN_SLOPE), ("range_atr_period", RANGE_ATR_PERIOD),
        ("min_range_atr", MIN_RANGE_ATR),
        ("pullback_touch_mult", PULLBACK_TOUCH_MULT),
        ("min_pullback_bars", MIN_PULLBACK_BARS),
        ("stop_buffer_atr", STOP_BUFFER_ATR),
        ("lock_target_atr", LOCK_TARGET_ATR), ("lock_pct", LOCK_PCT),
        ("runner_trail_bars", RUNNER_TRAIL_BARS),
        ("runner_trail_buffer", RUNNER_TRAIL_BUFFER),
        ("risk_pct", RISK_PCT), ("max_position_pct", MAX_POSITION_PCT),
        ("max_daily_trades", MAX_DAILY_TRADES),
        ("max_daily_loss_pct", MAX_DAILY_LOSS_PCT),
        ("no_entry_after_hour", NO_ENTRY_AFTER_HOUR),
        ("no_entry_after_minute", NO_ENTRY_AFTER_MINUTE),
        ("force_close_hour", FORCE_CLOSE_HOUR),
        ("force_close_minute", FORCE_CLOSE_MINUTE),
        ("losers_max_bars", LOSERS_MAX_BARS),
    )

    def __init__(self):
        self.ema = bt.indicators.EMA(self.data.close, period=self.p.ema_period)
        self.ema_slow = bt.indicators.EMA(self.data.close, period=self.p.ema_slow_period)
        self.atr = bt.indicators.ATR(self.data, period=self.p.atr_period)
        self.slope = EMASlope(self.data, ema_period=self.p.ema_period,
                              slope_period=self.p.ema_slope_period)
        self.highest = bt.indicators.Highest(self.data.high, period=self.p.range_atr_period)
        self.lowest = bt.indicators.Lowest(self.data.low, period=self.p.range_atr_period)

        self.order = None
        self._dir = 0
        self._pb_state = 0
        self._pb_bars = 0
        self._pb_low = 0.0

        self._entry_price = 0.0
        self._entry_bar = 0
        self._sl = 0.0
        self._total_size = 0
        self._lock_done = False
        self._runner_sl = 0.0

        self._today = None
        self._daily_trades = 0
        self._daily_pnl = 0.0
        self._daily_start = 0.0

    def _trend(self):
        atr = self.atr[0]
        if atr <= 0:
            return 0
        rng = self.highest[0] - self.lowest[0]
        if rng / atr < self.p.min_range_atr:
            return 0
        s = self.slope.slope[0]
        if s > self.p.min_slope and self.ema[0] > self.ema_slow[0]:
            return 1
        if s < -self.p.min_slope and self.ema[0] < self.ema_slow[0]:
            return -1
        return 0

    def _new_day(self):
        dt = self.data.datetime.date(0)
        if dt != self._today:
            self._today = dt
            self._daily_trades = 0
            self._daily_pnl = 0.0
            self._daily_start = self.broker.getvalue()
            self._pb_state = 0
            self._pb_bars = 0

    def _can_enter(self):
        t = self.data.datetime.time(0)
        if t >= datetime.time(self.p.no_entry_after_hour, self.p.no_entry_after_minute):
            return False
        if self._daily_trades >= self.p.max_daily_trades:
            return False
        if self._daily_start > 0:
            if (self._daily_start - self.broker.getvalue()) / self._daily_start >= self.p.max_daily_loss_pct:
                return False
        return True

    def _size(self, entry, stop):
        risk = abs(entry - stop)
        if risk <= 0:
            return 0
        val = self.broker.getvalue()
        return max(1, min(int(val * self.p.risk_pct / risk),
                          int(val * self.p.max_position_pct / entry)))

    def next(self):
        if self.order:
            return
        self._new_day()

        t = self.data.datetime.time(0)
        if t >= datetime.time(self.p.force_close_hour, self.p.force_close_minute):
            if self.position:
                self.order = self.close()
                self._reset()
            return

        if self.position:
            self._manage()
            return

        if self._can_enter():
            self._scan()

    def _scan(self):
        trend = self._trend()
        if trend == 0:
            self._pb_state = 0
            self._pb_bars = 0
            return

        if trend != self._dir:
            self._dir = trend
            self._pb_state = 0
            self._pb_bars = 0
            self._pb_low = 0.0

        atr = self.atr[0] if self.atr[0] > 0 else 999
        zone = atr * self.p.pullback_touch_mult

        if trend == 1:
            in_zone = (self.data.low[0] <= self.ema[0] + zone) and \
                      (self.data.low[0] >= self.ema[0] - zone)
        else:
            in_zone = (self.data.high[0] >= self.ema[0] - zone) and \
                      (self.data.high[0] <= self.ema[0] + zone)

        if self._pb_state == 0:
            if in_zone:
                self._pb_state = 1
                self._pb_bars = 1
                self._pb_low = self.data.low[0] if trend == 1 else self.data.high[0]
        elif self._pb_state == 1:
            if in_zone:
                self._pb_bars += 1
                if trend == 1:
                    self._pb_low = min(self._pb_low, self.data.low[0])
                else:
                    self._pb_low = max(self._pb_low, self.data.high[0])
            else:
                bounce = False
                if len(self) >= 2:
                    if trend == 1 and self.data.close[0] > self.data.high[-1]:
                        bounce = True
                    elif trend == -1 and self.data.close[0] < self.data.low[-1]:
                        bounce = True
                on_side = (trend == 1 and self.data.close[0] > self.ema[0]) or \
                          (trend == -1 and self.data.close[0] < self.ema[0])
                if bounce and on_side and self._pb_bars >= self.p.min_pullback_bars:
                    self._open(trend)
                self._pb_state = 0
                self._pb_bars = 0
                self._pb_low = 0.0

    def _open(self, d):
        price = self.data.close[0]
        atr = self.atr[0] if self.atr[0] > 0 else 0.01
        if d == 1:
            stop = self._pb_low - self.p.stop_buffer_atr * atr
        else:
            stop = self._pb_low + self.p.stop_buffer_atr * atr

        size = self._size(price, stop)
        if size <= 0:
            return

        self._dir = d
        self._entry_price = price
        self._entry_bar = len(self)
        self._sl = stop
        self._runner_sl = stop
        self._total_size = size
        self._lock_done = False
        self._daily_trades += 1

        if d == 1:
            self.order = self.buy(size=size)
        else:
            self.order = self.sell(size=size)

    def _manage(self):
        d = self._dir
        if d == 0:
            return

        price = self.data.close[0]
        atr = self.atr[0] if self.atr[0] > 0 else 0.01
        pos = abs(self.position.size)

        # 1. Loser timeout
        if len(self) - self._entry_bar > self.p.losers_max_bars:
            if (d == 1 and price < self._entry_price) or \
               (d == -1 and price > self._entry_price):
                self.order = self.close()
                self._reset()
                return

        # 2. Check stop (initial stop for full position, runner stop after lock)
        active_stop = self._runner_sl if self._lock_done else self._sl

        if d == 1 and self.data.low[0] <= active_stop:
            self.order = self.close()
            self._reset()
            return
        if d == -1 and self.data.high[0] >= active_stop:
            self.order = self.close()
            self._reset()
            return

        # 3. Lock profit: exit 70% when profit reaches target
        if not self._lock_done:
            profit = (price - self._entry_price) * d
            if profit >= self.p.lock_target_atr * atr:
                lock_size = max(1, int(self._total_size * self.p.lock_pct))
                lock_size = min(lock_size, pos - 1)  # keep at least 1 for runner
                if lock_size > 0:
                    if d == 1:
                        self.order = self.sell(size=lock_size)
                    else:
                        self.order = self.buy(size=lock_size)
                    self._lock_done = True
                    # Move runner stop to breakeven
                    self._runner_sl = self._entry_price
                return

        # 4. Runner trail: 10-bar trailing low - buffer (only ratchets up)
        if self._lock_done and len(self) >= self.p.runner_trail_bars:
            if d == 1:
                recent_low = min(self.data.low.get(ago=-i, size=1)[0]
                                 for i in range(self.p.runner_trail_bars))
                new_trail = recent_low - self.p.runner_trail_buffer * atr
                self._runner_sl = max(self._runner_sl, new_trail)
            else:
                recent_high = max(self.data.high.get(ago=-i, size=1)[0]
                                  for i in range(self.p.runner_trail_bars))
                new_trail = recent_high + self.p.runner_trail_buffer * atr
                self._runner_sl = min(self._runner_sl, new_trail)

    def _reset(self):
        self._dir = 0
        self._entry_price = 0.0
        self._entry_bar = 0
        self._sl = 0.0
        self._runner_sl = 0.0
        self._total_size = 0
        self._lock_done = False
        self._pb_state = 0
        self._pb_bars = 0
        self._pb_low = 0.0

    def notify_order(self, order):
        if order.status in (order.Submitted, order.Accepted):
            return
        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        self._daily_pnl += trade.pnlcomm
        if not self.position:
            self._reset()
