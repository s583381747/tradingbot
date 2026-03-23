"""
EMA20 Trend Following — Gen 13: Signal Line Entry + Lock/Runner.

Entry: After wick tests EMA20, place buy stop at test_candle_high + $0.05.
  - Only if dual filter passes (slope>=0.008, range/ATR>=5.0)
  - Stop order valid for 3 bars, then cancelled
  - Entry price is known in advance = precise risk calculation

Exit: Lock 70% at +1.0 ATR profit, run 30% with trailing stop.

Key: Uses buy stop orders instead of market orders at close.
"""

from __future__ import annotations
import datetime
import backtrader as bt

EMA_PERIOD = 20
EMA_SLOW_PERIOD = 50
ATR_PERIOD = 14
EMA_SLOPE_PERIOD = 5
MIN_SLOPE = 0.008
RANGE_ATR_PERIOD = 20
MIN_RANGE_ATR = 5.0

PULLBACK_TOUCH_MULT = 1.0
MIN_PULLBACK_BARS = 1
SIGNAL_OFFSET = 0.05            # entry at test_candle_high + this

STOP_BUFFER_ATR = 0.3
LOCK_TARGET_ATR = 1.0
LOCK_PCT = 0.70
RUNNER_TRAIL_BARS = 10
RUNNER_TRAIL_BUFFER_ATR = 0.3

RISK_PCT = 0.01
MAX_POSITION_PCT = 0.25
MAX_DAILY_TRADES = 6
MAX_DAILY_LOSS_PCT = 0.02

NO_ENTRY_AFTER_HOUR = 15
NO_ENTRY_AFTER_MINUTE = 30
FORCE_CLOSE_HOUR = 15
FORCE_CLOSE_MINUTE = 58
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
        ("signal_offset", SIGNAL_OFFSET),
        ("stop_buffer_atr", STOP_BUFFER_ATR),
        ("lock_target_atr", LOCK_TARGET_ATR), ("lock_pct", LOCK_PCT),
        ("runner_trail_bars", RUNNER_TRAIL_BARS),
        ("runner_trail_buffer_atr", RUNNER_TRAIL_BUFFER_ATR),
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
        self.lowest_ind = bt.indicators.Lowest(self.data.low, period=self.p.range_atr_period)

        self.order = None
        self._dir = 0

        # Pullback scan state
        self._pb_state = 0
        self._pb_bars = 0
        self._pb_low = 0.0

        # Pending signal line
        self._signal_price = 0.0     # buy/sell stop price
        self._signal_stop = 0.0      # stop loss if filled
        self._signal_dir = 0
        self._signal_bar = 0         # bar when signal was placed
        self._signal_size = 0

        # Position state
        self._entry_price = 0.0
        self._entry_bar = 0
        self._sl = 0.0
        self._total_size = 0
        self._lock_done = False
        self._runner_sl = 0.0

        # Daily
        self._today = None
        self._daily_trades = 0
        self._daily_start = 0.0

    def _trend(self):
        atr = self.atr[0]
        if atr <= 0: return 0
        rng = self.highest[0] - self.lowest_ind[0]
        if rng / atr < self.p.min_range_atr: return 0
        s = self.slope.slope[0]
        if s > self.p.min_slope and self.ema[0] > self.ema_slow[0]: return 1
        if s < -self.p.min_slope and self.ema[0] < self.ema_slow[0]: return -1
        return 0

    def _new_day(self):
        dt = self.data.datetime.date(0)
        if dt != self._today:
            self._today = dt
            self._daily_trades = 0
            self._daily_start = self.broker.getvalue()
            self._pb_state = 0
            self._signal_price = 0.0
            self._signal_dir = 0

    def _can_enter(self):
        t = self.data.datetime.time(0)
        if t >= datetime.time(self.p.no_entry_after_hour, self.p.no_entry_after_minute):
            return False
        if self._daily_trades >= self.p.max_daily_trades: return False
        if self._daily_start > 0:
            if (self._daily_start - self.broker.getvalue()) / self._daily_start >= self.p.max_daily_loss_pct:
                return False
        return True

    def _size(self, entry, stop):
        risk = abs(entry - stop)
        if risk <= 0: return 0
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
                self._full_reset()
            self._signal_price = 0.0
            return

        # Manage existing position
        if self.position:
            self._manage()
            return

        if not self._can_enter():
            self._signal_price = 0.0
            return

        # Check if pending signal expired
        if self._signal_price > 0:
            expired = (len(self) - self._signal_bar) > 3
            if expired:
                if self.order:
                    self.cancel(self.order)
                self._signal_price = 0.0

        # Scan for new pullback setups
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
                # Left zone — place stop order at signal line
                if self._pb_bars >= self.p.min_pullback_bars and not self.order:
                    atr_now = self.atr[0] if self.atr[0] > 0 else 0.01
                    if trend == 1:
                        sig_price = self.data.high[0] + self.p.signal_offset
                        sig_stop = self._pb_low - self.p.stop_buffer_atr * atr_now
                    else:
                        sig_price = self.data.low[0] - self.p.signal_offset
                        sig_stop = self._pb_low + self.p.stop_buffer_atr * atr_now

                    size = self._size(sig_price, sig_stop)
                    if size > 0:
                        self._signal_price = sig_price
                        self._signal_stop = sig_stop
                        self._signal_dir = trend
                        self._signal_bar = len(self)
                        self._signal_size = size
                        # Place the stop order immediately
                        if trend == 1:
                            self.order = self.buy(size=size, exectype=bt.Order.Stop, price=sig_price)
                        else:
                            self.order = self.sell(size=size, exectype=bt.Order.Stop, price=sig_price)

                self._pb_state = 0
                self._pb_bars = 0
                self._pb_low = 0.0

    def _manage(self):
        d = self._dir
        if d == 0:
            return

        price = self.data.close[0]
        atr = self.atr[0] if self.atr[0] > 0 else 0.01
        pos = abs(self.position.size)

        # Loser timeout
        if len(self) - self._entry_bar > self.p.losers_max_bars:
            if (d == 1 and price < self._entry_price) or \
               (d == -1 and price > self._entry_price):
                self.order = self.close()
                self._full_reset()
                return

        # Check stop (use HIGH for entry accuracy since we entered via stop order)
        active_stop = self._runner_sl if self._lock_done else self._sl
        if d == 1 and self.data.low[0] <= active_stop:
            self.order = self.close()
            self._full_reset()
            return
        if d == -1 and self.data.high[0] >= active_stop:
            self.order = self.close()
            self._full_reset()
            return

        # Lock: check if HIGH reached target (not just close)
        if not self._lock_done:
            if d == 1:
                max_profit = self.data.high[0] - self._entry_price
            else:
                max_profit = self._entry_price - self.data.low[0]

            if max_profit >= self.p.lock_target_atr * atr:
                lock_size = max(1, int(self._total_size * self.p.lock_pct))
                lock_size = min(lock_size, pos - 1)
                if lock_size > 0:
                    if d == 1:
                        self.order = self.sell(size=lock_size)
                    else:
                        self.order = self.buy(size=lock_size)
                    self._lock_done = True
                    self._runner_sl = self._entry_price  # move to breakeven
                return

        # Runner trail
        if self._lock_done and len(self) - self._entry_bar >= self.p.runner_trail_bars:
            if d == 1:
                recent_low = min(self.data.low.get(ago=-i, size=1)[0]
                                 for i in range(self.p.runner_trail_bars))
                new_trail = recent_low - self.p.runner_trail_buffer_atr * atr
                self._runner_sl = max(self._runner_sl, new_trail)
            else:
                recent_high = max(self.data.high.get(ago=-i, size=1)[0]
                                  for i in range(self.p.runner_trail_bars))
                new_trail = recent_high + self.p.runner_trail_buffer_atr * atr
                self._runner_sl = min(self._runner_sl, new_trail)

    def _full_reset(self):
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
        self._signal_price = 0.0
        self._signal_dir = 0

    def notify_order(self, order):
        if order.status in (order.Submitted, order.Accepted):
            return

        if order.status in (order.Completed,):
            # If this was an entry order (signal fill)
            if self._signal_price > 0 and not self.position.size == 0:
                if abs(self.position.size) > 0 and self._entry_bar == 0:
                    # Entry just filled
                    self._entry_price = order.executed.price
                    self._entry_bar = len(self)
                    self._sl = self._signal_stop
                    self._runner_sl = self._signal_stop
                    self._total_size = abs(order.executed.size)
                    self._lock_done = False
                    self._dir = self._signal_dir
                    self._daily_trades += 1
                    self._signal_price = 0.0

        if order.status == order.Canceled:
            self._signal_price = 0.0

        self.order = None

    def notify_trade(self, trade):
        if trade.isclosed and not self.position:
            self._full_reset()
