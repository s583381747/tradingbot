"""
EMA20 Trend Following — Gen 15: 2-Lock + Runner (30/20/50).

Data-driven configuration from probability analysis:
  - 67% reach +1 ATR, 50% reach +2 ATR, 40% reach +3 ATR
  - Conditional: once at 1 ATR, 87%+ continue to 2+
  - Runner captures fat tail (P90=10.7 ATR, P95=14.9 ATR)

Exit structure:
  Lock1: 30% at +1.0 ATR → move stop to breakeven
  Lock2: 20% at +2.5 ATR
  Runner: 50% rides with 10-bar trailing low

Entry: Dual filter + signal line at prev_high + $0.05
Stop: test candle low - 0.3*ATR
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
SIGNAL_OFFSET = 0.05

STOP_BUFFER_ATR = 0.3
LOCK1_ATR = 1.0
LOCK2_ATR = 2.5
LOCK1_PCT = 0.30
LOCK2_PCT = 0.20
# Runner = 50%

RUNNER_TRAIL_BARS = 10
RUNNER_TRAIL_BUFFER = 0.3

RISK_PCT = 0.01
MAX_POSITION_PCT = 0.25
MAX_DAILY_TRADES = 10
MAX_DAILY_LOSS_PCT = 0.02

NO_ENTRY_AFTER_HOUR = 15
NO_ENTRY_AFTER_MINUTE = 30
FORCE_CLOSE_HOUR = 15
FORCE_CLOSE_MINUTE = 58
LOSERS_MAX_BARS = 60


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
        ("lock1_atr", LOCK1_ATR), ("lock2_atr", LOCK2_ATR),
        ("lock1_pct", LOCK1_PCT), ("lock2_pct", LOCK2_PCT),
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
        self.lowest_ind = bt.indicators.Lowest(self.data.low, period=self.p.range_atr_period)

        self.order = None
        self._dir = 0
        self._pb_state = 0
        self._pb_bars = 0
        self._pb_low = 0.0

        # Position
        self._entry_price = 0.0
        self._entry_bar = 0
        self._initial_sl = 0.0
        self._runner_sl = 0.0
        self._total_size = 0
        self._lock1_done = False
        self._lock2_done = False

        # Pending signal order
        self._sig_stop = 0.0
        self._sig_dir = 0
        self._sig_bar = 0
        self._sig_pending = False

        self._today = None
        self._daily_trades = 0
        self._daily_start = 0.0

    def _trend(self):
        a = self.atr[0]
        if a <= 0: return 0
        if (self.highest[0] - self.lowest_ind[0]) / a < self.p.min_range_atr: return 0
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
        return max(3, min(int(val * self.p.risk_pct / risk),
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
            elif self._sig_pending:
                self._sig_pending = False
            return

        # Manage position
        if self.position:
            self._manage()
            if self.order:
                return

        # Cancel expired signal
        if self._sig_pending and (len(self) - self._sig_bar) > 3:
            self._sig_pending = False

        # Scan for entries (even while in position for future pyramiding)
        if not self.position and self._can_enter() and not self._sig_pending:
            self._scan()

    def _scan(self):
        trend = self._trend()
        if trend == 0:
            self._pb_state = 0; self._pb_bars = 0; return

        if trend != self._dir:
            self._dir = trend; self._pb_state = 0; self._pb_bars = 0; self._pb_low = 0.0

        a = self.atr[0] if self.atr[0] > 0 else 999
        zone = a * self.p.pullback_touch_mult

        if trend == 1:
            in_zone = (self.data.low[0] <= self.ema[0] + zone) and \
                      (self.data.low[0] >= self.ema[0] - zone)
        else:
            in_zone = (self.data.high[0] >= self.ema[0] - zone) and \
                      (self.data.high[0] <= self.ema[0] + zone)

        if self._pb_state == 0:
            if in_zone:
                self._pb_state = 1; self._pb_bars = 1
                self._pb_low = self.data.low[0] if trend == 1 else self.data.high[0]
        elif self._pb_state == 1:
            if in_zone:
                self._pb_bars += 1
                if trend == 1: self._pb_low = min(self._pb_low, self.data.low[0])
                else: self._pb_low = max(self._pb_low, self.data.high[0])
            else:
                if self._pb_bars >= self.p.min_pullback_bars:
                    a_now = self.atr[0] if self.atr[0] > 0 else 0.01
                    if trend == 1:
                        sig = self.data.high[0] + self.p.signal_offset
                        stp = self._pb_low - self.p.stop_buffer_atr * a_now
                    else:
                        sig = self.data.low[0] - self.p.signal_offset
                        stp = self._pb_low + self.p.stop_buffer_atr * a_now

                    sz = self._size(sig, stp)
                    if sz > 0:
                        self._sig_stop = stp
                        self._sig_dir = trend
                        self._sig_bar = len(self)
                        self._sig_pending = True
                        if trend == 1:
                            self.order = self.buy(size=sz, exectype=bt.Order.Stop, price=sig)
                        else:
                            self.order = self.sell(size=sz, exectype=bt.Order.Stop, price=sig)

                self._pb_state = 0; self._pb_bars = 0; self._pb_low = 0.0

    def _manage(self):
        d = self._dir
        if d == 0: return

        a = self.atr[0] if self.atr[0] > 0 else 0.01
        pos = abs(self.position.size)

        # Active stop
        sl = self._runner_sl if self._lock1_done else self._initial_sl

        # 1. Stop check (using bar high/low, not close)
        if d == 1 and self.data.low[0] <= sl:
            self.order = self.close(); self._reset(); return
        if d == -1 and self.data.high[0] >= sl:
            self.order = self.close(); self._reset(); return

        # 2. Loser timeout
        if len(self) - self._entry_bar > self.p.losers_max_bars:
            if (d == 1 and self.data.close[0] < self._entry_price) or \
               (d == -1 and self.data.close[0] > self._entry_price):
                self.order = self.close(); self._reset(); return

        # 3. Lock1: 30% at +1.0 ATR (use bar high/low to detect)
        if not self._lock1_done:
            if d == 1: reached = self.data.high[0] >= self._entry_price + self.p.lock1_atr * a
            else: reached = self.data.low[0] <= self._entry_price - self.p.lock1_atr * a
            if reached:
                l1_size = max(1, int(self._total_size * self.p.lock1_pct))
                l1_size = min(l1_size, pos - 1)
                if l1_size > 0:
                    if d == 1: self.order = self.sell(size=l1_size)
                    else: self.order = self.buy(size=l1_size)
                    self._lock1_done = True
                    # Move to breakeven
                    if d == 1: self._runner_sl = max(self._initial_sl, self._entry_price)
                    else: self._runner_sl = min(self._initial_sl, self._entry_price)
                return

        # 4. Lock2: 20% at +2.5 ATR
        if self._lock1_done and not self._lock2_done:
            if d == 1: reached = self.data.high[0] >= self._entry_price + self.p.lock2_atr * a
            else: reached = self.data.low[0] <= self._entry_price - self.p.lock2_atr * a
            if reached:
                l2_size = max(1, int(self._total_size * self.p.lock2_pct))
                l2_size = min(l2_size, pos - 1)
                if l2_size > 0:
                    if d == 1: self.order = self.sell(size=l2_size)
                    else: self.order = self.buy(size=l2_size)
                    self._lock2_done = True
                return

        # 5. Runner trail: 10-bar trailing low
        if self._lock1_done and len(self) - self._entry_bar >= self.p.runner_trail_bars:
            if d == 1:
                rl = min(self.data.low.get(ago=-i, size=1)[0]
                         for i in range(self.p.runner_trail_bars))
                self._runner_sl = max(self._runner_sl, rl - self.p.runner_trail_buffer * a)
            else:
                rh = max(self.data.high.get(ago=-i, size=1)[0]
                         for i in range(self.p.runner_trail_bars))
                self._runner_sl = min(self._runner_sl, rh + self.p.runner_trail_buffer * a)

    def _reset(self):
        self._dir = 0
        self._entry_price = 0.0
        self._entry_bar = 0
        self._initial_sl = 0.0
        self._runner_sl = 0.0
        self._total_size = 0
        self._lock1_done = False
        self._lock2_done = False
        self._pb_state = 0
        self._pb_bars = 0
        self._pb_low = 0.0
        self._sig_pending = False

    def notify_order(self, order):
        if order.status in (order.Submitted, order.Accepted):
            return
        if order.status == order.Completed and self._sig_pending:
            self._entry_price = order.executed.price
            self._entry_bar = len(self)
            self._initial_sl = self._sig_stop
            self._runner_sl = self._sig_stop
            self._total_size = abs(order.executed.size)
            self._lock1_done = False
            self._lock2_done = False
            self._dir = self._sig_dir
            self._daily_trades += 1
            self._sig_pending = False
        if order.status in (order.Canceled, order.Margin, order.Rejected):
            self._sig_pending = False
        self.order = None

    def notify_trade(self, trade):
        if trade.isclosed and not self.position:
            self._reset()
