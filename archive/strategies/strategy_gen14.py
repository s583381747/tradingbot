"""
EMA20 Trend Following — Gen 14: Pyramiding.

Key insight: pandas showed PF>1 because each signal was tested independently.
In real trading: pyramiding = add to position on each new bounce.

Logic:
  1. First bounce in trend → enter (position layer 1)
  2. Price runs, pulls back to EMA20, bounces again → ADD (layer 2)
  3. Each add: raise stop to new bounce low - buffer
  4. Lock 70% of EACH LAYER at +1 ATR from that layer's entry
  5. Runner portions all share the staircase trailing stop
  6. When stop hit → close everything remaining

Each layer has its own lock target but shares the global trailing stop.
Max layers = 10 per day.
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
LOCK_TARGET_ATR = 1.0
LOCK_PCT = 0.70
MAX_LAYERS = 10

RISK_PCT = 0.01
MAX_POSITION_PCT = 0.50  # higher limit for pyramided position
MAX_DAILY_TRADES = 10
MAX_DAILY_LOSS_PCT = 0.02

NO_ENTRY_AFTER_HOUR = 15
NO_ENTRY_AFTER_MINUTE = 30
FORCE_CLOSE_HOUR = 15
FORCE_CLOSE_MINUTE = 58
LOSERS_MAX_BARS = 60  # longer timeout since we're pyramiding


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
        ("max_layers", MAX_LAYERS),
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

        # Pullback scan
        self._pb_state = 0
        self._pb_bars = 0
        self._pb_low = 0.0

        # Global position state
        self._sl = 0.0              # current stop level (ratchets up)
        self._first_entry_bar = 0
        self._layers = []           # [{entry_price, size, locked}]
        self._n_layers = 0

        # Pending signal
        self._pending_entry = False
        self._pending_stop = 0.0
        self._pending_dir = 0
        self._pending_size = 0

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

    def _can_enter(self):
        t = self.data.datetime.time(0)
        if t >= datetime.time(self.p.no_entry_after_hour, self.p.no_entry_after_minute):
            return False
        if self._daily_trades >= self.p.max_daily_trades: return False
        if self._n_layers >= self.p.max_layers: return False
        if self._daily_start > 0:
            if (self._daily_start - self.broker.getvalue()) / self._daily_start >= self.p.max_daily_loss_pct:
                return False
        return True

    def _layer_size(self, entry, stop):
        risk = abs(entry - stop)
        if risk <= 0: return 0
        val = self.broker.getvalue()
        by_risk = int(val * self.p.risk_pct / risk)
        by_cap = int(val * self.p.max_position_pct / entry)
        current_pos_value = abs(self.position.size) * entry if self.position else 0
        remaining_cap = int((val * self.p.max_position_pct - current_pos_value) / entry)
        return max(1, min(by_risk, remaining_cap))

    def next(self):
        if self.order:
            return
        self._new_day()

        t = self.data.datetime.time(0)

        # Force close
        if t >= datetime.time(self.p.force_close_hour, self.p.force_close_minute):
            if self.position:
                self.order = self.close()
                self._full_reset()
            return

        # Manage position (stop check + lock)
        if self.position:
            self._manage()
            if self.order:
                return

        # Scan for new entries (even while in position!)
        if self._can_enter():
            self._scan()

    def _scan(self):
        trend = self._trend()
        if trend == 0:
            self._pb_state = 0
            self._pb_bars = 0
            return

        # If we're in a position, only add in same direction
        if self.position and trend != self._dir and self._dir != 0:
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
                if self._pb_bars >= self.p.min_pullback_bars:
                    atr_now = self.atr[0] if self.atr[0] > 0 else 0.01
                    if trend == 1:
                        sig_price = self.data.high[0] + self.p.signal_offset
                        sig_stop = self._pb_low - self.p.stop_buffer_atr * atr_now
                    else:
                        sig_price = self.data.low[0] - self.p.signal_offset
                        sig_stop = self._pb_low + self.p.stop_buffer_atr * atr_now

                    size = self._layer_size(sig_price, sig_stop)
                    if size > 0:
                        self._pending_entry = True
                        self._pending_stop = sig_stop
                        self._pending_dir = trend
                        self._pending_size = size
                        if trend == 1:
                            self.order = self.buy(size=size, exectype=bt.Order.Stop, price=sig_price)
                        else:
                            self.order = self.sell(size=size, exectype=bt.Order.Stop, price=sig_price)

                self._pb_state = 0
                self._pb_bars = 0
                self._pb_low = 0.0

    def _manage(self):
        d = self._dir
        if d == 0 or not self._layers:
            return

        atr = self.atr[0] if self.atr[0] > 0 else 0.01
        pos = abs(self.position.size)

        # 1. Check global stop
        if d == 1 and self.data.low[0] <= self._sl:
            self.order = self.close()
            self._full_reset()
            return
        if d == -1 and self.data.high[0] >= self._sl:
            self.order = self.close()
            self._full_reset()
            return

        # 2. Loser timeout (from first entry)
        if self._first_entry_bar > 0 and len(self) - self._first_entry_bar > self.p.losers_max_bars:
            total_cost = sum(l["entry"] * l["size"] for l in self._layers)
            total_shares = sum(l["size"] for l in self._layers)
            avg_entry = total_cost / total_shares if total_shares > 0 else 0
            if (d == 1 and self.data.close[0] < avg_entry) or \
               (d == -1 and self.data.close[0] > avg_entry):
                self.order = self.close()
                self._full_reset()
                return

        # 3. Lock profit on individual layers
        for layer in self._layers:
            if layer["locked"]:
                continue
            profit = (self.data.high[0] - layer["entry"]) * d if d == 1 else \
                     (layer["entry"] - self.data.low[0]) * d
            if profit >= self.p.lock_target_atr * atr:
                lock_size = max(1, int(layer["size"] * self.p.lock_pct))
                lock_size = min(lock_size, pos - 1)
                if lock_size > 0:
                    if d == 1:
                        self.order = self.sell(size=lock_size)
                    else:
                        self.order = self.buy(size=lock_size)
                    layer["locked"] = True
                    # Move global stop to at least breakeven of this layer
                    if d == 1:
                        self._sl = max(self._sl, layer["entry"])
                    else:
                        self._sl = min(self._sl, layer["entry"])
                    return

    def _full_reset(self):
        self._dir = 0
        self._sl = 0.0
        self._first_entry_bar = 0
        self._layers = []
        self._n_layers = 0
        self._pb_state = 0
        self._pb_bars = 0
        self._pb_low = 0.0
        self._pending_entry = False

    def notify_order(self, order):
        if order.status in (order.Submitted, order.Accepted):
            return

        if order.status == order.Completed and self._pending_entry:
            # New layer entered
            entry_price = order.executed.price
            entry_size = abs(order.executed.size)

            # Update global stop (raise to new bounce low)
            if self._dir == 1:
                new_sl = self._pending_stop
                self._sl = max(self._sl, new_sl) if self._sl > 0 else new_sl
            else:
                new_sl = self._pending_stop
                self._sl = min(self._sl, new_sl) if self._sl < 999999 else new_sl

            self._layers.append({"entry": entry_price, "size": entry_size, "locked": False})
            self._n_layers += 1
            self._daily_trades += 1

            if self._first_entry_bar == 0:
                self._first_entry_bar = len(self)
                self._dir = self._pending_dir

            self._pending_entry = False

        if order.status in (order.Canceled, order.Margin, order.Rejected):
            self._pending_entry = False

        self.order = None

    def notify_trade(self, trade):
        if trade.isclosed and not self.position:
            self._full_reset()
