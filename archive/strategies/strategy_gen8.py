"""
EMA20 Trend Following — Gen 8: Let Profits Run.

Core principle: 快砍亏损，让利润奔跑。
- NO fixed target. NO cascading TP. NO complexity.
- Entry: EMA20 pullback bounce (wick test + close > prev high)
- Stop: Below pullback low - buffer
- Trail: EMA20 - N*ATR (ride the trend until EMA breaks)
- Trend = price > EMA20 > EMA50 (simple alignment)
- Exit only when: trail hit, session end, or loser timeout

The EMA20 IS the strategy. Entry tests it, trail follows it.
"""

from __future__ import annotations
import datetime

import backtrader as bt

# ══════════════════════════════════════════════════════════════════
# HYPERPARAMETERS
# ══════════════════════════════════════════════════════════════════

# Core
EMA_PERIOD = 20
EMA_SLOW_PERIOD = 50
ATR_PERIOD = 14

# Entry: EMA20 pullback zone
PULLBACK_TOUCH_MULT = 1.0       # wick must reach within 1.0 ATR of EMA20
MIN_PULLBACK_BARS = 1           # min bars touching EMA zone before bounce

# Stop Loss: just below test candle extreme + small ATR buffer
STOP_BUFFER_ATR = 0.3           # buffer below test candle low (small — structural stop)

# Trailing Stop (the core of "let profits run")
TRAIL_ACTIVATE_ATR = 1.0        # start trailing after 1 ATR profit
TRAIL_EMA_ATR_MULT = 1.5        # trail at EMA20 - this * ATR (breathing room)

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


# ══════════════════════════════════════════════════════════════════
# STRATEGY
# ══════════════════════════════════════════════════════════════════

class Strategy(bt.Strategy):

    params = (
        ("ema_period", EMA_PERIOD),
        ("ema_slow_period", EMA_SLOW_PERIOD),
        ("atr_period", ATR_PERIOD),
        ("pullback_touch_mult", PULLBACK_TOUCH_MULT),
        ("min_pullback_bars", MIN_PULLBACK_BARS),
        ("stop_buffer_atr", STOP_BUFFER_ATR),
        ("trail_activate_atr", TRAIL_ACTIVATE_ATR),
        ("trail_ema_atr_mult", TRAIL_EMA_ATR_MULT),
        ("risk_pct", RISK_PCT),
        ("max_position_pct", MAX_POSITION_PCT),
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

        self.order = None

        # Entry state
        self._direction = 0         # 1=long, -1=short, 0=flat
        self._pb_state = 0          # 0=waiting, 1=in pullback zone
        self._pb_bars = 0
        self._pb_extreme = 0.0      # pullback low (long) or high (short)

        # Position state
        self._entry_price = 0.0
        self._entry_bar = 0
        self._initial_stop = 0.0
        self._trail_stop = 0.0
        self._trail_active = False

        # Daily
        self._today = None
        self._daily_trades = 0
        self._daily_pnl = 0.0
        self._daily_start = 0.0

    # ─── helpers ───
    def _trend(self) -> int:
        """Bull = close > EMA20 > EMA50. Bear = close < EMA20 < EMA50."""
        if self.data.close[0] > self.ema[0] and self.ema[0] > self.ema_slow[0]:
            return 1
        if self.data.close[0] < self.ema[0] and self.ema[0] < self.ema_slow[0]:
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

    def _can_enter(self) -> bool:
        t = self.data.datetime.time(0)
        if t >= datetime.time(self.p.no_entry_after_hour, self.p.no_entry_after_minute):
            return False
        if self._daily_trades >= self.p.max_daily_trades:
            return False
        if self._daily_start > 0:
            loss = (self._daily_start - self.broker.getvalue()) / self._daily_start
            if loss >= self.p.max_daily_loss_pct:
                return False
        return True

    def _calc_size(self, entry, stop):
        risk = abs(entry - stop)
        if risk <= 0:
            return 0
        val = self.broker.getvalue()
        s1 = int(val * self.p.risk_pct / risk)
        s2 = int(val * self.p.max_position_pct / entry)
        return max(1, min(s1, s2))

    # ─── main loop ───
    def next(self):
        if self.order:
            return

        self._new_day()

        # Force close
        t = self.data.datetime.time(0)
        if t >= datetime.time(self.p.force_close_hour, self.p.force_close_minute):
            if self.position:
                self.order = self.close()
                self._reset()
            return

        if self.position:
            self._manage()
            return

        if not self._can_enter():
            return

        self._entry_logic()

    # ─── entry ───
    def _entry_logic(self):
        trend = self._trend()
        if trend == 0:
            self._pb_state = 0
            self._pb_bars = 0
            return

        # Reset on direction change
        if trend != self._direction:
            self._direction = trend
            self._pb_state = 0
            self._pb_bars = 0
            self._pb_extreme = 0.0

        atr = self.atr[0] if self.atr[0] > 0 else 999
        zone = atr * self.p.pullback_touch_mult

        # Wick-based pullback detection
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
                self._pb_extreme = self.data.low[0] if trend == 1 else self.data.high[0]

        elif self._pb_state == 1:
            if in_zone:
                self._pb_bars += 1
                if trend == 1:
                    self._pb_extreme = min(self._pb_extreme, self.data.low[0])
                else:
                    self._pb_extreme = max(self._pb_extreme, self.data.high[0])
            else:
                # Left zone → check bounce
                bounce = False
                if len(self) >= 2:
                    if trend == 1 and self.data.close[0] > self.data.high[-1]:
                        bounce = True
                    elif trend == -1 and self.data.close[0] < self.data.low[-1]:
                        bounce = True

                on_side = (trend == 1 and self.data.close[0] > self.ema[0]) or \
                          (trend == -1 and self.data.close[0] < self.ema[0])

                if bounce and on_side and self._pb_bars >= self.p.min_pullback_bars:
                    self._enter(trend)

                self._pb_state = 0
                self._pb_bars = 0
                self._pb_extreme = 0.0

    def _enter(self, direction):
        price = self.data.close[0]
        atr = self.atr[0] if self.atr[0] > 0 else 0.01

        # Stop just below test candle low (long) / above test candle high (short)
        # _pb_extreme = lowest low during pullback (long) or highest high (short)
        if direction == 1:
            stop = self._pb_extreme - self.p.stop_buffer_atr * atr
        else:
            stop = self._pb_extreme + self.p.stop_buffer_atr * atr

        size = self._calc_size(price, stop)
        if size <= 0:
            return

        self._entry_price = price
        self._entry_bar = len(self)
        self._initial_stop = stop
        self._trail_stop = stop
        self._trail_active = False
        self._direction = direction
        self._daily_trades += 1

        if direction == 1:
            self.order = self.buy(size=size)
        else:
            self.order = self.sell(size=size)

    # ─── position management: let profits run ───
    def _manage(self):
        d = self._direction
        if d == 0:
            return

        price = self.data.close[0]
        atr = self.atr[0] if self.atr[0] > 0 else 0.01

        # 1. Loser timeout
        if len(self) - self._entry_bar > self.p.losers_max_bars:
            pnl = (price - self._entry_price) * d
            if pnl < 0:
                self.order = self.close()
                self._reset()
                return

        # 2. Activate trail when profit reaches threshold
        if not self._trail_active:
            profit = (price - self._entry_price) * d
            if profit >= self.p.trail_activate_atr * atr:
                self._trail_active = True

        # 3. Update trailing stop: EMA20 - N*ATR (only ratchets in favorable direction)
        if self._trail_active:
            if d == 1:
                new_trail = self.ema[0] - self.p.trail_ema_atr_mult * atr
                self._trail_stop = max(self._trail_stop, new_trail)
            else:
                new_trail = self.ema[0] + self.p.trail_ema_atr_mult * atr
                self._trail_stop = min(self._trail_stop, new_trail)

        # 4. Check stop hit
        if d == 1:
            hit = self.data.low[0] <= self._trail_stop
        else:
            hit = self.data.high[0] >= self._trail_stop

        if hit:
            self.order = self.close()
            self._reset()

    def _reset(self):
        self._direction = 0
        self._entry_price = 0.0
        self._entry_bar = 0
        self._initial_stop = 0.0
        self._trail_stop = 0.0
        self._trail_active = False
        self._pb_state = 0
        self._pb_bars = 0
        self._pb_extreme = 0.0

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
