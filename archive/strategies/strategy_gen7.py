"""
Chop-Box Trend-Following Strategy — Gen 7.

Gen 6 had price-range chop box but too loose trend filter (1801 trades, PF=0.23).
Gen 7 fix: add directional breakout requirement.

Chop box logic:
  1. Range/ATR < threshold → in chop → record box_high / box_low
  2. Price CLOSES above box_high → bullish breakout → track breakout direction
  3. Price CLOSES below box_low → bearish breakout
  4. After breakout + EMA alignment → look for pullback entry
  5. If range compresses again → new chop box

Trend = breakout direction + EMA20 vs EMA50 alignment.
No entry unless both agree.
"""

from __future__ import annotations

import datetime
from collections import deque

import backtrader as bt

# ══════════════════════════════════════════════════════════════════
# HYPERPARAMETERS
# ══════════════════════════════════════════════════════════════════

# Core Indicators
EMA_PERIOD = 20                 # FIXED — 20EMA is the core concept
EMA_SLOW_PERIOD = 50
ATR_PERIOD = 14

# Chop Box Detection (REWRITTEN — price range based)
CHOP_RANGE_PERIOD = 20          # bars to measure price range
CHOP_RANGE_ATR_RATIO = 3.5     # range/ATR below this = chop (consolidation)
CHOP_BOX_MIN_BARS = 10         # min bars in box before breakout counts

# Pullback Entry
PULLBACK_TOUCH_MULT = 1.2       # x ATR from EMA = pullback zone
MIN_PULLBACK_BARS = 1           # min bars near EMA before bounce

# Initial Stop (below pullback extreme = EMA test candle low)
INITIAL_STOP_ATR_MULT = 2.0

# 3-Portion Take Profit Trail (30/40/30)
TP_ACTIVATE_ATR = 3.0           # profit in ATR before trails activate
TP1_PCT = 0.30                  # tightest trail (candle low)
TP2_PCT = 0.40                  # medium trail (EMA - mid ATR)
TP3_PCT = 0.30                  # widest trail (EMA - wide ATR)
TP1_CANDLE_OFFSET = 0.50        # $ offset below candle low for TP1
TP2_EMA_ATR_MULT = 3.0          # ATR below EMA for TP2
TP3_EMA_ATR_MULT = 6.0          # ATR below EMA for TP3

# Addon Entries (sized to breakeven)
ENABLE_ADDON = True
MAX_ADDONS = 1
ADDON_PULLBACK_MULT = 1.2       # x ATR for addon pullback zone
ADDON_MIN_BARS = 2              # min bars in zone before addon

# Risk Management
RISK_PCT = 0.01
MAX_POSITION_PCT = 0.25
MAX_DAILY_TRADES = 6
MAX_DAILY_LOSS_PCT = 0.02

# Session Times
NO_ENTRY_AFTER_HOUR = 15
NO_ENTRY_AFTER_MINUTE = 30
FORCE_CLOSE_HOUR = 15
FORCE_CLOSE_MINUTE = 58

# Time-Based Exit
LOSERS_MAX_BARS = 45


# ══════════════════════════════════════════════════════════════════
# STRATEGY
# ══════════════════════════════════════════════════════════════════

class Strategy(bt.Strategy):
    """Price-range chop-box trend-following with cascading TP and breakeven addons."""

    params = (
        ("ema_period", EMA_PERIOD),
        ("ema_slow_period", EMA_SLOW_PERIOD),
        ("atr_period", ATR_PERIOD),
        ("chop_range_period", CHOP_RANGE_PERIOD),
        ("chop_range_atr_ratio", CHOP_RANGE_ATR_RATIO),
        ("chop_box_min_bars", CHOP_BOX_MIN_BARS),
        ("pullback_touch_mult", PULLBACK_TOUCH_MULT),
        ("min_pullback_bars", MIN_PULLBACK_BARS),
        ("initial_stop_atr_mult", INITIAL_STOP_ATR_MULT),
        ("tp_activate_atr", TP_ACTIVATE_ATR),
        ("tp1_pct", TP1_PCT),
        ("tp2_pct", TP2_PCT),
        ("tp3_pct", TP3_PCT),
        ("tp1_candle_offset", TP1_CANDLE_OFFSET),
        ("tp2_ema_atr_mult", TP2_EMA_ATR_MULT),
        ("tp3_ema_atr_mult", TP3_EMA_ATR_MULT),
        ("enable_addon", ENABLE_ADDON),
        ("max_addons", MAX_ADDONS),
        ("addon_pullback_mult", ADDON_PULLBACK_MULT),
        ("addon_min_bars", ADDON_MIN_BARS),
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

    # ─────────────────────────────────────── init
    def __init__(self):
        self.ema = bt.indicators.EMA(self.data.close, period=self.p.ema_period)
        self.ema_slow = bt.indicators.EMA(self.data.close, period=self.p.ema_slow_period)
        self.atr = bt.indicators.ATR(self.data, period=self.p.atr_period)
        self.highest = bt.indicators.Highest(self.data.high, period=self.p.chop_range_period)
        self.lowest = bt.indicators.Lowest(self.data.low, period=self.p.chop_range_period)

        self.order = None
        self.trade_direction = 0

        # Chop box state (price-range based)
        self._in_chop = True
        self._box_high = 0.0
        self._box_low = 999999.0
        self._chop_bar_count = 0
        self._breakout_dir = 0  # +1 bullish breakout, -1 bearish, 0 none

        # Pullback state
        self._pb_state = 0
        self._pb_dir = 0
        self._pb_bar_count = 0
        self._pb_extreme = 0.0

        # Position tracking
        self._entries = []
        self._total_size = 0
        self._avg_entry_price = 0.0
        self._addon_count = 0
        self._addon_pb_bars = 0
        self._entry_bar = 0

        # 3-portion TP state
        self._tp_active = False
        self._tp1_size = 0
        self._tp2_size = 0
        self._tp3_size = 0
        self._tp1_done = False
        self._tp2_done = False
        self._tp1_stop = 0.0
        self._tp2_stop = 0.0
        self._tp3_stop = 0.0
        self._initial_stop = 0.0

        # Daily state
        self._today = None
        self._daily_trades = 0
        self._daily_pnl = 0.0
        self._daily_start_value = 0.0

    # ─────────────────────────────────────── daily reset
    def _check_new_day(self):
        dt = self.data.datetime.date(0)
        if dt != self._today:
            self._today = dt
            self._daily_trades = 0
            self._daily_pnl = 0.0
            self._daily_start_value = self.broker.getvalue()
            # Reset chop box for new session (overnight gap makes old box irrelevant)
            self._in_chop = True
            self._box_high = 0.0
            self._box_low = 999999.0
            self._chop_bar_count = 0
            self._breakout_dir = 0
            self._pb_state = 0
            self._pb_dir = 0
            self._pb_bar_count = 0
            self._pb_extreme = 0.0
            self._addon_pb_bars = 0

    # ─────────────────────────────────────── session helpers
    def _past_entry_cutoff(self) -> bool:
        t = self.data.datetime.time(0)
        return t >= datetime.time(self.p.no_entry_after_hour, self.p.no_entry_after_minute)

    def _past_force_close(self) -> bool:
        t = self.data.datetime.time(0)
        return t >= datetime.time(self.p.force_close_hour, self.p.force_close_minute)

    def _daily_limit_reached(self) -> bool:
        if self._daily_trades >= self.p.max_daily_trades:
            return True
        if self._daily_start_value > 0:
            loss = (self._daily_start_value - self.broker.getvalue()) / self._daily_start_value
            if loss >= self.p.max_daily_loss_pct:
                return True
        return False

    # ─────────────────────────────────────── chop box (price-range + directional breakout)
    def _update_chop_box(self):
        """
        Price-range chop detection with directional breakout tracking.
        Chop = N-bar range / ATR < threshold.
        Breakout = price closes beyond box boundary in a direction.
        """
        atr_val = self.atr[0]
        if atr_val <= 0 or len(self) < self.p.chop_range_period:
            return

        range_high = self.highest[0]
        range_low = self.lowest[0]
        price_range = range_high - range_low
        range_ratio = price_range / atr_val

        if self._in_chop:
            self._chop_bar_count += 1

            # Check for directional breakout BEFORE updating box bounds
            # Use frozen box bounds from when chop started
            if self._chop_bar_count >= self.p.chop_box_min_bars and self._box_high > 0:
                if self.data.close[0] > self._box_high:
                    self._in_chop = False
                    self._breakout_dir = 1
                    return
                elif self.data.close[0] < self._box_low:
                    self._in_chop = False
                    self._breakout_dir = -1
                    return

            # Expand box only while range stays compressed
            if range_ratio < self.p.chop_range_atr_ratio:
                self._box_high = max(self._box_high, self.data.high[0])
                self._box_low = min(self._box_low, self.data.low[0])
        else:
            # In trend: check if range compresses → new chop box
            if range_ratio < self.p.chop_range_atr_ratio:
                self._in_chop = True
                self._box_high = range_high
                self._box_low = range_low
                self._chop_bar_count = 1
                self._breakout_dir = 0

    # ─────────────────────────────────────── trend detection
    def _detect_trend(self) -> int:
        """Trend = breakout direction + EMA alignment must agree."""
        # Need a breakout direction
        if self._breakout_dir == 0:
            return 0

        # EMA alignment must confirm breakout direction
        if self._breakout_dir == 1:
            if self.ema[0] > self.ema_slow[0]:
                return 1
        elif self._breakout_dir == -1:
            if self.ema[0] < self.ema_slow[0]:
                return -1

        return 0

    # ─────────────────────────────────────── position sizing
    def _calc_size(self, entry_price: float, stop_price: float) -> int:
        risk = abs(entry_price - stop_price)
        if risk <= 0:
            return 0
        account = self.broker.getvalue()
        size_by_risk = int(account * self.p.risk_pct / risk)
        size_by_capital = int(account * self.p.max_position_pct / entry_price)
        return max(1, min(size_by_risk, size_by_capital))

    # ═══════════════════════════════════════ NEXT
    def next(self):
        if self.order:
            return

        self._check_new_day()
        self._update_chop_box()

        # Force close at end of session
        if self._past_force_close() and self.position:
            self.order = self.close()
            self._reset_position_state()
            return

        # Manage open position
        if self.position:
            self._manage_position()
            return

        # No entries: in chop, past cutoff, or at daily limit
        if self._in_chop or self._past_entry_cutoff() or self._daily_limit_reached():
            return

        self._run_entry_logic()

    # ─────────────────────────────────────── entry logic
    def _run_entry_logic(self):
        trend = self._detect_trend()
        if trend == 0:
            self._pb_state = 0
            self._pb_bar_count = 0
            return

        # Reset on trend flip
        if trend != self._pb_dir:
            self._pb_dir = trend
            self._pb_state = 0
            self._pb_bar_count = 0
            self._pb_extreme = 0.0

        atr_val = self.atr[0] if self.atr[0] > 0 else 999
        dist = abs(self.data.close[0] - self.ema[0])
        in_zone = dist <= atr_val * self.p.pullback_touch_mult

        if self._pb_state == 0:
            if in_zone:
                self._pb_state = 1
                self._pb_bar_count = 1
                self._pb_extreme = self.data.low[0] if trend == 1 else self.data.high[0]

        elif self._pb_state == 1:
            if in_zone:
                self._pb_bar_count += 1
                if trend == 1:
                    self._pb_extreme = min(self._pb_extreme, self.data.low[0])
                else:
                    self._pb_extreme = max(self._pb_extreme, self.data.high[0])
            else:
                # Left the zone — check bounce confirmation
                on_side = (
                    (trend == 1 and self.data.close[0] > self.ema[0])
                    or (trend == -1 and self.data.close[0] < self.ema[0])
                )
                momentum = False
                if len(self) >= 2:
                    if trend == 1 and self.data.close[0] > self.data.high[-1]:
                        momentum = True
                    elif trend == -1 and self.data.close[0] < self.data.low[-1]:
                        momentum = True

                if on_side and self._pb_bar_count >= self.p.min_pullback_bars and momentum:
                    self._execute_entry(trend)

                self._pb_state = 0
                self._pb_bar_count = 0
                self._pb_extreme = 0.0

    # ─────────────────────────────────────── execute entry
    def _execute_entry(self, direction: int):
        entry_price = self.data.close[0]
        atr_val = self.atr[0] if self.atr[0] > 0 else 0.01

        if direction == 1:
            stop = self._pb_extreme - self.p.initial_stop_atr_mult * atr_val
        else:
            stop = self._pb_extreme + self.p.initial_stop_atr_mult * atr_val

        size = self._calc_size(entry_price, stop)
        if size <= 0:
            return

        self.trade_direction = direction
        self._entries = [{"price": entry_price, "size": size, "bar": len(self)}]
        self._total_size = size
        self._avg_entry_price = entry_price
        self._addon_count = 0
        self._addon_pb_bars = 0
        self._entry_bar = len(self)

        self._initial_stop = stop
        self._tp_active = False
        self._tp1_done = False
        self._tp2_done = False

        if size >= 3:
            self._tp1_size = max(1, int(size * self.p.tp1_pct))
            self._tp2_size = max(1, int(size * self.p.tp2_pct))
            self._tp3_size = size - self._tp1_size - self._tp2_size
            if self._tp3_size <= 0:
                self._tp3_size = 1
                self._tp2_size = size - self._tp1_size - 1
        else:
            self._tp1_size = 0
            self._tp2_size = 0
            self._tp3_size = size
            self._tp1_done = True
            self._tp2_done = True

        self._tp1_stop = stop
        self._tp2_stop = stop
        self._tp3_stop = stop

        self._daily_trades += 1

        if direction == 1:
            self.order = self.buy(size=size)
        else:
            self.order = self.sell(size=size)

    # ─────────────────────────────────────── manage position
    def _manage_position(self):
        d = self.trade_direction
        if d == 0:
            return

        price = self.data.close[0]
        bar_low = self.data.low[0]
        bar_high = self.data.high[0]
        atr_val = self.atr[0] if self.atr[0] > 0 else 0.01

        if len(self) - self._entry_bar > self.p.losers_max_bars:
            is_losing = (d == 1 and price < self._avg_entry_price) or \
                        (d == -1 and price > self._avg_entry_price)
            if is_losing:
                self.order = self.close()
                self._reset_position_state()
                return

        if not self._tp_active:
            profit_atr = (price - self._avg_entry_price) * d / atr_val
            if profit_atr >= self.p.tp_activate_atr:
                self._tp_active = True

        self._update_tp_trails()

        if self._tp_active:
            if not self._tp1_done and self._tp1_size > 0:
                hit = (d == 1 and bar_low <= self._tp1_stop) or \
                      (d == -1 and bar_high >= self._tp1_stop)
                if hit:
                    close_size = min(self._tp1_size, abs(self.position.size) - 1)
                    if close_size > 0:
                        self.order = self.sell(size=close_size) if d == 1 else self.buy(size=close_size)
                        self._tp1_done = True
                        return

            if self._tp1_done and not self._tp2_done and self._tp2_size > 0:
                hit = (d == 1 and bar_low <= self._tp2_stop) or \
                      (d == -1 and bar_high >= self._tp2_stop)
                if hit:
                    close_size = min(self._tp2_size, abs(self.position.size) - 1)
                    if close_size > 0:
                        self.order = self.sell(size=close_size) if d == 1 else self.buy(size=close_size)
                        self._tp2_done = True
                        return

            if self._tp1_done and self._tp2_done:
                hit = (d == 1 and bar_low <= self._tp3_stop) or \
                      (d == -1 and bar_high >= self._tp3_stop)
                if hit:
                    self.order = self.close()
                    self._reset_position_state()
                    return
        else:
            hit = (d == 1 and bar_low <= self._initial_stop) or \
                  (d == -1 and bar_high >= self._initial_stop)
            if hit:
                self.order = self.close()
                self._reset_position_state()
                return

        if self.p.enable_addon and self._addon_count < self.p.max_addons:
            if not self._past_entry_cutoff():
                self._check_addon_entry()

    # ─────────────────────────────────────── update TP trails
    def _update_tp_trails(self):
        if not self._tp_active:
            return

        d = self.trade_direction
        atr_val = self.atr[0] if self.atr[0] > 0 else 0.01

        if not self._tp1_done and len(self) >= 2:
            if d == 1:
                new = self.data.low[-1] - self.p.tp1_candle_offset
                self._tp1_stop = max(self._tp1_stop, new)
            else:
                new = self.data.high[-1] + self.p.tp1_candle_offset
                self._tp1_stop = min(self._tp1_stop, new)

        if not self._tp2_done:
            if d == 1:
                new = self.ema[0] - self.p.tp2_ema_atr_mult * atr_val
                self._tp2_stop = max(self._tp2_stop, new)
            else:
                new = self.ema[0] + self.p.tp2_ema_atr_mult * atr_val
                self._tp2_stop = min(self._tp2_stop, new)

        if d == 1:
            new = self.ema[0] - self.p.tp3_ema_atr_mult * atr_val
            self._tp3_stop = max(self._tp3_stop, new)
        else:
            new = self.ema[0] + self.p.tp3_ema_atr_mult * atr_val
            self._tp3_stop = min(self._tp3_stop, new)

    # ─────────────────────────────────────── addon entry
    def _check_addon_entry(self):
        d = self.trade_direction
        trend = self._detect_trend()
        if trend != d:
            self._addon_pb_bars = 0
            return

        price = self.data.close[0]
        current_size = abs(self.position.size)
        unrealized = (price - self._avg_entry_price) * d * current_size
        if unrealized <= 0:
            self._addon_pb_bars = 0
            return

        atr_val = self.atr[0] if self.atr[0] > 0 else 999
        dist = abs(price - self.ema[0])
        if dist <= atr_val * self.p.addon_pullback_mult:
            self._addon_pb_bars += 1
        else:
            self._addon_pb_bars = 0
            return

        if self._addon_pb_bars < self.p.addon_min_bars:
            return

        if len(self) < 2:
            return
        if d == 1 and price <= self.data.high[-1]:
            return
        if d == -1 and price >= self.data.low[-1]:
            return

        self._execute_addon()

    def _execute_addon(self):
        d = self.trade_direction
        entry_price = self.data.close[0]
        atr_val = self.atr[0] if self.atr[0] > 0 else 0.01

        if d == 1:
            addon_stop = self.data.low[0] - self.p.initial_stop_atr_mult * atr_val
        else:
            addon_stop = self.data.high[0] + self.p.initial_stop_atr_mult * atr_val

        risk_per_share = abs(entry_price - addon_stop)
        if risk_per_share <= 0:
            return

        current_size = abs(self.position.size)
        unrealized = (entry_price - self._avg_entry_price) * d * current_size
        addon_size = int(unrealized / risk_per_share)
        if addon_size <= 0:
            return

        max_total = int(self.broker.getvalue() * self.p.max_position_pct / entry_price)
        addon_size = min(addon_size, max_total - current_size)
        if addon_size <= 0:
            return

        old_cost = self._avg_entry_price * self._total_size
        self._total_size += addon_size
        self._avg_entry_price = (old_cost + entry_price * addon_size) / self._total_size
        self._addon_count += 1
        self._addon_pb_bars = 0
        self._entries.append({"price": entry_price, "size": addon_size, "bar": len(self)})

        self._tp3_size += addon_size

        if d == 1:
            self.order = self.buy(size=addon_size)
        else:
            self.order = self.sell(size=addon_size)

    # ─────────────────────────────────────── reset
    def _reset_position_state(self):
        self.trade_direction = 0
        self._entries = []
        self._total_size = 0
        self._avg_entry_price = 0.0
        self._addon_count = 0
        self._addon_pb_bars = 0
        self._entry_bar = 0
        self._tp_active = False
        self._tp1_done = False
        self._tp2_done = False
        self._tp1_size = 0
        self._tp2_size = 0
        self._tp3_size = 0
        self._tp1_stop = 0.0
        self._tp2_stop = 0.0
        self._tp3_stop = 0.0
        self._initial_stop = 0.0
        self._pb_state = 0

    # ─────────────────────────────────────── notifications
    def notify_order(self, order):
        if order.status in (order.Submitted, order.Accepted):
            return
        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        self._daily_pnl += trade.pnlcomm
        if not self.position:
            self._reset_position_state()
