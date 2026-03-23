"""
Chop-Box Trend-Following Strategy — Gen 3 (SIP ATR recalibration + wide stop).

Gen 3 changes from Gen 0:
  - INITIAL_STOP_ATR_MULT 2.0→5.0: wide stop proven better on 2y data
  - CHOP_SLOPE_THRESHOLD: adjusted for SIP ATR (SIP ATR ~27% larger than IEX)
  - TP trails slightly tightened to compensate for larger SIP ATR
  - Logic unchanged — only parameter recalibration for SIP data quality
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
EMA_SLOPE_PERIOD = 3            # ← was 5; faster = better trend detection (PF +113%)

# Chop Box Detection
CHOP_SLOPE_AVG_PERIOD = 20      # bars to average abs slope
CHOP_SLOPE_THRESHOLD = 0.012    # ← was 0.005; 0.012 filters garbage (PF 0.10→1.41)
CHOP_BOX_MIN_BARS = 5           # min flat bars before breakout allowed

# Pullback Entry
PULLBACK_TOUCH_MULT = 1.2       # x ATR from EMA = pullback zone
MIN_PULLBACK_BARS = 1           # min bars near EMA before bounce

# Initial Stop (below pullback extreme = EMA test candle low)
INITIAL_STOP_ATR_MULT = 5.0     # Gen 3: wide stop — 2y IEX showed 7.0 best; 5.0 for SIP (ATR 27% bigger)

# 3-Portion Take Profit Trail (30/40/30)
TP_ACTIVATE_ATR = 5.0           # ← was 2.0; patient activation preserves edge
TP1_PCT = 0.30                  # tightest trail (candle low - offset)
TP2_PCT = 0.40                  # medium trail (EMA - mid ATR)
TP3_PCT = 0.30                  # widest trail (EMA - wide ATR)
TP1_CANDLE_OFFSET = 0.50        # Gen 3: SIP has full candle range, 0.50 should be enough
TP2_EMA_ATR_MULT = 4.0          # Gen 3: tighter for SIP (SIP ATR 27% bigger, so 4.0*1.27≈5.0 effective)
TP3_EMA_ATR_MULT = 6.0          # Gen 3: tighter for SIP (6.0*1.27≈7.6 effective)

# Trend Flicker Tolerance
TREND_FLICKER_BARS = 3          # tolerate this many trend=0 bars before resetting pullback

# Addon Entries (sized to breakeven)
ENABLE_ADDON = True
MAX_ADDONS = 1                  # ← was 2; conservative pyramiding
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
LOSERS_MAX_BARS = 45            # ← was 60; cut losers faster (PF 1.41→1.64)


# ══════════════════════════════════════════════════════════════════
# INDICATORS
# ══════════════════════════════════════════════════════════════════

class EMASlope(bt.Indicator):
    """Normalized slope of EMA over N bars (% per bar)."""
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


# ══════════════════════════════════════════════════════════════════
# STRATEGY
# ══════════════════════════════════════════════════════════════════

class Strategy(bt.Strategy):
    """Chop-box trend-following with cascading 3-portion TP and breakeven addons."""

    params = (
        ("ema_period", EMA_PERIOD),
        ("ema_slow_period", EMA_SLOW_PERIOD),
        ("atr_period", ATR_PERIOD),
        ("ema_slope_period", EMA_SLOPE_PERIOD),
        ("chop_slope_avg_period", CHOP_SLOPE_AVG_PERIOD),
        ("chop_slope_threshold", CHOP_SLOPE_THRESHOLD),
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
        ("trend_flicker_bars", TREND_FLICKER_BARS),
    )

    # ─────────────────────────────────────── init
    def __init__(self):
        self.ema = bt.indicators.EMA(self.data.close, period=self.p.ema_period)
        self.ema_slow = bt.indicators.EMA(self.data.close, period=self.p.ema_slow_period)
        self.atr = bt.indicators.ATR(self.data, period=self.p.atr_period)
        self.ema_slope = EMASlope(
            self.data,
            ema_period=self.p.ema_period,
            slope_period=self.p.ema_slope_period,
        )

        self.order = None
        self.trade_direction = 0

        # Chop box state
        self._slope_history = deque(maxlen=self.p.chop_slope_avg_period)
        self._in_chop = True
        self._box_high = 0.0
        self._box_low = 999999.0
        self._chop_bar_count = 0

        # Pullback state
        self._pb_state = 0
        self._pb_dir = 0
        self._pb_bar_count = 0
        self._pb_extreme = 0.0
        self._trend_zero_count = 0  # flicker tolerance counter

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
            self._in_chop = True
            self._box_high = 0.0
            self._box_low = 999999.0
            self._chop_bar_count = 0
            self._slope_history.clear()
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

    # ─────────────────────────────────────── chop box
    def _update_chop_box(self):
        """Track chop box state. Chop = avg abs slope below threshold."""
        slope = self.ema_slope.slope[0]
        self._slope_history.append(abs(slope))

        # Need at least 5 bars of history to start evaluating
        if len(self._slope_history) < 5:
            self._box_high = max(self._box_high, self.data.high[0])
            self._box_low = min(self._box_low, self.data.low[0])
            self._chop_bar_count += 1
            return

        avg_abs_slope = sum(self._slope_history) / len(self._slope_history)

        if self._in_chop:
            # Check for breakout: slope picked up AND price outside box
            if (avg_abs_slope >= self.p.chop_slope_threshold
                    and self._chop_bar_count >= self.p.chop_box_min_bars):
                if (self.data.close[0] > self._box_high
                        or self.data.close[0] < self._box_low):
                    self._in_chop = False
                    return

            # Still in chop: expand box only during flat periods
            if avg_abs_slope < self.p.chop_slope_threshold:
                self._box_high = max(self._box_high, self.data.high[0])
                self._box_low = min(self._box_low, self.data.low[0])
                self._chop_bar_count += 1
        else:
            # In trend mode: check if slope went flat → new chop box
            if avg_abs_slope < self.p.chop_slope_threshold:
                self._in_chop = True
                self._box_high = self.data.high[0]
                self._box_low = self.data.low[0]
                self._chop_bar_count = 1

    # ─────────────────────────────────────── trend detection
    def _detect_trend(self) -> int:
        """Detect trend direction using slope + EMA alignment."""
        slope = self.ema_slope.slope[0]
        if abs(slope) < self.p.chop_slope_threshold:
            return 0
        direction = 1 if slope > 0 else -1
        if direction == 1 and self.ema[0] < self.ema_slow[0]:
            return 0
        if direction == -1 and self.ema[0] > self.ema_slow[0]:
            return 0
        return direction

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

        # Stop below pullback extreme (EMA test candle low)
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

        # Allocate TP portions
        if size >= 3:
            self._tp1_size = max(1, int(size * self.p.tp1_pct))
            self._tp2_size = max(1, int(size * self.p.tp2_pct))
            self._tp3_size = size - self._tp1_size - self._tp2_size
            if self._tp3_size <= 0:
                self._tp3_size = 1
                self._tp2_size = size - self._tp1_size - 1
        else:
            # Too small for 3 portions: all on widest trail
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

        # 1. Time-based exit for losers
        if len(self) - self._entry_bar > self.p.losers_max_bars:
            is_losing = (d == 1 and price < self._avg_entry_price) or \
                        (d == -1 and price > self._avg_entry_price)
            if is_losing:
                self.order = self.close()
                self._reset_position_state()
                return

        # 2. Check TP activation threshold
        if not self._tp_active:
            profit_atr = (price - self._avg_entry_price) * d / atr_val
            if profit_atr >= self.p.tp_activate_atr:
                self._tp_active = True

        # 3. Update trailing stops
        self._update_tp_trails()

        # 4. Check stop hits
        if self._tp_active:
            # TP1: tightest (candle trail) — exits first
            if not self._tp1_done and self._tp1_size > 0:
                hit = (d == 1 and bar_low <= self._tp1_stop) or \
                      (d == -1 and bar_high >= self._tp1_stop)
                if hit:
                    close_size = min(self._tp1_size, abs(self.position.size) - 1)
                    if close_size > 0:
                        self.order = self.sell(size=close_size) if d == 1 else self.buy(size=close_size)
                        self._tp1_done = True
                        return

            # TP2: medium (EMA trail) — exits second
            if self._tp1_done and not self._tp2_done and self._tp2_size > 0:
                hit = (d == 1 and bar_low <= self._tp2_stop) or \
                      (d == -1 and bar_high >= self._tp2_stop)
                if hit:
                    close_size = min(self._tp2_size, abs(self.position.size) - 1)
                    if close_size > 0:
                        self.order = self.sell(size=close_size) if d == 1 else self.buy(size=close_size)
                        self._tp2_done = True
                        return

            # TP3: widest (EMA wide trail) — exits last, closes everything
            if self._tp1_done and self._tp2_done:
                hit = (d == 1 and bar_low <= self._tp3_stop) or \
                      (d == -1 and bar_high >= self._tp3_stop)
                if hit:
                    self.order = self.close()
                    self._reset_position_state()
                    return
        else:
            # Before TP activation: initial stop protects all
            hit = (d == 1 and bar_low <= self._initial_stop) or \
                  (d == -1 and bar_high >= self._initial_stop)
            if hit:
                self.order = self.close()
                self._reset_position_state()
                return

        # 5. Check addon opportunity
        if self.p.enable_addon and self._addon_count < self.p.max_addons:
            if not self._past_entry_cutoff():
                self._check_addon_entry()

    # ─────────────────────────────────────── update TP trails
    def _update_tp_trails(self):
        if not self._tp_active:
            return

        d = self.trade_direction
        atr_val = self.atr[0] if self.atr[0] > 0 else 0.01

        # TP1: candle trail (previous candle low/high - offset)
        if not self._tp1_done and len(self) >= 2:
            if d == 1:
                new = self.data.low[-1] - self.p.tp1_candle_offset
                self._tp1_stop = max(self._tp1_stop, new)
            else:
                new = self.data.high[-1] + self.p.tp1_candle_offset
                self._tp1_stop = min(self._tp1_stop, new)

        # TP2: EMA - medium ATR
        if not self._tp2_done:
            if d == 1:
                new = self.ema[0] - self.p.tp2_ema_atr_mult * atr_val
                self._tp2_stop = max(self._tp2_stop, new)
            else:
                new = self.ema[0] + self.p.tp2_ema_atr_mult * atr_val
                self._tp2_stop = min(self._tp2_stop, new)

        # TP3: EMA - wide ATR
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

        # Must be in profit
        price = self.data.close[0]
        current_size = abs(self.position.size)
        unrealized = (price - self._avg_entry_price) * d * current_size
        if unrealized <= 0:
            self._addon_pb_bars = 0
            return

        # Pullback to EMA zone
        atr_val = self.atr[0] if self.atr[0] > 0 else 999
        dist = abs(price - self.ema[0])
        if dist <= atr_val * self.p.addon_pullback_mult:
            self._addon_pb_bars += 1
        else:
            self._addon_pb_bars = 0
            return

        if self._addon_pb_bars < self.p.addon_min_bars:
            return

        # Momentum confirmation
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

        # Addon stop: below current candle low + breathing room
        if d == 1:
            addon_stop = self.data.low[0] - self.p.initial_stop_atr_mult * atr_val
        else:
            addon_stop = self.data.high[0] + self.p.initial_stop_atr_mult * atr_val

        risk_per_share = abs(entry_price - addon_stop)
        if risk_per_share <= 0:
            return

        # Breakeven sizing: addon loss = existing unrealized profit
        current_size = abs(self.position.size)
        unrealized = (entry_price - self._avg_entry_price) * d * current_size
        addon_size = int(unrealized / risk_per_share)
        if addon_size <= 0:
            return

        # Cap by max position
        max_total = int(self.broker.getvalue() * self.p.max_position_pct / entry_price)
        addon_size = min(addon_size, max_total - current_size)
        if addon_size <= 0:
            return

        # Update position tracking
        old_cost = self._avg_entry_price * self._total_size
        self._total_size += addon_size
        self._avg_entry_price = (old_cost + entry_price * addon_size) / self._total_size
        self._addon_count += 1
        self._addon_pb_bars = 0
        self._entries.append({"price": entry_price, "size": addon_size, "bar": len(self)})

        # Add to TP3 (widest trail) portion
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
