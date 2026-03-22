"""
Self-contained trend-following strategy for autonomous optimization.
The agent modifies this file each iteration. Class MUST be named Strategy.
"""

from __future__ import annotations

import datetime

import backtrader as bt

# ══════════════════════════════════════════════════════════════════
# HYPERPARAMETERS — tune these
# ══════════════════════════════════════════════════════════════════

# Core Indicators
EMA_PERIOD = 20
EMA_SLOW_PERIOD = 50
ATR_PERIOD = 14
VOLUME_AVG_PERIOD = 20
RSI_PERIOD = 14

# Trend Detection
EMA_SLOPE_PERIOD = 5
EMA_SLOPE_THRESHOLD = 0.012

# Pullback Detection
PULLBACK_TOUCH_MULT = 1.2        # x ATR from EMA
MIN_PULLBACK_BARS = 1

# Entry Filters
RSI_OVERBOUGHT = 63
RSI_OVERSOLD = 32

# Initial Stop
INITIAL_STOP_TYPE = "atr"         # swing / atr
INITIAL_STOP_ATR_MULT = 2.5
INITIAL_STOP_OFFSET = 0.05       # $ below swing low

# Trailing Stop Phases
CANDLE_TRAIL_AFTER_BARS = 0      # 0 = candle trail disabled (EMA trail only)
CANDLE_TRAIL_OFFSET = 0.0        # $ below prior candle low
EMA_TRAIL_DISTANCE = 0.0         # 0 = always use EMA trail (no phase switching)
EMA_TRAIL_OFFSET = 6.0           # ATR below EMA for EMA trail (always-active baseline)

# 3-Stage Take Profit
TP1_ATR_MULT = 100.0
TP2_ATR_MULT = 200.0
TP1_PCT = 0.15
TP2_PCT = 0.25
TP3_PCT = 0.60                   # rides trailing stop

# Add-On Entries
ENABLE_ADDON = False
MAX_ADDONS = 2
ADDON_PULLBACK_MULT = 1.0        # x ATR for add-on pullback zone
ADDON_MIN_BARS = 2

# Position Sizing / Risk
RISK_PCT = 0.01
MAX_POSITION_PCT = 0.25

# Daily Limits
MAX_DAILY_TRADES = 6
MAX_DAILY_LOSS_PCT = 0.02
MAX_SAME_LEVEL_ATTEMPTS = 2

# Opening Range Filter
OPENING_RANGE_BARS = 5
OPENING_RANGE_ATR_MULT = 1.5

# Session Times
NO_ENTRY_AFTER_HOUR = 15
NO_ENTRY_AFTER_MINUTE = 30
FORCE_CLOSE_HOUR = 15
FORCE_CLOSE_MINUTE = 58

# Time-Based Exit
LOSERS_MAX_BARS = 45


# ══════════════════════════════════════════════════════════════════
# INLINED INDICATORS (no imports from src/)
# ══════════════════════════════════════════════════════════════════

class VWAP(bt.Indicator):
    """Volume Weighted Average Price with daily reset."""
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
    """Normalized slope of EMA over N bars (% per bar)."""
    lines = ("slope",)
    params = (
        ("ema_period", 20),
        ("slope_period", 5),
    )

    def __init__(self):
        self.ema = bt.indicators.EMA(self.data.close, period=self.p.ema_period)

    def next(self):
        if len(self) <= self.p.slope_period:
            self.lines.slope[0] = 0.0
            return
        raw = (self.ema[0] - self.ema[-self.p.slope_period]) / self.p.slope_period
        self.lines.slope[0] = raw / self.data.close[0] * 100 if self.data.close[0] > 0 else 0.0


class SwingDetector(bt.Indicator):
    """Track rolling N-bar swing high and swing low."""
    lines = ("swing_high", "swing_low")
    params = (("period", 5),)

    def __init__(self):
        self.addminperiod(self.p.period)

    def next(self):
        hi = max(self.data.high.get(size=self.p.period))
        lo = min(self.data.low.get(size=self.p.period))
        self.lines.swing_high[0] = hi
        self.lines.swing_low[0] = lo


# ══════════════════════════════════════════════════════════════════
# STRATEGY
# ══════════════════════════════════════════════════════════════════

class Strategy(bt.Strategy):
    """EMA20 trend-following with 3-phase trail, 3-stage TP, add-on entries."""

    params = (
        ("ema_period", EMA_PERIOD),
        ("ema_slow_period", EMA_SLOW_PERIOD),
        ("atr_period", ATR_PERIOD),
        ("volume_avg_period", VOLUME_AVG_PERIOD),
        ("rsi_period", RSI_PERIOD),
        ("ema_slope_period", EMA_SLOPE_PERIOD),
        ("ema_slope_threshold", EMA_SLOPE_THRESHOLD),
        ("pullback_touch_mult", PULLBACK_TOUCH_MULT),
        ("min_pullback_bars", MIN_PULLBACK_BARS),
        ("rsi_overbought", RSI_OVERBOUGHT),
        ("rsi_oversold", RSI_OVERSOLD),
        ("initial_stop_type", INITIAL_STOP_TYPE),
        ("initial_stop_atr_mult", INITIAL_STOP_ATR_MULT),
        ("initial_stop_offset", INITIAL_STOP_OFFSET),
        ("candle_trail_after_bars", CANDLE_TRAIL_AFTER_BARS),
        ("candle_trail_offset", CANDLE_TRAIL_OFFSET),
        ("ema_trail_distance", EMA_TRAIL_DISTANCE),
        ("ema_trail_offset", EMA_TRAIL_OFFSET),
        ("tp1_atr_mult", TP1_ATR_MULT),
        ("tp2_atr_mult", TP2_ATR_MULT),
        ("tp1_pct", TP1_PCT),
        ("tp2_pct", TP2_PCT),
        ("tp3_pct", TP3_PCT),
        ("enable_addon", ENABLE_ADDON),
        ("max_addons", MAX_ADDONS),
        ("addon_pullback_mult", ADDON_PULLBACK_MULT),
        ("addon_min_bars", ADDON_MIN_BARS),
        ("risk_pct", RISK_PCT),
        ("max_position_pct", MAX_POSITION_PCT),
        ("max_daily_trades", MAX_DAILY_TRADES),
        ("max_daily_loss_pct", MAX_DAILY_LOSS_PCT),
        ("max_same_level_attempts", MAX_SAME_LEVEL_ATTEMPTS),
        ("opening_range_bars", OPENING_RANGE_BARS),
        ("opening_range_atr_mult", OPENING_RANGE_ATR_MULT),
        ("no_entry_after_hour", NO_ENTRY_AFTER_HOUR),
        ("no_entry_after_minute", NO_ENTRY_AFTER_MINUTE),
        ("force_close_hour", FORCE_CLOSE_HOUR),
        ("force_close_minute", FORCE_CLOSE_MINUTE),
        ("losers_max_bars", LOSERS_MAX_BARS),
    )

    # ------------------------------------------------------------------ init
    def __init__(self):
        self.ema = bt.indicators.EMA(self.data.close, period=self.p.ema_period)
        self.ema_slow = bt.indicators.EMA(self.data.close, period=self.p.ema_slow_period)
        self.atr = bt.indicators.ATR(self.data, period=self.p.atr_period)
        self.vwap = VWAP(self.data)
        self.ema_slope = EMASlope(
            self.data,
            ema_period=self.p.ema_period,
            slope_period=self.p.ema_slope_period,
        )
        self.vol_avg = bt.indicators.SMA(
            self.data.volume, period=self.p.volume_avg_period
        )
        self.swing = SwingDetector(self.data, period=5)
        self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsi_period)

        # Order tracking
        self.order = None
        self.trade_direction = 0  # +1 long, -1 short, 0 flat

        # Position tracking
        self._entries = []
        self._total_size = 0
        self._avg_entry_price = 0.0
        self._addon_count = 0
        self._addon_pb_bars = 0

        # Take profit state
        self._tp1_price = 0.0
        self._tp2_price = 0.0
        self._tp1_done = False
        self._tp2_done = False
        self._tp1_size = 0
        self._tp2_size = 0
        self._initial_size = 0

        # Trailing stop state
        self._trail_phase = 0  # 0=initial, 1=candle, 2=ema
        self._stop_price = 0.0
        self._profitable_bars = 0
        self._entry_bar = 0

        # Daily state (reset each session)
        self._today = None
        self._daily_trades = 0
        self._daily_pnl = 0.0
        self._daily_start_value = 0.0
        self._same_level_attempts = 0
        self._last_entry_bar = -999

        # Pullback state (for entry)
        self._state = 0
        self._pullback_extreme = 0.0
        self._pullback_dir = 0
        self._pb_bar_count = 0

        # Opening range
        self._or_high = 0.0
        self._or_low = 999999.0
        self._or_bars_counted = 0
        self._or_is_wide = False

    # ----------------------------------------------------------- daily reset
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
            self._addon_pb_bars = 0

    # ------------------------------------------------------ session helpers
    def _current_time(self) -> datetime.time:
        return self.data.datetime.time(0)

    def _past_entry_cutoff(self) -> bool:
        t = self._current_time()
        cutoff = datetime.time(self.p.no_entry_after_hour, self.p.no_entry_after_minute)
        return t >= cutoff

    def _past_force_close(self) -> bool:
        t = self._current_time()
        close_t = datetime.time(self.p.force_close_hour, self.p.force_close_minute)
        return t >= close_t

    # -------------------------------------------------------- daily limits
    def _daily_limit_reached(self) -> bool:
        if self._daily_trades >= self.p.max_daily_trades:
            return True
        current_value = self.broker.getvalue()
        if self._daily_start_value > 0:
            loss_pct = (self._daily_start_value - current_value) / self._daily_start_value
            if loss_pct >= self.p.max_daily_loss_pct:
                return True
        return False

    # ----------------------------------------------------- opening range
    def _update_opening_range(self):
        if self._or_bars_counted < self.p.opening_range_bars:
            self._or_high = max(self._or_high, self.data.high[0])
            self._or_low = min(self._or_low, self.data.low[0])
            self._or_bars_counted += 1
            if self._or_bars_counted == self.p.opening_range_bars:
                or_range = self._or_high - self._or_low
                atr_val = self.atr[0] if self.atr[0] > 0 else 1.0
                self._or_is_wide = or_range > atr_val * self.p.opening_range_atr_mult

    def _in_opening_range_zone(self) -> bool:
        if not self._or_is_wide:
            return False
        if self._or_bars_counted < self.p.opening_range_bars:
            return True
        return self._or_low <= self.data.close[0] <= self._or_high

    # -------------------------------------------------------- trend detect
    def _detect_trend(self) -> int:
        """Detect trend using slope + EMA alignment."""
        slope = self.ema_slope.slope[0]
        if abs(slope) < self.p.ema_slope_threshold:
            return 0

        direction = 1 if slope > 0 else -1

        # EMA50 trend filter: fast EMA must be on correct side of slow EMA
        if direction == 1 and self.ema[0] < self.ema_slow[0]:
            return 0
        if direction == -1 and self.ema[0] > self.ema_slow[0]:
            return 0

        return direction

    # --------------------------------------------------- position sizing
    def _calc_size(self, entry_price: float, stop_price: float) -> int:
        risk_per_share = abs(entry_price - stop_price)
        if risk_per_share <= 0:
            return 0
        account_value = self.broker.getvalue()
        risk_amount = account_value * self.p.risk_pct
        size_by_risk = int(risk_amount / risk_per_share)
        max_by_capital = int(account_value * self.p.max_position_pct / entry_price)
        return max(1, min(size_by_risk, max_by_capital))

    # ------------------------------------------------------- initial stop
    def _calc_initial_stop(self, direction: int, entry_price: float) -> float:
        if self.p.initial_stop_type == "swing":
            if direction == 1:
                return self.swing.swing_low[0] - self.p.initial_stop_offset
            else:
                return self.swing.swing_high[0] + self.p.initial_stop_offset
        else:  # atr
            atr_val = self.atr[0] if self.atr[0] > 0 else 0.01
            offset = atr_val * self.p.initial_stop_atr_mult
            if direction == 1:
                return entry_price - offset
            else:
                return entry_price + offset

    # ------------------------------------------------------ TP setup
    def _setup_tp_levels(self):
        """Set TP prices and sizes from avg entry and total size."""
        d = self.trade_direction
        atr_val = self.atr[0] if self.atr[0] > 0 else 0.01

        if d == 1:
            self._tp1_price = self._avg_entry_price + self.p.tp1_atr_mult * atr_val
            self._tp2_price = self._avg_entry_price + self.p.tp2_atr_mult * atr_val
        else:
            self._tp1_price = self._avg_entry_price - self.p.tp1_atr_mult * atr_val
            self._tp2_price = self._avg_entry_price - self.p.tp2_atr_mult * atr_val

        total = self._total_size
        self._tp1_size = max(1, int(total * self.p.tp1_pct))
        self._tp2_size = max(1, int(total * self.p.tp2_pct))

        # Ensure TP1 + TP2 leaves at least 1 share for trailing
        if self._tp1_size + self._tp2_size >= total:
            self._tp2_size = max(0, total - self._tp1_size - 1)
            if self._tp1_size + self._tp2_size >= total:
                self._tp1_size = max(0, total - 1)
                self._tp2_size = 0

    def _recalc_tp_after_addon(self):
        """Recalculate TP levels after an add-on entry."""
        d = self.trade_direction
        atr_val = self.atr[0] if self.atr[0] > 0 else 0.01

        if d == 1:
            self._tp1_price = self._avg_entry_price + self.p.tp1_atr_mult * atr_val
            self._tp2_price = self._avg_entry_price + self.p.tp2_atr_mult * atr_val
        else:
            self._tp1_price = self._avg_entry_price - self.p.tp1_atr_mult * atr_val
            self._tp2_price = self._avg_entry_price - self.p.tp2_atr_mult * atr_val

        total = self._total_size
        if not self._tp1_done:
            self._tp1_size = max(1, int(total * self.p.tp1_pct))
        self._tp2_size = max(1, int(total * self.p.tp2_pct))

        # Guard: don't close everything via TPs
        tp1_pending = 0 if self._tp1_done else self._tp1_size
        if tp1_pending + self._tp2_size >= total:
            self._tp2_size = max(0, total - tp1_pending - 1)

    # ============================================================== NEXT
    def next(self):
        if self.order:
            return

        self._check_new_day()
        self._update_opening_range()

        # Force close at end of session
        if self._past_force_close() and self.position:
            self.order = self.close()
            self._reset_position_state()
            return

        # Manage open position
        if self.position:
            self._manage_position()
            return

        # No new entries conditions
        if self._past_entry_cutoff():
            return
        if self._daily_limit_reached():
            return
        if self._in_opening_range_zone():
            return

        # State machine for entry
        self._run_entry_logic()

    # -------------------------------------------------------- entry logic
    def _run_entry_logic(self):
        trend = self._detect_trend()

        if trend == 0:
            self._state = 0
            self._pb_bar_count = 0
            return

        # Reset if trend flipped
        if trend != self._pullback_dir:
            self._pullback_dir = trend
            self._state = 0
            self._pb_bar_count = 0
            self._pullback_extreme = 0.0

        atr_val = self.atr[0] if self.atr[0] > 0 else 999
        dist = abs(self.data.close[0] - self.ema[0])
        touch_dist = atr_val * self.p.pullback_touch_mult
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
                # RSI filter
                rsi_ok = True
                if trend == 1 and self.rsi[0] > self.p.rsi_overbought:
                    rsi_ok = False
                if trend == -1 and self.rsi[0] < self.p.rsi_oversold:
                    rsi_ok = False
                # Momentum confirmation: close breaks prior bar high/low
                momentum_ok = False
                if len(self) >= 2:
                    if trend == 1 and self.data.close[0] > self.data.high[-1]:
                        momentum_ok = True
                    elif trend == -1 and self.data.close[0] < self.data.low[-1]:
                        momentum_ok = True

                if (on_correct_side
                        and self._pb_bar_count >= self.p.min_pullback_bars
                        and rsi_ok and momentum_ok):
                    self._execute_entry(trend)

                # Reset for next pullback
                self._state = 0
                self._pb_bar_count = 0
                self._pullback_extreme = 0.0

    # -------------------------------------------------------- execute entry
    def _execute_entry(self, direction: int):
        entry_price = self.data.close[0]
        stop = self._calc_initial_stop(direction, entry_price)
        size = self._calc_size(entry_price, stop)

        if size <= 0:
            return

        if abs(len(self) - self._last_entry_bar) < 10:
            self._same_level_attempts += 1
            if self._same_level_attempts > self.p.max_same_level_attempts:
                return
        else:
            self._same_level_attempts = 1

        # Set position state
        self.trade_direction = direction
        self._entries = [{"price": entry_price, "size": size, "bar": len(self), "stop": stop}]
        self._total_size = size
        self._avg_entry_price = entry_price
        self._addon_count = 0
        self._addon_pb_bars = 0

        self._stop_price = stop
        self._trail_phase = 0  # INITIAL
        self._profitable_bars = 0
        self._entry_bar = len(self)

        self._tp1_done = False
        self._tp2_done = False
        self._initial_size = size

        self._last_entry_bar = len(self)
        self._daily_trades += 1

        if direction == 1:
            self.order = self.buy(size=size)
        else:
            self.order = self.sell(size=size)

        self._setup_tp_levels()

    # ---------------------------------------------------- manage position
    def _manage_position(self):
        d = self.trade_direction
        if d == 0:
            return

        bar_low = self.data.low[0]
        bar_high = self.data.high[0]
        price = self.data.close[0]

        # 1. Time-based exit: close if in trade too long and losing
        bars_in_trade = len(self) - self._entry_bar
        if bars_in_trade > self.p.losers_max_bars:
            is_losing = (
                (d == 1 and price < self._avg_entry_price)
                or (d == -1 and price > self._avg_entry_price)
            )
            if is_losing:
                self.order = self.close()
                self._reset_position_state()
                return

        # 2. Check stop hit
        if d == 1 and bar_low <= self._stop_price:
            self.order = self.close()
            self._reset_position_state()
            return
        if d == -1 and bar_high >= self._stop_price:
            self.order = self.close()
            self._reset_position_state()
            return

        # 3. Check take profits
        if self._check_take_profits():
            return

        # 4. Count profitable bars (cumulative, not consecutive)
        if d == 1 and price > self._avg_entry_price:
            self._profitable_bars += 1
        elif d == -1 and price < self._avg_entry_price:
            self._profitable_bars += 1

        # 5. Update trailing stop (phase-dependent)
        self._update_trailing_stop()

        # 6. Check add-on entry opportunity
        if self.p.enable_addon and self._addon_count < self.p.max_addons:
            if not self._past_entry_cutoff():
                self._check_addon_entry()

    # ---------------------------------------------------- take profits
    def _check_take_profits(self) -> bool:
        """Check and execute TP levels. Returns True if order placed."""
        d = self.trade_direction
        bar_high = self.data.high[0]
        bar_low = self.data.low[0]
        current_size = abs(self.position.size)

        # TP1
        if not self._tp1_done and self._tp1_size > 0:
            hit_tp1 = (
                (d == 1 and bar_high >= self._tp1_price)
                or (d == -1 and bar_low <= self._tp1_price)
            )
            if hit_tp1:
                tp_size = min(self._tp1_size, current_size - 1)
                if tp_size > 0:
                    if d == 1:
                        self.order = self.sell(size=tp_size)
                    else:
                        self.order = self.buy(size=tp_size)
                    self._tp1_done = True
                    # Move stop to breakeven (only if it improves)
                    if d == 1:
                        self._stop_price = max(self._stop_price, self._avg_entry_price)
                    else:
                        self._stop_price = min(self._stop_price, self._avg_entry_price)
                    return True

        # TP2
        if self._tp1_done and not self._tp2_done and self._tp2_size > 0:
            hit_tp2 = (
                (d == 1 and bar_high >= self._tp2_price)
                or (d == -1 and bar_low <= self._tp2_price)
            )
            if hit_tp2:
                tp_size = min(self._tp2_size, current_size - 1)
                if tp_size > 0:
                    if d == 1:
                        self.order = self.sell(size=tp_size)
                    else:
                        self.order = self.buy(size=tp_size)
                    self._tp2_done = True
                    return True

        return False

    # ---------------------------------------------------- trailing stop
    def _update_trailing_stop(self):
        """Update trailing stop based on current phase.

        EMA trail is always active as baseline (all phases).
        Phase 1 adds candle trail on top (takes tighter of EMA vs candle).
        Phase 2 reverts to EMA-only trail (looser, survives pullback to EMA).
        """
        d = self.trade_direction
        atr_val = self.atr[0] if self.atr[0] > 0 else 0.01

        # Phase 0 -> 1: after enough cumulative profitable bars
        if self._trail_phase == 0:
            if self._profitable_bars >= self.p.candle_trail_after_bars:
                self._trail_phase = 1

        # Phase 1 <-> 2: based on distance from EMA
        if self._trail_phase >= 1:
            dist_from_ema = abs(self.data.close[0] - self.ema[0])
            if dist_from_ema > self.p.ema_trail_distance * atr_val:
                self._trail_phase = 2
            elif self._trail_phase == 2:
                self._trail_phase = 1

        # EMA trail — always active as baseline (all phases)
        if d == 1:
            ema_stop = self.ema[0] - self.p.ema_trail_offset * atr_val
        else:
            ema_stop = self.ema[0] + self.p.ema_trail_offset * atr_val

        new_stop = ema_stop

        # Phase 1: candle trail tightens stop (takes better of EMA vs candle)
        if self._trail_phase == 1 and len(self) >= 2:
            if d == 1:
                candle_stop = self.data.low[-1] - self.p.candle_trail_offset
                new_stop = max(ema_stop, candle_stop)
            else:
                candle_stop = self.data.high[-1] + self.p.candle_trail_offset
                new_stop = min(ema_stop, candle_stop)

        # Ratchet only: stop can only move in favorable direction
        if d == 1 and new_stop > self._stop_price:
            self._stop_price = new_stop
        elif d == -1 and new_stop < self._stop_price:
            self._stop_price = new_stop

    # ---------------------------------------------------- add-on entries
    def _check_addon_entry(self):
        """Check if conditions are met for an add-on entry."""
        d = self.trade_direction
        trend = self._detect_trend()

        # 1. Trend must agree with position direction
        if trend != d:
            self._addon_pb_bars = 0
            return

        # 2. Price pulls back to EMA zone
        atr_val = self.atr[0] if self.atr[0] > 0 else 999
        dist = abs(self.data.close[0] - self.ema[0])
        addon_zone = atr_val * self.p.addon_pullback_mult

        if dist <= addon_zone:
            self._addon_pb_bars += 1
        else:
            self._addon_pb_bars = 0
            return

        # 3. Spent enough bars in zone
        if self._addon_pb_bars < self.p.addon_min_bars:
            return

        # 4. Momentum confirmation: close breaks prior bar high/low
        if len(self) < 2:
            return
        if d == 1 and self.data.close[0] <= self.data.high[-1]:
            return
        if d == -1 and self.data.close[0] >= self.data.low[-1]:
            return

        # 5. Execute add-on
        self._execute_addon()

    def _execute_addon(self):
        """Execute an add-on entry."""
        d = self.trade_direction
        entry_price = self.data.close[0]

        # New stop for this add-on entry
        if d == 1:
            new_stop = self.data.low[0] - self.p.initial_stop_offset
        else:
            new_stop = self.data.high[0] + self.p.initial_stop_offset

        size = self._calc_size(entry_price, new_stop)
        if size <= 0:
            return

        # Update position tracking
        self._entries.append({
            "price": entry_price, "size": size,
            "bar": len(self), "stop": new_stop,
        })

        # Update average entry price
        old_cost = self._avg_entry_price * self._total_size
        new_cost = entry_price * size
        self._total_size += size
        self._avg_entry_price = (old_cost + new_cost) / self._total_size

        self._addon_count += 1
        self._addon_pb_bars = 0

        # Reset trail phase to INITIAL with new stop
        self._trail_phase = 0
        self._profitable_bars = 0
        self._stop_price = new_stop

        # Place order (add-ons do NOT count as daily trades)
        if d == 1:
            self.order = self.buy(size=size)
        else:
            self.order = self.sell(size=size)

        # Recalculate TP levels from new average
        self._recalc_tp_after_addon()

    # ---------------------------------------------------- position state reset
    def _reset_position_state(self):
        """Reset all position-related state."""
        self.trade_direction = 0
        self._entries = []
        self._total_size = 0
        self._avg_entry_price = 0.0
        self._addon_count = 0
        self._addon_pb_bars = 0
        self._tp1_price = 0.0
        self._tp2_price = 0.0
        self._tp1_done = False
        self._tp2_done = False
        self._tp1_size = 0
        self._tp2_size = 0
        self._initial_size = 0
        self._trail_phase = 0
        self._stop_price = 0.0
        self._profitable_bars = 0
        self._entry_bar = 0
        self._state = 0

    # --------------------------------------------------- order notification
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
