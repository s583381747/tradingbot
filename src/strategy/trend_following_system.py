"""
Comprehensive trend-following day trading strategy for US equities.

Synthesized from 52 video analyses of @rijindoudoujin's trading education channel.

Core logic:
  1. Trend detection  — EMA slope + price vs EMA/VWAP
  2. Pullback to EMA  — price approaches 20 EMA within ATR-based threshold
  3. Breakout entry    — price breaks pullback candle high (long) / low (short)
  4. Risk management   — swing-based stop, partial TP, trailing stop, 1% sizing
  5. Daily discipline  — max trades, max loss, session-aware forced close

Parameters marked ⚠ are hard to quantify from video content and should be
optimized via parameter sweep (see scripts/run_intraday_backtest.py).
"""

from __future__ import annotations

import datetime
import logging

import backtrader as bt

from src.strategy.indicators import VWAP, EMASlope, SwingDetector

logger = logging.getLogger(__name__)


class TrendFollowingSystem(bt.Strategy):
    """Day-trading trend-following strategy with full risk management."""

    params = (
        # ── Core indicators ─────────────────────────────
        ("ema_period", 20),
        ("atr_period", 14),
        ("volume_avg_period", 20),

        # ── Trend detection ─────────────────────────────
        # ⚠ What constitutes a "clear" trend?
        ("ema_slope_period", 5),
        ("ema_slope_threshold", 0.01),   # test: 0.0 / 0.005 / 0.01 / 0.02
        ("require_above_vwap", True),

        # ── Pullback detection ──────────────────────────
        # ⚠ How close to EMA = "pullback touch"?
        ("pullback_touch_mult", 0.5),    # test: 0.2 / 0.5 / 0.8 / 1.0 (x ATR)

        # ── Entry ───────────────────────────────────────
        # ⚠ What volume = "放量 breakout"?
        ("use_volume_filter", True),
        ("volume_breakout_mult", 1.5),   # test: 1.0 / 1.3 / 1.5 / 2.0

        # ── Stop loss ───────────────────────────────────
        ("sl_type", "swing"),            # swing / atr / fixed_pct
        ("sl_offset", 0.05),             # $ offset for swing type
        ("sl_atr_mult", 1.5),            # multiplier for atr type
        ("sl_fixed_pct", 0.005),         # 0.5% for fixed_pct type

        # ── Take profit ─────────────────────────────────
        # ⚠ Partial TP ratio: 60% or 80%?
        ("use_partial_tp", True),
        ("partial_tp_pct", 0.6),         # test: 0.5 / 0.6 / 0.8
        ("tp_atr_mult", 2.0),            # first target = entry ± N*ATR

        # ── Trailing stop ───────────────────────────────
        # ⚠ What offset from EMA?
        ("trail_type", "ema"),           # ema / atr / none
        ("trail_atr_mult", 1.0),         # test: 0.5 / 1.0 / 1.5

        # ── Position sizing ─────────────────────────────
        ("risk_pct", 0.01),              # 1% rule
        ("max_position_pct", 0.25),      # max 25% in one trade

        # ── Daily limits ────────────────────────────────
        ("max_daily_trades", 6),
        ("max_daily_loss_pct", 0.02),    # 2% daily stop
        ("max_same_level_attempts", 2),

        # ── Opening range filter ────────────────────────
        # ⚠ What is a "long" first candle?
        ("opening_range_bars", 5),       # first N bars
        ("opening_range_atr_mult", 1.5), # test: 1.0 / 1.5 / 2.0

        # ── Session ─────────────────────────────────────
        ("no_entry_after_hour", 15),
        ("no_entry_after_minute", 30),
        ("force_close_hour", 15),
        ("force_close_minute", 55),

        # ── Logging ─────────────────────────────────────
        ("printlog", False),
    )

    # ------------------------------------------------------------------ init
    def __init__(self):
        # — Indicators —
        self.ema = bt.indicators.EMA(self.data.close, period=self.p.ema_period)
        self.atr = bt.indicators.ATR(self.data, period=self.p.atr_period)
        self.vwap = VWAP(self.data)
        self.ema_slope = EMASlope(
            self.data,
            ema_period=self.p.ema_period,
            slope_period=self.p.ema_slope_period,
        )
        self.vol_avg = bt.indicators.SMA(self.data.volume, period=self.p.volume_avg_period)
        self.swing = SwingDetector(self.data, period=5)

        # — Order tracking —
        self.order = None
        self.entry_price = 0.0
        self.stop_price = 0.0
        self.target_price = 0.0
        self.trade_direction = 0   # +1 long, -1 short, 0 flat

        # — Daily state (reset each session) —
        self._today = None
        self._daily_trades = 0
        self._daily_pnl = 0.0
        self._daily_start_value = 0.0
        self._same_level_attempts = 0
        self._last_entry_bar = -999
        self._partial_done = False

        # — Pullback state —
        # 0=waiting_trend  1=tracking
        self._state = 0
        self._pullback_extreme = 0.0   # swing low (long) / high (short)
        self._pullback_dir = 0         # +1 = pullback in uptrend, -1 = downtrend
        self._pb_bar_count = 0         # bars spent in pullback zone

        # — Opening range —
        self._or_high = 0.0
        self._or_low = 999999.0
        self._or_bars_counted = 0
        self._or_is_wide = False

    # --------------------------------------------------------------- logging
    def log(self, txt, do_print=False):
        if self.p.printlog or do_print:
            dt = self.data.datetime.datetime(0)
            logger.info("%s  %s", dt.strftime("%Y-%m-%d %H:%M"), txt)

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
            self._partial_done = False

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
        """True if OR is wide and price is still inside the range."""
        if not self._or_is_wide:
            return False
        if self._or_bars_counted < self.p.opening_range_bars:
            return True  # still forming
        return self._or_low <= self.data.close[0] <= self._or_high

    # -------------------------------------------------------- trend detect
    def _detect_trend(self) -> int:
        """Return +1 uptrend, -1 downtrend, 0 neutral."""
        slope = self.ema_slope.slope[0]
        if abs(slope) < self.p.ema_slope_threshold:
            return 0
        direction = 1 if slope > 0 else -1

        # Price must be on correct side of EMA
        if direction == 1 and self.data.close[0] < self.ema[0]:
            return 0
        if direction == -1 and self.data.close[0] > self.ema[0]:
            return 0

        # Optional VWAP filter
        if self.p.require_above_vwap:
            if direction == 1 and self.data.close[0] < self.vwap.vwap[0]:
                return 0
            if direction == -1 and self.data.close[0] > self.vwap.vwap[0]:
                return 0

        return direction

    # ----------------------------------------------------- pullback detect
    def _check_pullback(self, trend: int) -> bool:
        """True if price has pulled back close to EMA."""
        if trend == 0:
            return False
        dist = abs(self.data.close[0] - self.ema[0])
        threshold = self.atr[0] * self.p.pullback_touch_mult
        return dist <= threshold

    def _check_breakout(self, direction: int) -> bool:
        """True if price breaks the previous candle's high (long) or low (short)."""
        if len(self) < 2:
            return False
        if direction == 1:
            return self.data.high[0] > self.data.high[-1]
        else:
            return self.data.low[0] < self.data.low[-1]

    def _check_volume(self) -> bool:
        """True if current volume exceeds average * multiplier."""
        if not self.p.use_volume_filter:
            return True
        if self.vol_avg[0] <= 0:
            return True
        return self.data.volume[0] > self.vol_avg[0] * self.p.volume_breakout_mult

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

    # ------------------------------------------------------- stop loss calc
    def _calc_stop(self, direction: int, entry_price: float) -> float:
        if self.p.sl_type == "swing":
            if direction == 1:
                return self._pullback_extreme - self.p.sl_offset
            else:
                return self._pullback_extreme + self.p.sl_offset

        elif self.p.sl_type == "atr":
            offset = self.atr[0] * self.p.sl_atr_mult
            if direction == 1:
                return entry_price - offset
            else:
                return entry_price + offset

        else:  # fixed_pct
            if direction == 1:
                return entry_price * (1 - self.p.sl_fixed_pct)
            else:
                return entry_price * (1 + self.p.sl_fixed_pct)

    # -------------------------------------------------------- target calc
    def _calc_target(self, direction: int, entry_price: float) -> float:
        offset = self.atr[0] * self.p.tp_atr_mult
        if direction == 1:
            return entry_price + offset
        else:
            return entry_price - offset

    # ------------------------------------------------------ trailing stop
    def _calc_trail_stop(self, direction: int) -> float:
        if self.p.trail_type == "none":
            return self.stop_price  # unchanged

        atr_offset = self.atr[0] * self.p.trail_atr_mult

        if self.p.trail_type == "ema":
            if direction == 1:
                new_stop = self.ema[0] - atr_offset
            else:
                new_stop = self.ema[0] + atr_offset
        else:  # atr from price
            if direction == 1:
                new_stop = self.data.close[0] - atr_offset
            else:
                new_stop = self.data.close[0] + atr_offset

        # Only ratchet: never move stop against the position
        if direction == 1:
            return max(self.stop_price, new_stop)
        else:
            return min(self.stop_price, new_stop)

    # ============================================================== NEXT
    def next(self):
        if self.order:
            return

        self._check_new_day()
        self._update_opening_range()

        # ── Force close at end of session ──
        if self._past_force_close() and self.position:
            self.log(f"FORCE CLOSE at session end | pos={self.position.size}")
            self.order = self.close()
            self.trade_direction = 0
            self._state = 0
            return

        # ── Manage open position ──
        if self.position:
            self._manage_position()
            return

        # ── No new entries conditions ──
        if self._past_entry_cutoff():
            return
        if self._daily_limit_reached():
            return
        if self._in_opening_range_zone():
            return

        # ── State machine for entry ──
        self._run_entry_logic()

    # -------------------------------------------------------- entry logic
    def _run_entry_logic(self):
        trend = self._detect_trend()

        # Simplified 2-state approach: either looking for pullback or not.
        # The original 4-state machine was too strict and missed entries.

        if trend == 0:
            self._state = 0
            return

        # Always update pullback direction to current trend
        if self._state == 0:
            self._pullback_dir = trend
            self._state = 1
            self._pullback_extreme = 0.0
            self._pb_bar_count = 0

        # If trend flipped, reset
        if trend != self._pullback_dir:
            self._state = 0
            return

        # ── Check if price is near EMA (pullback zone) ──
        dist = abs(self.data.close[0] - self.ema[0])
        touch_dist = self.atr[0] * self.p.pullback_touch_mult if self.atr[0] > 0 else 999

        in_pullback_zone = dist <= touch_dist

        if in_pullback_zone:
            # Track pullback extreme while near EMA
            if self._pullback_dir == 1:
                if self._pullback_extreme == 0.0:
                    self._pullback_extreme = self.data.low[0]
                self._pullback_extreme = min(self._pullback_extreme, self.data.low[0])
            else:
                if self._pullback_extreme == 0.0:
                    self._pullback_extreme = self.data.high[0]
                self._pullback_extreme = max(self._pullback_extreme, self.data.high[0])
            self._pb_bar_count += 1

        elif self._pb_bar_count > 0:
            # Price moved away from EMA after being in pullback zone
            # Check for breakout in trend direction
            on_correct_side = (
                (self._pullback_dir == 1 and self.data.close[0] > self.ema[0]) or
                (self._pullback_dir == -1 and self.data.close[0] < self.ema[0])
            )
            if on_correct_side and self._check_breakout(self._pullback_dir):
                if self._check_volume():
                    self._execute_entry(self._pullback_dir)
                    self._pb_bar_count = 0
                    self._pullback_extreme = 0.0
                    return

            # If moved too far without breakout, reset pullback tracking
            if dist > touch_dist * 3:
                self._pb_bar_count = 0
                self._pullback_extreme = 0.0

    # -------------------------------------------------------- execute entry
    def _execute_entry(self, direction: int):
        entry_price = self.data.close[0]
        stop = self._calc_stop(direction, entry_price)
        target = self._calc_target(direction, entry_price)
        size = self._calc_size(entry_price, stop)

        if size <= 0:
            return

        # Same-level attempt limiter
        if abs(len(self) - self._last_entry_bar) < 10:
            self._same_level_attempts += 1
            if self._same_level_attempts > self.p.max_same_level_attempts:
                return
        else:
            self._same_level_attempts = 1

        self.entry_price = entry_price
        self.stop_price = stop
        self.target_price = target
        self.trade_direction = direction
        self._partial_done = False
        self._last_entry_bar = len(self)
        self._daily_trades += 1

        if direction == 1:
            self.log(f"BUY  size={size} @ {entry_price:.2f}  SL={stop:.2f}  TP={target:.2f}")
            self.order = self.buy(size=size)
        else:
            self.log(f"SELL size={size} @ {entry_price:.2f}  SL={stop:.2f}  TP={target:.2f}")
            self.order = self.sell(size=size)

    # ---------------------------------------------------- manage position
    def _manage_position(self):
        d = self.trade_direction

        # Use high/low for more accurate intraday stop/target checks
        bar_low = self.data.low[0]
        bar_high = self.data.high[0]

        # ── Check stop loss ──
        if d == 1 and bar_low <= self.stop_price:
            self.log(f"STOP HIT (long) low={bar_low:.2f}  stop={self.stop_price:.2f}")
            self.order = self.close()
            self.trade_direction = 0
            self._state = 0
            return

        if d == -1 and bar_high >= self.stop_price:
            self.log(f"STOP HIT (short) high={bar_high:.2f}  stop={self.stop_price:.2f}")
            self.order = self.close()
            self.trade_direction = 0
            self._state = 0
            return

        # ── Partial take profit ──
        if self.p.use_partial_tp and not self._partial_done:
            hit_target = (d == 1 and bar_high >= self.target_price) or \
                         (d == -1 and bar_low <= self.target_price)
            if hit_target:
                partial_size = int(abs(self.position.size) * self.p.partial_tp_pct)
                if partial_size > 0:
                    self.log(f"PARTIAL TP  close {partial_size} @ {self.data.close[0]:.2f}")
                    if d == 1:
                        self.order = self.sell(size=partial_size)
                    else:
                        self.order = self.buy(size=partial_size)
                    # Move stop to breakeven after partial TP
                    self.stop_price = self.entry_price
                    self._partial_done = True
                    return

        # ── Update trailing stop ──
        if self._partial_done or self.p.trail_type != "none":
            new_stop = self._calc_trail_stop(d)
            if d == 1 and new_stop > self.stop_price:
                self.stop_price = new_stop
            elif d == -1 and new_stop < self.stop_price:
                self.stop_price = new_stop

    # --------------------------------------------------- order notification
    def notify_order(self, order):
        if order.status in (order.Submitted, order.Accepted):
            return

        if order.status == order.Completed:
            if order.isbuy():
                self.log(f"BUY  EXEC @ {order.executed.price:.2f}  "
                         f"size={order.executed.size}  comm={order.executed.comm:.2f}")
            else:
                self.log(f"SELL EXEC @ {order.executed.price:.2f}  "
                         f"size={order.executed.size}  comm={order.executed.comm:.2f}")
        elif order.status in (order.Canceled, order.Margin, order.Rejected):
            self.log(f"Order CANCELED/MARGIN/REJECTED (status={order.status})")

        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        self._daily_pnl += trade.pnlcomm
        self.log(f"TRADE CLOSED  gross={trade.pnl:.2f}  net={trade.pnlcomm:.2f}  "
                 f"daily_pnl={self._daily_pnl:.2f}")
        if not self.position:
            self.trade_direction = 0
