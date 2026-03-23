"""
EMA20 Trend Following — Gen 16: Exact Pandas Alignment.

This implementation matches the pandas simulation BAR FOR BAR.
See ALIGNMENT_SPEC.md for the 5 alignment rules.

Key differences from Gen 13-15:
  - Prices from TOUCH bar, not exit-zone bar
  - Risk and ATR FIXED at entry time
  - No order-blocking: manage multiple signals via list
  - Stop checked EVERY bar regardless of pending orders
  - Runner trail uses entry_atr, not current atr
"""

from __future__ import annotations
import datetime
import backtrader as bt
import math

EMA_PERIOD = 20
EMA_SLOW_PERIOD = 50
ATR_PERIOD = 14

TOUCH_TOL = 0.15           # wick must be within 0.15*ATR of EMA20
TOUCH_BELOW_MAX = 0.5      # wick can't be more than 0.5*ATR below EMA20
SIGNAL_OFFSET = 0.05       # entry above touch bar high
STOP_BUFFER_ATR = 0.3      # below touch bar low
SIGNAL_VALID_BARS = 3      # signal expires after 3 bars

LOCK1_RR = 0.5             # lock 30% at 0.5:1 R:R
LOCK2_RR = 1.5             # lock 20% at 1.5:1 R:R
LOCK1_PCT = 0.30
LOCK2_PCT = 0.20
# Runner = 50%

RUNNER_TRAIL_BARS = 10
RUNNER_TRAIL_BUFFER_ATR = 0.3

RISK_PCT = 0.01
MAX_POSITION_PCT = 0.25
MAX_DAILY_LOSS_PCT = 0.02

NO_ENTRY_AFTER_HOUR = 15
NO_ENTRY_AFTER_MINUTE = 30
FORCE_CLOSE_HOUR = 15
FORCE_CLOSE_MINUTE = 58


class Strategy(bt.Strategy):

    params = (
        ("ema_period", EMA_PERIOD), ("ema_slow_period", EMA_SLOW_PERIOD),
        ("atr_period", ATR_PERIOD),
        ("touch_tol", TOUCH_TOL), ("touch_below_max", TOUCH_BELOW_MAX),
        ("signal_offset", SIGNAL_OFFSET), ("stop_buffer_atr", STOP_BUFFER_ATR),
        ("signal_valid_bars", SIGNAL_VALID_BARS),
        ("lock1_rr", LOCK1_RR), ("lock2_rr", LOCK2_RR),
        ("lock1_pct", LOCK1_PCT), ("lock2_pct", LOCK2_PCT),
        ("runner_trail_bars", RUNNER_TRAIL_BARS),
        ("runner_trail_buffer_atr", RUNNER_TRAIL_BUFFER_ATR),
        ("risk_pct", RISK_PCT), ("max_position_pct", MAX_POSITION_PCT),
        ("max_daily_loss_pct", MAX_DAILY_LOSS_PCT),
        ("no_entry_after_hour", NO_ENTRY_AFTER_HOUR),
        ("no_entry_after_minute", NO_ENTRY_AFTER_MINUTE),
        ("force_close_hour", FORCE_CLOSE_HOUR),
        ("force_close_minute", FORCE_CLOSE_MINUTE),
    )

    def __init__(self):
        self.ema = bt.indicators.EMA(self.data.close, period=self.p.ema_period)
        self.ema_slow = bt.indicators.EMA(self.data.close, period=self.p.ema_slow_period)
        self.atr = bt.indicators.ATR(self.data, period=self.p.atr_period)

        # We manage orders manually — no self.order gating
        self._pending_orders = []  # list of bt order objects

        # Active trades: each is a dict with all state
        # We can have multiple simultaneous trades
        self._active = []

        # Scan state
        self._touch_bar = -1
        self._touch_high = 0.0
        self._touch_low = 0.0
        self._touch_atr = 0.0
        self._touch_dir = 0
        self._bounce_confirmed = False

        # Multiple pending signals
        self._pending_signals = []  # list of (order, bar, stop, dir, atr, risk)

        # Daily
        self._today = None
        self._daily_start = 0.0

    def _new_day(self):
        dt = self.data.datetime.date(0)
        if dt != self._today:
            self._today = dt
            self._daily_start = self.broker.getvalue()
            self._touch_bar = -1
            self._bounce_confirmed = False
            self._signal_placed = False

    def _time_ok(self):
        t = self.data.datetime.time(0)
        return t < datetime.time(self.p.no_entry_after_hour, self.p.no_entry_after_minute)

    def _force_close_time(self):
        t = self.data.datetime.time(0)
        return t >= datetime.time(self.p.force_close_hour, self.p.force_close_minute)

    def _daily_loss_ok(self):
        if self._daily_start <= 0:
            return True
        return (self._daily_start - self.broker.getvalue()) / self._daily_start < self.p.max_daily_loss_pct

    def _calc_size(self, entry, stop):
        risk = abs(entry - stop)
        if risk <= 0:
            return 0
        val = self.broker.getvalue()
        by_risk = int(val * self.p.risk_pct / risk)
        by_cap = int(val * self.p.max_position_pct / entry)
        return max(3, min(by_risk, by_cap))

    def next(self):
        self._new_day()

        # ════════════════════════════════════════════
        # STEP 1: Force close at session end
        # ════════════════════════════════════════════
        if self._force_close_time():
            if self.position:
                self.close()
                self._active.clear()
            for order, *_ in self._pending_signals:
                self.cancel(order)
            self._pending_signals.clear()
            return

        # ════════════════════════════════════════════
        # STEP 2: Manage ALL active trades (stop/lock/trail)
        # ════════════════════════════════════════════
        if self._active:
            self._manage_trades()

        # ════════════════════════════════════════════
        # STEP 3: Cancel expired signals
        # ════════════════════════════════════════════
        remaining = []
        for item in self._pending_signals:
            order, bar, *rest = item
            if len(self) - bar > self.p.signal_valid_bars:
                self.cancel(order)
            else:
                remaining.append(item)
        self._pending_signals = remaining

        # ════════════════════════════════════════════
        # STEP 4: Scan for new touch/bounce/signal
        # ════════════════════════════════════════════
        if self._time_ok() and self._daily_loss_ok():
            self._scan()

    def _scan(self):
        a = self.atr[0]
        if a <= 0:
            return

        # Trend: simple EMA alignment
        trend = 0
        if self.data.close[0] > self.ema[0] and self.ema[0] > self.ema_slow[0]:
            trend = 1
        elif self.data.close[0] < self.ema[0] and self.ema[0] < self.ema_slow[0]:
            trend = -1
        if trend == 0:
            self._touch_bar = -1
            self._bounce_confirmed = False
            return

        tol = a * self.p.touch_tol

        # ─── BOUNCE FIRST: check previous bar's touch ───
        if self._touch_bar == len(self) - 1 and not self._bounce_confirmed:
            d = self._touch_dir
            if d == 1 and self.data.close[0] > self._touch_high:
                self._bounce_confirmed = True
            elif d == -1 and self.data.close[0] < self._touch_low:
                self._bounce_confirmed = True

            if self._bounce_confirmed:
                d = self._touch_dir
                if d == 1:
                    sig_price = self._touch_high + self.p.signal_offset
                    sig_stop = self._touch_low - self.p.stop_buffer_atr * self._touch_atr
                else:
                    sig_price = self._touch_low - self.p.signal_offset
                    sig_stop = self._touch_high + self.p.stop_buffer_atr * self._touch_atr

                size = self._calc_size(sig_price, sig_stop)
                if size > 0:
                    if d == 1:
                        order = self.buy(size=size, exectype=bt.Order.Stop, price=sig_price)
                    else:
                        order = self.sell(size=size, exectype=bt.Order.Stop, price=sig_price)

                    sig_risk = abs(sig_price - sig_stop)
                    self._pending_signals.append(
                        (order, len(self), sig_stop, d, self._touch_atr, sig_risk)
                    )

        # ─── THEN TOUCH: check if THIS bar is a new touch ───
        # (runs even if bounce just fired — no return above)
        if trend == 1:
            is_touch = (self.data.low[0] <= self.ema[0] + tol) and \
                       (self.data.low[0] >= self.ema[0] - a * self.p.touch_below_max)
        else:
            is_touch = (self.data.high[0] >= self.ema[0] - tol) and \
                       (self.data.high[0] <= self.ema[0] + a * self.p.touch_below_max)

        if is_touch:
            self._touch_bar = len(self)
            self._touch_high = self.data.high[0]
            self._touch_low = self.data.low[0]
            self._touch_atr = a
            self._touch_dir = trend
            self._bounce_confirmed = False

    def _manage_trades(self):
        """Check stop/lock/trail for ALL active trades."""
        closed_indices = []

        for idx, t in enumerate(self._active):
            d = t["dir"]
            entry = t["entry"]
            risk = t["risk"]
            entry_atr = t["entry_atr"]

            # ─── Stop check (EVERY bar, no skipping) ───
            if d == 1 and self.data.low[0] <= t["runner_stop"]:
                # Stopped out — close this trade's remaining shares
                remaining = t["size"] - t["locked_shares"]
                if remaining > 0 and self.position:
                    self.sell(size=min(remaining, abs(self.position.size)))
                closed_indices.append(idx)
                continue
            if d == -1 and self.data.high[0] >= t["runner_stop"]:
                remaining = t["size"] - t["locked_shares"]
                if remaining > 0 and self.position:
                    self.buy(size=min(remaining, abs(self.position.size)))
                closed_indices.append(idx)
                continue

            # ─── Lock1: 30% at 0.5:1 R:R ───
            if not t["lock1_done"]:
                target1 = entry + self.p.lock1_rr * risk * d
                hit = (d == 1 and self.data.high[0] >= target1) or \
                      (d == -1 and self.data.low[0] <= target1)
                if hit:
                    l1_size = max(1, int(t["size"] * self.p.lock1_pct))
                    l1_size = min(l1_size, abs(self.position.size) - 1) if self.position else 0
                    if l1_size > 0:
                        if d == 1:
                            self.sell(size=l1_size)
                        else:
                            self.buy(size=l1_size)
                        t["locked_shares"] += l1_size
                    t["lock1_done"] = True
                    # Move stop to breakeven
                    if d == 1:
                        t["runner_stop"] = max(t["runner_stop"], entry)
                    else:
                        t["runner_stop"] = min(t["runner_stop"], entry)

            # ─── Lock2: 20% at 1.5:1 R:R ───
            if t["lock1_done"] and not t["lock2_done"]:
                target2 = entry + self.p.lock2_rr * risk * d
                hit = (d == 1 and self.data.high[0] >= target2) or \
                      (d == -1 and self.data.low[0] <= target2)
                if hit:
                    l2_size = max(1, int(t["size"] * self.p.lock2_pct))
                    l2_size = min(l2_size, abs(self.position.size) - 1) if self.position else 0
                    if l2_size > 0:
                        if d == 1:
                            self.sell(size=l2_size)
                        else:
                            self.buy(size=l2_size)
                        t["locked_shares"] += l2_size
                    t["lock2_done"] = True

            # ─── Runner trail: 10-bar trailing low (uses ENTRY atr, fixed) ───
            bars_held = len(self) - t["entry_bar"]
            if t["lock1_done"] and bars_held >= self.p.runner_trail_bars:
                if d == 1:
                    rl = min(self.data.low.get(ago=-j, size=1)[0]
                             for j in range(self.p.runner_trail_bars))
                    new_stop = rl - self.p.runner_trail_buffer_atr * entry_atr
                    t["runner_stop"] = max(t["runner_stop"], new_stop)
                else:
                    rh = max(self.data.high.get(ago=-j, size=1)[0]
                             for j in range(self.p.runner_trail_bars))
                    new_stop = rh + self.p.runner_trail_buffer_atr * entry_atr
                    t["runner_stop"] = min(t["runner_stop"], new_stop)

        # Remove closed trades (reverse order to preserve indices)
        for idx in sorted(closed_indices, reverse=True):
            self._active.pop(idx)

    def notify_order(self, order):
        if order.status in (order.Submitted, order.Accepted):
            return

        if order.status == order.Completed:
            # Check if this is a signal entry fill
            for i, (pend_order, bar, sig_stop, sig_dir, sig_atr, sig_risk) in enumerate(self._pending_signals):
                if order is pend_order:
                    self._active.append({
                        "dir": sig_dir,
                        "entry": order.executed.price,
                        "entry_bar": len(self),
                        "entry_atr": sig_atr,
                        "risk": sig_risk,
                        "size": abs(order.executed.size),
                        "runner_stop": sig_stop,
                        "lock1_done": False,
                        "lock2_done": False,
                        "locked_shares": 0,
                    })
                    self._pending_signals.pop(i)
                    break

        if order.status in (order.Canceled, order.Margin, order.Rejected):
            self._pending_signals = [(o, b, s, d, a, r) for o, b, s, d, a, r
                                     in self._pending_signals if o is not order]

    def notify_trade(self, trade):
        pass  # we manage state via self._active
