"""20 EMA Pullback Trend Following Strategy - Full Implementation.

Based on analysis of 35 trading education videos from @rijindoudoujin.
Core logic: Trend confirmation -> Pullback to 20EMA/VWAP -> Breakout entry -> Trail exit.
"""

from __future__ import annotations

import backtrader as bt


class EmaPullbackSystem(bt.Strategy):
    """Intraday trend following strategy using 20 EMA pullback.

    Entry: Price pulls back to 20 EMA in a confirmed trend, then breaks out.
    Exit: Trailing stop along EMA or fixed stop-loss at swing point.
    """

    params = (
        # EMA
        ("ema_fast", 20),
        ("ema_slow", 40),
        ("ema_filter", 60),

        # Trend confirmation
        ("trend_bars", 5),
        ("trend_atr_mult", 1.0),

        # Pullback detection
        ("pullback_atr_tol", 0.3),
        ("max_pullback_bars", 8),

        # Volume filter
        ("use_vol_filter", True),
        ("vol_ma_len", 20),
        ("vol_thresh", 0.8),

        # Stop loss
        ("sl_atr_mult", 1.5),
        ("use_trail", True),
        ("trail_atr_offset", 0.3),

        # Take profit
        ("tp_atr_mult", 3.0),
        ("use_partial_tp", True),
        ("partial_tp_pct", 0.5),
        ("partial_tp_rr", 1.5),

        # Risk
        ("risk_per_trade", 0.02),
        ("max_daily_trades", 3),

        # Time filter (minutes from midnight)
        ("use_time_filter", False),
        ("trade_start", 630),   # 10:30
        ("trade_end", 930),     # 15:30
        ("force_close", 955),   # 15:55

        # Direction
        ("long_only", False),
        ("short_only", False),
        ("inverse", False),       # Flip all signals: buy→sell, sell→buy

        # ORB filter: first candle too big = range day risk
        ("use_orb_filter", True),
        ("orb_atr_mult", 2.0),       # first candle range > N*ATR = dangerous
        ("orb_cooldown_bars", 30),    # bars to wait after wide open

        # Logging
        ("printlog", False),
    )

    def __init__(self):
        # Indicators
        self.ema_fast = bt.indicators.EMA(self.data.close, period=self.p.ema_fast)
        self.ema_slow = bt.indicators.EMA(self.data.close, period=self.p.ema_slow)
        self.ema_filter = bt.indicators.EMA(self.data.close, period=self.p.ema_filter)
        self.atr = bt.indicators.ATR(self.data, period=14)
        self.vol_ma = bt.indicators.SMA(self.data.volume, period=self.p.vol_ma_len)

        # State tracking
        self.order = None
        self.entry_price = 0.0
        self.stop_loss = 0.0
        self.trail_stop = 0.0
        self.take_profit = 0.0
        self.initial_risk = 0.0
        self.partial_done = False
        self.bars_since_pullback = 999
        self.daily_trades = 0
        self.current_day = None
        self.had_bullish_wick = False
        self.had_bearish_wick = False

        # ORB (Open Range Breakout) filter state
        self.orb_first_bar_range = 0.0
        self.orb_wide_open = False
        self.orb_bar_count = 0

    def log(self, txt, dt=None):
        if self.p.printlog:
            dt = dt or self.datas[0].datetime.datetime(0)
            print(f"{dt:%Y-%m-%d %H:%M} {txt}")

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f"BUY @ {order.executed.price:.2f}, Size={order.executed.size:.0f}")
            else:
                self.log(f"SELL @ {order.executed.price:.2f}, Size={order.executed.size:.0f}")
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f"Order {order.Status[order.status]}")
        self.order = None

    def notify_trade(self, trade):
        if trade.isclosed:
            self.log(f"TRADE P&L: Gross={trade.pnl:.2f}, Net={trade.pnlcomm:.2f}")

    def _get_time_minutes(self):
        dt = self.data.datetime.datetime(0)
        return dt.hour * 60 + dt.minute

    def _is_new_day(self):
        dt = self.data.datetime.datetime(0)
        day = dt.date()
        if self.current_day != day:
            self.current_day = day
            self.orb_bar_count = 0
            self.orb_wide_open = False
            self.orb_first_bar_range = 0.0
            return True
        return False

    def _orb_ok(self):
        """Check if first candle of day was too wide (range/chop risk).
        NY open first candle too big -> likely consolidation day, be careful.
        """
        if not self.p.use_orb_filter:
            return True
        # During cooldown after wide open, block trades
        if self.orb_wide_open and self.orb_bar_count < self.p.orb_cooldown_bars:
            return False
        return True

    def _is_bullish(self):
        """Confirmed uptrend: EMA alignment + price displacement."""
        if self.ema_fast[0] <= self.ema_slow[0]:
            return False
        if self.data.close[0] <= self.ema_fast[0]:
            return False
        # Check price displacement
        if len(self.data) > self.p.trend_bars:
            disp = self.data.close[0] - self.data.close[-self.p.trend_bars]
            if disp <= self.atr[0] * self.p.trend_atr_mult:
                return False
        return True

    def _is_bearish(self):
        """Confirmed downtrend: EMA alignment + price displacement."""
        if self.ema_fast[0] >= self.ema_slow[0]:
            return False
        if self.data.close[0] >= self.ema_fast[0]:
            return False
        if len(self.data) > self.p.trend_bars:
            disp = self.data.close[-self.p.trend_bars] - self.data.close[0]
            if disp <= self.atr[0] * self.p.trend_atr_mult:
                return False
        return True

    def _is_near_ema(self):
        """Price is near 20 EMA (within ATR tolerance)."""
        tol = self.atr[0] * self.p.pullback_atr_tol
        ema_val = self.ema_fast[0]
        return (abs(self.data.low[0] - ema_val) <= tol or
                abs(self.data.high[0] - ema_val) <= tol or
                (self.data.low[0] <= ema_val <= self.data.high[0]))

    def _vol_ok(self):
        """Volume confirmation."""
        if not self.p.use_vol_filter:
            return True
        return self.data.volume[0] >= self.vol_ma[0] * self.p.vol_thresh

    def _check_wick(self):
        """Check for rejection wicks during pullback window."""
        body = abs(self.data.close[0] - self.data.open[0])
        if body < 0.0001:
            body = 0.0001
        upper_wick = self.data.high[0] - max(self.data.close[0], self.data.open[0])
        lower_wick = min(self.data.close[0], self.data.open[0]) - self.data.low[0]

        if lower_wick > body * 1.5:
            self.had_bullish_wick = True
        if upper_wick > body * 1.5:
            self.had_bearish_wick = True

    def _swing_low(self, bars):
        """Lowest low over N bars."""
        lo = self.data.low[0]
        for i in range(1, min(bars + 1, len(self.data))):
            lo = min(lo, self.data.low[-i])
        return lo

    def _swing_high(self, bars):
        """Highest high over N bars."""
        hi = self.data.high[0]
        for i in range(1, min(bars + 1, len(self.data))):
            hi = max(hi, self.data.high[-i])
        return hi

    def _calc_position_size(self, risk_amount):
        """Calculate position size based on risk per trade."""
        if risk_amount <= 0:
            return 0
        cash = self.broker.getcash()
        risk_dollars = cash * self.p.risk_per_trade
        size = int(risk_dollars / risk_amount)
        return max(1, size)

    def next(self):
        if self.order:
            return

        # Not enough data
        if len(self.data) < self.p.ema_filter + 10:
            return

        # New day reset
        if self._is_new_day():
            self.daily_trades = 0

        # ORB tracking: detect wide first candle
        self.orb_bar_count += 1
        if self.orb_bar_count == 1:
            self.orb_first_bar_range = self.data.high[0] - self.data.low[0]
            if self.atr[0] > 0 and self.orb_first_bar_range > self.atr[0] * self.p.orb_atr_mult:
                self.orb_wide_open = True
                self.log(f"ORB WARNING: wide first bar range={self.orb_first_bar_range:.2f} > {self.atr[0]*self.p.orb_atr_mult:.2f}")

        # Time filter
        if self.p.use_time_filter:
            t = self._get_time_minutes()
            # Force close at end of day
            if t >= self.p.force_close and self.position:
                self.log("FORCE CLOSE end of day")
                self.close()
                return

        # ====== POSITION MANAGEMENT ======
        if self.position:
            if self.position.size > 0:  # Long position
                # Trailing stop along EMA
                if self.p.use_trail:
                    new_trail = self.ema_fast[0] - self.atr[0] * self.p.trail_atr_offset
                    if new_trail > self.trail_stop and new_trail < self.data.close[0]:
                        self.trail_stop = new_trail

                # Check trail stop
                if self.data.close[0] <= self.trail_stop:
                    self.log(f"TRAIL STOP @ {self.data.close[0]:.2f}")
                    self.close()
                    return

                # Partial take profit
                if self.p.use_partial_tp and not self.partial_done and self.initial_risk > 0:
                    tp1 = self.entry_price + self.initial_risk * self.p.partial_tp_rr
                    if self.data.close[0] >= tp1:
                        sell_size = int(abs(self.position.size) * self.p.partial_tp_pct)
                        if sell_size > 0:
                            self.log(f"PARTIAL TP @ {self.data.close[0]:.2f}")
                            self.sell(size=sell_size)
                            self.partial_done = True
                            self.trail_stop = max(self.trail_stop, self.entry_price)

                # Full take profit
                if self.data.close[0] >= self.take_profit:
                    self.log(f"FULL TP @ {self.data.close[0]:.2f}")
                    self.close()
                    return

                # Hard stop
                if self.data.close[0] <= self.stop_loss:
                    self.log(f"STOP LOSS @ {self.data.close[0]:.2f}")
                    self.close()
                    return

            elif self.position.size < 0:  # Short position
                if self.p.use_trail:
                    new_trail = self.ema_fast[0] + self.atr[0] * self.p.trail_atr_offset
                    if new_trail < self.trail_stop and new_trail > self.data.close[0]:
                        self.trail_stop = new_trail

                if self.data.close[0] >= self.trail_stop:
                    self.log(f"TRAIL STOP @ {self.data.close[0]:.2f}")
                    self.close()
                    return

                if self.p.use_partial_tp and not self.partial_done and self.initial_risk > 0:
                    tp1 = self.entry_price - self.initial_risk * self.p.partial_tp_rr
                    if self.data.close[0] <= tp1:
                        buy_size = int(abs(self.position.size) * self.p.partial_tp_pct)
                        if buy_size > 0:
                            self.log(f"PARTIAL TP @ {self.data.close[0]:.2f}")
                            self.buy(size=buy_size)
                            self.partial_done = True
                            self.trail_stop = min(self.trail_stop, self.entry_price)

                if self.data.close[0] <= self.take_profit:
                    self.log(f"FULL TP @ {self.data.close[0]:.2f}")
                    self.close()
                    return

                if self.data.close[0] >= self.stop_loss:
                    self.log(f"STOP LOSS @ {self.data.close[0]:.2f}")
                    self.close()
                    return

            return  # Don't enter new positions while holding

        # ====== ENTRY LOGIC ======

        # Time window check
        if self.p.use_time_filter:
            t = self._get_time_minutes()
            if t < self.p.trade_start or t > self.p.trade_end:
                return

        # Daily trade limit
        if self.daily_trades >= self.p.max_daily_trades:
            return

        # Pullback tracking
        if self._is_near_ema():
            self.bars_since_pullback = 0
            self.had_bullish_wick = False
            self.had_bearish_wick = False
        else:
            self.bars_since_pullback += 1

        # Check wicks during pullback window
        if self.bars_since_pullback <= self.p.max_pullback_bars:
            self._check_wick()

        in_pullback_window = 0 < self.bars_since_pullback <= self.p.max_pullback_bars

        # Detect signals (direction-neutral)
        long_signal = (not self.p.short_only and
            self._is_bullish() and
            in_pullback_window and
            self.data.close[0] > self.data.high[-1] and
            self.data.close[0] > self.ema_fast[0] and
            self._vol_ok() and
            self._orb_ok())

        short_signal = (not self.p.long_only and
              self._is_bearish() and
              in_pullback_window and
              self.data.close[0] < self.data.low[-1] and
              self.data.close[0] < self.ema_fast[0] and
              self._vol_ok() and
              self._orb_ok())

        # INVERSE MODE: flip signals
        if self.p.inverse:
            long_signal, short_signal = short_signal, long_signal

        # ---- LONG ENTRY ----
        if long_signal:
            swing_lo = self._swing_low(self.p.max_pullback_bars)
            self.stop_loss = swing_lo - self.atr[0] * 0.1
            self.initial_risk = self.data.close[0] - self.stop_loss

            if self.initial_risk > 0:
                size = self._calc_position_size(self.initial_risk)
                self.entry_price = self.data.close[0]
                self.trail_stop = self.stop_loss
                self.take_profit = self.entry_price + self.atr[0] * self.p.tp_atr_mult
                self.partial_done = False
                self.daily_trades += 1

                self.log(f"LONG ENTRY @ {self.entry_price:.2f}, SL={self.stop_loss:.2f}, TP={self.take_profit:.2f}, Size={size}")
                self.buy(size=size)

        # ---- SHORT ENTRY ----
        elif short_signal:
            swing_hi = self._swing_high(self.p.max_pullback_bars)
            self.stop_loss = swing_hi + self.atr[0] * 0.1
            self.initial_risk = self.stop_loss - self.data.close[0]

            if self.initial_risk > 0:
                size = self._calc_position_size(self.initial_risk)
                self.entry_price = self.data.close[0]
                self.trail_stop = self.stop_loss
                self.take_profit = self.entry_price - self.atr[0] * self.p.tp_atr_mult
                self.partial_done = False
                self.daily_trades += 1

                self.log(f"SHORT ENTRY @ {self.entry_price:.2f}, SL={self.stop_loss:.2f}, TP={self.take_profit:.2f}, Size={size}")
                self.sell(size=size)
