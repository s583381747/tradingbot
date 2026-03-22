"""Custom intraday indicators for backtrader."""

from __future__ import annotations

import backtrader as bt


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
