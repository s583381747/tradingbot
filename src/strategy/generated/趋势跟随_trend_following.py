"""Auto-generated backtrader strategy: 趋势跟随 (Trend Following)."""

from __future__ import annotations

import backtrader as bt

from src.strategy.templates.backtrader_base import BaseGeneratedStrategy


class 趋势跟随TrendFollowing(BaseGeneratedStrategy):
    """Generated strategy: 趋势跟随 (Trend Following).

    
    """

    params = (
        ("printlog", False),
    )

    def __init__(self) -> None:
        super().__init__()
        pass  # no indicators

    def next(self) -> None:
        # Skip if an order is pending
        if self.has_pending_order():
            return

        # Entry logic
        if not self.position:
            if True:  # TODO: define entry conditions
                self.log('BUY CREATE %.2f' % self.data.close[0])
                self.order = self.buy()

        # Exit logic
        else:
            if True:  # TODO: define exit conditions
                self.log('SELL CREATE %.2f' % self.data.close[0])
                self.order = self.sell()
