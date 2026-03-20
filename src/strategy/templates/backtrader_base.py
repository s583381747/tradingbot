"""Base class for generated backtrader strategies.

All AI-generated and template-generated strategies inherit from
``BaseGeneratedStrategy`` which provides common boilerplate: order
notification, trade logging, and a standardised parameter tuple.
"""

from __future__ import annotations

import logging

import backtrader as bt

logger = logging.getLogger(__name__)


class BaseGeneratedStrategy(bt.Strategy):
    """Reusable base for every auto-generated backtrader strategy.

    Subclasses must override ``__init__`` (to define indicators) and
    ``next`` (to define entry / exit logic).  Everything else — order
    management callbacks, logging, parameter defaults — is handled here.

    Usage in generated code::

        from src.strategy.templates.backtrader_base import BaseGeneratedStrategy

        class MyStrategy(BaseGeneratedStrategy):
            params = (
                ("fast_period", 10),
                ("slow_period", 30),
                # ... strategy-specific params ...
            )

            def __init__(self):
                super().__init__()
                # --- INDICATORS GO HERE ---
                self.sma_fast = bt.indicators.SMA(self.data.close, period=self.p.fast_period)
                self.sma_slow = bt.indicators.SMA(self.data.close, period=self.p.slow_period)

            def next(self):
                # --- ENTRY / EXIT LOGIC GOES HERE ---
                if self.sma_fast[0] > self.sma_slow[0] and not self.position:
                    self.buy()
                elif self.sma_fast[0] < self.sma_slow[0] and self.position:
                    self.sell()
    """

    # Common parameters shared by every generated strategy.
    params = (
        ("printlog", False),
    )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        """Initialise order-tracking state.

        Subclasses **must** call ``super().__init__()`` and then define
        their own indicators.
        """
        self.order: bt.Order | None = None
        self.buy_price: float | None = None
        self.buy_comm: float | None = None

    def next(self) -> None:
        """Called on every bar.  Override in subclasses."""
        pass  # pragma: no cover — subclasses implement logic

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log(self, txt: str, dt: object = None, do_print: bool = False) -> None:
        """Write a log line with the current bar date.

        Parameters
        ----------
        txt:
            Message to log.
        dt:
            Optional datetime override (defaults to current bar date).
        do_print:
            Force printing even when ``self.p.printlog`` is False.
        """
        if self.p.printlog or do_print:
            dt = dt or self.datas[0].datetime.date(0)
            logger.info("%s  %s", dt.isoformat(), txt)

    # ------------------------------------------------------------------
    # Notifications
    # ------------------------------------------------------------------

    def notify_order(self, order: bt.Order) -> None:
        """Handle order state transitions."""
        if order.status in (order.Submitted, order.Accepted):
            # Order submitted/accepted — nothing to do yet.
            return

        if order.status in (order.Completed,):
            if order.isbuy():
                self.log(
                    f"BUY EXECUTED  Price: {order.executed.price:.2f}  "
                    f"Cost: {order.executed.value:.2f}  Comm: {order.executed.comm:.2f}"
                )
                self.buy_price = order.executed.price
                self.buy_comm = order.executed.comm
            else:
                self.log(
                    f"SELL EXECUTED  Price: {order.executed.price:.2f}  "
                    f"Cost: {order.executed.value:.2f}  Comm: {order.executed.comm:.2f}"
                )

        elif order.status in (order.Canceled, order.Margin, order.Rejected):
            self.log("Order Canceled/Margin/Rejected")

        # Reset — no pending order.
        self.order = None

    def notify_trade(self, trade: bt.Trade) -> None:  # type: ignore[override]
        """Log trade profit / loss when a position is closed."""
        if not trade.isclosed:
            return
        self.log(
            f"TRADE CLOSED  Gross: {trade.pnl:.2f}  Net: {trade.pnlcomm:.2f}"
        )

    # ------------------------------------------------------------------
    # Helpers for subclasses
    # ------------------------------------------------------------------

    def has_pending_order(self) -> bool:
        """Return True when an order is pending execution."""
        return self.order is not None
