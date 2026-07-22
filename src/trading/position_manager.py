"""
Position management module for Module 14: Paper Trading Engine.

Tracks open positions, entry prices, mark-to-market prices, holding days,
trailing stops, stop-loss & take-profit execution triggers, and partial position closing.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from src.trading.exceptions import PositionNotFoundException
from src.trading.models import Position, PositionType

logger = logging.getLogger(__name__)


class PositionManager:
    """Manages active open positions, mark-to-market updates, trailing stops, and partial closing.

    Attributes:
        positions: Dictionary mapping ticker string to Position object.
    """

    def __init__(self) -> None:
        """Initialises an empty PositionManager."""
        self.positions: Dict[str, Position] = {}

    def open_or_update_position(
        self,
        ticker: str,
        quantity: float,
        price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        trailing_stop: Optional[float] = None,
        position_type: PositionType = PositionType.LONG,
    ) -> Position:
        """Opens a new position or adds to an existing position.

        If position exists, averages entry price and adds quantity.

        Args:
            ticker: Asset ticker symbol.
            quantity: Quantity being added.
            price: Execution entry price.
            stop_loss: Optional Stop Loss level.
            take_profit: Optional Take Profit level.
            trailing_stop: Optional trailing stop offset.
            position_type: LONG or SHORT.

        Returns:
            Updated ``Position`` instance.
        """
        existing = self.positions.get(ticker)

        if existing is None:
            pos = Position(
                ticker=ticker,
                entry_price=price,
                current_price=price,
                quantity=quantity,
                stop_loss=stop_loss,
                take_profit=take_profit,
                trailing_stop=trailing_stop,
                position_type=position_type,
            )
            self.positions[ticker] = pos
            logger.info("Opened new %s position for %s: Qty=%.2f @ ₹%.2f.", position_type.value, ticker, quantity, price)
            return pos

        # Average existing position entry price
        new_qty = existing.quantity + quantity
        new_entry = ((existing.quantity * existing.entry_price) + (quantity * price)) / new_qty

        existing.quantity = round(new_qty, 4)
        existing.entry_price = round(new_entry, 2)
        existing.current_price = price

        if stop_loss is not None:
            existing.stop_loss = stop_loss
        if take_profit is not None:
            existing.take_profit = take_profit
        if trailing_stop is not None:
            existing.trailing_stop = trailing_stop

        logger.info("Updated position for %s: New Qty=%.2f, Avg Entry=₹%.2f.", ticker, existing.quantity, existing.entry_price)
        return existing

    def reduce_position(
        self, ticker: str, quantity: float, exit_price: float
    ) -> Tuple[float, Position | None]:
        """Reduces or closes an open position (supports partial selling).

        Args:
            ticker: Ticker symbol.
            quantity: Quantity to sell/close.
            exit_price: Exit execution price.

        Returns:
            Tuple of (realized_pnl, remaining_position_or_None).

        Raises:
            PositionNotFoundException: If position does not exist.
        """
        pos = self.positions.get(ticker)
        if pos is None:
            raise PositionNotFoundException(ticker)

        if quantity > pos.quantity:
            quantity = pos.quantity

        # Calculate realized PnL for the portion sold
        if pos.position_type == PositionType.SHORT:
            realized_pnl = (pos.entry_price - exit_price) * quantity
        else:
            realized_pnl = (exit_price - pos.entry_price) * quantity

        realized_pnl = round(realized_pnl, 2)
        remaining_qty = round(pos.quantity - quantity, 4)

        if remaining_qty <= 0:
            del self.positions[ticker]
            logger.info("Closed position for %s @ ₹%.2f. Realized PnL: ₹%.2f.", ticker, exit_price, realized_pnl)
            return realized_pnl, None

        pos.quantity = remaining_qty
        pos.current_price = exit_price
        logger.info(
            "Partially closed %s: Sold=%.2f, Remaining=%.2f @ ₹%.2f. PnL: ₹%.2f.",
            ticker,
            quantity,
            remaining_qty,
            exit_price,
            realized_pnl,
        )
        return realized_pnl, pos

    def update_market_prices(
        self, current_prices: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Updates mark-to-market prices and evaluates trailing stops & SL/TP triggers.

        Args:
            current_prices: Dictionary mapping ticker to latest market price.

        Returns:
            List of triggered order event dicts (e.g. SL/TP/Trailing stop hit).
        """
        triggered_events: List[Dict[str, Any]] = []

        for ticker, pos in list(self.positions.items()):
            if ticker not in current_prices:
                continue

            price = current_prices[ticker]
            pos.current_price = price

            # Update holding days
            try:
                created_dt = datetime.fromisoformat(pos.created_at.replace("Z", "+00:00"))
                pos.holding_days = max(0, (datetime.utcnow() - created_dt.replace(tzinfo=None)).days)
            except Exception:
                pass

            # Trailing stop update logic
            if pos.trailing_stop is not None and pos.trailing_stop > 0:
                if pos.position_type == PositionType.LONG:
                    # Raise stop loss as price rises
                    new_sl = round(price - pos.trailing_stop, 2)
                    if pos.stop_loss is None or new_sl > pos.stop_loss:
                        pos.stop_loss = new_sl
                        logger.debug("Adjusted trailing stop for %s LONG to ₹%.2f.", ticker, new_sl)

            # Check SL/TP triggers
            if pos.position_type == PositionType.LONG:
                if pos.stop_loss is not None and price <= pos.stop_loss:
                    triggered_events.append({"ticker": ticker, "trigger": "STOP_LOSS", "price": price})
                elif pos.take_profit is not None and price >= pos.take_profit:
                    triggered_events.append({"ticker": ticker, "trigger": "TAKE_PROFIT", "price": price})

        return triggered_events

    def get_position(self, ticker: str) -> Optional[Position]:
        """Retrieves position for a ticker."""
        return self.positions.get(ticker)

    def get_all_positions(self) -> List[Position]:
        """Returns list of all active open positions."""
        return list(self.positions.values())

    def get_total_invested(self) -> float:
        """Returns total current market value of all open positions."""
        return round(sum(p.current_value for p in self.positions.values()), 2)

    def get_total_unrealized_pnl(self) -> float:
        """Returns sum of unrealized PnL across all open positions."""
        return round(sum(p.pnl for p in self.positions.values()), 2)
