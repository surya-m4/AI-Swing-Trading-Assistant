"""
Order management & execution engine for paper trading.

Handles validation and execution of BUY and SELL orders, updating cash,
updating holdings via PositionManager, and logging trades via TradeHistoryManager.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from src.trading.exceptions import InvalidOrderException, PositionNotFoundException
from src.trading.models import OrderRequest, OrderResponse, OrderStatus, OrderType, Position, Trade
from src.trading.portfolio import Portfolio
from src.trading.position_manager import PositionManager
from src.trading.risk_checks import RiskChecker
from src.trading.trade_history import TradeHistoryManager

logger = logging.getLogger(__name__)


class OrderManager:
    """Orchestrates order validation, execution, portfolio cash updates, and transaction logging.

    Attributes:
        portfolio: Portfolio instance.
        position_manager: PositionManager instance.
        trade_history: TradeHistoryManager instance.
        risk_checker: RiskChecker instance.
    """

    def __init__(
        self,
        portfolio: Portfolio,
        position_manager: PositionManager,
        trade_history: TradeHistoryManager,
        risk_checker: Optional[RiskChecker] = None,
    ) -> None:
        """Initialises the OrderManager.

        Args:
            portfolio: Portfolio instance.
            position_manager: PositionManager instance.
            trade_history: TradeHistoryManager instance.
            risk_checker: Optional RiskChecker. Creates default if None.
        """
        self.portfolio = portfolio
        self.position_manager = position_manager
        self.trade_history = trade_history
        self.risk_checker = risk_checker if risk_checker is not None else RiskChecker()
        logger.info("OrderManager initialized.")

    def execute_buy(
        self,
        ticker: str,
        quantity: float,
        price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        trailing_stop: Optional[float] = None,
    ) -> OrderResponse:
        """Validates and executes a BUY paper order.

        Steps:
            1. Validate risk controls (cash check, ticker check, position size limit).
            2. Deduct total cost (quantity * price) from portfolio cash.
            3. Add/update open position in PositionManager.
            4. Log Trade transaction in TradeHistoryManager.

        Args:
            ticker: Ticker symbol.
            quantity: Quantity to buy.
            price: Execution price.
            stop_loss: Optional Stop Loss price level.
            take_profit: Optional Take Profit price level.
            trailing_stop: Optional trailing stop offset.

        Returns:
            ``OrderResponse`` Pydantic model with order status details.
        """
        logger.info("Placing BUY order for %s: Qty=%.2f @ ₹%.2f.", ticker, quantity, price)

        current_val = self.portfolio.cash + self.position_manager.get_total_invested()

        # Pre-trade risk validation
        self.risk_checker.validate_buy_order(
            ticker=ticker,
            quantity=quantity,
            price=price,
            available_cash=self.portfolio.cash,
            portfolio_value=current_val,
            raise_exception=True,
        )

        cost = round(quantity * price, 2)
        self.portfolio.update_cash(-cost)

        pos = self.position_manager.open_or_update_position(
            ticker=ticker,
            quantity=quantity,
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            trailing_stop=trailing_stop,
        )

        trade_id = f"TRD_{uuid.uuid4().hex[:8].upper()}"
        trade = Trade(
            trade_id=trade_id,
            ticker=ticker,
            action="BUY",
            price=price,
            quantity=quantity,
            timestamp=datetime.utcnow().isoformat() + "Z",
            broker="Paper",
            status="EXECUTED",
            pnl=0.0,
        )
        self.trade_history.record_trade(trade)

        logger.info("BUY Order EXECUTED for %s. Cost: ₹%s. Trade ID: %s.", ticker, f"{cost:,.2f}", trade_id)

        return OrderResponse(
            trade_id=trade_id,
            ticker=ticker,
            action="BUY",
            quantity=quantity,
            price=price,
            status="EXECUTED",
            pnl=0.0,
            timestamp=trade.timestamp,
            message=f"BUY order executed successfully for {quantity} {ticker} @ ₹{price:,.2f}.",
        )

    def execute_sell(
        self,
        ticker: str,
        quantity: float,
        price: float,
    ) -> OrderResponse:
        """Validates and executes a SELL paper order (supports partial selling).

        Steps:
            1. Validate holdings existence and quantity.
            2. Reduce or close position via PositionManager.
            3. Calculate realized PnL.
            4. Credit cash proceeds (cost_basis + realized_pnl) to portfolio cash.
            5. Log Trade transaction in TradeHistoryManager.

        Args:
            ticker: Ticker symbol.
            quantity: Quantity to sell.
            price: Execution exit price.

        Returns:
            ``OrderResponse`` Pydantic model with realized PnL and trade details.
        """
        logger.info("Placing SELL order for %s: Qty=%.2f @ ₹%.2f.", ticker, quantity, price)

        pos = self.position_manager.get_position(ticker)
        if pos is None:
            logger.warning("Sell rejected: No open position for %s.", ticker)
            raise PositionNotFoundException(ticker)

        self.risk_checker.validate_sell_order(
            ticker=ticker,
            quantity=quantity,
            price=price,
            owned_quantity=pos.quantity,
            raise_exception=True,
        )

        realized_pnl, remaining_pos = self.position_manager.reduce_position(
            ticker=ticker, quantity=quantity, exit_price=price
        )

        proceeds = round((quantity * pos.entry_price) + realized_pnl, 2)
        self.portfolio.update_cash(proceeds)

        trade_id = f"TRD_{uuid.uuid4().hex[:8].upper()}"
        trade = Trade(
            trade_id=trade_id,
            ticker=ticker,
            action="SELL",
            price=price,
            quantity=quantity,
            timestamp=datetime.utcnow().isoformat() + "Z",
            broker="Paper",
            status="EXECUTED",
            pnl=realized_pnl,
        )
        self.trade_history.record_trade(trade)

        logger.info(
            "SELL Order EXECUTED for %s. Qty=%.2f, Exit=₹%.2f, Realized PnL=₹%s. Trade ID: %s.",
            ticker,
            quantity,
            price,
            f"{realized_pnl:+,.2f}",
            trade_id,
        )

        return OrderResponse(
            trade_id=trade_id,
            ticker=ticker,
            action="SELL",
            quantity=quantity,
            price=price,
            status="EXECUTED",
            pnl=realized_pnl,
            timestamp=trade.timestamp,
            message=f"SELL order executed for {quantity} {ticker} @ ₹{price:,.2f}. Realized PnL: ₹{realized_pnl:+,.2f}.",
        )

    def close_position(self, ticker: str, current_price: float) -> OrderResponse:
        """Completely closes an active open position at the current market price."""
        pos = self.position_manager.get_position(ticker)
        if pos is None:
            raise PositionNotFoundException(ticker)
        return self.execute_sell(ticker=ticker, quantity=pos.quantity, price=current_price)
