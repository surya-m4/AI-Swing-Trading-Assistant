"""
Risk validation engine for paper trading.

Enforces pre-trade risk checks including cash availability, position size limits,
duplicate order prevention, non-negative quantities, and market status checks.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, Optional, Tuple

from src.trading.exceptions import (
    DuplicateOrderException,
    InsufficientFundsException,
    InvalidOrderException,
    MarketClosedException,
    RiskLimitExceededException,
)
from src.trading.models import RiskConfig

logger = logging.getLogger(__name__)


class RiskChecker:
    """Validates pre-trade risk controls and raises custom exceptions on violation.

    Attributes:
        config: RiskConfig instance specifying risk thresholds.
        market_open_mock: Boolean indicating if market is simulated open (default True).
    """

    def __init__(
        self,
        config: Optional[RiskConfig] = None,
        market_open_mock: bool = True,
    ) -> None:
        """Initialises the RiskChecker.

        Args:
            config: Optional ``RiskConfig``. Defaults to default settings.
            market_open_mock: Simulated market open status.
        """
        self.config = config if config is not None else RiskConfig()
        self.market_open_mock = market_open_mock
        self._recent_orders: Dict[str, float] = {}

    def set_market_status(self, is_open: bool) -> None:
        """Sets simulated market open status."""
        self.market_open_mock = is_open

    def validate_buy_order(
        self,
        ticker: str,
        quantity: float,
        price: float,
        available_cash: float,
        portfolio_value: float,
        raise_exception: bool = True,
    ) -> Tuple[bool, str]:
        """Validates all risk constraints before executing a BUY order.

        Args:
            ticker: Ticker symbol.
            quantity: Order quantity.
            price: Order price.
            available_cash: Current available cash.
            portfolio_value: Current total portfolio valuation.
            raise_exception: If True, raises custom exceptions on failure.

        Returns:
            Tuple of (is_valid, status_message).
        """
        # 1. Market Open Check
        if not self.market_open_mock:
            msg = f"Market is closed. Cannot execute order for '{ticker}'."
            if raise_exception:
                raise MarketClosedException(msg)
            return False, msg

        # 2. Ticker and Quantity Validation
        if not ticker or not ticker.strip():
            msg = "Ticker symbol cannot be empty."
            if raise_exception:
                raise InvalidOrderException(msg)
            return False, msg

        if quantity <= 0 or price <= 0:
            msg = f"Invalid order parameters: quantity={quantity}, price={price}."
            if raise_exception:
                raise InvalidOrderException(msg)
            return False, msg

        # 3. Duplicate Order Check (Cooldown)
        now = time.time()
        last_time = self._recent_orders.get(ticker, 0.0)
        if now - last_time < self.config.cooldown_seconds:
            if raise_exception:
                raise DuplicateOrderException(ticker)
            return False, f"Duplicate order for '{ticker}' within cooldown window."

        # 4. Insufficient Funds Check
        required_cash = quantity * price
        if required_cash > available_cash:
            if raise_exception:
                raise InsufficientFundsException(required=required_cash, available=available_cash)
            return False, f"Insufficient funds: Required ₹{required_cash:,.2f}, Available ₹{available_cash:,.2f}."

        # 5. Position Size Cap Check (% of Portfolio Value)
        if portfolio_value > 0:
            position_pct = (required_cash / portfolio_value) * 100.0
            if position_pct > self.config.max_position_size_pct:
                msg = (
                    f"Position size ₹{required_cash:,.2f} ({position_pct:.1f}%) exceeds "
                    f"max allowed limit of {self.config.max_position_size_pct:.1f}%."
                )
                if raise_exception:
                    raise RiskLimitExceededException(msg)
                return False, msg

        # Record timestamp for duplicate check
        self._recent_orders[ticker] = now
        return True, "Risk validation passed."

    def validate_sell_order(
        self,
        ticker: str,
        quantity: float,
        price: float,
        owned_quantity: float,
        raise_exception: bool = True,
    ) -> Tuple[bool, str]:
        """Validates risk constraints before executing a SELL order.

        Args:
            ticker: Ticker symbol.
            quantity: Quantity to sell.
            price: Exit execution price.
            owned_quantity: Current open quantity owned.
            raise_exception: If True, raises exceptions on failure.

        Returns:
            Tuple of (is_valid, status_message).
        """
        if not self.market_open_mock:
            msg = f"Market is closed. Cannot execute sell for '{ticker}'."
            if raise_exception:
                raise MarketClosedException(msg)
            return False, msg

        if quantity <= 0 or price <= 0:
            msg = f"Invalid sell parameters: quantity={quantity}, price={price}."
            if raise_exception:
                raise InvalidOrderException(msg)
            return False, msg

        if quantity > owned_quantity:
            msg = f"Cannot sell {quantity} of '{ticker}'; only {owned_quantity} owned."
            if raise_exception:
                raise InvalidOrderException(msg)
            return False, msg

        return True, "Sell order risk validation passed."
