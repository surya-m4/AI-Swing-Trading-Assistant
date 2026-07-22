"""
Custom exception hierarchy for Module 14: Paper Trading Engine.
"""

from __future__ import annotations


class TradingException(Exception):
    """Base exception for all paper trading errors."""

    def __init__(self, message: str, code: str = "TRADING_ERROR") -> None:
        super().__init__(message)
        self.message = message
        self.code = code

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"


class InsufficientFundsException(TradingException):
    """Raised when available cash balance is insufficient to execute a BUY order."""

    def __init__(self, required: float, available: float) -> None:
        msg = f"Insufficient funds: Required ₹{required:,.2f}, Available ₹{available:,.2f}."
        super().__init__(msg, code="INSUFFICIENT_FUNDS")
        self.required = required
        self.available = available


class InvalidOrderException(TradingException):
    """Raised when order parameters are invalid (e.g., negative quantity, non-existent ticker)."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code="INVALID_ORDER")


class PositionNotFoundException(TradingException):
    """Raised when attempting to close or modify a non-existent position."""

    def __init__(self, ticker: str) -> None:
        msg = f"No open position found for ticker '{ticker}'."
        super().__init__(msg, code="POSITION_NOT_FOUND")
        self.ticker = ticker


class RiskLimitExceededException(TradingException):
    """Raised when an order violates risk limits (e.g., position size cap exceeded)."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code="RISK_LIMIT_EXCEEDED")


class MarketClosedException(TradingException):
    """Raised when attempting an order outside market hours."""

    def __init__(self, message: str = "Market is currently closed.") -> None:
        super().__init__(message, code="MARKET_CLOSED")


class DuplicateOrderException(TradingException):
    """Raised when a duplicate order is detected within the cooldown window."""

    def __init__(self, ticker: str) -> None:
        msg = f"Duplicate order detected for '{ticker}' within the execution window."
        super().__init__(msg, code="DUPLICATE_ORDER")
        self.ticker = ticker
