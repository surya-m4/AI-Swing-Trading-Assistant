"""
Module 14: Paper Trading Engine Package.

Provides virtual portfolio management, order execution, position tracking,
risk validation, trade history logging, and quantitative analytics.
"""

from .exceptions import (
    DuplicateOrderException,
    InsufficientFundsException,
    InvalidOrderException,
    MarketClosedException,
    PositionNotFoundException,
    RiskLimitExceededException,
    TradingException,
)
from .models import (
    OrderRequest,
    OrderResponse,
    OrderStatus,
    OrderType,
    PortfolioAnalyticsModel,
    PortfolioSummaryModel,
    Position,
    PositionType,
    RiskConfig,
    RiskStatusModel,
    Trade,
)
from .order_manager import OrderManager
from .paper_trader import PaperTrader
from .portfolio import DEFAULT_STARTING_CAPITAL, Portfolio
from .position_manager import PositionManager
from .risk_checks import RiskChecker
from .trade_history import TradeHistoryManager

__all__ = [
    "PaperTrader",
    "Portfolio",
    "OrderManager",
    "PositionManager",
    "RiskChecker",
    "TradeHistoryManager",
    "TradingException",
    "InsufficientFundsException",
    "InvalidOrderException",
    "PositionNotFoundException",
    "RiskLimitExceededException",
    "MarketClosedException",
    "DuplicateOrderException",
    "OrderRequest",
    "OrderResponse",
    "OrderType",
    "OrderStatus",
    "PositionType",
    "Position",
    "Trade",
    "PortfolioSummaryModel",
    "PortfolioAnalyticsModel",
    "RiskConfig",
    "RiskStatusModel",
    "DEFAULT_STARTING_CAPITAL",
]
