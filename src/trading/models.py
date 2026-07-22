"""
Data models, enums, dataclasses, and Pydantic schemas for Module 14: Paper Trading Engine.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


# ── Enums ─────────────────────────────────────────────────────────────


class OrderType(str, enum.Enum):
    """Supported order actions."""

    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(str, enum.Enum):
    """Lifecycle status of a trading order."""

    PENDING = "PENDING"
    EXECUTED = "EXECUTED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"


class PositionType(str, enum.Enum):
    """Position direction."""

    LONG = "LONG"
    SHORT = "SHORT"


# ── Dataclasses / Domain Models ──────────────────────────────────────


@dataclass
class Position:
    """Dataclass representing an active open position.

    Attributes:
        ticker: Symbol of the security or pair.
        entry_price: Average entry execution price.
        current_price: Mark-to-market current price.
        quantity: Current position quantity (integer or float for forex).
        stop_loss: Optional Stop Loss price level.
        take_profit: Optional Take Profit price level.
        trailing_stop: Optional trailing stop offset or trigger level.
        holding_days: Number of days the position has been held.
        created_at: ISO timestamp string when position was opened.
        position_type: Position direction (LONG or SHORT).
    """

    ticker: str
    entry_price: float
    current_price: float
    quantity: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop: Optional[float] = None
    holding_days: int = 0
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    position_type: PositionType = PositionType.LONG

    @property
    def current_value(self) -> float:
        """Returns current mark-to-market position value."""
        return round(self.quantity * self.current_price, 2)

    @property
    def pnl(self) -> float:
        """Returns unrealized profit/loss."""
        if self.position_type == PositionType.SHORT:
            return round((self.entry_price - self.current_price) * self.quantity, 2)
        return round((self.current_price - self.entry_price) * self.quantity, 2)

    @property
    def pnl_pct(self) -> float:
        """Returns unrealized percentage return."""
        cost = self.quantity * self.entry_price
        if cost <= 0:
            return 0.0
        return round((self.pnl / cost) * 100.0, 2)

    def to_dict(self) -> Dict[str, Any]:
        """Converts position to a dictionary."""
        return {
            "ticker": self.ticker,
            "entry_price": self.entry_price,
            "current_price": self.current_price,
            "quantity": self.quantity,
            "current_value": self.current_value,
            "pnl": self.pnl,
            "pnl_pct": self.pnl_pct,
            "holding_days": self.holding_days,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "trailing_stop": self.trailing_stop,
            "created_at": self.created_at,
            "position_type": self.position_type.value,
        }


@dataclass
class Trade:
    """Dataclass representing a completed or historical transaction.

    Attributes:
        trade_id: Unique identifier string for the trade.
        ticker: Ticker symbol.
        action: Order action string (BUY, SELL, CLOSE).
        price: Execution price.
        quantity: Quantity traded.
        timestamp: ISO timestamp string of execution.
        broker: Broker name ("Paper").
        status: Execution status (EXECUTED, REJECTED, etc.).
        pnl: Realized profit/loss for closed trades.
    """

    trade_id: str
    ticker: str
    action: str
    price: float
    quantity: float
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    broker: str = "Paper"
    status: str = "EXECUTED"
    pnl: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Converts trade to dictionary."""
        return {
            "trade_id": self.trade_id,
            "ticker": self.ticker,
            "action": self.action,
            "price": self.price,
            "quantity": self.quantity,
            "timestamp": self.timestamp,
            "broker": self.broker,
            "status": self.status,
            "pnl": self.pnl,
        }


# ── Pydantic Request / Response Models for API & Analytics ───────────


class OrderRequest(BaseModel):
    """Schema for submitting a BUY or SELL order."""

    ticker: str = Field(..., description="Ticker symbol (e.g. RELIANCE.NS, EURUSD=X)")
    quantity: float = Field(..., gt=0, description="Order quantity")
    price: float = Field(..., gt=0, description="Execution price")
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop: Optional[float] = None


class OrderResponse(BaseModel):
    """Schema returned after placing an order."""

    trade_id: str
    ticker: str
    action: str
    quantity: float
    price: float
    status: str
    pnl: float = 0.0
    timestamp: str
    message: str = "Order executed successfully."


class PortfolioSummaryModel(BaseModel):
    """Schema representing live portfolio status and basic metrics."""

    cash: float
    invested_amount: float
    available_margin: float
    portfolio_value: float
    daily_pnl: float
    overall_pnl: float
    win_rate: float
    loss_rate: float
    max_drawdown: float
    roi: float


class PortfolioAnalyticsModel(BaseModel):
    """Schema for detailed quantitative portfolio performance analytics."""

    portfolio_return: float
    total_profit: float
    total_loss: float
    sharpe_ratio: float
    sortino_ratio: float
    profit_factor: float
    avg_winning_trade: float
    avg_losing_trade: float
    max_drawdown: float
    win_percentage: float


class RiskConfig(BaseModel):
    """Risk configuration settings."""

    max_position_size_pct: float = 20.0  # Max % of portfolio per ticker
    max_portfolio_risk_pct: float = 2.0  # Max risk per trade
    allow_margin: bool = False
    cooldown_seconds: float = 2.0


class RiskStatusModel(BaseModel):
    """Schema returning current risk limits and check status."""

    max_position_size_pct: float
    cash_available: float
    is_market_open: bool
    risk_limits_ok: bool
    status_message: str = "All risk checks passed."
