"""
Pydantic schemas for the FastAPI backend.

Contains request / response models for predictions, model info, health
checks, live market data, asset search, and paper trading.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


# ── Prediction ───────────────────────────────────────────────────────


class PredictRequest(BaseModel):
    """Schema for the prediction request.

    Expects a dictionary of features mapping feature names to their
    numeric values.
    """

    features: Dict[str, float]

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "features": {
                    "Open": 150.0,
                    "High": 155.0,
                    "Low": 149.0,
                    "Close": 153.0,
                    "Volume": 1000000,
                    "RSI": 45.5,
                    "MACD": 1.2,
                }
            }
        }
    )


class PredictResponse(BaseModel):
    """Schema for the prediction response."""

    action: str
    confidence_score: float
    model_name: str
    timestamp: str


# ── Model info ───────────────────────────────────────────────────────


class ModelInfoResponse(BaseModel):
    """Schema for model information response."""

    model_name: str
    version: Optional[str] = "1.0"
    training_date: Optional[str] = None
    hyperparameters: Dict[str, Any]
    accuracy: float
    precision: float
    recall: float
    f1_score: float


# ── Health ───────────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    """Schema for health check response."""

    status: str


# ── Market data ──────────────────────────────────────────────────────


class MarketQuoteResponse(BaseModel):
    """A single market quote with change percentage."""

    Ticker: str
    Open: float = 0.0
    High: float = 0.0
    Low: float = 0.0
    Close: float = 0.0
    Volume: int = 0
    Timestamp: str = ""
    Change_Pct: float = 0.0


class LiveMarketResponse(BaseModel):
    """Aggregated live market data response."""

    count: int
    quotes: List[MarketQuoteResponse]
    source: str = "cache"


# ── Live prediction ──────────────────────────────────────────────────


class LivePredictionItem(BaseModel):
    """A single live prediction for one symbol."""

    ticker: str
    action: str = "HOLD"
    confidence: float = 0.0
    probability: Dict[str, float] = Field(default_factory=dict)
    expected_return: float = 0.0
    risk_score: float = 0.0
    risk_level: str = "MEDIUM"
    model_name: str = ""
    close_price: float = 0.0


class LivePredictionResponse(BaseModel):
    """Aggregated live predictions response."""

    count: int
    predictions: List[LivePredictionItem]


# ── Asset search ─────────────────────────────────────────────────────


class AssetSearchItem(BaseModel):
    """A single search result."""

    symbol: str
    name: str
    category: str
    exchange: str = ""


class AssetSearchResponse(BaseModel):
    """Asset search results."""

    count: int
    results: List[AssetSearchItem]


# ── Manager status ───────────────────────────────────────────────────


class MarketStatusResponse(BaseModel):
    """Diagnostic status of the MarketManager."""

    total_assets: int = 0
    cache_size: int = 0
    scheduler_running: bool = False
    scheduler_ticks: int = 0
    sse_connections: int = 0
    favorites_count: int = 0
    watchlists_count: int = 0
    model_loaded: bool = False
    model_name: str = ""


# ── Paper Trading ────────────────────────────────────────────────────


class PaperOrderRequest(BaseModel):
    """Request schema for placing a paper trade order."""

    symbol: str
    action: str = "BUY"
    quantity: int = Field(..., gt=0, description="Order quantity")
    price: float = Field(..., gt=0, description="Execution price")
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


class PaperCloseRequest(BaseModel):
    """Request schema for closing an open position."""

    symbol: str
    exit_price: float = Field(..., gt=0, description="Exit execution price")


class PaperModifyRequest(BaseModel):
    """Request schema for modifying Stop Loss / Take Profit."""

    symbol: str
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


class PaperResetRequest(BaseModel):
    """Request schema for resetting virtual paper trading account."""

    initial_capital: float = Field(100000.0, gt=0, description="Starting capital balance")


class PaperRecommendationResponse(BaseModel):
    """AI Trade Recommendation schema with position sizing & risk levels."""

    symbol: str
    action: str
    recommended_size: int
    risk_pct: float
    stop_loss: float
    take_profit: float
    confidence_score: float
    expected_return: float = 0.0
    risk_level: str = "MEDIUM"
    entry_price: float = 0.0


class PaperPortfolioSummaryResponse(BaseModel):
    """Portfolio metrics and summary response."""

    initial_capital: float
    cash: float
    portfolio_value: float
    unrealized_pnl: float
    realized_pnl: float
    total_pnl: float
    win_rate: float
    avg_return: float
    max_drawdown: float
    open_positions_count: int
    closed_trades_count: int
