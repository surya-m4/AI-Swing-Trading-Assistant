"""
FastAPI Routes for the AI Swing Trading Assistant API.

Includes health, predictions, model info, live market data, and Module 14 Paper Trading endpoints.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse, StreamingResponse

from .dependencies import get_market_manager, get_paper_broker, get_paper_trader, get_predictor
from .health import check_health
from .predictor import ModelPredictor
from .schemas import (
    AssetSearchItem,
    AssetSearchResponse,
    HealthResponse,
    LiveMarketResponse,
    LivePredictionItem,
    LivePredictionResponse,
    MarketQuoteResponse,
    MarketStatusResponse,
    ModelInfoResponse,
    PaperCloseRequest,
    PaperModifyRequest,
    PaperOrderRequest,
    PaperPortfolioSummaryResponse,
    PaperRecommendationResponse,
    PaperResetRequest,
    PredictRequest,
    PredictResponse,
)
from src.trading.exceptions import TradingException
from src.trading.models import (
    OrderRequest,
    OrderResponse,
    PortfolioAnalyticsModel,
    PortfolioSummaryModel,
    RiskStatusModel,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# ── General ──────────────────────────────────────────────────────────


@router.get("/", tags=["General"])
async def root():
    """Welcome endpoint."""
    return {"message": "Welcome to the AI Swing Trading Assistant API"}


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    """Health check endpoint."""
    health_status = check_health()
    if health_status.get("status") != "ok":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service Unhealthy",
        )
    return {"status": "ok"}


# ── Prediction (existing) ───────────────────────────────────────────


@router.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict(
    request: PredictRequest,
    predictor: ModelPredictor = Depends(get_predictor),
):
    """Predicts the trading action based on provided stock features."""
    if not request.features:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Features cannot be empty.",
        )

    try:
        action, confidence = predictor.predict(request.features)

        return PredictResponse(
            action=action,
            confidence_score=confidence,
            model_name=predictor.model_name,
            timestamp=datetime.utcnow().isoformat() + "Z",
        )
    except RuntimeError as e:
        logger.error("Prediction failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e)
        )
    except Exception as e:
        logger.error("Unexpected error during prediction: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


# ── Model info (existing) ───────────────────────────────────────────


@router.get(
    "/model-info", response_model=ModelInfoResponse, tags=["Model Info"]
)
async def get_model_info(
    predictor: ModelPredictor = Depends(get_predictor),
):
    """Retrieves information about the currently loaded model."""
    artifacts_dir = os.path.join("artifacts")

    hyperparameters: Dict[str, Any] = {}
    params_path = os.path.join(
        artifacts_dir,
        f"{predictor.model_name.replace('_optimized', '')}_best_params.json",
    )
    if os.path.exists(params_path):
        with open(params_path, "r") as f:
            hyperparameters = json.load(f)

    metrics: Dict[str, Any] = {
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1_score": 0.0,
    }
    metrics_path = os.path.join(artifacts_dir, "evaluation_report.json")

    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            report = json.load(f)
            if predictor.model_name in report:
                metrics = report[predictor.model_name]
            elif any("tuned" in str(k) for k in report.keys()):
                for k, v in report.items():
                    if "tuned" in k:
                        if "tuned_metrics" in v:
                            metrics = v["tuned_metrics"]
                        else:
                            metrics = v
                        break
            else:
                if "accuracy" in report:
                    metrics = report

    return ModelInfoResponse(
        model_name=predictor.model_name,
        hyperparameters=hyperparameters,
        accuracy=metrics.get("accuracy", 0.0),
        precision=metrics.get("precision", 0.0),
        recall=metrics.get("recall", 0.0),
        f1_score=metrics.get(
            "f1_macro", metrics.get("f1_score", metrics.get("f1_weighted", 0.0))
        ),
    )


@router.get("/metrics", tags=["Model Info"])
async def get_metrics():
    """Retrieves the full evaluation metrics report."""
    metrics_path = os.path.join("artifacts", "evaluation_report.json")
    if not os.path.exists(metrics_path):
        opt_path = os.path.join("artifacts", "optimization_results.json")
        if os.path.exists(opt_path):
            with open(opt_path, "r") as f:
                return json.load(f)

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Metrics report not found.",
        )

    with open(metrics_path, "r") as f:
        return json.load(f)


# ── Live Market Data ─────────────────────────────────────────────────


@router.get(
    "/market/indian",
    response_model=LiveMarketResponse,
    tags=["Market Data"],
)
async def get_indian_market():
    """Returns latest Indian stock quotes with change percentage."""
    mgr = get_market_manager()
    quotes = mgr.get_indian_quotes()
    return LiveMarketResponse(
        count=len(quotes),
        quotes=[MarketQuoteResponse(**q) for q in quotes],
        source="cache" if mgr.scheduler.is_running else "live",
    )


@router.get(
    "/market/forex",
    response_model=LiveMarketResponse,
    tags=["Market Data"],
)
async def get_forex_market():
    """Returns latest Forex quotes with change percentage."""
    mgr = get_market_manager()
    quotes = mgr.get_forex_quotes()
    return LiveMarketResponse(
        count=len(quotes),
        quotes=[MarketQuoteResponse(**q) for q in quotes],
        source="cache" if mgr.scheduler.is_running else "live",
    )


@router.get(
    "/market/live",
    response_model=LiveMarketResponse,
    tags=["Market Data"],
)
async def get_live_market():
    """Returns combined Indian + Forex live market data."""
    mgr = get_market_manager()
    quotes = mgr.get_all_live()
    return LiveMarketResponse(
        count=len(quotes),
        quotes=[MarketQuoteResponse(**q) for q in quotes],
        source="cache" if mgr.scheduler.is_running else "live",
    )


# ── Live Predictions ─────────────────────────────────────────────────


def _risk_level(score: float) -> str:
    """Converts a numeric risk score to a human-readable level."""
    if score < 0.3:
        return "LOW"
    if score < 0.6:
        return "MEDIUM"
    return "HIGH"


@router.get(
    "/prediction/live",
    response_model=LivePredictionResponse,
    tags=["Prediction"],
)
async def get_live_predictions():
    """Returns live AI predictions for tracked symbols."""
    mgr = get_market_manager()
    raw = mgr.get_predictions()
    items = []
    for p in raw:
        items.append(
            LivePredictionItem(
                ticker=p.get("ticker", ""),
                action=p.get("action", "HOLD"),
                confidence=p.get("confidence", 0.0),
                probability=p.get("probability", {}),
                expected_return=p.get("expected_return", 0.0),
                risk_score=p.get("risk_score", 0.0),
                risk_level=_risk_level(p.get("risk_score", 0.5)),
                model_name=p.get("model_name", ""),
                close_price=p.get("close_price", 0.0),
            )
        )
    return LivePredictionResponse(count=len(items), predictions=items)


# ── Asset Search ─────────────────────────────────────────────────────


@router.get(
    "/market/search",
    response_model=AssetSearchResponse,
    tags=["Market Data"],
)
async def search_assets(q: str = Query(..., min_length=1, description="Search query")):
    """Searches assets by symbol or name."""
    mgr = get_market_manager()
    results = mgr.search_assets(q)
    return AssetSearchResponse(
        count=len(results),
        results=[AssetSearchItem(**r) for r in results],
    )


# ── Top Gainers / Losers / Most Active ───────────────────────────────


@router.get("/market/top-gainers", tags=["Market Data"])
async def top_gainers(n: int = Query(10, ge=1, le=50)):
    """Returns the top N gainers by change percentage."""
    mgr = get_market_manager()
    return mgr.get_top_gainers(n)


@router.get("/market/top-losers", tags=["Market Data"])
async def top_losers(n: int = Query(10, ge=1, le=50)):
    """Returns the top N losers by change percentage."""
    mgr = get_market_manager()
    return mgr.get_top_losers(n)


@router.get("/market/most-active", tags=["Market Data"])
async def most_active(n: int = Query(10, ge=1, le=50)):
    """Returns the top N most actively traded assets."""
    mgr = get_market_manager()
    return mgr.get_most_active(n)


# ── SSE Stream ───────────────────────────────────────────────────────


@router.get("/market/stream", tags=["Market Data"])
async def market_stream():
    """Server-Sent Events stream for real-time market updates."""
    mgr = get_market_manager()
    queue = mgr.connection_manager.connect()

    async def event_generator():
        try:
            while True:
                data = await queue.get()
                yield f"data: {data}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            mgr.connection_manager.disconnect(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── Manager Status ───────────────────────────────────────────────────


@router.get(
    "/market/status",
    response_model=MarketStatusResponse,
    tags=["Market Data"],
)
async def market_status():
    """Returns the current MarketManager diagnostic status."""
    mgr = get_market_manager()
    return MarketStatusResponse(**mgr.status())


# ── Module 14: Paper Trading Engine Endpoints ────────────────────────


@router.get(
    "/portfolio",
    response_model=PortfolioSummaryModel,
    tags=["Trading Engine"],
)
async def get_portfolio_endpoint(trader=Depends(get_paper_trader)):
    """Retrieves live virtual portfolio status (Cash, Value, Margin, P&L, Win Rate, ROI)."""
    return trader.get_portfolio()


@router.get(
    "/positions",
    tags=["Trading Engine"],
)
async def get_positions_endpoint(trader=Depends(get_paper_trader)):
    """Returns list of active open positions."""
    return trader.get_positions()


@router.get(
    "/trade-history",
    tags=["Trading Engine"],
)
async def get_trade_history_endpoint(trader=Depends(get_paper_trader)):
    """Returns complete trade history log."""
    return trader.get_trade_history()


@router.post(
    "/buy",
    response_model=OrderResponse,
    tags=["Trading Engine"],
)
async def buy_order_endpoint(
    req: OrderRequest, trader=Depends(get_paper_trader)
):
    """Executes a BUY paper trading order after risk checks."""
    try:
        return trader.buy(
            ticker=req.ticker,
            quantity=req.quantity,
            price=req.price,
            stop_loss=req.stop_loss,
            take_profit=req.take_profit,
            trailing_stop=req.trailing_stop,
        )
    except TradingException as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)
        )


@router.post(
    "/sell",
    response_model=OrderResponse,
    tags=["Trading Engine"],
)
async def sell_order_endpoint(
    req: OrderRequest, trader=Depends(get_paper_trader)
):
    """Executes a SELL paper trading order (supports partial selling)."""
    try:
        return trader.sell(
            ticker=req.ticker, quantity=req.quantity, price=req.price
        )
    except TradingException as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)
        )


@router.delete(
    "/positions/{ticker}",
    response_model=OrderResponse,
    tags=["Trading Engine"],
)
async def close_position_endpoint(
    ticker: str, price: float = Query(..., gt=0), trader=Depends(get_paper_trader)
):
    """Completely closes an active position for *ticker* at execution *price*."""
    try:
        return trader.close_position(ticker=ticker, current_price=price)
    except TradingException as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)
        )


@router.get(
    "/analytics",
    response_model=PortfolioAnalyticsModel,
    tags=["Trading Engine"],
)
async def get_analytics_endpoint(trader=Depends(get_paper_trader)):
    """Computes quantitative portfolio analytics (Sharpe, Sortino, Profit Factor, Max DD)."""
    return trader.get_analytics()


@router.get(
    "/risk",
    response_model=RiskStatusModel,
    tags=["Trading Engine"],
)
async def get_risk_endpoint(trader=Depends(get_paper_trader)):
    """Returns current pre-trade risk configuration and account status."""
    return trader.get_risk_status()


@router.get(
    "/trade-history/export",
    tags=["Trading Engine"],
)
async def export_trade_history_csv(trader=Depends(get_paper_trader)):
    """Exports all trade history records to a CSV file and downloads it."""
    csv_path = trader.export_history_csv("data/trade_history_export.csv")
    return FileResponse(
        csv_path, media_type="text/csv", filename="paper_trade_history.csv"
    )
