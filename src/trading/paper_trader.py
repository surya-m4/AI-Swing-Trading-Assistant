"""
Main PaperTrader facade for Module 14: Paper Trading Engine.

Combines Portfolio, PositionManager, OrderManager, RiskChecker, TradeHistoryManager,
and Prediction Engine integration into a single cohesive facade.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from src.market_data.live_data import LiveDataProcessor
from src.trading.exceptions import (
    InsufficientFundsException,
    InvalidOrderException,
    PositionNotFoundException,
    RiskLimitExceededException,
    TradingException,
)
from src.trading.models import (
    OrderRequest,
    OrderResponse,
    PortfolioAnalyticsModel,
    PortfolioSummaryModel,
    Position,
    RiskConfig,
    RiskStatusModel,
    Trade,
)
from src.trading.order_manager import OrderManager
from src.trading.portfolio import DEFAULT_STARTING_CAPITAL, Portfolio
from src.trading.position_manager import PositionManager
from src.trading.risk_checks import RiskChecker
from src.trading.trade_history import TradeHistoryManager

logger = logging.getLogger(__name__)


class PaperTrader:
    """Production-grade PaperTrader façade.

    Attributes:
        portfolio: Portfolio state manager.
        position_manager: Active open positions manager.
        trade_history: TradeHistoryManager instance.
        risk_checker: Risk validation engine.
        order_manager: Order execution engine.
        processor: AI LiveDataProcessor instance.
    """

    _instance: Optional["PaperTrader"] = None

    def __init__(
        self,
        initial_capital: float = DEFAULT_STARTING_CAPITAL,
        db_path: Optional[str] = None,
        models_dir: str = "models",
        risk_config: Optional[RiskConfig] = None,
    ) -> None:
        """Initialises the PaperTrader.

        Args:
            initial_capital: Virtual starting capital balance in ₹ (default ₹10,00,000).
            db_path: Optional custom path to SQLite database file.
            models_dir: Directory containing trained model files.
            risk_config: Optional RiskConfig instance.
        """
        self.portfolio = Portfolio(initial_capital=initial_capital)
        self.position_manager = PositionManager()
        self.trade_history = (
            TradeHistoryManager(db_path=db_path)
            if db_path
            else TradeHistoryManager()
        )
        self.risk_checker = RiskChecker(config=risk_config)
        self.order_manager = OrderManager(
            portfolio=self.portfolio,
            position_manager=self.position_manager,
            trade_history=self.trade_history,
            risk_checker=self.risk_checker,
        )
        self.processor = LiveDataProcessor(models_dir=models_dir)
        logger.info("PaperTrader façade initialized with initial capital ₹%s.", f"{initial_capital:,.2f}")

    # ── Trading Actions ──────────────────────────────────────────────

    def buy(
        self,
        ticker: str,
        quantity: float,
        price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        trailing_stop: Optional[float] = None,
    ) -> OrderResponse:
        """Executes a BUY paper order.

        Args:
            ticker: Asset symbol.
            quantity: Quantity to buy.
            price: Execution price.
            stop_loss: Optional Stop Loss price level.
            take_profit: Optional Take Profit price level.
            trailing_stop: Optional trailing stop offset.

        Returns:
            OrderResponse Pydantic model.
        """
        return self.order_manager.execute_buy(
            ticker=ticker,
            quantity=quantity,
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            trailing_stop=trailing_stop,
        )

    def sell(
        self,
        ticker: str,
        quantity: float,
        price: float,
    ) -> OrderResponse:
        """Executes a SELL paper order (partial or full).

        Args:
            ticker: Asset symbol.
            quantity: Quantity to sell.
            price: Exit execution price.

        Returns:
            OrderResponse Pydantic model.
        """
        return self.order_manager.execute_sell(
            ticker=ticker, quantity=quantity, price=price
        )

    def close_position(self, ticker: str, current_price: float) -> OrderResponse:
        """Completely closes an active open position for *ticker* at *current_price*."""
        return self.order_manager.close_position(ticker=ticker, current_price=current_price)

    # ── Portfolio & Analytics Queries ────────────────────────────────

    def get_portfolio(
        self, current_prices: Optional[Dict[str, float]] = None
    ) -> PortfolioSummaryModel:
        """Returns portfolio summary metrics."""
        if current_prices:
            self.position_manager.update_market_prices(current_prices)
        positions = self.position_manager.get_all_positions()
        trades = self.trade_history.get_all_trades()
        return self.portfolio.get_summary(positions=positions, trades=trades)

    def get_positions(self) -> List[Dict[str, Any]]:
        """Returns list of active open positions as dictionaries."""
        return [p.to_dict() for p in self.position_manager.get_all_positions()]

    def get_trade_history() -> List[Dict[str, Any]]:
        """Returns list of all logged transactions."""
        return [t.to_dict() for t in self.trade_history.get_all_trades()]

    def get_analytics(
        self, current_prices: Optional[Dict[str, float]] = None
    ) -> PortfolioAnalyticsModel:
        """Computes quantitative portfolio analytics (Sharpe, Sortino, Profit Factor)."""
        if current_prices:
            self.position_manager.update_market_prices(current_prices)
        positions = self.position_manager.get_all_positions()
        trades = self.trade_history.get_all_trades()
        return self.portfolio.calculate_analytics(positions=positions, trades=trades)

    def get_risk_status(self) -> RiskStatusModel:
        """Returns current risk status and limits."""
        return RiskStatusModel(
            max_position_size_pct=self.risk_checker.config.max_position_size_pct,
            cash_available=self.portfolio.cash,
            is_market_open=self.risk_checker.market_open_mock,
            risk_limits_ok=True,
            status_message="Risk controls active.",
        )

    def export_history_csv(self, filepath: str = "data/trade_history.csv") -> str:
        """Exports trade history to a CSV file."""
        return self.trade_history.export_to_csv(filepath)

    # ── AI Model Prediction Integration ──────────────────────────────

    def get_ai_recommendation(
        self, ticker: str, period: str = "3mo"
    ) -> Dict[str, Any]:
        """Integrates with LiveDataProcessor to generate AI prediction & sizing.

        Args:
            ticker: Ticker symbol.
            period: Lookback period for feature calculations.

        Returns:
            Recommendation payload dictionary.
        """
        pred = self.processor.predict(ticker, period=period)
        if "error" in pred:
            return pred

        action = str(pred.get("action", "HOLD")).upper()
        confidence = float(pred.get("confidence", 0.5))
        close_price = float(pred.get("close_price", 0.0))

        # Position sizing based on 2% risk limit and capital
        port_summary = self.get_portfolio({ticker: close_price})
        risk_budget = port_summary.portfolio_value * 0.02 * min(1.0, max(0.5, confidence))

        sl_distance = close_price * 0.025
        qty = max(1, int(risk_budget / sl_distance)) if sl_distance > 0 else 1
        qty = min(qty, int(self.portfolio.cash / close_price)) if close_price > 0 else 0

        stop_loss = round(close_price - sl_distance, 2) if action == "BUY" else round(close_price + sl_distance, 2)
        take_profit = round(close_price + (sl_distance * 2.0), 2) if action == "BUY" else round(close_price - (sl_distance * 2.0), 2)

        return {
            "ticker": ticker,
            "action": action,
            "recommended_quantity": qty,
            "confidence_score": round(confidence, 4),
            "close_price": close_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "risk_pct": 2.0,
            "expected_return": pred.get("expected_return", 0.0),
            "model_name": pred.get("model_name", "AI Model"),
        }

    def reset_account(self, initial_capital: float = DEFAULT_STARTING_CAPITAL) -> None:
        """Resets virtual portfolio capital, clears open positions and trade history."""
        self.portfolio.reset(initial_capital)
        self.position_manager.positions.clear()
        self.trade_history.clear_history()
        logger.info("PaperTrader account reset to ₹%s.", f"{initial_capital:,.2f}")
