"""
PaperBroker facade — unified entry point for paper trading and AI integration.

Connects the Portfolio state, OrderManager, RiskManager, and LiveDataProcessor
to provide automated AI recommendations, paper trade execution, and performance tracking.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.market_data.live_data import LiveDataProcessor
from src.paper_trading.order_manager import OrderManager
from src.paper_trading.portfolio import Portfolio
from src.paper_trading.risk_manager import RiskManager
from src.paper_trading.trade_history import TradeHistoryDB

logger = logging.getLogger(__name__)


class PaperBroker:
    """Unified Paper Broker facade for trading operations and AI model integration.

    Attributes:
        portfolio: Portfolio state manager.
        risk_manager: Risk rules engine.
        order_manager: Order execution engine.
        processor: AI LiveDataProcessor instance.
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        db_path: Optional[str] = None,
        models_dir: str = "models",
    ) -> None:
        """Initialises the PaperBroker.

        Args:
            initial_capital: Virtual account starting capital.
            db_path: Optional path to custom SQLite DB file.
            models_dir: Directory containing trained model files.
        """
        db = TradeHistoryDB(db_path) if db_path else TradeHistoryDB()
        self.portfolio = Portfolio(initial_capital=initial_capital, db=db)
        self.risk_manager = RiskManager()
        self.order_manager = OrderManager(
            portfolio=self.portfolio, risk_manager=self.risk_manager
        )
        self.processor = LiveDataProcessor(models_dir=models_dir)
        logger.info("PaperBroker initialised.")

    # ── AI Model Recommendation Integration ──────────────────────────

    def get_ai_recommendation(
        self, symbol: str, period: str = "3mo"
    ) -> Dict[str, Any]:
        """Fetches AI prediction for *symbol* and computes risk-managed trade sizing.

        Payload includes:
            - ``symbol``
            - ``action`` (BUY / SELL / HOLD)
            - ``recommended_size`` (Quantity count)
            - ``risk_pct`` (% of portfolio equity)
            - ``stop_loss`` (Price level)
            - ``take_profit`` (Price level)
            - ``confidence_score`` (Model confidence)
            - ``expected_return`` (Estimated % gain)
            - ``risk_level`` (LOW / MEDIUM / HIGH)
            - ``current_price``

        Args:
            symbol: Asset ticker symbol.
            period: Lookback period for features.

        Returns:
            Recommendation payload dictionary.
        """
        # Call AI prediction engine
        prediction = self.processor.predict(symbol, period=period)
        if "error" in prediction:
            logger.warning("AI prediction error for %s: %s", symbol, prediction["error"])
            return prediction

        action = str(prediction.get("action", "HOLD")).upper()
        confidence = float(prediction.get("confidence", 0.5))
        close_price = float(prediction.get("close_price", 0.0))

        # Get summary for current equity & cash
        summary = self.portfolio.get_summary({symbol: close_price})
        equity = summary["portfolio_value"]
        cash = summary["cash"]

        # Generate recommendation sizing and SL/TP
        rec = self.risk_manager.generate_trade_recommendation(
            symbol=symbol,
            entry_price=close_price,
            action=action,
            confidence=confidence,
            portfolio_equity=equity,
            available_cash=cash,
        )

        rec["expected_return"] = prediction.get("expected_return", 0.0)
        rec["model_name"] = prediction.get("model_name", "AI Model")

        risk_score = prediction.get("risk_score", 0.5)
        if risk_score < 0.3:
            rec["risk_level"] = "LOW"
        elif risk_score < 0.6:
            rec["risk_level"] = "MEDIUM"
        else:
            rec["risk_level"] = "HIGH"

        return rec

    def execute_ai_trade(
        self, symbol: str, min_confidence: float = 0.60
    ) -> Dict[str, Any]:
        """Automatically fetches AI recommendation and places trade if signal is strong.

        Args:
            symbol: Ticker symbol.
            min_confidence: Minimum confidence threshold to trigger order.

        Returns:
            Order execution payload or status dictionary.
        """
        rec = self.get_ai_recommendation(symbol)
        if "error" in rec:
            return rec

        action = rec.get("action", "HOLD").upper()
        confidence = rec.get("confidence_score", 0.0)
        qty = int(rec.get("recommended_size", 0))
        price = float(rec.get("entry_price", 0.0))
        sl = rec.get("stop_loss")
        tp = rec.get("take_profit")

        if action not in ("BUY", "SELL"):
            return {
                "status": "SKIPPED",
                "reason": f"AI signal is {action}.",
                "recommendation": rec,
            }

        if confidence < min_confidence:
            return {
                "status": "SKIPPED",
                "reason": f"Confidence {confidence:.2f} below threshold {min_confidence:.2f}.",
                "recommendation": rec,
            }

        if qty <= 0:
            return {
                "status": "SKIPPED",
                "reason": "Calculated position size is zero.",
                "recommendation": rec,
            }

        # Place the order
        order_res = self.order_manager.place_order(
            symbol=symbol,
            action=action,
            quantity=qty,
            price=price,
            stop_loss=sl,
            take_profit=tp,
        )
        order_res["recommendation"] = rec
        return order_res

    # ── High-Level Paper Trading API ─────────────────────────────────

    def buy(
        self,
        symbol: str,
        quantity: int,
        price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Executes a manual BUY paper order.

        Args:
            symbol: Ticker symbol.
            quantity: Quantity of shares/units.
            price: Execution price.
            stop_loss: Optional Stop Loss level.
            take_profit: Optional Take Profit level.

        Returns:
            Order payload dict.
        """
        return self.order_manager.place_order(
            symbol=symbol,
            action="BUY",
            quantity=quantity,
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

    def sell(self, symbol: str, price: float) -> Dict[str, Any]:
        """Executes a manual SELL / CLOSE position paper order.

        Args:
            symbol: Ticker symbol.
            price: Exit execution price.

        Returns:
            Trade summary dict.
        """
        return self.order_manager.close_position(symbol=symbol, exit_price=price)

    def close_position(self, symbol: str, price: float) -> Dict[str, Any]:
        """Alias for closing an active position."""
        return self.sell(symbol, price)

    def modify_position(
        self,
        symbol: str,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Modifies Stop Loss and Take Profit for an open position."""
        return self.order_manager.modify_sl_tp(
            symbol=symbol, stop_loss=stop_loss, take_profit=take_profit
        )

    def get_portfolio_summary(
        self, current_prices: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """Retrieves full portfolio performance and valuation summary."""
        return self.portfolio.get_summary(current_prices)

    def reset_account(self, initial_capital: float = 100000.0) -> None:
        """Resets virtual account to new capital and clears paper trading data."""
        self.portfolio.reset(initial_capital)
