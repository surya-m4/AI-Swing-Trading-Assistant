"""
Dependencies for the FastAPI application.

Provides singleton instances of ``ModelPredictor``, ``MarketManager``, ``PaperBroker``,
and ``PaperTrader`` via dependency injection.
"""

import logging
from typing import Optional

from .predictor import ModelPredictor

logger = logging.getLogger(__name__)

# Global singleton instances
_predictor_instance: Optional[ModelPredictor] = None
_market_manager_instance = None
_paper_broker_instance = None
_paper_trader_instance = None


def get_predictor() -> ModelPredictor:
    """Dependency injection function to get the ModelPredictor instance."""
    global _predictor_instance
    if _predictor_instance is None:
        logger.info("Initializing ModelPredictor singleton.")
        _predictor_instance = ModelPredictor()
    return _predictor_instance


def get_market_manager():
    """Dependency injection function to get the MarketManager instance."""
    global _market_manager_instance
    if _market_manager_instance is None:
        logger.info("Initializing MarketManager singleton.")
        from src.market_data.market_manager import MarketManager

        _market_manager_instance = MarketManager(auto_start=False)
    return _market_manager_instance


def get_paper_broker():
    """Dependency injection function to get the PaperBroker instance."""
    global _paper_broker_instance
    if _paper_broker_instance is None:
        logger.info("Initializing PaperBroker singleton.")
        from src.paper_trading.paper_broker import PaperBroker

        _paper_broker_instance = PaperBroker()
    return _paper_broker_instance


def get_paper_trader():
    """Dependency injection function to get the PaperTrader instance."""
    global _paper_trader_instance
    if _paper_trader_instance is None:
        logger.info("Initializing PaperTrader singleton.")
        from src.trading.paper_trader import PaperTrader

        _paper_trader_instance = PaperTrader()
    return _paper_trader_instance
