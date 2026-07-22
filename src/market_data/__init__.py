"""
Live Market Data Integration Package.

Provides live market data fetching, feature computation,
and prediction generation for Indian stocks, Forex pairs,
indices, commodities, and crypto.

Components
----------
- ``AssetRegistry``        — Dynamic asset catalogue with search.
- ``MarketDataCache``      — Thread-safe TTL cache.
- ``IndianMarketFetcher``  — NSE equity data via yfinance.
- ``ForexMarketFetcher``   — Forex data via yfinance / Twelve Data.
- ``LiveDataProcessor``    — Feature engineering + model inference.
- ``DataScheduler``        — Generic interval scheduler.
- ``LiveRefreshScheduler`` — Full pipeline scheduler.
- ``MarketManager``        — Unified facade for all operations.
- ``ConnectionManager``    — SSE connection manager.
"""

from .assets_config import AssetCategory, AssetInfo, AssetRegistry
from .cache import MarketDataCache
from .indian_market import IndianMarketFetcher
from .forex_market import ForexMarketFetcher
from .live_data import LiveDataProcessor
from .scheduler import DataScheduler, LiveRefreshScheduler
from .market_manager import MarketManager
from .websocket_manager import ConnectionManager

__all__ = [
    "AssetCategory",
    "AssetInfo",
    "AssetRegistry",
    "MarketDataCache",
    "IndianMarketFetcher",
    "ForexMarketFetcher",
    "LiveDataProcessor",
    "DataScheduler",
    "LiveRefreshScheduler",
    "MarketManager",
    "ConnectionManager",
]
