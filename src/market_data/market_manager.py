"""
Unified Market Manager — the single entry point for all live market
data operations.

Combines ``IndianMarketFetcher``, ``ForexMarketFetcher``,
``LiveDataProcessor``, ``MarketDataCache``, ``ConnectionManager``,
and ``LiveRefreshScheduler`` behind one cohesive API.

Dashboard pages, FastAPI endpoints, and the scheduler all interact
with market data exclusively through this class.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any, Dict, List, Optional

from src.market_data.assets_config import AssetCategory, AssetRegistry
from src.market_data.cache import MarketDataCache
from src.market_data.indian_market import IndianMarketFetcher
from src.market_data.forex_market import ForexMarketFetcher
from src.market_data.live_data import LiveDataProcessor
from src.market_data.websocket_manager import ConnectionManager
from src.market_data.scheduler import LiveRefreshScheduler

logger = logging.getLogger(__name__)

# Persistent storage path for favorites / watchlists
_WATCHLISTS_FILE = os.path.join("data", "watchlists.json")


class MarketManager:
    """Unified facade for all live market data operations.

    Provides methods for fetching quotes, searching assets, managing
    favorites / custom watchlists, computing top gainers / losers /
    most-active, and starting the auto-refresh pipeline.

    This class is designed to be used as a **singleton** — instantiated
    once at application startup and injected into FastAPI and Streamlit
    via dependency injection or session state.

    Attributes:
        registry: Asset registry for search and filtering.
        cache: Shared TTL cache for market data.
        indian_fetcher: Indian stock data fetcher.
        forex_fetcher: Forex data fetcher.
        processor: Live data processor (features + predictions).
        connection_manager: SSE connection manager.
        scheduler: Live refresh scheduler.
    """

    _instance: Optional["MarketManager"] = None
    _lock = threading.Lock()

    def __new__(cls, *args: Any, **kwargs: Any) -> "MarketManager":
        """Thread-safe singleton constructor."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        models_dir: str = "models",
        cache_ttl: int = 30,
        auto_start: bool = False,
    ) -> None:
        """Initialises the manager and its sub-components.

        Args:
            models_dir: Directory containing trained model ``.pkl`` files.
            cache_ttl: Default cache TTL in seconds.
            auto_start: If ``True``, starts the refresh scheduler
                immediately.
        """
        if hasattr(self, "_initialised"):
            return
        self._initialised = True

        # Core components
        self.registry = AssetRegistry()
        self.cache = MarketDataCache(default_ttl=cache_ttl)
        self.indian_fetcher = IndianMarketFetcher(cache=self.cache)
        self.forex_fetcher = ForexMarketFetcher(cache=self.cache)
        self.processor = LiveDataProcessor(models_dir=models_dir)
        self.connection_manager = ConnectionManager()

        # Scheduler
        self.scheduler = LiveRefreshScheduler(market_manager=self)

        # In-memory favorites and watchlists
        self._favorites: List[str] = []
        self._watchlists: Dict[str, List[str]] = {}
        self._load_watchlists()

        logger.info("MarketManager initialised.")

        if auto_start:
            self.start()

    # ── Lifecycle ────────────────────────────────────────────────────

    def start(self) -> None:
        """Starts the live refresh scheduler."""
        if not self.scheduler.is_running:
            self.scheduler.start()
            logger.info("MarketManager started live refresh.")

    def stop(self) -> None:
        """Stops the live refresh scheduler."""
        if self.scheduler.is_running:
            self.scheduler.stop()
            logger.info("MarketManager stopped live refresh.")

    # ── Quote retrieval ──────────────────────────────────────────────

    def get_indian_quotes(self) -> List[Dict[str, Any]]:
        """Returns cached Indian stock quotes.

        Falls back to a fresh batch fetch if the cache is empty.

        Returns:
            List of quote dictionaries.
        """
        cached = self.cache.get("all:indian_quotes")
        if cached is not None:
            return cached
        quotes = self.indian_fetcher.fetch_batch_quotes()
        self.cache.set("all:indian_quotes", quotes)
        return quotes

    def get_forex_quotes(self) -> List[Dict[str, Any]]:
        """Returns cached Forex quotes.

        Falls back to a fresh batch fetch if the cache is empty.

        Returns:
            List of quote dictionaries.
        """
        cached = self.cache.get("all:forex_quotes")
        if cached is not None:
            return cached
        quotes = self.forex_fetcher.fetch_batch_quotes(
            self.forex_fetcher.pairs[:8]
        )
        self.cache.set("all:forex_quotes", quotes)
        return quotes

    def get_all_live(self) -> List[Dict[str, Any]]:
        """Returns combined Indian + Forex live quotes.

        Returns:
            List of all quote dictionaries.
        """
        cached = self.cache.get("all:live_quotes")
        if cached is not None:
            return cached
        return self.get_indian_quotes() + self.get_forex_quotes()

    def get_quote(self, symbol: str) -> Dict[str, Any]:
        """Returns the latest quote for a single symbol.

        Args:
            symbol: Ticker symbol (e.g. ``RELIANCE.NS``).

        Returns:
            Quote dictionary or empty dict.
        """
        cached = self.cache.get(f"quote:{symbol}")
        if cached is not None:
            return cached

        if symbol.endswith(".NS") or symbol.startswith("^"):
            return self.indian_fetcher.fetch_latest_quote(symbol)
        return self.forex_fetcher.fetch_latest_quote(symbol)

    # ── Predictions ──────────────────────────────────────────────────

    def get_predictions(self) -> List[Dict[str, Any]]:
        """Returns cached live predictions for tracked symbols.

        Returns:
            List of prediction dictionaries.
        """
        cached = self.cache.get("all:predictions")
        return cached if cached is not None else []

    def get_prediction(self, symbol: str) -> Dict[str, Any]:
        """Returns a live prediction for a single symbol.

        Args:
            symbol: Ticker symbol.

        Returns:
            Prediction dictionary.
        """
        return self.processor.predict(symbol)

    # ── Sorting / Rankings ───────────────────────────────────────────

    def get_top_gainers(self, n: int = 10) -> List[Dict[str, Any]]:
        """Returns the top *n* gainers by Change_Pct.

        Args:
            n: Number of results.

        Returns:
            Sorted list of quote dictionaries (descending by change %).
        """
        quotes = self.get_all_live()
        sorted_q = sorted(
            quotes,
            key=lambda q: q.get("Change_Pct", 0),
            reverse=True,
        )
        return sorted_q[:n]

    def get_top_losers(self, n: int = 10) -> List[Dict[str, Any]]:
        """Returns the top *n* losers by Change_Pct.

        Args:
            n: Number of results.

        Returns:
            Sorted list of quote dictionaries (ascending by change %).
        """
        quotes = self.get_all_live()
        sorted_q = sorted(
            quotes,
            key=lambda q: q.get("Change_Pct", 0),
        )
        return sorted_q[:n]

    def get_most_active(self, n: int = 10) -> List[Dict[str, Any]]:
        """Returns the top *n* most active by volume.

        Args:
            n: Number of results.

        Returns:
            Sorted list of quote dictionaries (descending by volume).
        """
        quotes = self.get_all_live()
        sorted_q = sorted(
            quotes,
            key=lambda q: q.get("Volume", 0),
            reverse=True,
        )
        return sorted_q[:n]

    # ── Asset search ─────────────────────────────────────────────────

    def search_assets(self, query: str) -> List[Dict[str, str]]:
        """Searches assets by name or symbol.

        Args:
            query: Search string.

        Returns:
            List of ``{symbol, name, category}`` dicts.
        """
        results = self.registry.search(query)
        return [
            {
                "symbol": a.symbol,
                "name": a.name,
                "category": a.category.value,
                "exchange": a.exchange,
            }
            for a in results
        ]

    def get_assets_by_category(
        self, category: AssetCategory
    ) -> List[Dict[str, str]]:
        """Returns all assets in a category.

        Args:
            category: ``AssetCategory`` value.

        Returns:
            List of asset info dicts.
        """
        assets = self.registry.get_by_category(category)
        return [
            {
                "symbol": a.symbol,
                "name": a.name,
                "category": a.category.value,
                "exchange": a.exchange,
            }
            for a in assets
        ]

    # ── Favorites ────────────────────────────────────────────────────

    def get_favorites(self) -> List[str]:
        """Returns the list of favorited symbols.

        Returns:
            List of ticker symbols.
        """
        return list(self._favorites)

    def add_favorite(self, symbol: str) -> bool:
        """Adds a symbol to favorites.

        Args:
            symbol: Ticker symbol.

        Returns:
            ``True`` if added, ``False`` if already present.
        """
        if symbol not in self._favorites:
            self._favorites.append(symbol)
            self._save_watchlists()
            logger.info("Added %s to favorites.", symbol)
            return True
        return False

    def remove_favorite(self, symbol: str) -> bool:
        """Removes a symbol from favorites.

        Args:
            symbol: Ticker symbol.

        Returns:
            ``True`` if removed, ``False`` if not found.
        """
        if symbol in self._favorites:
            self._favorites.remove(symbol)
            self._save_watchlists()
            logger.info("Removed %s from favorites.", symbol)
            return True
        return False

    def get_favorite_quotes(self) -> List[Dict[str, Any]]:
        """Returns latest quotes for all favorited symbols.

        Returns:
            List of quote dictionaries.
        """
        return [
            self.get_quote(sym)
            for sym in self._favorites
            if self.get_quote(sym)
        ]

    # ── Watchlists ───────────────────────────────────────────────────

    def get_watchlists(self) -> Dict[str, List[str]]:
        """Returns all custom watchlists.

        Returns:
            Dict mapping watchlist name to list of symbols.
        """
        return dict(self._watchlists)

    def get_watchlist(self, name: str) -> List[str]:
        """Returns symbols in a named watchlist.

        Args:
            name: Watchlist name.

        Returns:
            List of ticker symbols (empty if not found).
        """
        return list(self._watchlists.get(name, []))

    def create_watchlist(self, name: str, symbols: List[str]) -> bool:
        """Creates or updates a custom watchlist.

        Args:
            name: Watchlist name.
            symbols: List of ticker symbols.

        Returns:
            ``True`` on success.
        """
        self._watchlists[name] = list(symbols)
        self._save_watchlists()
        logger.info("Created watchlist '%s' with %d symbols.", name, len(symbols))
        return True

    def delete_watchlist(self, name: str) -> bool:
        """Deletes a custom watchlist.

        Args:
            name: Watchlist name.

        Returns:
            ``True`` if deleted, ``False`` if not found.
        """
        if name in self._watchlists:
            del self._watchlists[name]
            self._save_watchlists()
            logger.info("Deleted watchlist '%s'.", name)
            return True
        return False

    def add_to_watchlist(self, name: str, symbol: str) -> bool:
        """Adds a symbol to an existing watchlist.

        Args:
            name: Watchlist name.
            symbol: Ticker symbol.

        Returns:
            ``True`` on success, ``False`` if watchlist not found.
        """
        if name not in self._watchlists:
            return False
        if symbol not in self._watchlists[name]:
            self._watchlists[name].append(symbol)
            self._save_watchlists()
        return True

    def remove_from_watchlist(self, name: str, symbol: str) -> bool:
        """Removes a symbol from an existing watchlist.

        Args:
            name: Watchlist name.
            symbol: Ticker symbol.

        Returns:
            ``True`` on success, ``False`` if watchlist or symbol not found.
        """
        if name not in self._watchlists:
            return False
        if symbol in self._watchlists[name]:
            self._watchlists[name].remove(symbol)
            self._save_watchlists()
            return True
        return False

    def get_watchlist_quotes(self, name: str) -> List[Dict[str, Any]]:
        """Returns latest quotes for all symbols in a watchlist.

        Args:
            name: Watchlist name.

        Returns:
            List of quote dictionaries.
        """
        symbols = self.get_watchlist(name)
        return [self.get_quote(s) for s in symbols if self.get_quote(s)]

    # ── Persistence ──────────────────────────────────────────────────

    def _load_watchlists(self) -> None:
        """Loads favorites and watchlists from disk."""
        if not os.path.exists(_WATCHLISTS_FILE):
            return
        try:
            with open(_WATCHLISTS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._favorites = data.get("favorites", [])
            self._watchlists = data.get("watchlists", {})
            logger.info(
                "Loaded %d favorites and %d watchlists.",
                len(self._favorites),
                len(self._watchlists),
            )
        except Exception as exc:
            logger.warning("Failed to load watchlists: %s", exc)

    def _save_watchlists(self) -> None:
        """Persists favorites and watchlists to disk."""
        os.makedirs(os.path.dirname(_WATCHLISTS_FILE), exist_ok=True)
        try:
            with open(_WATCHLISTS_FILE, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "favorites": self._favorites,
                        "watchlists": self._watchlists,
                    },
                    f,
                    indent=2,
                )
        except Exception as exc:
            logger.warning("Failed to save watchlists: %s", exc)

    # ── Status / Diagnostics ─────────────────────────────────────────

    def status(self) -> Dict[str, Any]:
        """Returns a diagnostic summary of the MarketManager state.

        Returns:
            Dictionary with component status information.
        """
        return {
            "total_assets": len(self.registry),
            "cache_size": self.cache.size,
            "scheduler_running": self.scheduler.is_running,
            "scheduler_ticks": self.scheduler.tick_count,
            "sse_connections": self.connection_manager.active_connections,
            "favorites_count": len(self._favorites),
            "watchlists_count": len(self._watchlists),
            "model_loaded": self.processor.model is not None,
            "model_name": self.processor.model_name,
        }
