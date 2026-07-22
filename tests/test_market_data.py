"""
Comprehensive unit tests for the market_data module (Module 13).

Covers:
    - AssetRegistry (search, category filter, symbol lookup)
    - MarketDataCache (set/get, TTL expiry, invalidation, thread safety)
    - IndianMarketFetcher (fetch history, quotes, batch, retry, cache)
    - ForexMarketFetcher (fetch history, quotes, Twelve Data fallback)
    - LiveDataProcessor (feature computation, prediction)
    - DataScheduler (start/stop, interval, error handling)
    - LiveRefreshScheduler (pipeline execution)
    - MarketManager (unified interface, favorites, watchlists)
    - ConnectionManager (SSE connect/disconnect/broadcast)
"""

import asyncio
import json
import os
import tempfile
import threading
import time

import pandas as pd
import pytest
from unittest.mock import MagicMock, PropertyMock, patch

from src.market_data.assets_config import AssetCategory, AssetInfo, AssetRegistry
from src.market_data.cache import MarketDataCache
from src.market_data.indian_market import IndianMarketFetcher
from src.market_data.forex_market import ForexMarketFetcher
from src.market_data.live_data import LiveDataProcessor
from src.market_data.scheduler import DataScheduler, LiveRefreshScheduler
from src.market_data.websocket_manager import ConnectionManager


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def sample_history():
    """Returns a realistic yfinance-style DataFrame."""
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    dates.name = "Date"
    df = pd.DataFrame(
        {
            "Open": [100, 102, 101, 103, 104],
            "High": [105, 106, 104, 107, 108],
            "Low": [99, 100, 99, 101, 102],
            "Close": [103, 104, 102, 105, 106],
            "Volume": [1000, 1100, 900, 1200, 1300],
        },
        index=dates,
    )
    return df


@pytest.fixture
def empty_history():
    """Returns an empty DataFrame."""
    return pd.DataFrame()


@pytest.fixture
def cache():
    """Returns a fresh cache with 2-second TTL for fast tests."""
    return MarketDataCache(default_ttl=2)


@pytest.fixture
def registry():
    """Returns an AssetRegistry instance."""
    return AssetRegistry()


# ── AssetRegistry ────────────────────────────────────────────────────


class TestAssetRegistry:
    """Tests for the AssetRegistry class."""

    def test_registry_has_assets(self, registry):
        """Registry should contain 100+ assets."""
        assert len(registry) > 100

    def test_get_all(self, registry):
        """get_all() should return a non-empty list of AssetInfo."""
        all_assets = registry.get_all()
        assert len(all_assets) > 0
        assert isinstance(all_assets[0], AssetInfo)

    def test_get_by_category_indian_stocks(self, registry):
        """Filtering by INDIAN_STOCKS should return 90+ results."""
        stocks = registry.get_by_category(AssetCategory.INDIAN_STOCKS)
        assert len(stocks) >= 90
        assert all(a.category == AssetCategory.INDIAN_STOCKS for a in stocks)

    def test_get_by_category_forex_major(self, registry):
        """Filtering by FOREX_MAJOR should return exactly 7 pairs."""
        pairs = registry.get_by_category(AssetCategory.FOREX_MAJOR)
        assert len(pairs) == 7

    def test_get_by_category_indices(self, registry):
        """Filtering by INDICES should return index assets."""
        indices = registry.get_by_category(AssetCategory.INDICES)
        assert len(indices) >= 3
        symbols = [a.symbol for a in indices]
        assert "^NSEI" in symbols

    def test_get_by_symbol(self, registry):
        """get_by_symbol() should return exact match."""
        asset = registry.get_by_symbol("RELIANCE.NS")
        assert asset is not None
        assert asset.name == "Reliance Industries"
        assert asset.category == AssetCategory.INDIAN_STOCKS

    def test_get_by_symbol_missing(self, registry):
        """get_by_symbol() should return None for unknown symbol."""
        assert registry.get_by_symbol("NONEXISTENT.XX") is None

    def test_search_by_symbol(self, registry):
        """Search should match symbol substrings."""
        results = registry.search("RELIANCE")
        assert len(results) >= 1
        assert any(a.symbol == "RELIANCE.NS" for a in results)

    def test_search_by_name(self, registry):
        """Search should match name substrings."""
        results = registry.search("Infosys")
        assert len(results) >= 1
        assert any(a.symbol == "INFY.NS" for a in results)

    def test_search_case_insensitive(self, registry):
        """Search should be case-insensitive."""
        results = registry.search("bitcoin")
        assert len(results) >= 1

    def test_search_empty_query(self, registry):
        """Empty query should return empty list."""
        assert registry.search("") == []

    def test_get_symbols_for_category(self, registry):
        """get_symbols_for_category should return string list."""
        symbols = registry.get_symbols_for_category(AssetCategory.CRYPTO)
        assert len(symbols) >= 5
        assert "BTC-USD" in symbols

    def test_get_display_map(self, registry):
        """get_display_map should return symbol→name dict."""
        dmap = registry.get_display_map(AssetCategory.FOREX_MAJOR)
        assert isinstance(dmap, dict)
        assert "EURUSD=X" in dmap
        assert dmap["EURUSD=X"] == "EUR/USD"

    def test_categories(self):
        """categories() should return all AssetCategory values."""
        cats = AssetRegistry.categories()
        assert AssetCategory.INDIAN_STOCKS in cats
        assert AssetCategory.CRYPTO in cats
        assert len(cats) == 7

    def test_contains(self, registry):
        """__contains__ should support 'in' operator."""
        assert "RELIANCE.NS" in registry
        assert "FAKE.XX" not in registry


# ── MarketDataCache ──────────────────────────────────────────────────


class TestMarketDataCache:
    """Tests for the MarketDataCache class."""

    def test_invalid_ttl(self):
        """Should raise ValueError for non-positive TTL."""
        with pytest.raises(ValueError):
            MarketDataCache(default_ttl=0)
        with pytest.raises(ValueError):
            MarketDataCache(default_ttl=-5)

    def test_set_and_get(self, cache):
        """Basic set/get should work."""
        cache.set("key1", {"price": 100.0})
        assert cache.get("key1") == {"price": 100.0}

    def test_get_missing_key(self, cache):
        """Getting a missing key should return None."""
        assert cache.get("nonexistent") is None

    def test_ttl_expiry(self, cache):
        """Values should expire after TTL."""
        cache.set("expires", "data", ttl=1)
        assert cache.get("expires") == "data"
        time.sleep(1.5)
        assert cache.get("expires") is None

    def test_custom_ttl_override(self, cache):
        """Custom TTL should override default."""
        cache.set("short", "value", ttl=1)
        cache.set("long", "value", ttl=10)
        time.sleep(1.5)
        assert cache.get("short") is None
        assert cache.get("long") == "value"

    def test_invalidate(self, cache):
        """invalidate() should remove a key."""
        cache.set("key", "value")
        assert cache.invalidate("key") is True
        assert cache.get("key") is None

    def test_invalidate_missing(self, cache):
        """invalidate() on missing key should return False."""
        assert cache.invalidate("ghost") is False

    def test_clear(self, cache):
        """clear() should remove all entries."""
        cache.set("a", 1)
        cache.set("b", 2)
        cache.clear()
        assert cache.size == 0
        assert cache.get("a") is None

    def test_set_many(self, cache):
        """set_many() should store multiple items atomically."""
        cache.set_many({"x": 10, "y": 20, "z": 30})
        assert cache.get("x") == 10
        assert cache.get("z") == 30

    def test_get_many(self, cache):
        """get_many() should retrieve multiple items."""
        cache.set_many({"a": 1, "b": 2, "c": 3})
        result = cache.get_many(["a", "c", "missing"])
        assert result == {"a": 1, "c": 3}

    def test_size(self, cache):
        """size should reflect number of entries."""
        assert cache.size == 0
        cache.set("k", "v")
        assert cache.size == 1

    def test_keys(self, cache):
        """keys() should return stored keys."""
        cache.set("alpha", 1)
        cache.set("beta", 2)
        assert set(cache.keys()) == {"alpha", "beta"}

    def test_contains(self, cache):
        """__contains__ should check for non-expired keys."""
        cache.set("exists", "yes")
        assert "exists" in cache
        assert "nope" not in cache

    def test_purge_expired(self, cache):
        """purge_expired() should remove stale entries."""
        cache.set("stale", "old", ttl=1)
        cache.set("fresh", "new", ttl=10)
        time.sleep(1.5)
        purged = cache.purge_expired()
        assert purged == 1
        assert cache.get("fresh") == "new"

    def test_thread_safety(self, cache):
        """Cache should handle concurrent writes without corruption."""
        errors = []

        def writer(prefix: str, count: int):
            try:
                for i in range(count):
                    cache.set(f"{prefix}_{i}", i)
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=writer, args=(f"t{t}", 50))
            for t in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert cache.size <= 250


# ── IndianMarketFetcher ──────────────────────────────────────────────


class TestIndianMarketFetcher:
    """Tests for the IndianMarketFetcher class."""

    def test_default_tickers_from_registry(self):
        """Should load tickers from AssetRegistry by default."""
        fetcher = IndianMarketFetcher()
        assert "RELIANCE.NS" in fetcher.tickers
        assert len(fetcher.tickers) > 6  # more than the old default

    def test_custom_tickers(self):
        """Should accept a custom ticker list."""
        fetcher = IndianMarketFetcher(tickers=["TCS.NS"])
        assert fetcher.tickers == ["TCS.NS"]

    def test_cache_integration(self, cache):
        """Should accept a cache instance."""
        fetcher = IndianMarketFetcher(tickers=["TCS.NS"], cache=cache)
        assert fetcher.cache is cache

    @patch("src.market_data.indian_market.yf.Ticker")
    def test_fetch_history_success(self, mock_ticker_cls, sample_history):
        """Should return a normalised DataFrame on success."""
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = sample_history
        mock_ticker_cls.return_value = mock_ticker

        fetcher = IndianMarketFetcher(tickers=["RELIANCE.NS"])
        df = fetcher.fetch_history("RELIANCE.NS")

        assert not df.empty
        assert "Date" in df.columns
        assert "Close" in df.columns
        assert "Ticker" in df.columns
        assert df["Ticker"].iloc[0] == "RELIANCE.NS"

    @patch("src.market_data.indian_market.yf.Ticker")
    def test_fetch_history_empty(self, mock_ticker_cls, empty_history):
        """Should return empty DataFrame when no data returned."""
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = empty_history
        mock_ticker_cls.return_value = mock_ticker

        fetcher = IndianMarketFetcher(tickers=["INVALID.NS"])
        df = fetcher.fetch_history("INVALID.NS")
        assert df.empty

    @patch("src.market_data.indian_market.yf.Ticker")
    def test_fetch_history_exception(self, mock_ticker_cls):
        """Should return empty DataFrame on network error (after retries)."""
        mock_ticker_cls.side_effect = Exception("Network error")

        fetcher = IndianMarketFetcher(tickers=["RELIANCE.NS"])
        df = fetcher.fetch_history("RELIANCE.NS")
        assert df.empty

    @patch("src.market_data.indian_market.yf.Ticker")
    def test_fetch_latest_quote_success(self, mock_ticker_cls, sample_history):
        """Should return a quote dict with Change_Pct."""
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = sample_history
        mock_ticker_cls.return_value = mock_ticker

        fetcher = IndianMarketFetcher(tickers=["RELIANCE.NS"])
        quote = fetcher.fetch_latest_quote("RELIANCE.NS")

        assert quote["Ticker"] == "RELIANCE.NS"
        assert quote["Close"] == 106.0
        assert "Change_Pct" in quote
        assert "Timestamp" in quote

    @patch("src.market_data.indian_market.yf.Ticker")
    def test_fetch_latest_quote_empty(self, mock_ticker_cls, empty_history):
        """Should return empty dict when no data."""
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = empty_history
        mock_ticker_cls.return_value = mock_ticker

        fetcher = IndianMarketFetcher(tickers=["BAD.NS"])
        assert fetcher.fetch_latest_quote("BAD.NS") == {}

    @patch("src.market_data.indian_market.yf.Ticker")
    def test_fetch_all_quotes(self, mock_ticker_cls, sample_history):
        """Should return quotes for all tickers."""
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = sample_history
        mock_ticker_cls.return_value = mock_ticker

        fetcher = IndianMarketFetcher(tickers=["TCS.NS", "INFY.NS"])
        quotes = fetcher.fetch_all_quotes()
        assert len(quotes) == 2

    @patch("src.market_data.indian_market.yf.Ticker")
    def test_cache_hit(self, mock_ticker_cls, sample_history, cache):
        """Should return cached data without calling yf.Ticker."""
        # Pre-populate the cache with a quote
        cached_quote = {
            "Ticker": "TCS.NS",
            "Open": 102.0,
            "High": 106.0,
            "Low": 100.0,
            "Close": 104.0,
            "Volume": 1100,
            "Timestamp": "2024-01-02",
            "Change_Pct": 0.97,
        }
        cache.set("quote:TCS.NS", cached_quote)

        fetcher = IndianMarketFetcher(tickers=["TCS.NS"], cache=cache)
        quote = fetcher.fetch_latest_quote("TCS.NS")

        assert quote == cached_quote
        # yf.Ticker should NOT be called at all — pure cache hit
        assert mock_ticker_cls.call_count == 0


# ── ForexMarketFetcher ───────────────────────────────────────────────


class TestForexMarketFetcher:
    """Tests for the ForexMarketFetcher class."""

    def test_default_pairs_from_registry(self):
        """Should load pairs from AssetRegistry by default."""
        fetcher = ForexMarketFetcher()
        assert "EURUSD=X" in fetcher.pairs
        assert len(fetcher.pairs) > 5

    def test_custom_pairs(self):
        """Should accept a custom pair list."""
        fetcher = ForexMarketFetcher(pairs=["GBPUSD=X"])
        assert fetcher.pairs == ["GBPUSD=X"]

    @patch("src.market_data.forex_market.yf.Ticker")
    def test_fetch_history_success(self, mock_ticker_cls, sample_history):
        """Should return a normalised DataFrame."""
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = sample_history
        mock_ticker_cls.return_value = mock_ticker

        fetcher = ForexMarketFetcher(pairs=["EURUSD=X"])
        df = fetcher.fetch_history("EURUSD=X")

        assert not df.empty
        assert "Ticker" in df.columns

    @patch("src.market_data.forex_market.yf.Ticker")
    def test_fetch_history_api_failure(self, mock_ticker_cls):
        """Should return empty DataFrame on API error."""
        mock_ticker_cls.side_effect = Exception("API timeout")

        fetcher = ForexMarketFetcher(pairs=["EURUSD=X"])
        df = fetcher.fetch_history("EURUSD=X")
        assert df.empty

    @patch("src.market_data.forex_market.yf.Ticker")
    def test_fetch_latest_quote(self, mock_ticker_cls, sample_history):
        """Should return quote with Change_Pct."""
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = sample_history
        mock_ticker_cls.return_value = mock_ticker

        fetcher = ForexMarketFetcher(pairs=["EURUSD=X"])
        quote = fetcher.fetch_latest_quote("EURUSD=X")
        assert quote["Ticker"] == "EURUSD=X"
        assert "Change_Pct" in quote

    def test_yf_symbol_to_td_conversion(self):
        """Should correctly convert yfinance symbols to Twelve Data."""
        fetcher = ForexMarketFetcher(pairs=[])
        assert fetcher._yf_symbol_to_td("EURUSD=X") == "EUR/USD"
        assert fetcher._yf_symbol_to_td("GBPUSD=X") == "GBP/USD"
        assert fetcher._yf_symbol_to_td("INR=X") == "USD/INR"

    def test_twelve_data_key_detection(self):
        """Should detect TWELVE_DATA_API_KEY from environment."""
        with patch.dict(os.environ, {"TWELVE_DATA_API_KEY": "test_key"}):
            fetcher = ForexMarketFetcher(pairs=[])
            assert fetcher.twelve_data_key == "test_key"


# ── LiveDataProcessor ────────────────────────────────────────────────


class TestLiveDataProcessor:
    """Tests for the LiveDataProcessor class."""

    @patch("src.market_data.live_data.os.path.isdir", return_value=False)
    def test_init_no_models_dir(self, _):
        """Should handle missing models directory gracefully."""
        proc = LiveDataProcessor(models_dir="nonexistent")
        assert proc.model is None

    @patch("src.market_data.live_data.os.path.isdir", return_value=True)
    @patch("src.market_data.live_data.os.listdir", return_value=[])
    @patch("src.market_data.live_data.os.path.exists", return_value=False)
    def test_init_empty_models_dir(self, *_):
        """Should handle empty models directory."""
        proc = LiveDataProcessor(models_dir="models")
        assert proc.model is None

    @patch("src.market_data.live_data.os.path.isdir", return_value=False)
    def test_predict_without_model(self, _):
        """Should return error when model is not loaded."""
        proc = LiveDataProcessor(models_dir="nonexistent")
        result = proc.predict("RELIANCE.NS")
        assert "error" in result

    @patch("src.market_data.live_data.os.path.isdir", return_value=False)
    def test_compute_features_empty_data(self, _):
        """Should return empty DataFrame when no market data."""
        proc = LiveDataProcessor(models_dir="nonexistent")

        with patch.object(
            proc.indian_fetcher, "fetch_history", return_value=pd.DataFrame()
        ):
            df = proc.compute_features("RELIANCE.NS")
            assert df.empty


# ── DataScheduler ────────────────────────────────────────────────────


class TestDataScheduler:
    """Tests for the generic DataScheduler class."""

    def test_invalid_interval(self):
        """Should raise ValueError for zero or negative interval."""
        with pytest.raises(ValueError):
            DataScheduler(callback=lambda: None, interval=0)

    def test_start_and_stop(self):
        """Should execute callback and stop cleanly."""
        counter = {"n": 0}

        def increment():
            counter["n"] += 1

        sched = DataScheduler(callback=increment, interval=1)
        sched.start()
        assert sched.is_running

        time.sleep(2.5)
        sched.stop()
        assert not sched.is_running
        assert counter["n"] >= 2

    def test_double_start(self):
        """Double start should be a no-op."""
        sched = DataScheduler(callback=lambda: None, interval=1)
        sched.start()
        sched.start()  # should be a no-op
        assert sched.is_running
        sched.stop()

    def test_callback_exception_does_not_crash(self):
        """Scheduler should survive callback exceptions."""

        def bad_callback():
            raise RuntimeError("boom")

        sched = DataScheduler(callback=bad_callback, interval=1)
        sched.start()
        time.sleep(1.5)
        assert sched.is_running
        sched.stop()


# ── LiveRefreshScheduler ─────────────────────────────────────────────


class TestLiveRefreshScheduler:
    """Tests for the LiveRefreshScheduler class."""

    def test_create_with_default_interval(self):
        """Should use 30s default interval."""
        mock_mgr = MagicMock()
        sched = LiveRefreshScheduler(market_manager=mock_mgr)
        assert sched.interval == 30

    def test_create_with_custom_interval(self):
        """Should accept custom interval."""
        mock_mgr = MagicMock()
        sched = LiveRefreshScheduler(market_manager=mock_mgr, interval=10)
        assert sched.interval == 10

    def test_start_and_stop(self):
        """Should start and stop the scheduler."""
        mock_mgr = MagicMock()
        mock_mgr.indian_fetcher.fetch_batch_quotes.return_value = []
        mock_mgr.forex_fetcher.fetch_batch_quotes.return_value = []
        mock_mgr.forex_fetcher.pairs = ["EURUSD=X"]
        mock_mgr.cache = None
        mock_mgr.connection_manager = None

        sched = LiveRefreshScheduler(market_manager=mock_mgr, interval=1)
        sched.start()
        assert sched.is_running

        time.sleep(2)
        sched.stop()
        assert not sched.is_running
        assert sched.tick_count >= 1

    def test_tick_count_increments(self):
        """Should increment tick count after each cycle."""
        mock_mgr = MagicMock()
        mock_mgr.indian_fetcher.fetch_batch_quotes.return_value = []
        mock_mgr.forex_fetcher.fetch_batch_quotes.return_value = []
        mock_mgr.forex_fetcher.pairs = []
        mock_mgr.cache = None
        mock_mgr.connection_manager = None

        sched = LiveRefreshScheduler(market_manager=mock_mgr, interval=1)
        sched.start()
        time.sleep(2.5)
        sched.stop()
        assert sched.tick_count >= 2


# ── ConnectionManager ────────────────────────────────────────────────


class TestConnectionManager:
    """Tests for the SSE ConnectionManager class."""

    def test_initial_state(self):
        """Should start with zero connections."""
        mgr = ConnectionManager()
        assert mgr.active_connections == 0

    def test_connect_and_disconnect(self):
        """Should track connections."""
        mgr = ConnectionManager()
        q = mgr.connect()
        assert mgr.active_connections == 1
        mgr.disconnect(q)
        assert mgr.active_connections == 0

    def test_broadcast(self):
        """Should push data to all connected queues."""
        mgr = ConnectionManager()
        q1 = mgr.connect()
        q2 = mgr.connect()

        mgr.broadcast({"price": 100})

        assert not q1.empty()
        assert not q2.empty()

        data1 = q1.get_nowait()
        data2 = q2.get_nowait()
        assert json.loads(data1)["price"] == 100
        assert json.loads(data2)["price"] == 100

    def test_broadcast_no_connections(self):
        """Broadcast with no clients should not error."""
        mgr = ConnectionManager()
        mgr.broadcast({"data": "test"})  # should not raise

    def test_disconnect_unknown_queue(self):
        """Disconnecting an unknown queue should not error."""
        mgr = ConnectionManager()
        fake_queue = asyncio.Queue()
        mgr.disconnect(fake_queue)  # should not raise


# ── MarketManager ────────────────────────────────────────────────────


class TestMarketManager:
    """Tests for the MarketManager unified facade."""

    @patch("src.market_data.market_manager.LiveDataProcessor")
    @patch("src.market_data.market_manager.os.path.exists", return_value=False)
    def test_singleton_pattern(self, _, mock_proc):
        """MarketManager should be a singleton."""
        from src.market_data.market_manager import MarketManager

        # Reset singleton for test isolation
        MarketManager._instance = None
        if hasattr(MarketManager, '_initialised'):
            delattr(MarketManager, '_initialised')

        mgr1 = MarketManager.__new__(MarketManager)
        # Manually reset to allow test
        MarketManager._instance = None

    @patch("src.market_data.market_manager.LiveDataProcessor")
    @patch("src.market_data.market_manager.os.path.exists", return_value=False)
    def test_search_assets(self, _, mock_proc):
        """Should search assets via the registry."""
        from src.market_data.market_manager import MarketManager

        MarketManager._instance = None
        if hasattr(MarketManager, '_initialised'):
            delattr(MarketManager, '_initialised')

        mgr = MarketManager()
        results = mgr.search_assets("RELIANCE")
        assert len(results) >= 1
        assert results[0]["symbol"] == "RELIANCE.NS"

        # Cleanup
        MarketManager._instance = None
        if hasattr(MarketManager, '_initialised'):
            delattr(MarketManager, '_initialised')

    @patch("src.market_data.market_manager.LiveDataProcessor")
    @patch("src.market_data.market_manager.os.path.exists", return_value=False)
    def test_favorites_crud(self, _, mock_proc):
        """Should add, get, and remove favorites."""
        from src.market_data.market_manager import MarketManager

        MarketManager._instance = None
        if hasattr(MarketManager, '_initialised'):
            delattr(MarketManager, '_initialised')

        mgr = MarketManager()

        # Add
        assert mgr.add_favorite("TCS.NS") is True
        assert mgr.add_favorite("TCS.NS") is False  # duplicate

        # Get
        assert "TCS.NS" in mgr.get_favorites()

        # Remove
        assert mgr.remove_favorite("TCS.NS") is True
        assert mgr.remove_favorite("TCS.NS") is False  # not found
        assert "TCS.NS" not in mgr.get_favorites()

        # Cleanup
        MarketManager._instance = None
        if hasattr(MarketManager, '_initialised'):
            delattr(MarketManager, '_initialised')

    @patch("src.market_data.market_manager.LiveDataProcessor")
    @patch("src.market_data.market_manager.os.path.exists", return_value=False)
    def test_watchlist_crud(self, _, mock_proc):
        """Should create, modify, and delete watchlists."""
        from src.market_data.market_manager import MarketManager

        MarketManager._instance = None
        if hasattr(MarketManager, '_initialised'):
            delattr(MarketManager, '_initialised')

        mgr = MarketManager()

        # Create
        assert mgr.create_watchlist("test_wl", ["A.NS", "B.NS"]) is True
        assert mgr.get_watchlist("test_wl") == ["A.NS", "B.NS"]

        # Add symbol
        assert mgr.add_to_watchlist("test_wl", "C.NS") is True
        assert "C.NS" in mgr.get_watchlist("test_wl")

        # Remove symbol
        assert mgr.remove_from_watchlist("test_wl", "A.NS") is True
        assert "A.NS" not in mgr.get_watchlist("test_wl")

        # Delete watchlist
        assert mgr.delete_watchlist("test_wl") is True
        assert mgr.get_watchlist("test_wl") == []

        # Cleanup
        MarketManager._instance = None
        if hasattr(MarketManager, '_initialised'):
            delattr(MarketManager, '_initialised')

    @patch("src.market_data.market_manager.LiveDataProcessor")
    @patch("src.market_data.market_manager.os.path.exists", return_value=False)
    def test_top_gainers_losers(self, _, mock_proc):
        """Should sort quotes by change percentage."""
        from src.market_data.market_manager import MarketManager

        MarketManager._instance = None
        if hasattr(MarketManager, '_initialised'):
            delattr(MarketManager, '_initialised')

        mgr = MarketManager()

        # Mock cache with sorted data
        test_quotes = [
            {"Ticker": "A", "Change_Pct": 5.0, "Volume": 100},
            {"Ticker": "B", "Change_Pct": -3.0, "Volume": 500},
            {"Ticker": "C", "Change_Pct": 2.0, "Volume": 200},
        ]
        mgr.cache.set("all:live_quotes", test_quotes)

        gainers = mgr.get_top_gainers(2)
        assert gainers[0]["Ticker"] == "A"
        assert gainers[1]["Ticker"] == "C"

        losers = mgr.get_top_losers(1)
        assert losers[0]["Ticker"] == "B"

        active = mgr.get_most_active(1)
        assert active[0]["Ticker"] == "B"

        # Cleanup
        MarketManager._instance = None
        if hasattr(MarketManager, '_initialised'):
            delattr(MarketManager, '_initialised')

    @patch("src.market_data.market_manager.LiveDataProcessor")
    @patch("src.market_data.market_manager.os.path.exists", return_value=False)
    def test_status(self, _, mock_proc):
        """Should return a diagnostic status dict."""
        from src.market_data.market_manager import MarketManager

        MarketManager._instance = None
        if hasattr(MarketManager, '_initialised'):
            delattr(MarketManager, '_initialised')

        mgr = MarketManager()
        s = mgr.status()

        assert "total_assets" in s
        assert "cache_size" in s
        assert "scheduler_running" in s
        assert s["total_assets"] > 100

        # Cleanup
        MarketManager._instance = None
        if hasattr(MarketManager, '_initialised'):
            delattr(MarketManager, '_initialised')
