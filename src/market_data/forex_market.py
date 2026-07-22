"""
Forex market data fetcher with dual-provider support.

Primary provider: **yfinance** (free, no key required).
Secondary provider: **Twelve Data API** (activated when the environment
variable ``TWELVE_DATA_API_KEY`` is set).

Supports dynamic pair lists from ``AssetRegistry``, caching, and
exponential-backoff retry logic.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
import yfinance as yf

from src.market_data.assets_config import AssetCategory, AssetRegistry
from src.market_data.cache import MarketDataCache

logger = logging.getLogger(__name__)

# Retry configuration
_MAX_RETRIES: int = 3
_BACKOFF_BASE: float = 1.0

# Default forex pairs (backward-compatible)
DEFAULT_FOREX_PAIRS: List[str] = [
    "EURUSD=X",
    "GBPUSD=X",
    "USDJPY=X",
    "AUDUSD=X",
    "USDCHF=X",
    "USDCAD=X",
    "NZDUSD=X",
    "INR=X",
]

# Twelve Data base URL
_TWELVE_DATA_BASE = "https://api.twelvedata.com"


def _retry_on_failure(func):
    """Decorator that retries *func* up to ``_MAX_RETRIES`` times with
    exponential backoff.

    Args:
        func: Callable to wrap.

    Returns:
        Wrapped callable with retry logic.
    """

    def wrapper(*args, **kwargs):
        last_exc: Optional[Exception] = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                last_exc = exc
                wait = _BACKOFF_BASE * (2 ** (attempt - 1))
                logger.warning(
                    "Attempt %d/%d failed for %s: %s — retrying in %.1fs",
                    attempt,
                    _MAX_RETRIES,
                    func.__name__,
                    exc,
                    wait,
                )
                time.sleep(wait)
        logger.error(
            "All %d retries exhausted for %s: %s",
            _MAX_RETRIES,
            func.__name__,
            last_exc,
        )
        return None

    return wrapper


class ForexMarketFetcher:
    """Fetches OHLCV data for Forex currency pairs.

    Uses yfinance by default and falls back to Twelve Data when the
    ``TWELVE_DATA_API_KEY`` environment variable is configured.

    Attributes:
        pairs: List of Yahoo Finance forex symbols.
        cache: Optional ``MarketDataCache`` instance.
        twelve_data_key: Twelve Data API key (``None`` if not configured).
    """

    def __init__(
        self,
        pairs: Optional[List[str]] = None,
        cache: Optional[MarketDataCache] = None,
    ) -> None:
        """Initialises the fetcher.

        Args:
            pairs: Forex symbols.  If ``None``, loads all forex pairs
                from ``AssetRegistry`` (Major + Minor + INR).
            cache: Shared cache instance.  ``None`` disables caching.
        """
        if pairs is not None:
            self.pairs = pairs
        else:
            try:
                registry = AssetRegistry()
                self.pairs = (
                    registry.get_symbols_for_category(AssetCategory.FOREX_MAJOR)
                    + registry.get_symbols_for_category(AssetCategory.FOREX_MINOR)
                    + registry.get_symbols_for_category(AssetCategory.FOREX_INR)
                )
            except Exception:
                self.pairs = list(DEFAULT_FOREX_PAIRS)

        self.cache = cache
        self.twelve_data_key: Optional[str] = os.environ.get(
            "TWELVE_DATA_API_KEY"
        )
        if self.twelve_data_key:
            logger.info("Twelve Data API key detected — dual-provider mode.")

    # ── Provider selection ───────────────────────────────────────────

    def _yf_symbol_to_td(self, yf_symbol: str) -> str:
        """Converts a yfinance forex symbol to a Twelve Data symbol.

        Example: ``EURUSD=X`` → ``EUR/USD``, ``INR=X`` → ``USD/INR``.

        Args:
            yf_symbol: Yahoo Finance forex symbol.

        Returns:
            Twelve Data-compatible symbol string.
        """
        clean = yf_symbol.replace("=X", "")
        if clean == "INR":
            return "USD/INR"
        if len(clean) == 6:
            return f"{clean[:3]}/{clean[3:]}"
        return clean

    # ── yfinance provider ────────────────────────────────────────────

    def fetch_history(
        self,
        pair: str,
        period: str = "1mo",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Downloads historical OHLCV data for one forex pair.

        Args:
            pair: Yahoo Finance forex symbol (e.g. ``EURUSD=X``).
            period: Look-back period.
            interval: Bar size.

        Returns:
            DataFrame with columns ``[Date, Open, High, Low, Close, Volume]``.
            Empty DataFrame on failure.
        """
        cache_key = f"history:{pair}:{period}:{interval}"
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                logger.debug("Cache hit for %s history.", pair)
                return cached

        df = self._download_history_yf(pair, period, interval)
        if df is not None and not df.empty and self.cache:
            self.cache.set(cache_key, df, ttl=120)
        return df if df is not None else pd.DataFrame()

    @_retry_on_failure
    def _download_history_yf(
        self, pair: str, period: str, interval: str
    ) -> pd.DataFrame:
        """Internal yfinance download with retry.

        Args:
            pair: Yahoo Finance forex symbol.
            period: Look-back period.
            interval: Bar size.

        Returns:
            Normalised DataFrame or empty DataFrame.
        """
        ticker = yf.Ticker(pair)
        df = ticker.history(period=period, interval=interval)

        if df.empty:
            logger.warning("No data returned for %s.", pair)
            return pd.DataFrame()

        df = df.reset_index()
        if "Datetime" in df.columns:
            df = df.rename(columns={"Datetime": "Date"})
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])

        df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
        df["Ticker"] = pair
        logger.info("Fetched %d rows for %s.", len(df), pair)
        return df

    # ── Quote methods ────────────────────────────────────────────────

    def fetch_latest_quote(self, pair: str) -> Dict[str, Any]:
        """Returns the most recent OHLCV bar for *pair*.

        Tries Twelve Data first (if configured), then yfinance.

        Args:
            pair: Yahoo Finance forex symbol.

        Returns:
            Quote dictionary or empty dict on failure.
        """
        cache_key = f"quote:{pair}"
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

        quote: Optional[Dict[str, Any]] = None

        if self.twelve_data_key:
            quote = self._fetch_quote_twelve_data(pair)

        if not quote:
            quote = self._download_latest_quote_yf(pair)

        if quote and self.cache:
            self.cache.set(cache_key, quote)

        return quote if quote else {}

    @_retry_on_failure
    def _download_latest_quote_yf(self, pair: str) -> Optional[Dict[str, Any]]:
        """Fetches the latest quote from yfinance with retry.

        Args:
            pair: Yahoo Finance forex symbol.

        Returns:
            Quote dict or ``None``.
        """
        ticker = yf.Ticker(pair)
        hist = ticker.history(period="5d", interval="1d")

        if hist.empty:
            logger.warning("No recent data for %s.", pair)
            return None

        latest = hist.iloc[-1]
        prev_close = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else None
        close = float(latest["Close"])
        change_pct = (
            round(((close - prev_close) / prev_close) * 100, 4)
            if prev_close
            else 0.0
        )

        return {
            "Ticker": pair,
            "Open": float(latest["Open"]),
            "High": float(latest["High"]),
            "Low": float(latest["Low"]),
            "Close": close,
            "Volume": int(latest.get("Volume", 0)),
            "Timestamp": str(hist.index[-1]),
            "Change_Pct": change_pct,
        }

    # ── Twelve Data provider ─────────────────────────────────────────

    def _fetch_quote_twelve_data(self, yf_symbol: str) -> Optional[Dict[str, Any]]:
        """Fetches a single real-time quote from the Twelve Data API.

        Args:
            yf_symbol: Yahoo Finance forex symbol (converted internally).

        Returns:
            Quote dict or ``None`` on failure.
        """
        td_symbol = self._yf_symbol_to_td(yf_symbol)
        url = f"{_TWELVE_DATA_BASE}/quote"
        params = {"symbol": td_symbol, "apikey": self.twelve_data_key}

        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            if "code" in data and data["code"] != 200:
                logger.warning(
                    "Twelve Data error for %s: %s",
                    td_symbol,
                    data.get("message", "unknown"),
                )
                return None

            close = float(data.get("close", 0))
            prev_close = float(data.get("previous_close", 0))
            change_pct = (
                round(((close - prev_close) / prev_close) * 100, 4)
                if prev_close
                else float(data.get("percent_change", 0))
            )

            return {
                "Ticker": yf_symbol,
                "Open": float(data.get("open", 0)),
                "High": float(data.get("high", 0)),
                "Low": float(data.get("low", 0)),
                "Close": close,
                "Volume": int(data.get("volume", 0)),
                "Timestamp": data.get("datetime", ""),
                "Change_Pct": change_pct,
            }
        except Exception as exc:
            logger.warning(
                "Twelve Data fetch failed for %s: %s", td_symbol, exc
            )
            return None

    def fetch_history_twelve_data(
        self,
        yf_symbol: str,
        interval: str = "1day",
        outputsize: int = 60,
    ) -> pd.DataFrame:
        """Fetches historical data from Twelve Data API.

        Args:
            yf_symbol: Yahoo Finance forex symbol (converted internally).
            interval: Bar interval (``1min``, ``5min``, ``1h``, ``1day``).
            outputsize: Number of bars to retrieve (max 5000).

        Returns:
            DataFrame with ``[Date, Open, High, Low, Close, Volume]``.
            Empty DataFrame if API key is absent or on failure.
        """
        if not self.twelve_data_key:
            logger.warning("Twelve Data API key not set.")
            return pd.DataFrame()

        td_symbol = self._yf_symbol_to_td(yf_symbol)
        url = f"{_TWELVE_DATA_BASE}/time_series"
        params = {
            "symbol": td_symbol,
            "interval": interval,
            "outputsize": outputsize,
            "apikey": self.twelve_data_key,
        }

        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            if "values" not in data:
                logger.warning("No values in Twelve Data response for %s.", td_symbol)
                return pd.DataFrame()

            records = data["values"]
            df = pd.DataFrame(records)
            df = df.rename(
                columns={
                    "datetime": "Date",
                    "open": "Open",
                    "high": "High",
                    "low": "Low",
                    "close": "Close",
                    "volume": "Volume",
                }
            )
            for col in ["Open", "High", "Low", "Close"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0).astype(int)
            df["Date"] = pd.to_datetime(df["Date"])
            df["Ticker"] = yf_symbol
            df = df.sort_values("Date").reset_index(drop=True)

            logger.info(
                "Twelve Data: fetched %d rows for %s.", len(df), td_symbol
            )
            return df

        except Exception as exc:
            logger.error(
                "Twelve Data history fetch failed for %s: %s", td_symbol, exc
            )
            return pd.DataFrame()

    # ── Batch operations ─────────────────────────────────────────────

    def fetch_all_quotes(self) -> List[Dict[str, Any]]:
        """Fetches the latest quote for every pair.

        Returns:
            List of quote dictionaries.
        """
        quotes: List[Dict[str, Any]] = []
        for pair in self.pairs:
            quote = self.fetch_latest_quote(pair)
            if quote:
                quotes.append(quote)
        return quotes

    def fetch_batch_quotes(
        self, pairs: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Bulk-downloads latest quotes via ``yf.download``.

        Falls back to sequential fetching if bulk download fails.

        Args:
            pairs: Symbols to download.  Defaults to ``self.pairs``.

        Returns:
            List of quote dictionaries.
        """
        symbols = pairs or self.pairs
        if not symbols:
            return []

        try:
            df = yf.download(
                symbols,
                period="5d",
                interval="1d",
                group_by="ticker",
                threads=True,
            )
        except Exception as exc:
            logger.error("Forex batch download failed: %s", exc)
            return self.fetch_all_quotes()

        quotes: List[Dict[str, Any]] = []
        for pair in symbols:
            try:
                if len(symbols) == 1:
                    pair_df = df
                else:
                    pair_df = df[pair]

                if pair_df.empty or pair_df.dropna(how="all").empty:
                    continue

                pair_df = pair_df.dropna()
                if len(pair_df) < 1:
                    continue

                latest = pair_df.iloc[-1]
                prev_close = (
                    float(pair_df["Close"].iloc[-2])
                    if len(pair_df) >= 2
                    else None
                )
                close = float(latest["Close"])
                change_pct = (
                    round(((close - prev_close) / prev_close) * 100, 4)
                    if prev_close
                    else 0.0
                )

                quotes.append(
                    {
                        "Ticker": pair,
                        "Open": float(latest["Open"]),
                        "High": float(latest["High"]),
                        "Low": float(latest["Low"]),
                        "Close": close,
                        "Volume": int(latest.get("Volume", 0)),
                        "Timestamp": str(pair_df.index[-1]),
                        "Change_Pct": change_pct,
                    }
                )
            except Exception as exc:
                logger.warning("Skipping %s in batch: %s", pair, exc)

        if quotes and self.cache:
            for q in quotes:
                self.cache.set(f"quote:{q['Ticker']}", q)

        logger.info(
            "Batch-fetched %d/%d forex quotes.", len(quotes), len(symbols)
        )
        return quotes
