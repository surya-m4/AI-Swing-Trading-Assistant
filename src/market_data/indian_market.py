"""
Indian stock market data fetcher using yfinance.

Supports dynamic ticker lists from ``AssetRegistry``, per-request
caching via ``MarketDataCache``, exponential-backoff retry logic,
and efficient batch downloads.
"""

import logging
import time
from typing import Any, Dict, List, Optional

import pandas as pd
import yfinance as yf

from src.market_data.assets_config import AssetCategory, AssetRegistry
from src.market_data.cache import MarketDataCache

logger = logging.getLogger(__name__)

# Retry configuration
_MAX_RETRIES: int = 3
_BACKOFF_BASE: float = 1.0  # seconds

# Default tickers for Indian equity markets (backward-compatible)
DEFAULT_INDIAN_TICKERS: List[str] = [
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "SBIN.NS",
]


def _retry_on_failure(func):
    """Decorator that retries *func* up to ``_MAX_RETRIES`` times with
    exponential backoff on any ``Exception``.

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


class IndianMarketFetcher:
    """Fetches OHLCV data for Indian stocks from Yahoo Finance.

    Supports dynamic ticker lists loaded from ``AssetRegistry`` and
    transparent caching to avoid redundant API calls.

    Attributes:
        tickers: List of NSE ticker symbols.
        cache: Optional ``MarketDataCache`` instance.
    """

    def __init__(
        self,
        tickers: Optional[List[str]] = None,
        cache: Optional[MarketDataCache] = None,
    ) -> None:
        """Initialises the fetcher with a list of tickers.

        Args:
            tickers: NSE-suffixed ticker symbols.  If ``None``, loads
                all Indian stocks from ``AssetRegistry``.
            cache: Shared cache instance.  ``None`` disables caching.
        """
        if tickers is not None:
            self.tickers = tickers
        else:
            try:
                registry = AssetRegistry()
                self.tickers = registry.get_symbols_for_category(
                    AssetCategory.INDIAN_STOCKS
                )
            except Exception:
                self.tickers = list(DEFAULT_INDIAN_TICKERS)

        self.cache = cache

    # ── Single-ticker helpers ────────────────────────────────────────

    def fetch_history(
        self,
        ticker: str,
        period: str = "3mo",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Downloads historical OHLCV data for one ticker.

        Args:
            ticker: Yahoo Finance symbol (e.g. ``RELIANCE.NS``).
            period: Look-back period accepted by yfinance (``1d``, ``5d``,
                ``1mo``, ``3mo``, ``6mo``, ``1y``, ``2y``, ``5y``, ``10y``,
                ``ytd``, ``max``).
            interval: Bar size (``1m``, ``2m``, ``5m``, ``15m``, ``30m``,
                ``60m``, ``90m``, ``1h``, ``1d``, ``5d``, ``1wk``, ``1mo``,
                ``3mo``).

        Returns:
            DataFrame with columns ``[Date, Open, High, Low, Close, Volume]``.
            Empty DataFrame on failure.
        """
        cache_key = f"history:{ticker}:{period}:{interval}"
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                logger.debug("Cache hit for %s history.", ticker)
                return cached

        df = self._download_history(ticker, period, interval)
        if df is not None and not df.empty and self.cache:
            self.cache.set(cache_key, df, ttl=120)  # history cached 2 min
        return df if df is not None else pd.DataFrame()

    @_retry_on_failure
    def _download_history(
        self, ticker: str, period: str, interval: str
    ) -> pd.DataFrame:
        """Internal download with retry logic.

        Args:
            ticker: Yahoo Finance symbol.
            period: Look-back period.
            interval: Bar size.

        Returns:
            Normalised DataFrame or empty DataFrame.
        """
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)

        if df.empty:
            logger.warning("No data returned for %s.", ticker)
            return pd.DataFrame()

        df = df.reset_index()
        if "Datetime" in df.columns:
            df = df.rename(columns={"Datetime": "Date"})
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])

        df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
        df["Ticker"] = ticker
        logger.info("Fetched %d rows for %s.", len(df), ticker)
        return df

    def fetch_latest_quote(self, ticker: str) -> Dict[str, Any]:
        """Returns the most recent OHLCV bar for *ticker*.

        Args:
            ticker: Yahoo Finance symbol.

        Returns:
            Dictionary with ``Open``, ``High``, ``Low``, ``Close``,
            ``Volume``, ``Timestamp``, and ``Ticker`` keys.
            Empty dict on failure.
        """
        cache_key = f"quote:{ticker}"
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

        quote = self._download_latest_quote(ticker)
        if quote and self.cache:
            self.cache.set(cache_key, quote)
        return quote if quote else {}

    @_retry_on_failure
    def _download_latest_quote(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Internal quote download with retry.

        Args:
            ticker: Yahoo Finance symbol.

        Returns:
            Quote dict or ``None`` on failure.
        """
        stock = yf.Ticker(ticker)
        hist = stock.history(period="5d", interval="1d")

        if hist.empty:
            logger.warning("No recent data for %s.", ticker)
            return None

        latest = hist.iloc[-1]
        prev_close = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else None
        close = float(latest["Close"])
        change_pct = (
            round(((close - prev_close) / prev_close) * 100, 2)
            if prev_close
            else 0.0
        )

        return {
            "Ticker": ticker,
            "Open": float(latest["Open"]),
            "High": float(latest["High"]),
            "Low": float(latest["Low"]),
            "Close": close,
            "Volume": int(latest["Volume"]),
            "Timestamp": str(hist.index[-1]),
            "Change_Pct": change_pct,
        }

    # ── Batch operations ─────────────────────────────────────────────

    def fetch_all_quotes(self) -> List[Dict[str, Any]]:
        """Fetches the latest quote for every ticker in ``self.tickers``.

        Returns:
            List of quote dictionaries.
        """
        quotes: List[Dict[str, Any]] = []
        for ticker in self.tickers:
            quote = self.fetch_latest_quote(ticker)
            if quote:
                quotes.append(quote)
        return quotes

    def fetch_batch_quotes(
        self, tickers: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Bulk-downloads latest quotes using ``yf.download`` for speed.

        This is significantly faster than calling ``fetch_latest_quote``
        in a loop for large ticker lists.

        Args:
            tickers: Symbols to download.  Defaults to ``self.tickers``.

        Returns:
            List of quote dictionaries with change-percent included.
        """
        symbols = tickers or self.tickers
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
            logger.error("Batch download failed: %s", exc)
            return self.fetch_all_quotes()  # fall back to sequential

        quotes: List[Dict[str, Any]] = []
        for ticker in symbols:
            try:
                if len(symbols) == 1:
                    ticker_df = df
                else:
                    ticker_df = df[ticker]

                if ticker_df.empty or ticker_df.dropna(how="all").empty:
                    continue

                ticker_df = ticker_df.dropna()
                if len(ticker_df) < 1:
                    continue

                latest = ticker_df.iloc[-1]
                prev_close = (
                    float(ticker_df["Close"].iloc[-2])
                    if len(ticker_df) >= 2
                    else None
                )
                close = float(latest["Close"])
                change_pct = (
                    round(((close - prev_close) / prev_close) * 100, 2)
                    if prev_close
                    else 0.0
                )
                quotes.append(
                    {
                        "Ticker": ticker,
                        "Open": float(latest["Open"]),
                        "High": float(latest["High"]),
                        "Low": float(latest["Low"]),
                        "Close": close,
                        "Volume": int(latest["Volume"]),
                        "Timestamp": str(ticker_df.index[-1]),
                        "Change_Pct": change_pct,
                    }
                )
            except Exception as exc:
                logger.warning("Skipping %s in batch: %s", ticker, exc)

        if quotes and self.cache:
            for q in quotes:
                self.cache.set(f"quote:{q['Ticker']}", q)

        logger.info(
            "Batch-fetched %d/%d Indian quotes.", len(quotes), len(symbols)
        )
        return quotes
