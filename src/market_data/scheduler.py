"""
Scheduler for periodic market data refresh.

Contains two classes:

* ``DataScheduler`` — lightweight generic scheduler (preserved for
  backward compatibility).
* ``LiveRefreshScheduler`` — production pipeline that orchestrates:
  fetch → cache → features → predict → broadcast on a configurable
  interval.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class DataScheduler:
    """Runs a callback at a fixed interval in a background thread.

    This is used to auto-refresh live market data and predictions.
    The scheduler is safe to start and stop from any thread.

    Attributes:
        interval: Seconds between successive invocations.
        callback: Zero-argument callable executed on each tick.
    """

    def __init__(
        self,
        callback: Callable[[], None],
        interval: int = 30,
    ) -> None:
        """Initialises the scheduler.

        Args:
            callback: Function invoked on every tick.
            interval: Refresh interval in seconds (default 30).

        Raises:
            ValueError: If *interval* is not positive.
        """
        if interval <= 0:
            raise ValueError("Interval must be a positive integer.")
        self.interval = interval
        self.callback = callback
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @property
    def is_running(self) -> bool:
        """Whether the scheduler loop is active."""
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """Starts the scheduler on a daemon thread.

        Does nothing if the scheduler is already running.
        """
        if self.is_running:
            logger.warning("Scheduler is already running.")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("Scheduler started (interval=%ds).", self.interval)

    def stop(self) -> None:
        """Signals the scheduler to stop and waits for the thread to finish."""
        if not self.is_running:
            return
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval + 5)
        logger.info("Scheduler stopped.")

    def _run(self) -> None:
        """Internal loop: execute callback, then sleep."""
        while not self._stop_event.is_set():
            try:
                self.callback()
            except Exception as exc:
                logger.error("Scheduler callback error: %s", exc)
            self._stop_event.wait(timeout=self.interval)


class LiveRefreshScheduler:
    """Production-grade scheduler that orchestrates the full live pipeline.

    Pipeline executed on each tick:
        1. Fetch all market data (Indian stocks + Forex + Indices +
           Commodities + Crypto).
        2. Store results in ``MarketDataCache``.
        3. Compute technical indicators via ``FeatureEngineeringPipeline``.
        4. Feed features into the trained model for live predictions.
        5. Broadcast updates to connected SSE clients.

    Attributes:
        interval: Refresh interval in seconds.
        is_running: Whether the scheduler loop is active.
    """

    def __init__(
        self,
        market_manager: Any,
        interval: Optional[int] = None,
    ) -> None:
        """Initialises the live refresh scheduler.

        Args:
            market_manager: ``MarketManager`` instance that owns the
                fetchers, cache, processor, and connection manager.
            interval: Refresh interval in seconds.  Defaults to the
                ``REFRESH_INTERVAL_SECONDS`` environment variable or 30.
        """
        self._manager = market_manager
        self.interval: int = interval or int(
            os.environ.get("REFRESH_INTERVAL_SECONDS", "30")
        )
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._tick_count: int = 0
        logger.info(
            "LiveRefreshScheduler created (interval=%ds).", self.interval
        )

    @property
    def is_running(self) -> bool:
        """Whether the scheduler loop is active."""
        return self._thread is not None and self._thread.is_alive()

    @property
    def tick_count(self) -> int:
        """Number of completed refresh cycles since start."""
        return self._tick_count

    def start(self) -> None:
        """Starts the scheduler on a daemon thread."""
        if self.is_running:
            logger.warning("LiveRefreshScheduler is already running.")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="LiveRefreshScheduler"
        )
        self._thread.start()
        logger.info("LiveRefreshScheduler started.")

    def stop(self) -> None:
        """Signals the scheduler to stop and waits for the thread."""
        if not self.is_running:
            return
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval + 5)
        logger.info(
            "LiveRefreshScheduler stopped after %d ticks.", self._tick_count
        )

    def _run(self) -> None:
        """Internal loop: execute the full pipeline on each tick."""
        while not self._stop_event.is_set():
            t0 = time.monotonic()
            try:
                self._execute_pipeline()
                self._tick_count += 1
                elapsed = time.monotonic() - t0
                logger.info(
                    "Refresh cycle #%d completed in %.2fs.",
                    self._tick_count,
                    elapsed,
                )
            except Exception as exc:
                logger.error("Refresh pipeline error: %s", exc, exc_info=True)

            self._stop_event.wait(timeout=self.interval)

    def _execute_pipeline(self) -> None:
        """Runs the fetch → cache → features → predict → broadcast pipeline."""
        mgr = self._manager

        # 1. Fetch market data
        indian_quotes = mgr.indian_fetcher.fetch_batch_quotes()
        forex_quotes = mgr.forex_fetcher.fetch_batch_quotes(
            mgr.forex_fetcher.pairs[:8]  # limit to reduce API load
        )

        # 2. Cache quotes
        all_quotes = indian_quotes + forex_quotes
        if mgr.cache:
            quote_map: Dict[str, Any] = {}
            for q in all_quotes:
                quote_map[f"quote:{q['Ticker']}"] = q
            mgr.cache.set_many(quote_map)

        # Store aggregate lists
        if mgr.cache:
            mgr.cache.set("all:indian_quotes", indian_quotes, ttl=self.interval + 5)
            mgr.cache.set("all:forex_quotes", forex_quotes, ttl=self.interval + 5)
            mgr.cache.set("all:live_quotes", all_quotes, ttl=self.interval + 5)

        # 3 & 4. Compute features and predictions for top tracked symbols
        predictions: List[Dict[str, Any]] = []
        tracked_symbols = [q["Ticker"] for q in all_quotes[:20]]  # top 20

        for symbol in tracked_symbols:
            try:
                pred = mgr.processor.predict(symbol)
                if pred and "error" not in pred:
                    predictions.append(pred)
            except Exception as exc:
                logger.debug("Prediction skipped for %s: %s", symbol, exc)

        if mgr.cache:
            mgr.cache.set(
                "all:predictions", predictions, ttl=self.interval + 5
            )

        # 5. Broadcast to SSE clients
        if mgr.connection_manager:
            broadcast_data = {
                "type": "market_update",
                "tick": self._tick_count + 1,
                "indian_count": len(indian_quotes),
                "forex_count": len(forex_quotes),
                "prediction_count": len(predictions),
                "quotes": all_quotes[:10],  # summary
                "predictions": predictions[:5],  # summary
            }
            mgr.connection_manager.broadcast_sync(broadcast_data)
