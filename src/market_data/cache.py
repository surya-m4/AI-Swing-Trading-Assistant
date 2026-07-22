"""
Thread-safe TTL cache for market data.

Stores quotes, DataFrames, and prediction payloads with configurable
time-to-live.  Designed for the 30-second refresh cycle so that
multiple consumers (API endpoints, dashboard, scheduler) can read
cached data without triggering redundant API calls.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class MarketDataCache:
    """Thread-safe in-memory cache with per-key TTL expiry.

    Each entry is stored as a ``(value, expiry_timestamp)`` tuple.
    Reads that hit an expired key transparently return ``None`` and
    purge the stale entry.

    Attributes:
        default_ttl: Default time-to-live in seconds for new entries.

    Example::

        cache = MarketDataCache(default_ttl=30)
        cache.set("RELIANCE.NS", {"price": 2450.0})
        cache.get("RELIANCE.NS")  # → {"price": 2450.0}
        # … 31 seconds later …
        cache.get("RELIANCE.NS")  # → None
    """

    def __init__(self, default_ttl: int = 30) -> None:
        """Initialises the cache.

        Args:
            default_ttl: Default expiry in seconds (must be positive).

        Raises:
            ValueError: If *default_ttl* is not positive.
        """
        if default_ttl <= 0:
            raise ValueError("default_ttl must be a positive integer.")
        self.default_ttl = default_ttl
        self._store: Dict[str, tuple[Any, float]] = {}
        self._lock = threading.Lock()
        logger.info("MarketDataCache initialised (TTL=%ds).", default_ttl)

    # ── Core operations ──────────────────────────────────────────────

    def get(self, key: str) -> Optional[Any]:
        """Retrieves a cached value if it has not expired.

        Args:
            key: Cache key (typically a ticker symbol).

        Returns:
            The cached value, or ``None`` on miss / expiry.
        """
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            value, expiry = entry
            if time.monotonic() > expiry:
                del self._store[key]
                logger.debug("Cache expired for key '%s'.", key)
                return None
            return value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Stores a value with an optional custom TTL.

        Args:
            key: Cache key.
            value: Any serialisable object.
            ttl: Override TTL in seconds.  ``None`` uses ``default_ttl``.
        """
        effective_ttl = ttl if ttl is not None else self.default_ttl
        with self._lock:
            self._store[key] = (value, time.monotonic() + effective_ttl)
        logger.debug("Cached key '%s' (TTL=%ds).", key, effective_ttl)

    def invalidate(self, key: str) -> bool:
        """Removes a single key from the cache.

        Args:
            key: Cache key to remove.

        Returns:
            ``True`` if the key existed and was removed.
        """
        with self._lock:
            removed = self._store.pop(key, None)
        if removed is not None:
            logger.debug("Invalidated cache key '%s'.", key)
            return True
        return False

    def clear(self) -> None:
        """Removes all entries from the cache."""
        with self._lock:
            count = len(self._store)
            self._store.clear()
        logger.info("Cache cleared (%d entries removed).", count)

    # ── Bulk operations ──────────────────────────────────────────────

    def set_many(self, items: Dict[str, Any], ttl: Optional[int] = None) -> None:
        """Stores multiple key-value pairs atomically.

        Args:
            items: Mapping of keys to values.
            ttl: Override TTL in seconds.
        """
        effective_ttl = ttl if ttl is not None else self.default_ttl
        expiry = time.monotonic() + effective_ttl
        with self._lock:
            for key, value in items.items():
                self._store[key] = (value, expiry)
        logger.debug("Bulk-cached %d keys (TTL=%ds).", len(items), effective_ttl)

    def get_many(self, keys: list[str]) -> Dict[str, Any]:
        """Retrieves multiple values, skipping expired/missing keys.

        Args:
            keys: List of cache keys.

        Returns:
            Dictionary of ``{key: value}`` for non-expired hits.
        """
        result: Dict[str, Any] = {}
        now = time.monotonic()
        with self._lock:
            for key in keys:
                entry = self._store.get(key)
                if entry is not None:
                    value, expiry = entry
                    if now <= expiry:
                        result[key] = value
                    else:
                        del self._store[key]
        return result

    # ── Diagnostics ──────────────────────────────────────────────────

    @property
    def size(self) -> int:
        """Returns the number of entries (including potentially expired)."""
        with self._lock:
            return len(self._store)

    def keys(self) -> list[str]:
        """Returns all current cache keys (may include expired)."""
        with self._lock:
            return list(self._store.keys())

    def purge_expired(self) -> int:
        """Removes all expired entries and returns the count purged."""
        now = time.monotonic()
        purged = 0
        with self._lock:
            expired_keys = [
                k for k, (_, exp) in self._store.items() if now > exp
            ]
            for key in expired_keys:
                del self._store[key]
                purged += 1
        if purged:
            logger.info("Purged %d expired cache entries.", purged)
        return purged

    def __contains__(self, key: str) -> bool:
        """Checks if a key exists and is not expired."""
        return self.get(key) is not None

    def __len__(self) -> int:
        return self.size
