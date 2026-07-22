"""
Server-Sent Events (SSE) connection manager for real-time market updates.

Tracks active SSE connections and provides broadcast capability so that
the scheduler can push refreshed market data and predictions to all
connected API consumers without polling.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List, Set

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages Server-Sent Events (SSE) connections for live market streaming.

    Each connected client is represented by an ``asyncio.Queue``.  The
    scheduler calls ``broadcast()`` after every refresh cycle to push
    the latest data to all listeners.

    Example::

        manager = ConnectionManager()

        # In a FastAPI SSE endpoint:
        queue = manager.connect()
        try:
            while True:
                data = await queue.get()
                yield f"data: {data}\\n\\n"
        finally:
            manager.disconnect(queue)

        # From the scheduler:
        manager.broadcast({"RELIANCE.NS": {"Close": 2450.0}})
    """

    def __init__(self) -> None:
        """Initialises an empty connection pool."""
        self._connections: Set[asyncio.Queue] = set()
        logger.info("ConnectionManager initialised.")

    @property
    def active_connections(self) -> int:
        """Number of currently connected clients."""
        return len(self._connections)

    def connect(self) -> asyncio.Queue:
        """Registers a new SSE client connection.

        Returns:
            An ``asyncio.Queue`` that will receive broadcast messages.
        """
        queue: asyncio.Queue = asyncio.Queue()
        self._connections.add(queue)
        logger.info(
            "SSE client connected. Active connections: %d",
            self.active_connections,
        )
        return queue

    def disconnect(self, queue: asyncio.Queue) -> None:
        """Removes a disconnected client.

        Args:
            queue: The queue previously returned by ``connect()``.
        """
        self._connections.discard(queue)
        logger.info(
            "SSE client disconnected. Active connections: %d",
            self.active_connections,
        )

    def broadcast(self, data: Dict[str, Any]) -> None:
        """Pushes a data payload to every connected client.

        This method is thread-safe and can be called from the scheduler
        background thread.  It schedules the async puts on each queue.

        Args:
            data: JSON-serialisable dictionary to broadcast.
        """
        if not self._connections:
            return

        message = json.dumps(data, default=str)

        dead_queues: List[asyncio.Queue] = []
        for queue in self._connections:
            try:
                queue.put_nowait(message)
            except asyncio.QueueFull:
                logger.warning("SSE queue full — dropping client.")
                dead_queues.append(queue)
            except Exception as exc:
                logger.warning("SSE broadcast error: %s", exc)
                dead_queues.append(queue)

        for q in dead_queues:
            self._connections.discard(q)

        logger.debug(
            "Broadcast to %d clients (%d dropped).",
            self.active_connections,
            len(dead_queues),
        )

    def broadcast_sync(self, data: Dict[str, Any]) -> None:
        """Synchronous broadcast wrapper for use in background threads.

        Tries to find a running event loop and schedules the puts.
        Falls back to ``put_nowait`` if no loop is running.

        Args:
            data: JSON-serialisable dictionary to broadcast.
        """
        self.broadcast(data)
