"""
Trade history manager with SQLite persistence and CSV export capabilities.

Logs every trade transaction with fields:
Trade ID, Ticker, Action (Buy/Sell), Price, Quantity, Timestamp, Broker, Status, and PnL.
"""

from __future__ import annotations

import csv
import logging
import os
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.trading.models import Trade

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = os.path.join("data", "paper_trading.db")


class TradeHistoryManager:
    """SQLite-backed manager for logging trade records and exporting to CSV.

    Attributes:
        db_path: File path to SQLite database.
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH) -> None:
        """Initialises the TradeHistoryManager.

        Args:
            db_path: Path to SQLite database file.
        """
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Returns SQLite connection."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Creates trade logs table if it does not exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS trade_history (
                    trade_id TEXT PRIMARY KEY,
                    ticker TEXT NOT NULL,
                    action TEXT NOT NULL,
                    price REAL NOT NULL,
                    quantity REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    broker TEXT NOT NULL DEFAULT 'Paper',
                    status TEXT NOT NULL DEFAULT 'EXECUTED',
                    pnl REAL NOT NULL DEFAULT 0.0
                )
                """
            )
            conn.commit()

    def record_trade(self, trade: Trade) -> None:
        """Inserts a trade record into the database.

        Args:
            trade: Trade dataclass instance.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO trade_history
                (trade_id, ticker, action, price, quantity, timestamp, broker, status, pnl)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trade.trade_id,
                    trade.ticker,
                    trade.action,
                    trade.price,
                    trade.quantity,
                    trade.timestamp,
                    trade.broker,
                    trade.status,
                    trade.pnl,
                ),
            )
            conn.commit()
            logger.info("Recorded trade %s for %s (%s Qty=%.2f @ ₹%.2f).", trade.trade_id, trade.ticker, trade.action, trade.quantity, trade.price)

    def get_all_trades(self) -> List[Trade]:
        """Retrieves all historical trade records.

        Returns:
            List of ``Trade`` dataclasses ordered by timestamp descending.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM trade_history ORDER BY timestamp DESC")
            rows = cursor.fetchall()
            return [
                Trade(
                    trade_id=r["trade_id"],
                    ticker=r["ticker"],
                    action=r["action"],
                    price=float(r["price"]),
                    quantity=float(r["quantity"]),
                    timestamp=r["timestamp"],
                    broker=r["broker"],
                    status=r["status"],
                    pnl=float(r["pnl"]),
                )
                for r in rows
            ]

    def export_to_csv(self, filepath: str) -> str:
        """Exports all trade history records to a CSV file.

        Args:
            filepath: Destination CSV file path.

        Returns:
            Absolute path to the created CSV file.
        """
        trades = self.get_all_trades()
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

        fieldnames = [
            "Trade ID",
            "Ticker",
            "Action",
            "Price",
            "Quantity",
            "Timestamp",
            "Broker",
            "Status",
            "PnL",
        ]

        with open(filepath, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(fieldnames)
            for t in trades:
                writer.writerow(
                    [
                        t.trade_id,
                        t.ticker,
                        t.action,
                        t.price,
                        t.quantity,
                        t.timestamp,
                        t.broker,
                        t.status,
                        t.pnl,
                    ]
                )

        logger.info("Exported %d trade records to CSV: %s", len(trades), filepath)
        return os.path.abspath(filepath)

    def clear_history(self) -> None:
        """Clears trade history records."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM trade_history")
            conn.commit()
