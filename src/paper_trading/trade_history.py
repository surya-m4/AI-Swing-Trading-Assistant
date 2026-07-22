"""
SQLite trade history and persistent storage module for paper trading.

Manages tables for account balances, open positions, order logs, closed trades,
and periodic equity snapshots.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = os.path.join("data", "paper_trading.db")


class TradeHistoryDB:
    """SQLite database manager for persisting paper trading account data, trades, and orders.

    Attributes:
        db_path: Path to the SQLite database file.
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH) -> None:
        """Initialises the database manager and creates tables if they do not exist.

        Args:
            db_path: Absolute or relative path to the SQLite file.
        """
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Establishes and returns a SQLite connection."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Creates tables if they do not already exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Account state
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS account (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    initial_capital REAL NOT NULL,
                    cash REAL NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

            # Open positions
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS positions (
                    symbol TEXT PRIMARY KEY,
                    asset_class TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    entry_price REAL NOT NULL,
                    stop_loss REAL,
                    take_profit REAL,
                    position_type TEXT DEFAULT 'LONG',
                    created_at TEXT NOT NULL
                )
                """
            )

            # Orders log
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS orders (
                    order_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    action TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    price REAL NOT NULL,
                    status TEXT NOT NULL,
                    stop_loss REAL,
                    take_profit REAL,
                    created_at TEXT NOT NULL
                )
                """
            )

            # Closed trades history
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS trades (
                    trade_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    action TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL NOT NULL,
                    realized_pnl REAL NOT NULL,
                    return_pct REAL NOT NULL,
                    opened_at TEXT NOT NULL,
                    closed_at TEXT NOT NULL
                )
                """
            )

            # Equity snapshots
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS equity_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    portfolio_value REAL NOT NULL,
                    cash REAL NOT NULL,
                    unrealized_pnl REAL NOT NULL,
                    realized_pnl REAL NOT NULL
                )
                """
            )

            conn.commit()
            logger.info("SQLite Paper Trading DB initialized at %s.", self.db_path)

    # ── Account CRUD ─────────────────────────────────────────────────

    def save_account_state(self, initial_capital: float, cash: float) -> None:
        """Saves or updates account cash and capital state."""
        now = datetime.utcnow().isoformat() + "Z"
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM account")
            count = cursor.fetchone()[0]
            if count == 0:
                cursor.execute(
                    "INSERT INTO account (initial_capital, cash, updated_at) VALUES (?, ?, ?)",
                    (initial_capital, cash, now),
                )
            else:
                cursor.execute(
                    "UPDATE account SET initial_capital = ?, cash = ?, updated_at = ? WHERE id = 1",
                    (initial_capital, cash, now),
                )
            conn.commit()

    def get_account(self) -> Optional[Dict[str, Any]]:
        """Retrieves account state."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM account ORDER BY id DESC LIMIT 1")
            row = cursor.fetchone()
            if row:
                return dict(row)
        return None

    # ── Positions CRUD ───────────────────────────────────────────────

    def save_position(
        self,
        symbol: str,
        asset_class: str,
        quantity: int,
        entry_price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        position_type: str = "LONG",
    ) -> None:
        """Saves or replaces an open position."""
        now = datetime.utcnow().isoformat() + "Z"
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO positions
                (symbol, asset_class, quantity, entry_price, stop_loss, take_profit, position_type, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    symbol,
                    asset_class,
                    quantity,
                    entry_price,
                    stop_loss,
                    take_profit,
                    position_type,
                    now,
                ),
            )
            conn.commit()

    def remove_position(self, symbol: str) -> None:
        """Deletes an open position when closed."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM positions WHERE symbol = ?", (symbol,))
            conn.commit()

    def get_all_positions(self) -> List[Dict[str, Any]]:
        """Retrieves all active open positions."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM positions")
            return [dict(row) for row in cursor.fetchall()]

    # ── Orders & Trades CRUD ─────────────────────────────────────────

    def record_order(
        self,
        order_id: str,
        symbol: str,
        action: str,
        quantity: int,
        price: float,
        status: str = "FILLED",
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> None:
        """Records an order entry into the orders table."""
        now = datetime.utcnow().isoformat() + "Z"
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO orders (order_id, symbol, action, quantity, price, status, stop_loss, take_profit, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    order_id,
                    symbol,
                    action,
                    quantity,
                    price,
                    status,
                    stop_loss,
                    take_profit,
                    now,
                ),
            )
            conn.commit()

    def record_closed_trade(
        self,
        trade_id: str,
        symbol: str,
        action: str,
        quantity: int,
        entry_price: float,
        exit_price: float,
        realized_pnl: float,
        return_pct: float,
        opened_at: str,
    ) -> None:
        """Records a closed trade into the trades history table."""
        now = datetime.utcnow().isoformat() + "Z"
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO trades
                (trade_id, symbol, action, quantity, entry_price, exit_price, realized_pnl, return_pct, opened_at, closed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trade_id,
                    symbol,
                    action,
                    quantity,
                    entry_price,
                    exit_price,
                    realized_pnl,
                    return_pct,
                    opened_at,
                    now,
                ),
            )
            conn.commit()

    def get_all_trades(self) -> List[Dict[str, Any]]:
        """Retrieves all closed trades."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM trades ORDER BY closed_at DESC")
            return [dict(row) for row in cursor.fetchall()]

    # ── Equity Snapshots ─────────────────────────────────────────────

    def record_equity_snapshot(
        self,
        portfolio_value: float,
        cash: float,
        unrealized_pnl: float,
        realized_pnl: float,
    ) -> None:
        """Records a periodic equity curve snapshot."""
        now = datetime.utcnow().isoformat() + "Z"
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO equity_snapshots (timestamp, portfolio_value, cash, unrealized_pnl, realized_pnl)
                VALUES (?, ?, ?, ?, ?)
                """,
                (now, portfolio_value, cash, unrealized_pnl, realized_pnl),
            )
            conn.commit()

    def get_equity_curve(self) -> List[Dict[str, Any]]:
        """Retrieves equity history snapshots."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM equity_snapshots ORDER BY timestamp ASC"
            )
            return [dict(row) for row in cursor.fetchall()]

    def reset_database(self, initial_capital: float = 100000.0) -> None:
        """Wipes paper trading history and resets account capital."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM account")
            cursor.execute("DELETE FROM positions")
            cursor.execute("DELETE FROM orders")
            cursor.execute("DELETE FROM trades")
            cursor.execute("DELETE FROM equity_snapshots")
            conn.commit()
        self.save_account_state(initial_capital, initial_capital)
        logger.info("Paper Trading DB reset to initial capital ₹%.2f.", initial_capital)
