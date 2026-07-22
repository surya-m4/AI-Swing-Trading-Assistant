"""
PnL and performance analytics module for paper trading.

Provides functions and classes to calculate Realized P/L, Unrealized P/L,
Portfolio Equity, Win Rate, Average Return, and Maximum Drawdown.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class PnLCalculator:
    """Calculates profit/loss metrics and performance statistics for paper trading portfolios.

    Handles both Indian equity stocks and Forex currency pairs.
    """

    @staticmethod
    def calculate_unrealized_pnl(
        position: Dict[str, Any], current_price: float
    ) -> float:
        """Calculates unrealized P/L for a single open position.

        Args:
            position: Dictionary representing an open position containing keys:
                ``quantity``, ``entry_price``, ``position_type`` (LONG/SHORT).
            current_price: Mark-to-market current price of the asset.

        Returns:
            Unrealized profit/loss value (positive for profit, negative for loss).
        """
        qty = float(position.get("quantity", 0))
        entry_price = float(position.get("entry_price", 0.0))
        position_type = str(position.get("position_type", "LONG")).upper()

        if qty <= 0 or entry_price <= 0 or current_price <= 0:
            return 0.0

        if position_type == "SHORT":
            pnl = (entry_price - current_price) * qty
        else:
            pnl = (current_price - entry_price) * qty

        return round(pnl, 2)

    @staticmethod
    def calculate_total_unrealized_pnl(
        positions: List[Dict[str, Any]], current_prices: Dict[str, float]
    ) -> float:
        """Calculates total unrealized P/L across all open positions.

        Args:
            positions: List of open position dictionaries.
            current_prices: Dictionary mapping ticker symbol to current price.

        Returns:
            Sum of unrealized P/L across all positions.
        """
        total = 0.0
        for pos in positions:
            symbol = pos.get("symbol", "")
            current_price = current_prices.get(symbol, pos.get("entry_price", 0.0))
            total += PnLCalculator.calculate_unrealized_pnl(pos, current_price)
        return round(total, 2)

    @staticmethod
    def calculate_realized_pnl(trades: List[Dict[str, Any]]) -> float:
        """Calculates total realized P/L from closed trades.

        Args:
            trades: List of closed trade dictionaries, each containing ``realized_pnl``.

        Returns:
            Sum of realized profit/loss across closed trades.
        """
        total = 0.0
        for t in trades:
            total += float(t.get("realized_pnl", 0.0))
        return round(total, 2)

    @staticmethod
    def calculate_portfolio_value(
        cash: float,
        positions: List[Dict[str, Any]],
        current_prices: Dict[str, float],
    ) -> float:
        """Calculates total portfolio value (cash + mark-to-market positions).

        Args:
            cash: Available cash balance.
            positions: List of open position dictionaries.
            current_prices: Mapping of symbol to current price.

        Returns:
            Total portfolio valuation.
        """
        market_value = 0.0
        for pos in positions:
            symbol = pos.get("symbol", "")
            current_price = current_prices.get(symbol, pos.get("entry_price", 0.0))
            qty = float(pos.get("quantity", 0))
            market_value += qty * current_price

        return round(cash + market_value, 2)

    @staticmethod
    def calculate_win_rate(trades: List[Dict[str, Any]]) -> float:
        """Calculates win rate percentage from closed trades.

        Args:
            trades: List of closed trade dictionaries.

        Returns:
            Win rate percentage between 0.0 and 100.0.
        """
        if not trades:
            return 0.0

        wins = sum(1 for t in trades if float(t.get("realized_pnl", 0.0)) > 0)
        return round((wins / len(trades)) * 100.0, 2)

    @staticmethod
    def calculate_avg_return(trades: List[Dict[str, Any]]) -> float:
        """Calculates average percentage return per trade across closed trades.

        Args:
            trades: List of closed trade dictionaries containing ``return_pct``
                or ``realized_pnl`` and ``entry_cost``.

        Returns:
            Average return percentage.
        """
        if not trades:
            return 0.0

        returns: List[float] = []
        for t in trades:
            if "return_pct" in t:
                returns.append(float(t["return_pct"]))
            else:
                pnl = float(t.get("realized_pnl", 0.0))
                cost = float(t.get("entry_cost", 0.0))
                if cost > 0:
                    returns.append((pnl / cost) * 100.0)

        if not returns:
            return 0.0

        return round(float(np.mean(returns)), 2)

    @staticmethod
    def calculate_max_drawdown(equity_history: List[float]) -> float:
        """Calculates maximum drawdown percentage from historical equity series.

        Args:
            equity_history: Chronological list of portfolio equity values.

        Returns:
            Maximum drawdown percentage (positive number representing peak-to-trough decline).
        """
        if len(equity_history) < 2:
            return 0.0

        arr = np.array(equity_history, dtype=float)
        peaks = np.maximum.accumulate(arr)

        # Avoid division by zero if peak is 0 or negative
        peaks = np.where(peaks <= 0, 1e-8, peaks)

        drawdowns = (peaks - arr) / peaks
        max_dd = float(np.max(drawdowns)) * 100.0

        return round(max_dd, 2)
