"""
Portfolio state management module for paper trading.

Maintains in-memory portfolio state, open positions, cash balance,
and integrates with SQLite persistence and PnL metrics calculation.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.paper_trading.pnl import PnLCalculator
from src.paper_trading.trade_history import TradeHistoryDB

logger = logging.getLogger(__name__)


class Portfolio:
    """Manages paper trading account portfolio state, cash, positions, and analytics.

    Attributes:
        initial_capital: Starting virtual capital balance.
        cash: Current available cash.
        db: Persistence database manager.
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        db: Optional[TradeHistoryDB] = None,
    ) -> None:
        """Initialises the portfolio.

        Args:
            initial_capital: Virtual starting capital.
            db: Optional ``TradeHistoryDB`` instance. Creates default if None.
        """
        self.db = db if db is not None else TradeHistoryDB()
        self.initial_capital = initial_capital
        self.cash = initial_capital

        # Load persisted state if exists
        self._load_from_db()

    def _load_from_db(self) -> None:
        """Loads account balance and open positions from the database."""
        account = self.db.get_account()
        if account:
            self.initial_capital = float(account.get("initial_capital", self.initial_capital))
            self.cash = float(account.get("cash", self.cash))
        else:
            self.db.save_account_state(self.initial_capital, self.cash)

    def reset(self, initial_capital: float = 100000.0) -> None:
        """Resets the portfolio to a new initial capital balance.

        Args:
            initial_capital: New virtual starting capital balance.
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.db.reset_database(initial_capital)
        logger.info("Portfolio reset with capital ₹%.2f.", initial_capital)

    # ── Positions & State Access ─────────────────────────────────────

    def get_open_positions(self) -> List[Dict[str, Any]]:
        """Returns all currently open positions from DB."""
        return self.db.get_all_positions()

    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Retrieves an open position for a given symbol.

        Args:
            symbol: Ticker symbol.

        Returns:
            Position dictionary or None if not found.
        """
        positions = self.get_open_positions()
        for p in positions:
            if p.get("symbol") == symbol:
                return p
        return None

    def get_closed_trades(self) -> List[Dict[str, Any]]:
        """Returns all closed trade records from DB."""
        return self.db.get_all_trades()

    # ── Cash & Position Updates ──────────────────────────────────────

    def update_cash(self, delta: float) -> float:
        """Updates available cash balance.

        Args:
            delta: Amount to add (positive) or deduct (negative).

        Returns:
            New cash balance.
        """
        self.cash += delta
        self.db.save_account_state(self.initial_capital, self.cash)
        return self.cash

    def add_position(
        self,
        symbol: str,
        asset_class: str,
        quantity: int,
        entry_price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        position_type: str = "LONG",
    ) -> Dict[str, Any]:
        """Opens or adds to a position and deducts required cash.

        Args:
            symbol: Ticker symbol.
            asset_class: Asset class (INDIAN_STOCKS, FOREX, etc.).
            quantity: Position quantity.
            entry_price: Execution price.
            stop_loss: Optional Stop Loss level.
            take_profit: Optional Take Profit level.
            position_type: LONG or SHORT.

        Returns:
            Updated position dictionary.
        """
        existing = self.get_position(symbol)
        if existing:
            # Average entry price for existing position
            old_qty = int(existing.get("quantity", 0))
            old_entry = float(existing.get("entry_price", 0.0))
            new_qty = old_qty + quantity
            new_entry = ((old_qty * old_entry) + (quantity * entry_price)) / new_qty
            quantity = new_qty
            entry_price = round(new_entry, 2)
            if stop_loss is None:
                stop_loss = existing.get("stop_loss")
            if take_profit is None:
                take_profit = existing.get("take_profit")

        self.db.save_position(
            symbol=symbol,
            asset_class=asset_class,
            quantity=quantity,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_type=position_type,
        )

        return self.get_position(symbol) or {}

    def modify_position_sl_tp(
        self,
        symbol: str,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> bool:
        """Modifies Stop Loss and Take Profit levels for an open position.

        Args:
            symbol: Ticker symbol.
            stop_loss: New Stop Loss level.
            take_profit: New Take Profit level.

        Returns:
            True if modified, False if position not found.
        """
        pos = self.get_position(symbol)
        if not pos:
            return False

        sl = stop_loss if stop_loss is not None else pos.get("stop_loss")
        tp = take_profit if take_profit is not None else pos.get("take_profit")

        self.db.save_position(
            symbol=symbol,
            asset_class=pos.get("asset_class", "EQUITY"),
            quantity=int(pos.get("quantity", 0)),
            entry_price=float(pos.get("entry_price", 0.0)),
            stop_loss=sl,
            take_profit=tp,
            position_type=pos.get("position_type", "LONG"),
        )
        return True

    def remove_position(self, symbol: str) -> None:
        """Removes an open position from storage."""
        self.db.remove_position(symbol)

    # ── Summary & Performance Analytics ─────────────────────────────

    def get_summary(
        self, current_prices: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """Generates a complete portfolio valuation summary.

        Args:
            current_prices: Optional mapping of symbol to live mark-to-market prices.

        Returns:
            Dictionary containing:
                - ``initial_capital``
                - ``cash``
                - ``portfolio_value``
                - ``unrealized_pnl``
                - ``realized_pnl``
                - ``total_pnl``
                - ``win_rate``
                - ``avg_return``
                - ``max_drawdown``
                - ``open_positions_count``
                - ``closed_trades_count``
        """
        prices = current_prices or {}
        open_positions = self.get_open_positions()
        closed_trades = self.get_closed_trades()

        unrealized_pnl = PnLCalculator.calculate_total_unrealized_pnl(
            open_positions, prices
        )
        realized_pnl = PnLCalculator.calculate_realized_pnl(closed_trades)
        portfolio_value = PnLCalculator.calculate_portfolio_value(
            self.cash, open_positions, prices
        )

        win_rate = PnLCalculator.calculate_win_rate(closed_trades)
        avg_return = PnLCalculator.calculate_avg_return(closed_trades)

        # Snapshot equity history for drawdown calculation
        self.db.record_equity_snapshot(
            portfolio_value=portfolio_value,
            cash=self.cash,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=realized_pnl,
        )

        equity_curve = [
            float(snap["portfolio_value"]) for snap in self.db.get_equity_curve()
        ]
        max_dd = PnLCalculator.calculate_max_drawdown(equity_curve)

        return {
            "initial_capital": round(self.initial_capital, 2),
            "cash": round(self.cash, 2),
            "portfolio_value": round(portfolio_value, 2),
            "unrealized_pnl": round(unrealized_pnl, 2),
            "realized_pnl": round(realized_pnl, 2),
            "total_pnl": round(realized_pnl + unrealized_pnl, 2),
            "win_rate": win_rate,
            "avg_return": avg_return,
            "max_drawdown": max_dd,
            "open_positions_count": len(open_positions),
            "closed_trades_count": len(closed_trades),
        }
