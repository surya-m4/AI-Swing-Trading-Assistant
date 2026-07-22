"""
Virtual Portfolio & Quantitative Portfolio Analytics module for paper trading.

Tracks Cash, Invested Amount, Available Margin, Portfolio Value, Daily & Overall P&L,
Win Rate, Loss Rate, Max Drawdown, ROI, Sharpe Ratio, Sortino Ratio, and Profit Factor.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from src.trading.models import PortfolioAnalyticsModel, PortfolioSummaryModel, Position, Trade

logger = logging.getLogger(__name__)

DEFAULT_STARTING_CAPITAL = 1000000.0  # ₹10,00,000


class Portfolio:
    """Virtual Portfolio tracking cash, margin, returns, and quantitative risk metrics.

    Attributes:
        initial_capital: Starting virtual capital balance (default ₹10,00,000).
        cash: Current uninvested cash balance.
        equity_snapshots: Historical portfolio equity values for drawdown and Sharpe ratio calculations.
    """

    def __init__(self, initial_capital: float = DEFAULT_STARTING_CAPITAL) -> None:
        """Initialises the virtual portfolio.

        Args:
            initial_capital: Starting capital amount in ₹.
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.equity_snapshots: List[float] = [initial_capital]
        self._daily_start_value: float = initial_capital
        logger.info("Portfolio initialized with Starting Capital ₹%s.", f"{initial_capital:,.2f}")

    def update_cash(self, delta: float) -> float:
        """Adds or deducts cash from the portfolio.

        Args:
            delta: Cash amount change.

        Returns:
            Updated cash balance.
        """
        self.cash = round(self.cash + delta, 2)
        return self.cash

    def reset(self, initial_capital: Optional[float] = None) -> None:
        """Resets portfolio cash and equity tracking."""
        cap = initial_capital if initial_capital is not None else self.initial_capital
        self.initial_capital = cap
        self.cash = cap
        self.equity_snapshots = [cap]
        self._daily_start_value = cap
        logger.info("Portfolio reset to starting capital ₹%s.", f"{cap:,.2f}")

    # ── Portfolio Valuation & Metrics ─────────────────────────────────

    def get_summary(
        self, positions: List[Position], trades: List[Trade]
    ) -> PortfolioSummaryModel:
        """Computes summary metrics for virtual portfolio.

        Args:
            positions: List of active open Position objects.
            trades: List of historical completed Trade objects.

        Returns:
            Pydantic ``PortfolioSummaryModel`` instance.
        """
        invested_amount = round(sum(p.current_value for p in positions), 2)
        available_margin = round(self.cash, 2)  # Available cash as margin
        portfolio_value = round(self.cash + invested_amount, 2)

        # Snapshot for drawdown / analytics tracking
        self.equity_snapshots.append(portfolio_value)

        overall_pnl = round(portfolio_value - self.initial_capital, 2)
        daily_pnl = round(portfolio_value - self._daily_start_value, 2)

        roi = (
            round((overall_pnl / self.initial_capital) * 100.0, 2)
            if self.initial_capital > 0
            else 0.0
        )

        closed_trades = [t for t in trades if t.action in ("SELL", "CLOSE") or t.pnl != 0]
        total_closed = len(closed_trades)

        if total_closed > 0:
            wins = sum(1 for t in closed_trades if t.pnl > 0)
            losses = sum(1 for t in closed_trades if t.pnl < 0)
            win_rate = round((wins / total_closed) * 100.0, 2)
            loss_rate = round((losses / total_closed) * 100.0, 2)
        else:
            win_rate = 0.0
            loss_rate = 0.0

        max_drawdown = self.calculate_max_drawdown()

        return PortfolioSummaryModel(
            cash=round(self.cash, 2),
            invested_amount=invested_amount,
            available_margin=available_margin,
            portfolio_value=portfolio_value,
            daily_pnl=daily_pnl,
            overall_pnl=overall_pnl,
            win_rate=win_rate,
            loss_rate=loss_rate,
            max_drawdown=max_drawdown,
            roi=roi,
        )

    def calculate_analytics(
        self, positions: List[Position], trades: List[Trade], risk_free_rate: float = 0.05
    ) -> PortfolioAnalyticsModel:
        """Computes advanced quantitative analytics.

        Calculates Sharpe Ratio, Sortino Ratio, Profit Factor, Average Win/Loss, etc.

        Args:
            positions: List of active open positions.
            trades: List of closed trades.
            risk_free_rate: Annual risk-free interest rate (default 5.0%).

        Returns:
            Pydantic ``PortfolioAnalyticsModel`` instance.
        """
        invested = sum(p.current_value for p in positions)
        portfolio_value = self.cash + invested
        portfolio_return = (
            round(((portfolio_value - self.initial_capital) / self.initial_capital) * 100.0, 2)
            if self.initial_capital > 0
            else 0.0
        )

        closed_trades = [t for t in trades if t.action in ("SELL", "CLOSE") or t.pnl != 0]
        winning_trades = [t for t in closed_trades if t.pnl > 0]
        losing_trades = [t for t in closed_trades if t.pnl < 0]

        total_profit = round(sum(t.pnl for t in winning_trades), 2)
        total_loss = round(sum(abs(t.pnl) for t in losing_trades), 2)

        win_pct = (
            round((len(winning_trades) / len(closed_trades)) * 100.0, 2)
            if closed_trades
            else 0.0
        )

        avg_win = (
            round(total_profit / len(winning_trades), 2) if winning_trades else 0.0
        )
        avg_loss = (
            round(-total_loss / len(losing_trades), 2) if losing_trades else 0.0
        )

        # Profit Factor = Total Profit / Total Loss
        if total_loss > 0:
            profit_factor = round(total_profit / total_loss, 2)
        else:
            profit_factor = round(total_profit, 2) if total_profit > 0 else 1.0

        # Returns series for Sharpe & Sortino
        trade_pnls = [t.pnl for t in closed_trades]
        if len(trade_pnls) >= 2:
            returns = np.array(trade_pnls) / (self.initial_capital / len(trade_pnls))
            mean_ret = float(np.mean(returns))
            std_ret = float(np.std(returns, ddof=1))

            rf_per_trade = risk_free_rate / 252.0  # Daily risk free rate approx
            sharpe = (
                round(float((mean_ret - rf_per_trade) / std_ret * np.sqrt(252)), 2)
                if std_ret > 0
                else 0.0
            )

            downside_returns = returns[returns < 0]
            downside_std = (
                float(np.std(downside_returns, ddof=1)) if len(downside_returns) > 1 else 1e-6
            )
            sortino = (
                round(float((mean_ret - rf_per_trade) / downside_std * np.sqrt(252)), 2)
                if downside_std > 0
                else 0.0
            )
        else:
            sharpe = 0.0
            sortino = 0.0

        max_dd = self.calculate_max_drawdown()

        return PortfolioAnalyticsModel(
            portfolio_return=portfolio_return,
            total_profit=total_profit,
            total_loss=total_loss,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            profit_factor=profit_factor,
            avg_winning_trade=avg_win,
            avg_losing_trade=avg_loss,
            max_drawdown=max_dd,
            win_percentage=win_pct,
        )

    def calculate_max_drawdown(self) -> float:
        """Calculates maximum drawdown percentage from historical equity snapshots."""
        if len(self.equity_snapshots) < 2:
            return 0.0

        arr = np.array(self.equity_snapshots, dtype=float)
        peaks = np.maximum.accumulate(arr)
        peaks = np.where(peaks <= 0, 1e-8, peaks)

        drawdowns = (peaks - arr) / peaks
        return round(float(np.max(drawdowns)) * 100.0, 2)
