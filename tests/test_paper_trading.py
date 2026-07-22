"""
Comprehensive unit test suite for Module 14: Paper Trading Engine in src/trading/.

Tests RiskChecker, PositionManager, TradeHistoryManager, Portfolio, OrderManager,
PaperTrader, quantitative analytics, and custom exceptions.
"""

import os
import tempfile
import pytest

from src.trading.exceptions import (
    DuplicateOrderException,
    InsufficientFundsException,
    InvalidOrderException,
    MarketClosedException,
    PositionNotFoundException,
    RiskLimitExceededException,
    TradingException,
)
from src.trading.models import OrderType, Position, PositionType, RiskConfig, Trade
from src.trading.order_manager import OrderManager
from src.trading.paper_trader import PaperTrader
from src.trading.portfolio import DEFAULT_STARTING_CAPITAL, Portfolio
from src.trading.position_manager import PositionManager
from src.trading.risk_checks import RiskChecker
from src.trading.trade_history import TradeHistoryManager


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def temp_db_path():
    """Creates a temporary SQLite DB file path and removes it after test."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        path = tmp.name
    yield path
    if os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass


@pytest.fixture
def db(temp_db_path):
    """Returns a fresh TradeHistoryManager instance."""
    return TradeHistoryManager(db_path=temp_db_path)


@pytest.fixture
def portfolio():
    """Returns a fresh Portfolio initialized to ₹10,00,000."""
    return Portfolio(initial_capital=1000000.0)


@pytest.fixture
def position_manager():
    """Returns a fresh PositionManager instance."""
    return PositionManager()


@pytest.fixture
def risk_checker():
    """Returns a RiskChecker instance with 20% max position cap."""
    return RiskChecker(config=RiskConfig(max_position_size_pct=20.0, cooldown_seconds=0.01))


@pytest.fixture
def trader(temp_db_path):
    """Returns a fresh PaperTrader instance."""
    return PaperTrader(initial_capital=1000000.0, db_path=temp_db_path)


# ── Test RiskChecker ─────────────────────────────────────────────────


class TestRiskChecker:
    """Tests for pre-trade risk validation rules."""

    def test_buy_validation_success(self, risk_checker):
        valid, msg = risk_checker.validate_buy_order(
            ticker="RELIANCE.NS",
            quantity=10,
            price=2500.0,
            available_cash=1000000.0,
            portfolio_value=1000000.0,
            raise_exception=False,
        )
        assert valid is True

    def test_buy_insufficient_funds_raises(self, risk_checker):
        with pytest.raises(InsufficientFundsException):
            risk_checker.validate_buy_order(
                ticker="RELIANCE.NS",
                quantity=1000,
                price=2500.0,
                available_cash=50000.0,
                portfolio_value=1000000.0,
                raise_exception=True,
            )

    def test_buy_invalid_quantity_raises(self, risk_checker):
        with pytest.raises(InvalidOrderException):
            risk_checker.validate_buy_order(
                ticker="RELIANCE.NS",
                quantity=-5,
                price=2500.0,
                available_cash=1000000.0,
                portfolio_value=1000000.0,
                raise_exception=True,
            )

    def test_buy_position_size_exceeded_raises(self, risk_checker):
        # 30% of portfolio value (exceeds 20% cap)
        with pytest.raises(RiskLimitExceededException):
            risk_checker.validate_buy_order(
                ticker="RELIANCE.NS",
                quantity=120,
                price=2500.0,  # cost = 300,000 = 30% of 1M
                available_cash=1000000.0,
                portfolio_value=1000000.0,
                raise_exception=True,
            )

    def test_market_closed_raises(self, risk_checker):
        risk_checker.set_market_status(False)
        with pytest.raises(MarketClosedException):
            risk_checker.validate_buy_order(
                ticker="RELIANCE.NS",
                quantity=10,
                price=2500.0,
                available_cash=1000000.0,
                portfolio_value=1000000.0,
                raise_exception=True,
            )

    def test_sell_validation_insufficient_holdings(self, risk_checker):
        with pytest.raises(InvalidOrderException):
            risk_checker.validate_sell_order(
                ticker="TCS.NS", quantity=50, price=3000.0, owned_quantity=20, raise_exception=True
            )


# ── Test PositionManager ─────────────────────────────────────────────


class TestPositionManager:
    """Tests for position tracking, mark-to-market, and partial closing."""

    def test_open_position(self, position_manager):
        pos = position_manager.open_or_update_position("INFY.NS", quantity=50, price=1400.0)
        assert pos.ticker == "INFY.NS"
        assert pos.quantity == 50
        assert pos.entry_price == 1400.0
        assert pos.current_value == 70000.0

    def test_update_position_averaging(self, position_manager):
        position_manager.open_or_update_position("INFY.NS", quantity=50, price=1400.0)
        pos = position_manager.open_or_update_position("INFY.NS", quantity=50, price=1600.0)

        assert pos.quantity == 100
        assert pos.entry_price == 1500.0

    def test_partial_sell_position(self, position_manager):
        position_manager.open_or_update_position("SBIN.NS", quantity=100, price=600.0)
        pnl, remaining = position_manager.reduce_position("SBIN.NS", quantity=40, exit_price=650.0)

        # Realized PnL = (650 - 600) * 40 = 2000
        assert pnl == 2000.0
        assert remaining is not None
        assert remaining.quantity == 60

    def test_full_close_position(self, position_manager):
        position_manager.open_or_update_position("SBIN.NS", quantity=100, price=600.0)
        pnl, remaining = position_manager.reduce_position("SBIN.NS", quantity=100, exit_price=650.0)

        assert pnl == 5000.0
        assert remaining is None
        assert position_manager.get_position("SBIN.NS") is None

    def test_trailing_stop_update(self, position_manager):
        position_manager.open_or_update_position("TCS.NS", quantity=10, price=3000.0, stop_loss=2900.0, trailing_stop=100.0)

        # Update market price rising to 3200
        position_manager.update_market_prices({"TCS.NS": 3200.0})
        pos = position_manager.get_position("TCS.NS")

        # Trailing stop should adjust stop_loss to 3200 - 100 = 3100
        assert pos.stop_loss == 3100.0


# ── Test TradeHistoryManager ─────────────────────────────────────────


class TestTradeHistoryManager:
    """Tests for SQLite persistence and CSV export."""

    def test_record_and_get_trades(self, db):
        t1 = Trade("TRD_1", "RELIANCE.NS", "BUY", 2500.0, 10, pnl=0.0)
        t2 = Trade("TRD_2", "RELIANCE.NS", "SELL", 2600.0, 10, pnl=1000.0)

        db.record_trade(t1)
        db.record_trade(t2)

        trades = db.get_all_trades()
        assert len(trades) == 2

    def test_export_to_csv(self, db, temp_db_path):
        t1 = Trade("TRD_1", "TCS.NS", "BUY", 3000.0, 5, pnl=0.0)
        db.record_trade(t1)

        csv_path = os.path.join(os.path.dirname(temp_db_path), "test_export.csv")
        result_path = db.export_to_csv(csv_path)

        assert os.path.exists(result_path)
        with open(result_path, "r", encoding="utf-8") as f:
            content = f.read()
            assert "TCS.NS" in content
            assert "BUY" in content

        if os.path.exists(result_path):
            os.remove(result_path)


# ── Test Portfolio & Analytics ───────────────────────────────────────


class TestPortfolio:
    """Tests for virtual portfolio tracking & quantitative analytics."""

    def test_portfolio_init(self, portfolio):
        assert portfolio.initial_capital == 1000000.0
        assert portfolio.cash == 1000000.0

    def test_portfolio_analytics_calculations(self, portfolio):
        positions = [Position("RELIANCE.NS", entry_price=2500.0, current_price=2600.0, quantity=100)]
        trades = [
            Trade("1", "TCS.NS", "SELL", 3200.0, 10, pnl=2000.0),
            Trade("2", "INFY.NS", "SELL", 1400.0, 10, pnl=-500.0),
        ]

        analytics = portfolio.calculate_analytics(positions, trades)
        assert analytics.total_profit == 2000.0
        assert analytics.total_loss == 500.0
        assert analytics.profit_factor == 4.0
        assert analytics.win_percentage == 50.0


# ── Test PaperTrader End-to-End Workflow ────────────────────────────


class TestPaperTrader:
    """Tests for PaperTrader main façade."""

    def test_paper_trader_buy_flow(self, trader):
        resp = trader.buy("RELIANCE.NS", quantity=20, price=2500.0)

        assert resp.status == "EXECUTED"
        assert resp.action == "BUY"

        summary = trader.get_portfolio({"RELIANCE.NS": 2500.0})
        assert summary.cash == 950000.0  # 1,000,000 - 50,000
        assert summary.invested_amount == 50000.0
        assert summary.portfolio_value == 1000000.0

    def test_paper_trader_partial_sell_flow(self, trader):
        trader.buy("RELIANCE.NS", quantity=20, price=2500.0)
        sell_resp = trader.sell("RELIANCE.NS", quantity=10, price=2600.0)

        assert sell_resp.status == "EXECUTED"
        assert sell_resp.pnl == 1000.0  # (2600 - 2500) * 10

        positions = trader.get_positions()
        assert len(positions) == 1
        assert positions[0]["quantity"] == 10

    def test_paper_trader_close_position_flow(self, trader):
        trader.buy("TCS.NS", quantity=10, price=3000.0)
        close_resp = trader.close_position("TCS.NS", current_price=3100.0)

        assert close_resp.status == "EXECUTED"
        assert close_resp.pnl == 1000.0
        assert len(trader.get_positions()) == 0

    def test_export_csv_flow(self, trader):
        trader.buy("INFY.NS", quantity=10, price=1400.0)
        csv_file = trader.export_history_csv()
        assert os.path.exists(csv_file)
