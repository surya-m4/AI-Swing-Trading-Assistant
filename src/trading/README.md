# Module 14: Paper Trading Engine

Production-grade paper trading simulation system for the AI-Powered Swing Trading Assistant.

## Features

1. **Virtual Portfolio (Default ₹10,00,000 Starting Capital)**
   - Tracks Cash, Invested Amount, Available Margin, Portfolio Value, Daily P&L, Overall P&L, Win Rate, Loss Rate, Maximum Drawdown, and ROI.
2. **BUY Orders Execution & Risk Controls**
   - Validates ticker existence, positive quantity, cash availability, and risk limits.
3. **SELL Orders Execution & Partial Closing**
   - Validates owned holdings, supports partial position reductions, credits cash proceeds, and logs realized P&L.
4. **Position Management**
   - Tracks Ticker, Entry Price, Mark-to-Market Price, Quantity, Current Value, P&L, Holding Days, Stop Loss, Take Profit, and Trailing Stop.
5. **Trade History & CSV Export**
   - Stores transactions in SQLite (`data/paper_trading.db`).
   - Export trade logs to CSV via `export_history_csv()`.
6. **Quantitative Portfolio Analytics**
   - Calculates Portfolio Return, Total Profit, Total Loss, Sharpe Ratio, Sortino Ratio, Profit Factor, Average Winning Trade, Average Losing Trade, Maximum Drawdown, and Win Percentage.
7. **FastAPI & Streamlit Integration**
   - Endpoints: `GET /portfolio`, `GET /positions`, `GET /trade-history`, `POST /buy`, `POST /sell`, `DELETE /positions/{ticker}`, `GET /analytics`, `GET /risk`.
   - Dark-themed Streamlit interactive UI with Plotly performance charts.

## Architecture

```
src/trading/
├── __init__.py          # Public package exports
├── paper_trader.py      # Main PaperTrader facade & AI recommendation integration
├── portfolio.py         # Virtual portfolio & quantitative analytics engine
├── order_manager.py     # BUY and SELL order execution engine
├── position_manager.py  # Position state, mark-to-market, and trailing stops
├── trade_history.py     # SQLite storage & CSV export
├── risk_checks.py       # Risk validation & pre-trade checks
├── models.py            # Dataclasses & Pydantic models
└── exceptions.py        # Custom trading exceptions
```

## Quick Start Usage Example

```python
from src.trading import PaperTrader

# Initialize trader with starting capital ₹10,00,000
trader = PaperTrader(initial_capital=1000000.0)

# Place a BUY paper order for 100 shares of RELIANCE.NS @ ₹2,500
buy_resp = trader.buy(
    ticker="RELIANCE.NS",
    quantity=100,
    price=2500.0,
    stop_loss=2425.0,
    take_profit=2650.0
)
print("Order status:", buy_resp.status)

# Retrieve portfolio valuation summary
summary = trader.get_portfolio({"RELIANCE.NS": 2550.0})
print("Portfolio Value:", summary.portfolio_value)

# Close 50 shares (partial sell) @ ₹2,550
sell_resp = trader.sell(ticker="RELIANCE.NS", quantity=50, price=2550.0)
print("Realized PnL:", sell_resp.pnl)

# Export trade history to CSV
csv_path = trader.export_history_csv("data/my_trades.csv")
print("CSV exported to:", csv_path)
```
