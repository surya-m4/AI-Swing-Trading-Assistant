"""
Dynamic market data table component.

Loads ticker symbols from ``AssetRegistry`` instead of hard-coding.
Supports pagination for 100+ symbols and colour-coded change percentages.
"""

import streamlit as st
import pandas as pd
import yfinance as yf
from typing import Dict, List, Optional


@st.cache_data(ttl=60)
def fetch_market_data(tickers: Dict[str, str]) -> pd.DataFrame:
    """Fetches latest market data for a set of tickers.

    Args:
        tickers: Mapping of Yahoo Finance symbol to display name.

    Returns:
        DataFrame with Symbol, Price, Change %, and Volume columns.
    """
    data = []
    for ticker, name in tickers.items():
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="2d")
            if len(hist) >= 2:
                prev_close = hist["Close"].iloc[0]
                current = hist["Close"].iloc[1]
                change_pct = ((current - prev_close) / prev_close) * 100
                vol = hist["Volume"].iloc[1]
            else:
                current, change_pct, vol = 0, 0, 0

            data.append(
                {
                    "Symbol": name,
                    "Ticker": ticker,
                    "Price": f"{current:.2f}",
                    "Change %": f"{change_pct:+.2f}%",
                    "Volume": f"{int(vol):,}",
                    "change_raw": change_pct,
                }
            )
        except Exception:
            data.append(
                {
                    "Symbol": name,
                    "Ticker": ticker,
                    "Price": "N/A",
                    "Change %": "N/A",
                    "Volume": "N/A",
                    "change_raw": 0,
                }
            )
    return pd.DataFrame(data)


def _get_dynamic_tickers(market: str) -> Dict[str, str]:
    """Loads ticker → display-name mapping from AssetRegistry.

    Falls back to a small built-in set if the registry import fails.

    Args:
        market: One of ``"indian"``, ``"forex"``, ``"indices"``,
            ``"commodities"``, ``"crypto"``, or ``"all"``.

    Returns:
        Dict mapping Yahoo Finance symbol to display name.
    """
    try:
        from src.market_data.assets_config import AssetCategory, AssetRegistry

        registry = AssetRegistry()
        category_map = {
            "indian": AssetCategory.INDIAN_STOCKS,
            "forex": AssetCategory.FOREX_MAJOR,
            "forex_minor": AssetCategory.FOREX_MINOR,
            "forex_inr": AssetCategory.FOREX_INR,
            "indices": AssetCategory.INDICES,
            "commodities": AssetCategory.COMMODITIES,
            "crypto": AssetCategory.CRYPTO,
        }
        if market in category_map:
            return registry.get_display_map(category_map[market])
        # "all" or unknown → return all
        return registry.get_display_map()
    except Exception:
        # Fallback
        if market == "indian":
            return {
                "RELIANCE.NS": "RELIANCE",
                "TCS.NS": "TCS",
                "INFY.NS": "INFY",
                "HDFCBANK.NS": "HDFCBANK",
                "ICICIBANK.NS": "ICICIBANK",
                "SBIN.NS": "SBIN",
            }
        return {
            "EURUSD=X": "EUR/USD",
            "GBPUSD=X": "GBP/USD",
            "USDJPY=X": "USD/JPY",
            "AUDUSD=X": "AUD/USD",
            "INR=X": "USD/INR",
        }


def render_market_panel(
    market: str = "indian",
    tickers: Optional[Dict[str, str]] = None,
    page_size: int = 15,
    show_search: bool = False,
) -> None:
    """Renders the market data table with pagination.

    Args:
        market: Market category string.
        tickers: Optional override for ticker map.
        page_size: Number of rows per page.
        show_search: If ``True``, displays a search/filter bar.
    """
    ticker_map = tickers or _get_dynamic_tickers(market)

    # Optional search filter
    if show_search:
        search_q = st.text_input(
            "🔍 Filter symbols",
            key=f"filter_{market}",
            placeholder="Type to filter...",
        )
        if search_q:
            q = search_q.upper()
            ticker_map = {
                k: v for k, v in ticker_map.items()
                if q in k.upper() or q in v.upper()
            }

    if not ticker_map:
        st.info("No matching assets found.")
        return

    # Pagination
    total = len(ticker_map)
    total_pages = max(1, (total + page_size - 1) // page_size)
    page = st.number_input(
        "Page",
        min_value=1,
        max_value=total_pages,
        value=1,
        key=f"page_{market}",
        label_visibility="collapsed",
    ) if total_pages > 1 else 1

    start = (page - 1) * page_size
    end = start + page_size
    page_tickers = dict(list(ticker_map.items())[start:end])

    df = fetch_market_data(page_tickers)

    # Render as styled HTML table
    html_table = (
        "<table class='dataframe fade-in'>"
        "<tr><th>Symbol</th><th>Price</th><th>Change %</th><th>Volume</th></tr>"
    )
    for _, row in df.iterrows():
        change = row["Change %"]
        if isinstance(change, str):
            color = (
                "#00FF7F"
                if change.startswith("+")
                else "#FF3333"
                if change.startswith("-")
                else "#FFF"
            )
        else:
            color = "#FFF"
        html_table += "<tr>"
        html_table += f"<td><strong>{row['Symbol']}</strong></td>"
        html_table += f"<td>{row['Price']}</td>"
        html_table += f"<td style='color: {color}; font-weight: bold;'>{change}</td>"
        html_table += f"<td style='color: #94A3B8;'>{row['Volume']}</td>"
        html_table += "</tr>"
    html_table += "</table>"

    st.markdown(f"<div class='glass-card'>{html_table}</div>", unsafe_allow_html=True)

    if total_pages > 1:
        st.caption(
            f"Showing {start + 1}–{min(end, total)} of {total} assets  ·  Page {page}/{total_pages}"
        )
