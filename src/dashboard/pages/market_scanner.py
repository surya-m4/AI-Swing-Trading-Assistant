"""
Market Scanner dashboard page.

Provides a dynamic, filterable view of all assets with tabs for
Top Gainers, Top Losers, and Most Active.  Loads data from
``AssetRegistry`` and auto-refreshes.
"""

import streamlit as st
from components.market_table import render_market_panel, _get_dynamic_tickers, fetch_market_data


def render():
    """Renders the Market Scanner page."""
    st.markdown(
        "<h2 class='neon-blue'>Market Scanner</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color:#94A3B8;'>Scan 150+ assets across categories. "
        "Search, filter, and find opportunities.</p>",
        unsafe_allow_html=True,
    )

    # Category filter
    categories = [
        "Indian Stocks",
        "Forex Major",
        "Forex Minor",
        "Forex INR",
        "Indices",
        "Commodities",
        "Crypto",
    ]
    category_keys = [
        "indian",
        "forex",
        "forex_minor",
        "forex_inr",
        "indices",
        "commodities",
        "crypto",
    ]

    selected_cat = st.selectbox(
        "Filter by Category",
        options=categories,
        index=0,
        key="scanner_category",
    )
    market_key = category_keys[categories.index(selected_cat)]

    st.markdown("<br>", unsafe_allow_html=True)

    # Tabs for different views
    tab_all, tab_gainers, tab_losers, tab_active = st.tabs(
        ["📋 All Assets", "🚀 Top Gainers", "📉 Top Losers", "🔥 Most Active"]
    )

    with tab_all:
        render_market_panel(
            market=market_key,
            page_size=20,
            show_search=True,
        )

    with tab_gainers:
        _render_ranked_table(market_key, sort_by="change_raw", ascending=False, title="Top Gainers")

    with tab_losers:
        _render_ranked_table(market_key, sort_by="change_raw", ascending=True, title="Top Losers")

    with tab_active:
        _render_ranked_table(market_key, sort_by="volume_raw", ascending=False, title="Most Active")


def _render_ranked_table(market_key: str, sort_by: str, ascending: bool, title: str) -> None:
    """Renders a sorted market table based on a ranking criterion.

    Args:
        market_key: Market category key.
        sort_by: Column to sort by (``change_raw`` or ``volume_raw``).
        ascending: Sort order.
        title: Display title.
    """
    ticker_map = _get_dynamic_tickers(market_key)
    if not ticker_map:
        st.info("No assets found in this category.")
        return

    # Limit to top 20 for speed
    limited = dict(list(ticker_map.items())[:20])
    df = fetch_market_data(limited)

    if df.empty:
        st.info("No market data available.")
        return

    # Add numeric columns for sorting
    if "change_raw" not in df.columns:
        df["change_raw"] = df["Change %"].apply(
            lambda x: float(x.replace("%", "").replace("+", ""))
            if isinstance(x, str) and x != "N/A"
            else 0
        )
    if "volume_raw" not in df.columns:
        df["volume_raw"] = df["Volume"].apply(
            lambda x: int(x.replace(",", ""))
            if isinstance(x, str) and x != "N/A"
            else 0
        )

    actual_sort = sort_by if sort_by in df.columns else "change_raw"
    df = df.sort_values(actual_sort, ascending=ascending).head(10)

    # Render table
    html_table = (
        f"<table class='dataframe fade-in'>"
        f"<tr><th>#</th><th>Symbol</th><th>Price</th><th>Change %</th><th>Volume</th></tr>"
    )
    for idx, (_, row) in enumerate(df.iterrows(), 1):
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
        html_table += f"<td style='color: #94A3B8;'>{idx}</td>"
        html_table += f"<td><strong>{row['Symbol']}</strong></td>"
        html_table += f"<td>{row['Price']}</td>"
        html_table += f"<td style='color: {color}; font-weight: bold;'>{change}</td>"
        html_table += f"<td style='color: #94A3B8;'>{row['Volume']}</td>"
        html_table += "</tr>"
    html_table += "</table>"

    st.markdown(f"<div class='glass-card'>{html_table}</div>", unsafe_allow_html=True)
