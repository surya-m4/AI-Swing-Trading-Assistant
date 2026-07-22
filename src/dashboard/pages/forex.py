"""
Forex Markets dashboard page.

Loads pairs dynamically from ``AssetRegistry`` with support for
Major, Minor, and INR pair categories.
"""

import streamlit as st
from components.charts import render_main_chart
from components.market_table import render_market_panel
from components.prediction_card import render_prediction_card


def render():
    """Renders the Forex Markets page with dynamic pair loading."""
    st.markdown(
        "<h2 class='neon-purple'>Forex Markets</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color:#94A3B8;'>Real-time analysis and AI predictions "
        "for major, minor, and INR currency pairs.</p>",
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Category selector for forex sub-types
    forex_type = st.radio(
        "Pair Type",
        ["Major Pairs", "Minor Pairs", "INR Pairs"],
        horizontal=True,
        key="forex_type_selector",
    )
    market_key = {
        "Major Pairs": "forex",
        "Minor Pairs": "forex_minor",
        "INR Pairs": "forex_inr",
    }[forex_type]

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(f"### {forex_type}")
        render_market_panel(
            market=market_key,
            page_size=12,
            show_search=True,
        )
    with col2:
        st.markdown("### Detailed Chart")
        render_main_chart()

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### AI Prediction")
        render_prediction_card(ticker="EURUSD=X")
