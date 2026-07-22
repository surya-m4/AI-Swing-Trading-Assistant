"""
Indian Stocks dashboard page.

Loads symbols dynamically from ``AssetRegistry`` with auto-refresh support.
"""

import streamlit as st
from components.charts import render_main_chart
from components.market_table import render_market_panel
from components.prediction_card import render_prediction_card


def render():
    """Renders the Indian Stocks page with dynamic asset loading."""
    st.markdown(
        "<h2 class='neon-blue'>Indian Stocks</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color:#94A3B8;'>Real-time analysis and AI predictions "
        "for 100+ NSE-listed equities. Data refreshes automatically.</p>",
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("### Top Movers")
        render_market_panel(
            market="indian",
            page_size=15,
            show_search=True,
        )
    with col2:
        st.markdown("### Detailed Chart")
        render_main_chart()

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### AI Prediction")
        render_prediction_card()
