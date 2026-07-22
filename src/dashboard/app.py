"""
Streamlit Dashboard — Main Application Entry Point.

Provides the sidebar navigation, page routing, auto-refresh via
``st_autorefresh``, and custom CSS loading for the AI Swing Trading
Assistant.
"""

import os
import streamlit as st
from streamlit_option_menu import option_menu

from components.navbar import render_navbar
from pages import (
    dashboard,
    indian_stocks,
    forex,
    predictions,
    portfolio,
    backtesting,
    experiments,
    settings,
    market_scanner,
    watchlists,
)

# Set page config MUST be the first Streamlit command
st.set_page_config(
    page_title="AI Swing Trading Assistant",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_css():
    """Load custom CSS for premium styling."""
    css_path = os.path.join(os.path.dirname(__file__), "assets", "css", "style.css")
    if os.path.exists(css_path):
        with open(css_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def setup_auto_refresh():
    """Configures 30-second auto-refresh using st_autorefresh.

    Falls back gracefully if the streamlit-extras package is not
    installed.
    """
    try:
        from streamlit_extras.streaming_write import streaming_write  # noqa: F401
        # st_autorefresh may be in streamlit_extras or streamlit-autorefresh
    except ImportError:
        pass

    try:
        from streamlit_autorefresh import st_autorefresh

        st_autorefresh(interval=30_000, limit=None, key="market_auto_refresh")
    except ImportError:
        # Fallback: use a simple meta-refresh via HTML
        st.markdown(
            '<meta http-equiv="refresh" content="30">',
            unsafe_allow_html=True,
        )


def main():
    """Main application entry point."""
    load_css()
    render_navbar()

    # Auto-refresh for live data
    setup_auto_refresh()

    # Sidebar Navigation using option_menu
    with st.sidebar:
        st.markdown(
            "<h2 class='neon-blue' style='text-align: center;'>AI TRADER</h2>",
            unsafe_allow_html=True,
        )
        st.markdown("---")

        selected = option_menu(
            menu_title=None,
            options=[
                "Dashboard",
                "Indian Stocks",
                "Forex",
                "Market Scanner",
                "Watchlists",
                "AI Predictions",
                "Portfolio",
                "Backtesting",
                "MLflow Experiments",
                "Settings",
            ],
            icons=[
                "house",
                "graph-up-arrow",
                "currency-exchange",
                "search",
                "star",
                "robot",
                "briefcase",
                "activity",
                "cpu",
                "gear",
            ],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {
                    "padding": "0!important",
                    "background-color": "transparent",
                },
                "icon": {"color": "#00C8FF", "font-size": "18px"},
                "nav-link": {
                    "font-size": "15px",
                    "text-align": "left",
                    "margin": "5px",
                    "--hover-color": "#1E293B",
                },
                "nav-link-selected": {
                    "background-color": "rgba(0, 200, 255, 0.2)",
                    "border-left": "4px solid #00C8FF",
                },
            },
        )

        st.markdown("---")

        # Refresh status indicator
        st.markdown(
            "<div style='text-align: center;'>"
            "<span style='color: #00FF7F; font-size: 0.75rem;'>● Auto-Refresh Active (30s)</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div style='text-align: center; color: #888;'>v2.0.0</div>",
            unsafe_allow_html=True,
        )

    # Route to the selected page
    if selected == "Dashboard":
        dashboard.render()
    elif selected == "Indian Stocks":
        indian_stocks.render()
    elif selected == "Forex":
        forex.render()
    elif selected == "Market Scanner":
        market_scanner.render()
    elif selected == "Watchlists":
        watchlists.render()
    elif selected == "AI Predictions":
        predictions.render()
    elif selected == "Portfolio":
        portfolio.render()
    elif selected == "Backtesting":
        backtesting.render()
    elif selected == "MLflow Experiments":
        experiments.render()
    elif selected == "Settings":
        settings.render()


if __name__ == "__main__":
    main()
