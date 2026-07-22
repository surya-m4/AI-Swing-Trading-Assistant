import streamlit as st

def render_summary_cards():
    """Renders the top row summary cards."""
    col1, col2, col3, col4, col5 = st.columns(5)
    
    cards = [
        {"col": col1, "title": "Total Trades", "value": "1,248", "trend": "+12%", "trend_class": "metric-trend-up", "icon": "📊"},
        {"col": col2, "title": "Prediction Accuracy", "value": "82.4%", "trend": "+1.2%", "trend_class": "metric-trend-up", "icon": "🎯"},
        {"col": col3, "title": "Portfolio Value", "value": "₹ 15.4M", "trend": "+4.5%", "trend_class": "metric-trend-up", "icon": "💰"},
        {"col": col4, "title": "Win Rate", "value": "68%", "trend": "-2.1%", "trend_class": "metric-trend-down", "icon": "🏆"},
        {"col": col5, "title": "Today's P&L", "value": "₹ 45K", "trend": "+3.4%", "trend_class": "metric-trend-up", "icon": "📈"},
    ]
    
    for card in cards:
        with card["col"]:
            st.markdown(f"""
            <div class="glass-card fade-in">
                <div style="display: flex; justify-content: space-between; align-items: center; color: #94A3B8; font-size: 0.9rem;">
                    <span>{card["title"]}</span>
                    <span style="font-size: 1.2rem;">{card["icon"]}</span>
                </div>
                <div class="metric-value">{card["value"]}</div>
                <div class="{card["trend_class"]}">{card["trend"]} vs last month</div>
            </div>
            """, unsafe_allow_html=True)
