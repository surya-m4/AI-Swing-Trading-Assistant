import streamlit as st
from datetime import datetime

def render_navbar():
    """Renders the top navbar."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    navbar_html = f"""
    <div class="top-navbar fade-in">
        <div style="display: flex; align-items: center; gap: 20px;">
            <div class="nav-badges">
                <span class="badge-open">NSE OPEN</span>
                <span class="badge-open">FOREX OPEN</span>
            </div>
        </div>
        <div style="display: flex; align-items: center; gap: 20px; font-weight: 500;">
            <span>🔍 Search</span>
            <span>⏱️ {now_str}</span>
            <span>🔔</span>
            <span>👤 Trader Pro</span>
        </div>
    </div>
    """
    st.markdown(navbar_html, unsafe_allow_html=True)
