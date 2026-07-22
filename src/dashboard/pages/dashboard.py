import streamlit as st
import yfinance as yf
import pandas as pd
from components.summary_cards import render_summary_cards
from components.market_table import render_market_panel
from components.charts import render_main_chart
from components.prediction_card import render_prediction_card

def render():
    """Renders the main dashboard page."""
    
    # Hero Section
    st.markdown("""
        <div class="hero-section fade-in">
            <div class="hero-title">AI Powered Swing Trading Assistant</div>
            <div style="font-size: 1.2rem; color: #94A3B8;">End-to-End MLOps Trading Platform</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Summary Cards Row
    render_summary_cards()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Main Layout: 2 Columns (Market Panels vs Chart & Predictions)
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("<h3 class='neon-blue'>Live Market</h3>", unsafe_allow_html=True)
        render_market_panel(market="indian")
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<h3 class='neon-blue'>Forex</h3>", unsafe_allow_html=True)
        render_market_panel(market="forex")
        
    with col2:
        st.markdown("<h3 class='neon-purple'>Technical Analysis</h3>", unsafe_allow_html=True)
        # We will use yfinance to get some real data for the chart to look professional
        render_main_chart()
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<h3 class='neon-green'>AI Prediction</h3>", unsafe_allow_html=True)
        render_prediction_card()
