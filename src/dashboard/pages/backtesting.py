import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd

def render():
    st.markdown("<h2 class='neon-purple'>Backtesting Results</h2>", unsafe_allow_html=True)
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    metrics = [
        {"col": col1, "title": "CAGR", "value": "24.5%"},
        {"col": col2, "title": "Sharpe Ratio", "value": "1.85"},
        {"col": col3, "title": "Max Drawdown", "value": "-12.4%"},
        {"col": col4, "title": "Profit Factor", "value": "2.1"},
    ]
    
    for m in metrics:
        with m["col"]:
            st.markdown(f"""
            <div class='glass-card fade-in' style='text-align: center;'>
                <div style='color: #94A3B8; font-size: 0.9rem;'>{m["title"]}</div>
                <div class='neon-blue' style='font-size: 2rem; font-weight: bold;'>{m["value"]}</div>
            </div>
            """, unsafe_allow_html=True)
            
    # Equity Curve
    st.markdown("<div class='glass-card fade-in'>", unsafe_allow_html=True)
    st.markdown("### Equity Curve")
    
    np.random.seed(42)
    dates = pd.date_range(start='2022-01-01', periods=252)
    returns = np.random.normal(0.001, 0.015, 252)
    equity = 100000 * np.exp(np.cumsum(returns))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=equity, mode='lines', fill='tozeroy', 
                             line=dict(color='#00C8FF', width=3),
                             fillcolor='rgba(0, 200, 255, 0.1)',
                             name='Portfolio Value'))
                             
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=10, b=10, l=10, r=10), height=400)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
