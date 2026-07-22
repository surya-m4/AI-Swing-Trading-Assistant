import streamlit as st
import plotly.graph_objects as go
from components.prediction_card import render_prediction_card

def render():
    st.markdown("<h2 class='neon-blue'>AI Predictions & SHAP Explanations</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#94A3B8;'>Deep dive into the machine learning model's decision making process.</p>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        render_prediction_card()
        
    with col2:
        st.markdown("<div class='glass-card fade-in'>", unsafe_allow_html=True)
        st.markdown("### Feature Importance (SHAP)")
        
        # Dummy SHAP data
        features = ['MACD', 'RSI_14', 'Volume_EMA', 'BB_Lower', 'ATR_14', 'Lag_1_Close']
        importance = [0.25, 0.18, 0.15, 0.12, 0.08, 0.05]
        
        fig = go.Figure(go.Bar(
            x=importance,
            y=features,
            orientation='h',
            marker=dict(
                color=importance,
                colorscale='Viridis',
            )
        ))
        
        fig.update_layout(
            template="plotly_dark", 
            paper_bgcolor="rgba(0,0,0,0)", 
            plot_bgcolor="rgba(0,0,0,0)", 
            margin=dict(t=0, b=0, l=0, r=0),
            height=350,
            yaxis={'categoryorder':'total ascending'}
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
