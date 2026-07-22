import streamlit as st
import plotly.graph_objects as go
import yfinance as yf
from plotly.subplots import make_subplots

@st.cache_data(ttl=600)
def get_chart_data(ticker="RELIANCE.NS", period="3mo"):
    stock = yf.Ticker(ticker)
    return stock.history(period=period)

def render_main_chart():
    """Renders the main interactive Plotly chart."""
    
    st.markdown("<div class='glass-card fade-in'>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        ticker = st.selectbox("Select Asset", ["RELIANCE.NS", "TCS.NS", "INFY.NS", "EURUSD=X", "BTC-USD"], label_visibility="collapsed")
    with col2:
        period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y"], index=1, label_visibility="collapsed")
    with col3:
        indicator = st.selectbox("Indicator", ["None", "SMA 20", "EMA 20", "Bollinger Bands"], label_visibility="collapsed")
        
    df = get_chart_data(ticker, period)
    
    if df.empty:
        st.warning("No data available for this ticker.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # Create subplots: Candlestick + Volume
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, subplot_titles=(f'{ticker} Price', 'Volume'),
                        row_width=[0.2, 0.7])

    # Candlestick
    fig.add_trace(go.Candlestick(x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    name='Price'),
                    row=1, col=1)

    # Volume
    colors = ['#FF3333' if row['Open'] - row['Close'] >= 0 else '#00FF7F' for index, row in df.iterrows()]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors, name='Volume'),
                    row=2, col=1)

    # Add Indicators
    if indicator == "SMA 20":
        df['SMA20'] = df['Close'].rolling(window=20).mean()
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], line=dict(color='#00C8FF', width=2), name='SMA 20'), row=1, col=1)
    elif indicator == "EMA 20":
        df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], line=dict(color='#8A2BE2', width=2), name='EMA 20'), row=1, col=1)
    elif indicator == "Bollinger Bands":
        df['SMA20'] = df['Close'].rolling(window=20).mean()
        df['STD20'] = df['Close'].rolling(window=20).std()
        df['Upper'] = df['SMA20'] + (df['STD20'] * 2)
        df['Lower'] = df['SMA20'] - (df['STD20'] * 2)
        fig.add_trace(go.Scatter(x=df.index, y=df['Upper'], line=dict(color='rgba(255,255,255,0.2)'), name='Upper Band'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Lower'], line=dict(color='rgba(255,255,255,0.2)'), fill='tonexty', fillcolor='rgba(0, 200, 255, 0.1)', name='Lower Band'), row=1, col=1)

    # Layout styling for dark mode
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_rangeslider_visible=False,
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
