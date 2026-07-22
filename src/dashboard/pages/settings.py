import streamlit as st

def render():
    st.markdown("<h2 class='neon-blue'>Settings</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='glass-card fade-in'>", unsafe_allow_html=True)
        st.markdown("### Trading Parameters")
        st.number_input("Total Trading Capital (₹)", value=1000000)
        st.slider("Max Risk per Trade (%)", min_value=0.5, max_value=5.0, value=2.0, step=0.1)
        st.selectbox("Default Benchmark", ["NIFTY 50", "BANKNIFTY", "S&P 500"])
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        st.markdown("<div class='glass-card fade-in'>", unsafe_allow_html=True)
        st.markdown("### API Integration")
        st.text_input("Broker API Key", type="password")
        st.text_input("Broker API Secret", type="password")
        st.toggle("Enable Live Trading Execution", value=False)
        st.toggle("Enable Email Notifications", value=True)
        
        st.button("Save Settings", type="primary")
        st.markdown("</div>", unsafe_allow_html=True)
