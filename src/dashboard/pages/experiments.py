import streamlit as st
import os
import json

def render():
    st.markdown("<h2 class='neon-purple'>MLflow Experiments</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#94A3B8;'>Tracking machine learning models, hyperparameters, and metrics.</p>", unsafe_allow_html=True)
    
    st.markdown("<div class='glass-card fade-in'>", unsafe_allow_html=True)
    st.markdown("### Best Model Pipeline")
    
    # Read optimization results if available
    results_path = os.path.join('artifacts', 'optimization_results.json')
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            results = json.load(f)
            
        st.json(results, expanded=True)
    else:
        st.info("No optimization results found. Run the training pipeline first.")
        
    st.markdown("</div>", unsafe_allow_html=True)
