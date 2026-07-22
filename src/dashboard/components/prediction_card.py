"""
AI prediction card component with live data support.

Displays real model predictions when available, gracefully falling
back to a placeholder when the model is not loaded.
"""

import streamlit as st
from typing import Any, Dict, Optional


def _get_live_prediction(ticker: str = "RELIANCE.NS") -> Optional[Dict[str, Any]]:
    """Attempts to fetch a live prediction from the LiveDataProcessor.

    Args:
        ticker: Ticker symbol to predict on.

    Returns:
        Prediction dict or ``None`` on failure.
    """
    try:
        from src.market_data.live_data import LiveDataProcessor

        processor = LiveDataProcessor()
        if processor.model is not None:
            return processor.predict(ticker)
    except Exception:
        pass
    return None


def render_prediction_card(prediction: Optional[Dict[str, Any]] = None, ticker: str = "RELIANCE.NS") -> None:
    """Renders the AI prediction glowing card.

    Uses *prediction* if provided, otherwise tries to fetch a live
    prediction.  Falls back to a placeholder when no model is loaded.

    Args:
        prediction: Optional pre-computed prediction dictionary.
        ticker: Ticker symbol for the prediction.
    """
    pred = prediction
    if pred is None:
        pred = _get_live_prediction(ticker)

    # Extract values (with fallbacks for placeholder display)
    if pred and "error" not in pred:
        action = pred.get("action", "HOLD")
        confidence = pred.get("confidence", 0.0)
        risk_score = pred.get("risk_score", 0.5)
        expected_return = pred.get("expected_return", 0.0)
        close_price = pred.get("close_price", 0.0)
        model_name = pred.get("model_name", "AI Model")
        pred_ticker = pred.get("ticker", ticker)

        # Risk level from score
        if risk_score < 0.3:
            risk_level = "LOW"
            risk_color = "#00FF7F"
        elif risk_score < 0.6:
            risk_level = "MEDIUM"
            risk_color = "#FFA500"
        else:
            risk_level = "HIGH"
            risk_color = "#FF3333"

        # Action color
        action_color = {
            "BUY": "#00FF7F",
            "SELL": "#FF3333",
            "HOLD": "#FFA500",
        }.get(action.upper(), "#FFF")

        action_class = {
            "BUY": "neon-green",
            "SELL": "",
            "HOLD": "",
        }.get(action.upper(), "")

    else:
        action = "HOLD"
        confidence = 0.0
        risk_level = "N/A"
        risk_color = "#94A3B8"
        expected_return = 0.0
        close_price = 0.0
        model_name = "Model not loaded"
        pred_ticker = ticker
        action_color = "#FFA500"
        action_class = ""

    # Confidence as percentage
    conf_pct = f"{confidence * 100:.1f}%" if confidence else "N/A"

    st.markdown(
        f"""
        <div class="glass-card gradient-border fade-in">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <div style="color: #94A3B8; font-size: 0.9rem;">Model Output</div>
                    <div class="{action_class}" style="font-size: 3.5rem; font-weight: 800; line-height: 1; color: {action_color};">{action.upper()}</div>
                    <div style="color: {action_color}; font-weight: 600;">{pred_ticker}</div>
                </div>
                <div style="text-align: right;">
                    <div style="color: #94A3B8; font-size: 0.9rem;">Confidence</div>
                    <div style="font-size: 2.5rem; font-weight: bold; color: #FFF;">{conf_pct}</div>
                </div>
            </div>
            
            <hr style="border-color: rgba(255,255,255,0.1);">
            
            <div style="display: flex; justify-content: space-between;">
                <div>
                    <div style="color: #94A3B8; font-size: 0.8rem;">Risk Level</div>
                    <div style="color: {risk_color}; font-weight: bold;">{risk_level}</div>
                </div>
                <div>
                    <div style="color: #94A3B8; font-size: 0.8rem;">Expected Return</div>
                    <div style="color: #FFF; font-weight: bold;">{expected_return:+.2f}%</div>
                </div>
                <div>
                    <div style="color: #94A3B8; font-size: 0.8rem;">Close Price</div>
                    <div style="color: #00C8FF; font-weight: bold;">{close_price:,.2f}</div>
                </div>
                <div>
                    <div style="color: #94A3B8; font-size: 0.8rem;">Model</div>
                    <div style="color: #E2E8F0; font-weight: bold; font-size: 0.85rem;">{model_name}</div>
                </div>
            </div>
        </div>
    """,
        unsafe_allow_html=True,
    )
