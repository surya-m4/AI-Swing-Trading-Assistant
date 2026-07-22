"""
Live data processor that combines market fetching, feature engineering,
and model prediction into a single pipeline.
"""
import os
import logging
from typing import Dict, Any, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from src.market_data.indian_market import IndianMarketFetcher
from src.market_data.forex_market import ForexMarketFetcher
from src.features.feature_engineering import FeatureEngineeringPipeline

logger = logging.getLogger(__name__)


class LiveDataProcessor:
    """End-to-end processor: fetch → features → predict.

    This class fetches the latest OHLCV history for a ticker, runs the
    full feature-engineering pipeline, and feeds the resulting feature
    vector into the trained model to produce a trading signal.

    Attributes:
        indian_fetcher: Indian market data fetcher.
        forex_fetcher: Forex market data fetcher.
        feature_pipeline: Feature engineering pipeline.
        model: Trained sklearn model (loaded from disk).
        label_encoder: Fitted label encoder (loaded from disk).
        model_name: Name of the loaded model file.
    """

    def __init__(self, models_dir: str = "models"):
        """Initialises the processor and loads the model.

        Args:
            models_dir: Directory that contains ``*.pkl`` model artefacts.
        """
        self.indian_fetcher = IndianMarketFetcher()
        self.forex_fetcher = ForexMarketFetcher()
        self.feature_pipeline = FeatureEngineeringPipeline()

        self.model = None
        self.label_encoder = None
        self.model_name = "Unknown"
        self._load_model(models_dir)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def _load_model(self, models_dir: str) -> None:
        """Loads the best model and optional label encoder from *models_dir*.

        Args:
            models_dir: Path to the directory containing ``.pkl`` files.
        """
        if not os.path.isdir(models_dir):
            logger.warning(f"Models directory '{models_dir}' not found.")
            return

        # Label encoder
        le_path = os.path.join(models_dir, "label_encoder.pkl")
        if os.path.exists(le_path):
            self.label_encoder = joblib.load(le_path)
            logger.info("Loaded label encoder.")

        # Model – prefer optimised variant
        pkl_files = [
            f for f in os.listdir(models_dir)
            if f.endswith(".pkl") and f != "label_encoder.pkl"
        ]
        if not pkl_files:
            logger.warning("No trained models found.")
            return

        optimized = [f for f in pkl_files if "optimized" in f]
        target = optimized[0] if optimized else pkl_files[0]
        self.model = joblib.load(os.path.join(models_dir, target))
        self.model_name = target.replace(".pkl", "")
        logger.info(f"Loaded model: {self.model_name}")

    # ------------------------------------------------------------------
    # Feature engineering helpers
    # ------------------------------------------------------------------
    def compute_features(self, ticker: str, period: str = "3mo") -> pd.DataFrame:
        """Fetches market data and computes features for *ticker*.

        Args:
            ticker: Yahoo Finance symbol (e.g. ``RELIANCE.NS``).
            period: Historical look-back period passed to yfinance.

        Returns:
            Feature-engineered DataFrame.  Empty on failure.
        """
        # Determine the right fetcher based on ticker suffix
        if ticker.endswith(".NS"):
            df = self.indian_fetcher.fetch_history(ticker, period=period)
        else:
            df = self.forex_fetcher.fetch_history(ticker, period=period)

        if df.empty:
            logger.warning(f"Empty data for {ticker}; cannot compute features.")
            return pd.DataFrame()

        try:
            df = self.feature_pipeline.run(df)
            logger.info(f"Computed {len(df.columns)} features for {ticker}.")
            return df
        except Exception as exc:
            logger.error(f"Feature engineering failed for {ticker}: {exc}")
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, ticker: str, period: str = "3mo") -> Dict[str, Any]:
        """Generates a full prediction payload for *ticker*.

        Steps:
            1. Fetch OHLCV history.
            2. Compute features.
            3. Take the last row (most recent bar).
            4. Run the model.

        Args:
            ticker: Yahoo Finance symbol.
            period: Look-back for history.

        Returns:
            Dictionary containing ``action``, ``confidence``, ``probability``,
            ``expected_return``, ``risk_score``, ``model_name``, and
            ``ticker``.  Returns a dict with ``error`` key on failure.
        """
        if self.model is None:
            return {"error": "Model is not loaded.", "ticker": ticker}

        df = self.compute_features(ticker, period)
        if df.empty:
            return {"error": "No features available.", "ticker": ticker}

        # Use the last row as the prediction input
        exclude_cols = {"Date", "Ticker", "Label", "Future_Close", "Future_Return"}
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        latest_row = df[feature_cols].iloc[[-1]]

        try:
            prediction = self.model.predict(latest_row)[0]

            # Confidence / probability
            confidence = 0.0
            probabilities: Dict[str, float] = {}
            if hasattr(self.model, "predict_proba"):
                proba = self.model.predict_proba(latest_row)[0]
                confidence = float(np.max(proba))
                classes = (
                    self.model.classes_.tolist()
                    if hasattr(self.model, "classes_")
                    else list(range(len(proba)))
                )
                probabilities = {
                    str(c): round(float(p), 4) for c, p in zip(classes, proba)
                }

            # Decode label
            action = str(prediction)
            if self.label_encoder is not None:
                action = self.label_encoder.inverse_transform([prediction])[0]

            # Derived metrics
            close_price = float(df["Close"].iloc[-1])
            atr_value = float(df["ATR"].iloc[-1]) if "ATR" in df.columns else 0.0
            expected_return = round(atr_value / close_price * 100, 2) if close_price else 0.0
            risk_score = round(1.0 - confidence, 4)

            return {
                "ticker": ticker,
                "action": action,
                "confidence": round(confidence, 4),
                "probability": probabilities,
                "expected_return": expected_return,
                "risk_score": risk_score,
                "model_name": self.model_name,
                "close_price": close_price,
            }
        except Exception as exc:
            logger.error(f"Prediction failed for {ticker}: {exc}")
            return {"error": str(exc), "ticker": ticker}
