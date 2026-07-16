"""
Tests for the feature engineering module.
"""

import unittest
import numpy as np
import pandas as pd

from src.features.indicators import TechnicalIndicators
from src.features.candlestick import CandlestickFeatures
from src.features.feature_engineering import FeatureEngineeringPipeline


class TestFeatureEngineering(unittest.TestCase):
    """Test suite for feature engineering components."""

    def setUp(self):
        """Set up test environment and mock data."""
        # Create a sample DataFrame with synthetic price data
        dates = pd.date_range(start="2023-01-01", periods=50)
        np.random.seed(42)
        
        close = np.linspace(10, 20, 50) + np.random.normal(0, 1, 50)
        open_price = close - np.random.normal(0, 0.5, 50)
        high = np.maximum(open_price, close) + np.random.uniform(0, 1, 50)
        low = np.minimum(open_price, close) - np.random.uniform(0, 1, 50)
        volume = np.random.randint(1000, 5000, 50)
        
        self.df = pd.DataFrame({
            "Date": dates,
            "Open": open_price,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume
        })
        
        self.indicators = TechnicalIndicators()
        self.candlestick = CandlestickFeatures()
        self.pipeline = FeatureEngineeringPipeline()

    def test_sma(self):
        """Test Simple Moving Average (SMA)."""
        df_out = self.indicators.add_sma(self.df.copy(), window=5)
        self.assertIn("SMA_5", df_out.columns)
        self.assertTrue(pd.isna(df_out["SMA_5"].iloc[0]))
        self.assertFalse(pd.isna(df_out["SMA_5"].iloc[10]))

    def test_ema(self):
        """Test Exponential Moving Average (EMA)."""
        df_out = self.indicators.add_ema(self.df.copy(), window=5)
        self.assertIn("EMA_5", df_out.columns)
        self.assertTrue(pd.isna(df_out["EMA_5"].iloc[0]))
        self.assertFalse(pd.isna(df_out["EMA_5"].iloc[10]))

    def test_rsi(self):
        """Test Relative Strength Index (RSI)."""
        df_out = self.indicators.add_rsi(self.df.copy(), window=14)
        self.assertIn("RSI_14", df_out.columns)

    def test_macd(self):
        """Test MACD."""
        df_out = self.indicators.add_macd(self.df.copy())
        self.assertIn("MACD", df_out.columns)
        self.assertIn("MACD_Signal", df_out.columns)
        self.assertIn("MACD_Diff", df_out.columns)

    def test_bollinger_bands(self):
        """Test Bollinger Bands."""
        df_out = self.indicators.add_bollinger_bands(self.df.copy())
        self.assertIn("BB_High", df_out.columns)
        self.assertIn("BB_Low", df_out.columns)
        self.assertIn("BB_Mid", df_out.columns)

    def test_atr(self):
        """Test Average True Range (ATR)."""
        df_out = self.indicators.add_atr(self.df.copy())
        self.assertIn("ATR", df_out.columns)

    def test_adx(self):
        """Test Average Directional Index (ADX)."""
        df_out = self.indicators.add_adx(self.df.copy())
        self.assertIn("ADX", df_out.columns)

    def test_roc(self):
        """Test Rate of Change (ROC)."""
        df_out = self.pipeline._create_general_features(self.df.copy())
        self.assertIn("Rate_of_Change", df_out.columns)

    def test_candlestick_features(self):
        """Test candlestick pattern detection and features."""
        df_out = self.candlestick.generate_features(self.df.copy())
        expected_cols = [
            "Candle_Body", "Upper_Wick", "Lower_Wick", "Candle_Range",
            "Bullish_Candle", "Bearish_Candle", "Doji", "Hammer", 
            "Shooting_Star", "Bullish_Engulfing", "Bearish_Engulfing"
        ]
        for col in expected_cols:
            self.assertIn(col, df_out.columns)

    def test_lag_features(self):
        """Test generation of lag features."""
        df_out = self.pipeline._create_lag_features(self.df.copy())
        self.assertIn("Close_Lag_1", df_out.columns)
        self.assertIn("Close_Lag_2", df_out.columns)
        self.assertIn("Close_Lag_3", df_out.columns)
        self.assertIn("Volume_Lag_1", df_out.columns)

    def test_validation_errors_missing_columns(self):
        """Test that missing columns raise validation errors."""
        df_invalid = self.df.drop(columns=["Close"])
        with self.assertRaises(ValueError):
            self.indicators.add_sma(df_invalid)
            
        with self.assertRaises(ValueError):
            self.candlestick.generate_features(df_invalid)
            
        with self.assertRaises(ValueError):
            self.pipeline.run(df_invalid)

    def test_empty_dataframe(self):
        """Test pipeline behavior with an empty DataFrame."""
        df_empty = pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"])
        df_out = self.pipeline.run(df_empty)
        self.assertEqual(len(df_out), 0)

    def test_full_pipeline(self):
        """Test the full feature engineering pipeline execution."""
        df_out = self.pipeline.run(self.df.copy())
        
        # The first few rows should be dropped due to NaNs from rolling/lags
        self.assertLess(len(df_out), len(self.df))
        
        # There should be no NaN values left
        self.assertFalse(df_out.isnull().values.any())
        
        # Check that we have a significant number of columns (base + features)
        self.assertGreater(len(df_out.columns), 20)


if __name__ == "__main__":
    unittest.main()
