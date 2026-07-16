"""
Candlestick features generation module for the Swing Trading Assistant.
"""

import logging
import numpy as np
import pandas as pd

# Configure logger
logger = logging.getLogger(__name__)


class CandlestickFeatures:
    """Class for generating candlestick features from OHLC data."""

    @staticmethod
    def _validate_columns(df: pd.DataFrame) -> None:
        """Validate if required OHLC columns exist in the DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Raises:
            ValueError: If any required columns are missing.
        """
        required_columns = ["Open", "High", "Low", "Close"]
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            error_msg = f"Missing required columns for candlestick features: {missing}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate a variety of candlestick features and patterns.

        Features generated: Candle_Body, Upper_Wick, Lower_Wick, Candle_Range,
        Bullish_Candle, Bearish_Candle, Doji, Hammer, Shooting_Star,
        Bullish_Engulfing, Bearish_Engulfing.

        Args:
            df (pd.DataFrame): Input DataFrame containing OHLC data.

        Returns:
            pd.DataFrame: DataFrame with the new candlestick features added.
        """
        self._validate_columns(df)

        # Basic properties
        df["Candle_Body"] = df["Close"] - df["Open"]
        df["Upper_Wick"] = df["High"] - df[["Open", "Close"]].max(axis=1)
        df["Lower_Wick"] = df[["Open", "Close"]].min(axis=1) - df["Low"]
        df["Candle_Range"] = df["High"] - df["Low"]
        
        # Bullish or Bearish (binary indicator)
        df["Bullish_Candle"] = (df["Close"] > df["Open"]).astype(int)
        df["Bearish_Candle"] = (df["Close"] < df["Open"]).astype(int)

        body_abs = df["Candle_Body"].abs()
        # Avoid division by zero by replacing zero ranges with NaN
        range_valid = df["Candle_Range"].replace(0, np.nan)

        # Doji: The body is very small compared to the total range
        df["Doji"] = (body_abs / range_valid < 0.1).astype(int)

        # Hammer: Small body, long lower wick (at least twice the body), small upper wick
        df["Hammer"] = (
            (df["Lower_Wick"] > (2 * body_abs)) & 
            (df["Upper_Wick"] < body_abs)
        ).astype(int)

        # Shooting Star: Small body, long upper wick (at least twice the body), small lower wick
        df["Shooting_Star"] = (
            (df["Upper_Wick"] > (2 * body_abs)) & 
            (df["Lower_Wick"] < body_abs)
        ).astype(int)

        # Shifted values for 2-candle patterns
        prev_body = df["Candle_Body"].shift(1)
        prev_open = df["Open"].shift(1)
        prev_close = df["Close"].shift(1)
        prev_bearish = df["Bearish_Candle"].shift(1)
        prev_bullish = df["Bullish_Candle"].shift(1)

        # Bullish Engulfing: 
        # Previous candle was bearish, current is bullish, current body engulfs previous body
        df["Bullish_Engulfing"] = (
            (prev_bearish == 1) &
            (df["Bullish_Candle"] == 1) &
            (df["Open"] <= prev_close) &
            (df["Close"] >= prev_open) & 
            (body_abs > prev_body.abs())
        ).astype(int)

        # Bearish Engulfing:
        # Previous candle was bullish, current is bearish, current body engulfs previous body
        df["Bearish_Engulfing"] = (
            (prev_bullish == 1) &
            (df["Bearish_Candle"] == 1) &
            (df["Open"] >= prev_close) &
            (df["Close"] <= prev_open) &
            (body_abs > prev_body.abs())
        ).astype(int)

        return df
