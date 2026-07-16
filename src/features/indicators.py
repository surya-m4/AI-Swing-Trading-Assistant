"""
Technical indicators calculation module for the Swing Trading Assistant.
"""

import logging
from typing import List

import pandas as pd
import ta

# Configure logger
logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """Class for calculating technical indicators."""

    @staticmethod
    def _validate_columns(df: pd.DataFrame, required_columns: List[str]) -> None:
        """Validate if required columns exist in the DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame.
            required_columns (List[str]): List of required column names.

        Raises:
            ValueError: If any of the required columns are missing.
        """
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            error_msg = f"Missing required columns for indicator: {missing}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def add_sma(self, df: pd.DataFrame, window: int = 20, column: str = "Close") -> pd.DataFrame:
        """Add Simple Moving Average (SMA) feature to the DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame.
            window (int): Moving average window size.
            column (str): Target column for calculation.

        Returns:
            pd.DataFrame: DataFrame with the SMA column added.
        """
        self._validate_columns(df, [column])
        df[f"SMA_{window}"] = ta.trend.sma_indicator(df[column], window=window)
        return df

    def add_ema(self, df: pd.DataFrame, window: int = 20, column: str = "Close") -> pd.DataFrame:
        """Add Exponential Moving Average (EMA) feature to the DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame.
            window (int): Moving average window size.
            column (str): Target column for calculation.

        Returns:
            pd.DataFrame: DataFrame with the EMA column added.
        """
        self._validate_columns(df, [column])
        df[f"EMA_{window}"] = ta.trend.ema_indicator(df[column], window=window)
        return df

    def add_rsi(self, df: pd.DataFrame, window: int = 14, column: str = "Close") -> pd.DataFrame:
        """Add Relative Strength Index (RSI) feature to the DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame.
            window (int): RSI window size.
            column (str): Target column for calculation.

        Returns:
            pd.DataFrame: DataFrame with the RSI column added.
        """
        self._validate_columns(df, [column])
        df[f"RSI_{window}"] = ta.momentum.rsi(df[column], window=window)
        return df

    def add_macd(self, df: pd.DataFrame, window_slow: int = 26, window_fast: int = 12, window_sign: int = 9, column: str = "Close") -> pd.DataFrame:
        """Add Moving Average Convergence Divergence (MACD) features.

        Args:
            df (pd.DataFrame): Input DataFrame.
            window_slow (int): Slow EMA window.
            window_fast (int): Fast EMA window.
            window_sign (int): Signal line window.
            column (str): Target column for calculation.

        Returns:
            pd.DataFrame: DataFrame with MACD, MACD_Signal, and MACD_Diff columns.
        """
        self._validate_columns(df, [column])
        macd = ta.trend.MACD(close=df[column], window_slow=window_slow, window_fast=window_fast, window_sign=window_sign)
        df["MACD"] = macd.macd()
        df["MACD_Signal"] = macd.macd_signal()
        df["MACD_Diff"] = macd.macd_diff()
        return df

    def add_bollinger_bands(self, df: pd.DataFrame, window: int = 20, window_dev: int = 2, column: str = "Close") -> pd.DataFrame:
        """Add Bollinger Bands features.

        Args:
            df (pd.DataFrame): Input DataFrame.
            window (int): Moving average window size.
            window_dev (int): Number of standard deviations.
            column (str): Target column for calculation.

        Returns:
            pd.DataFrame: DataFrame with BB_High, BB_Low, and BB_Mid columns.
        """
        self._validate_columns(df, [column])
        bb = ta.volatility.BollingerBands(close=df[column], window=window, window_dev=window_dev)
        df["BB_High"] = bb.bollinger_hband()
        df["BB_Low"] = bb.bollinger_lband()
        df["BB_Mid"] = bb.bollinger_mavg()
        return df

    def add_atr(self, df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """Add Average True Range (ATR) feature.

        Args:
            df (pd.DataFrame): Input DataFrame containing High, Low, and Close.
            window (int): ATR window size.

        Returns:
            pd.DataFrame: DataFrame with the ATR column added.
        """
        self._validate_columns(df, ["High", "Low", "Close"])
        df["ATR"] = ta.volatility.average_true_range(high=df["High"], low=df["Low"], close=df["Close"], window=window)
        return df

    def add_adx(self, df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """Add Average Directional Index (ADX) feature.

        Args:
            df (pd.DataFrame): Input DataFrame containing High, Low, and Close.
            window (int): ADX window size.

        Returns:
            pd.DataFrame: DataFrame with the ADX column added.
        """
        self._validate_columns(df, ["High", "Low", "Close"])
        df["ADX"] = ta.trend.adx(high=df["High"], low=df["Low"], close=df["Close"], window=window)
        return df

    def add_stochastic_oscillator(self, df: pd.DataFrame, window: int = 14, smooth_window: int = 3) -> pd.DataFrame:
        """Add Stochastic Oscillator features.

        Args:
            df (pd.DataFrame): Input DataFrame.
            window (int): Lookback window.
            smooth_window (int): Smoothing window.

        Returns:
            pd.DataFrame: DataFrame with Stoch_K and Stoch_D columns.
        """
        self._validate_columns(df, ["High", "Low", "Close"])
        stoch = ta.momentum.StochasticOscillator(high=df["High"], low=df["Low"], close=df["Close"], window=window, smooth_window=smooth_window)
        df["Stoch_K"] = stoch.stoch()
        df["Stoch_D"] = stoch.stoch_signal()
        return df

    def add_cci(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """Add Commodity Channel Index (CCI) feature.

        Args:
            df (pd.DataFrame): Input DataFrame containing High, Low, and Close.
            window (int): CCI window size.

        Returns:
            pd.DataFrame: DataFrame with the CCI column added.
        """
        self._validate_columns(df, ["High", "Low", "Close"])
        df["CCI"] = ta.trend.cci(high=df["High"], low=df["Low"], close=df["Close"], window=window)
        return df

    def add_obv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add On Balance Volume (OBV) feature.

        Args:
            df (pd.DataFrame): Input DataFrame containing Close and Volume.

        Returns:
            pd.DataFrame: DataFrame with the OBV column added.
        """
        self._validate_columns(df, ["Close", "Volume"])
        df["OBV"] = ta.volume.on_balance_volume(close=df["Close"], volume=df["Volume"])
        return df
