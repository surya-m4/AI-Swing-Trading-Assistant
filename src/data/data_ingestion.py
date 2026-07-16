"""Data ingestion utilities for the Swing Trading Assistant.

Provides functions to load raw market data from CSV, JSON, or external APIs.
"""

import pandas as pd
import os
import yfinance as yf
import logging
from typing import Union

logger = logging.getLogger(__name__)

class DataIngestion:
    """Class to handle downloading, validating, and saving market data."""

    def __init__(self, raw_data_dir: str = "data/raw"):
        """
        Initializes the DataIngestion pipeline.

        Args:
            raw_data_dir (str): Directory where raw data will be saved.
        """
        self.raw_data_dir = raw_data_dir
        os.makedirs(self.raw_data_dir, exist_ok=True)

    def download_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Downloads historical market data using yfinance.

        Args:
            ticker (str): The stock ticker symbol.
            start_date (str): Start date (YYYY-MM-DD).
            end_date (str): End date (YYYY-MM-DD).

        Returns:
            pd.DataFrame: Downloaded data.

        Raises:
            ValueError: If any argument is empty.
            RuntimeError: If no data is returned.
        """
        if not ticker or not start_date or not end_date:
            raise ValueError("Ticker, start_date, and end_date must be provided and non-empty.")
            
        logger.info(f"Downloading data for {ticker} from {start_date} to {end_date}...")
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if df is None or df.empty:
            raise RuntimeError(f"Failed to download data for {ticker}.")
            
        # Flatten MultiIndex columns if present (yfinance >= 0.2.40 often returns MultiIndex)
        if isinstance(df.columns, pd.MultiIndex):
            # Typically level 0 is 'Price' which contains OHLCV, but let's check
            if 'Close' in df.columns.get_level_values(0):
                df.columns = df.columns.get_level_values(0)
            elif 'Close' in df.columns.get_level_values(1):
                df.columns = df.columns.get_level_values(1)
                
        # Ensure the index is named 'Date'
        if df.index.name is None or df.index.name != 'Date':
            df.index.name = 'Date'
            
        return df

    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validates the downloaded DataFrame.

        Args:
            df (pd.DataFrame): Data to validate.

        Returns:
            bool: True if valid, False otherwise.
        """
        if df is None or df.empty:
            logger.error("Validation failed: DataFrame is empty or None.")
            return False
            
        required_cols = {"Open", "High", "Low", "Close"}
        # yfinance columns might be MultiIndex in recent versions, or just normal Index.
        # Check if the intersection of required_cols and columns is complete
        # Handle both single level and multi level columns from yf
        columns = df.columns
        if isinstance(columns, pd.MultiIndex):
            columns = columns.get_level_values(0)
            
        if not required_cols.issubset(set(columns)):
            logger.error(f"Validation failed: Missing required columns. Found: {list(columns)}")
            return False
            
        return True

    def save_data(self, df: pd.DataFrame, path: str):
        """
        Saves the DataFrame to a CSV file.

        Args:
            df (pd.DataFrame): The data to save.
            path (str): The file path to save to.
        """
        logger.info(f"Saving data to {path}...")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path)

def load_csv(path: Union[str, os.PathLike]) -> pd.DataFrame:
    """Load a CSV file into a DataFrame.

    Args:
        path: Path to the CSV file.
    Returns:
        pandas.DataFrame with the loaded data.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV file not found: {path}")
    return pd.read_csv(path)
