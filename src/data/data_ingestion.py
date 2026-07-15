"""
Data Ingestion Module.

This module provides the DataIngestion class which downloads historical stock
data using the yfinance API, validates the downloaded data, and saves it to a
specified CSV file path.
"""

import logging
import os
from typing import Union
import pandas as pd
import yfinance as yf

# Configure logger for this module
logger = logging.getLogger(__name__)


class DataIngestion:
    """
    A class to handle downloading, validating, and saving historical stock data.

    Attributes:
        raw_data_dir (str): Default directory where raw data files will be
                            saved.
    """

    def __init__(self, raw_data_dir: str = "data/raw") -> None:
        """
        Initializes the DataIngestion helper.

        Args:
            raw_data_dir (str): Directory path to save raw data.
                                Defaults to 'data/raw'.
        """
        self.raw_data_dir = raw_data_dir
        try:
            os.makedirs(self.raw_data_dir, exist_ok=True)
            logger.info(
                f"DataIngestion initialized. Raw data directory: {raw_data_dir}"
            )
        except Exception as e:
            logger.error(
                f"Failed to create raw data directory {raw_data_dir}: {e}"
            )
            raise OSError(
                f"Could not create directory {raw_data_dir}: {e}"
            ) from e

    def download_data(
        self, ticker: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """
        Downloads historical stock data for a given ticker from Yahoo Finance.

        Args:
            ticker (str): The stock symbol (e.g., 'AAPL', 'MSFT').
            start_date (str): Start date for data in 'YYYY-MM-DD' format.
            end_date (str): End date for data in 'YYYY-MM-DD' format.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the historical stock
                          data.

        Raises:
            ValueError: If inputs are invalid (empty ticker or date strings).
            RuntimeError: If yfinance download fails or returns empty data.
        """
        if not ticker or not ticker.strip():
            logger.error("Download failed: Ticker symbol is empty or invalid.")
            raise ValueError("Ticker symbol cannot be empty.")
        if not start_date or not start_date.strip():
            logger.error("Download failed: Start date is empty or invalid.")
            raise ValueError("Start date cannot be empty.")
        if not end_date or not end_date.strip():
            logger.error("Download failed: End date is empty or invalid.")
            raise ValueError("End date cannot be empty.")

        ticker = ticker.strip().upper()
        logger.info(
            f"Downloading historical data for '{ticker}' "
            f"from {start_date} to {end_date}..."
        )

        try:
            # Download data using yfinance
            data = yf.download(
                ticker, start=start_date, end=end_date, progress=False
            )
        except Exception as e:
            logger.exception(
                f"Exception occurred while downloading ticker '{ticker}': {e}"
            )
            raise RuntimeError(
                f"Error downloading data for ticker '{ticker}': {e}"
            ) from e

        if data is None or data.empty:
            logger.error(
                f"No data returned for ticker '{ticker}' "
                f"between {start_date} and {end_date}."
            )
            raise RuntimeError(
                f"Downloaded DataFrame is empty for ticker '{ticker}' "
                f"between {start_date} and {end_date}."
            )

        # Flatten MultiIndex columns if present (yfinance return format)
        if isinstance(data.columns, pd.MultiIndex):
            if "Price" in data.columns.names:
                data.columns = data.columns.get_level_values("Price")
            else:
                data.columns = data.columns.get_level_values(0)

        logger.info(
            f"Successfully downloaded {len(data)} rows for ticker '{ticker}'."
        )
        return data

    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validates the downloaded historical stock data.

        Checks:
        1. Whether the DataFrame is not empty.
        2. Whether the required columns ('Open', 'High', 'Low', 'Close',
           'Volume') exist in the DataFrame columns.

        Args:
            data (pd.DataFrame): The DataFrame to validate.

        Returns:
            bool: True if validation succeeds, False otherwise.
        """
        if data is None:
            logger.error("Validation failed: DataFrame is None.")
            return False

        if data.empty:
            logger.error("Validation failed: DataFrame is empty.")
            return False

        # Extract column names, handling standard Index or MultiIndex (yfinance format)
        if isinstance(data.columns, pd.MultiIndex):
            columns = [col[0] for col in data.columns]
        else:
            columns = list(data.columns)

        required_columns = ["Open", "High", "Low", "Close", "Volume"]
        missing_columns = [col for col in required_columns if col not in columns]

        if missing_columns:
            logger.error(
                f"Validation failed: Missing required columns: {missing_columns}"
            )
            return False

        # Verify that there are actual data rows
        if len(data) == 0:
            logger.error("Validation failed: DataFrame contains zero rows.")
            return False

        logger.info("Validation successful: All checks passed.")
        return True

    def save_data(
        self, data: pd.DataFrame, output_path: Union[str, os.PathLike]
    ) -> None:
        """
        Saves the stock data DataFrame to a CSV file.

        Args:
            data (pd.DataFrame): The DataFrame to save.
            output_path (Union[str, os.PathLike]): The complete path where the
                                                   CSV file should be saved.

        Raises:
            ValueError: If the DataFrame is empty or the output path is invalid.
            IOError: If saving the file to the file system fails.
        """
        if data is None or data.empty:
            logger.error("Save failed: DataFrame is None or empty.")
            raise ValueError("Cannot save an empty or None DataFrame.")

        if not output_path:
            logger.error("Save failed: Output path is empty.")
            raise ValueError("Output path cannot be empty.")

        output_path_str = str(output_path)
        logger.info(f"Saving data to '{output_path_str}'...")

        try:
            # Ensure the directory of output_path exists
            directory = os.path.dirname(output_path_str)
            if directory:
                os.makedirs(directory, exist_ok=True)

            data.to_csv(output_path_str)
            logger.info(f"Data successfully saved to '{output_path_str}'.")
        except Exception as e:
            logger.exception(
                f"Exception occurred while saving data to '{output_path_str}': {e}"
            )
            raise IOError(
                f"Failed to save data to '{output_path_str}': {e}"
            ) from e
