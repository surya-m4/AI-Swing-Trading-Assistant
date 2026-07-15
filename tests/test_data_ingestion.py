"""
Test script for verifying DataIngestion class functionality.
"""

import logging
import os
import unittest
import pandas as pd
from src.data.data_ingestion import DataIngestion

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


class TestDataIngestion(unittest.TestCase):
    """Test suite for DataIngestion."""

    def setUp(self):
        self.test_raw_dir = "data/raw_test"
        self.ingestor = DataIngestion(raw_data_dir=self.test_raw_dir)
        self.test_ticker = "AAPL"
        self.start_date = "2023-01-01"
        self.end_date = "2023-01-10"
        self.output_path = os.path.join(self.test_raw_dir, f"{self.test_ticker}.csv")

    def tearDown(self):
        # Clean up created files
        if os.path.exists(self.output_path):
            os.remove(self.output_path)
        if os.path.exists(self.test_raw_dir):
            try:
                os.rmdir(self.test_raw_dir)
            except OSError:
                pass  # Directory might not be empty or already deleted

    def test_download_success(self):
        logger.info("Running test_download_success...")
        df = self.ingestor.download_data(
            self.test_ticker, self.start_date, self.end_date
        )
        self.assertIsNotNone(df)
        self.assertFalse(df.empty)
        logger.info(f"Downloaded columns: {df.columns}")

    def test_validate_success(self):
        logger.info("Running test_validate_success...")
        df = self.ingestor.download_data(
            self.test_ticker, self.start_date, self.end_date
        )
        is_valid = self.ingestor.validate_data(df)
        self.assertTrue(is_valid)

    def test_validate_failure_empty(self):
        logger.info("Running test_validate_failure_empty...")
        empty_df = pd.DataFrame()
        is_valid = self.ingestor.validate_data(empty_df)
        self.assertFalse(is_valid)

        is_valid_none = self.ingestor.validate_data(None)
        self.assertFalse(is_valid_none)

    def test_validate_failure_missing_cols(self):
        logger.info("Running test_validate_failure_missing_cols...")
        # Create DataFrame missing required columns
        df = pd.DataFrame({"Open": [1.0], "Close": [2.0]})
        is_valid = self.ingestor.validate_data(df)
        self.assertFalse(is_valid)

    def test_save_and_verify(self):
        logger.info("Running test_save_and_verify...")
        df = self.ingestor.download_data(
            self.test_ticker, self.start_date, self.end_date
        )
        
        self.ingestor.save_data(df, self.output_path)
        self.assertTrue(os.path.exists(self.output_path))
        
        # Load and verify it is not empty
        saved_df = pd.read_csv(self.output_path)
        self.assertFalse(saved_df.empty)
        
        # Check that Date column (which was the index) or similar index exists in the CSV
        self.assertIn("Date", saved_df.columns)

    def test_invalid_ticker_raises_runtime_error(self):
        logger.info("Running test_invalid_ticker_raises_runtime_error...")
        # yfinance often returns empty data for garbage tickers instead of raising errors.
        # Our class should detect the empty data and raise RuntimeError.
        with self.assertRaises(RuntimeError):
            self.ingestor.download_data("INVALID_TICKER_XYZ", "2023-01-01", "2023-01-05")

    def test_empty_arguments_raise_value_error(self):
        logger.info("Running test_empty_arguments_raise_value_error...")
        with self.assertRaises(ValueError):
            self.ingestor.download_data("", "2023-01-01", "2023-01-05")
        with self.assertRaises(ValueError):
            self.ingestor.download_data("AAPL", "", "2023-01-05")
        with self.assertRaises(ValueError):
            self.ingestor.download_data("AAPL", "2023-01-01", "")


if __name__ == "__main__":
    unittest.main()
