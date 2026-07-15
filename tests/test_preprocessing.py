"""
Test script for verifying DataPreprocessor class functionality.
"""

import logging
import os
import unittest
import pandas as pd
import numpy as np
from src.data.preprocessing import DataPreprocessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


class TestDataPreprocessor(unittest.TestCase):
    """Test suite for DataPreprocessor."""

    def setUp(self):
        self.preprocessor = DataPreprocessor()
        self.test_dir = "data/test_temp"
        os.makedirs(self.test_dir, exist_ok=True)
        self.raw_csv_path = os.path.join(self.test_dir, "raw_data.csv")
        self.processed_csv_path = os.path.join(self.test_dir, "processed_data.csv")

        # Create a mock dataframe for testing pipeline components
        self.mock_data = pd.DataFrame({
            "Date": ["2023-01-03", "2023-01-01", "2023-01-02", "2023-01-02"],  # out of order, duplicate
            "Open": [10.0, 12.0, 11.0, 11.0],
            "High": [12.0, 13.0, 12.0, 12.0],
            "Low": [9.0, 11.0, 10.0, 10.0],
            "Close": [11.0, 12.5, 11.5, 11.5],
            "Volume": [1000, 1500, 1200, 1200]
        })

    def tearDown(self):
        # Clean up files created
        for file in [self.raw_csv_path, self.processed_csv_path]:
            if os.path.exists(file):
                os.remove(file)
        if os.path.exists(self.test_dir):
            try:
                os.rmdir(self.test_dir)
            except OSError:
                pass

    def test_load_data_success(self):
        logger.info("Running test_load_data_success...")
        self.mock_data.to_csv(self.raw_csv_path, index=False)
        df = self.preprocessor.load_data(self.raw_csv_path)
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 4)

    def test_load_data_file_not_found(self):
        logger.info("Running test_load_data_file_not_found...")
        with self.assertRaises(FileNotFoundError):
            self.preprocessor.load_data("non_existent_file_path_xyz.csv")

    def test_validate_columns_success(self):
        logger.info("Running test_validate_columns_success...")
        # Columns match requirements
        self.preprocessor.validate_columns(self.mock_data)

    def test_validate_columns_failure(self):
        logger.info("Running test_validate_columns_failure...")
        invalid_data = self.mock_data.drop(columns=["Volume"])
        with self.assertRaises(ValueError):
            self.preprocessor.validate_columns(invalid_data)

    def test_convert_datatypes_success(self):
        logger.info("Running test_convert_datatypes_success...")
        # Create dataframe with strings in place of numbers or timestamps
        str_data = pd.DataFrame({
            "Date": ["2023-01-01"],
            "Open": ["10.5"],
            "High": ["11.2"],
            "Low": ["9.8"],
            "Close": ["10.9"],
            "Volume": ["5000"]
        })
        converted = self.preprocessor.convert_datatypes(str_data)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(converted["Date"]))
        self.assertTrue(pd.api.types.is_numeric_dtype(converted["Open"]))
        self.assertTrue(pd.api.types.is_numeric_dtype(converted["Volume"]))

    def test_convert_datatypes_failure(self):
        logger.info("Running test_convert_datatypes_failure...")
        invalid_data = pd.DataFrame({
            "Date": ["2023-01-01"],
            "Open": ["not-a-number"],
            "High": [11.2],
            "Low": [9.8],
            "Close": [10.9],
            "Volume": [5000]
        })
        with self.assertRaises(ValueError):
            self.preprocessor.convert_datatypes(invalid_data)

    def test_clean_missing_values(self):
        logger.info("Running test_clean_missing_values...")
        nan_data = pd.DataFrame({
            "Date": ["2023-01-01", "2023-01-02"],
            "Open": [10.0, np.nan],  # NaN Open
            "High": [11.0, 12.0],
            "Low": [9.0, 10.0],
            "Close": [10.5, 11.5],
            "Volume": [1000, 1200]
        })
        cleaned = self.preprocessor.clean_missing_values(nan_data)
        self.assertEqual(len(cleaned), 1)
        self.assertFalse(cleaned.isnull().values.any())

    def test_remove_duplicates(self):
        logger.info("Running test_remove_duplicates...")
        cleaned = self.preprocessor.remove_duplicates(self.mock_data)
        self.assertEqual(len(cleaned), 3)  # One duplicate row removed

    def test_validate_prices(self):
        logger.info("Running test_validate_prices...")
        bad_rows = pd.DataFrame({
            "Date": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05", "2023-01-06", "2023-01-07"],
            # High < Low
            "Open": [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            "High": [9.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0],
            "Low": [11.0, 9.0, 9.0, 11.0, 9.0, 9.0, 9.0],
            "Close": [10.5, 10.5, 10.5, 10.5, 10.5, 10.5, 10.5],
            "Volume": [100, 100, 100, 100, 100, 100, 100]
        })
        
        # Test case: High < Open
        bad_rows.loc[1, "High"] = 8.0 # High (8) < Open (10)
        
        # Test case: High < Close
        bad_rows.loc[2, "High"] = 8.0 # High (8) < Close (10.5)
        
        # Test case: Low > Open
        bad_rows.loc[3, "Low"] = 11.0 # Low (11) > Open (10)
        
        # Test case: Low > Close
        bad_rows.loc[4, "Low"] = 11.0 # Low (11) > Close (10.5)
        
        # Test case: Volume < 0
        bad_rows.loc[5, "Volume"] = -50
        
        # Row 6 is valid
        # Let's run price validation
        cleaned = self.preprocessor.validate_prices(bad_rows)
        self.assertEqual(len(cleaned), 1)  # Only row 6 should be left
        self.assertEqual(cleaned.iloc[0]["Date"], "2023-01-07")

    def test_sort_by_date(self):
        logger.info("Running test_sort_by_date...")
        df = self.mock_data.copy()
        df["Date"] = pd.to_datetime(df["Date"])
        sorted_df = self.preprocessor.sort_by_date(df)
        
        # Verify order of dates
        self.assertEqual(sorted_df.iloc[0]["Date"], pd.Timestamp("2023-01-01"))
        self.assertEqual(sorted_df.iloc[1]["Date"], pd.Timestamp("2023-01-02"))
        
        # Verify index was reset (should be 0, 1, 2...)
        self.assertEqual(sorted_df.index[0], 0)

    def test_save_and_load_processed(self):
        logger.info("Running test_save_and_load_processed...")
        self.preprocessor.save_processed_data(self.mock_data, self.processed_csv_path)
        self.assertTrue(os.path.exists(self.processed_csv_path))
        
        loaded = pd.read_csv(self.processed_csv_path)
        self.assertEqual(len(loaded), len(self.mock_data))

    def test_full_preprocessing_pipeline(self):
        logger.info("Running test_full_preprocessing_pipeline...")
        # Create dataset containing NaN, duplicate, unsorted date, invalid price
        dirty_data = pd.DataFrame({
            "Date": ["2023-01-03", "2023-01-01", "2023-01-02", "2023-01-02", "2023-01-04", "2023-01-05"],
            "Open": [10.0, 12.0, 11.0, 11.0, np.nan, 10.0],
            "High": [12.0, 13.0, 12.0, 12.0, 12.0, 9.0],  # 2023-01-05 High < Open
            "Low": [9.0, 11.0, 10.0, 10.0, 10.0, 8.0],
            "Close": [11.0, 12.5, 11.5, 11.5, 11.0, 8.5],
            "Volume": [1000, 1500, 1200, 1200, 1300, 100]
        })
        
        dirty_data.to_csv(self.raw_csv_path, index=False)
        
        # Run orchestrator
        cleaned = self.preprocessor.preprocess(self.raw_csv_path, self.processed_csv_path)
        
        # Expecting:
        # - Row with NaN Open (2023-01-04) removed
        # - Duplicate row (2023-01-02 duplicate) removed
        # - Row with invalid price (2023-01-05 High (9) < Open (10)) removed
        # - Sorted ascending: 2023-01-01, 2023-01-02, 2023-01-03
        self.assertEqual(len(cleaned), 3)
        self.assertEqual(cleaned.iloc[0]["Date"], pd.Timestamp("2023-01-01"))
        self.assertEqual(cleaned.iloc[1]["Date"], pd.Timestamp("2023-01-02"))
        self.assertEqual(cleaned.iloc[2]["Date"], pd.Timestamp("2023-01-03"))
        
        # Check CSV saved correctly
        self.assertTrue(os.path.exists(self.processed_csv_path))


if __name__ == "__main__":
    unittest.main()
