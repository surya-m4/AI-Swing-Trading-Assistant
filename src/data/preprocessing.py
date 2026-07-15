"""
Data Preprocessing Module.

This module provides the DataPreprocessor class which loads raw stock data,
applies quality validation and cleaning steps, sorts by date, and saves the
cleaned datasets to data/processed/.
"""

import logging
import os
from typing import Dict, Union
import pandas as pd

# Configure logger for this module
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    A class to handle loading, cleaning, validation, and saving stock data.
    """

    def __init__(self) -> None:
        """Initializes the DataPreprocessor."""
        logger.info("DataPreprocessor initialized.")

    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Loads CSV file containing historical stock data.

        Args:
            file_path (str): Path to the CSV file.

        Returns:
            pd.DataFrame: Loaded stock data.

        Raises:
            FileNotFoundError: If the CSV file does not exist.
            IOError: If reading the CSV file fails.
        """
        if not file_path:
            raise ValueError("File path cannot be empty.")

        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found at: {file_path}")

        logger.info(f"Loading data from '{file_path}'...")
        try:
            data = pd.read_csv(file_path)
            logger.info(f"Successfully loaded {len(data)} rows.")
            return data
        except Exception as e:
            logger.exception(f"Failed to read CSV file '{file_path}': {e}")
            raise IOError(f"Error reading CSV file '{file_path}': {e}") from e

    def validate_columns(self, data: pd.DataFrame) -> None:
        """
        Validates that all required columns are present in the DataFrame.

        Required columns: Date, Open, High, Low, Close, Volume

        Args:
            data (pd.DataFrame): The DataFrame to validate.

        Raises:
            ValueError: If any required column is missing.
        """
        if data is None:
            raise ValueError("DataFrame is None.")

        required_columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
        missing_columns = [col for col in required_columns if col not in data.columns]

        if missing_columns:
            logger.error(f"Column validation failed. Missing: {missing_columns}")
            raise ValueError(f"Missing required columns: {missing_columns}")

        logger.info("Column validation passed.")

    def convert_datatypes(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Converts column datatypes to correct formats.

        Converts 'Date' to datetime and numeric columns to float/int.

        Args:
            data (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with updated datatypes.

        Raises:
            ValueError: If conversion fails.
        """
        logger.info("Converting column datatypes...")
        df = data.copy()
        try:
            df["Date"] = pd.to_datetime(df["Date"])
        except Exception as e:
            logger.error(f"Failed to convert 'Date' column to datetime: {e}")
            raise ValueError(f"Failed to convert 'Date' column to datetime: {e}") from e

        numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
        for col in numeric_cols:
            try:
                df[col] = pd.to_numeric(df[col], errors="raise")
            except Exception as e:
                logger.error(f"Failed to convert column '{col}' to numeric: {e}")
                raise ValueError(f"Failed to convert column '{col}' to numeric: {e}") from e

        logger.info("Datatype conversion successful.")
        return df

    def clean_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Removes rows with missing values in required columns.

        Args:
            data (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: Cleaned DataFrame.
        """
        logger.info("Cleaning missing values...")
        required_columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
        before_count = len(data)
        
        # Drop rows with NaNs in required columns
        cleaned_df = data.dropna(subset=required_columns)
        after_count = len(cleaned_df)
        
        removed = before_count - after_count
        logger.info(f"Removed {removed} rows with missing values.")
        return cleaned_df

    def remove_duplicates(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Removes duplicate rows.

        Args:
            data (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: Cleaned DataFrame.
        """
        logger.info("Removing duplicate rows...")
        before_count = len(data)
        cleaned_df = data.drop_duplicates()
        after_count = len(cleaned_df)
        
        removed = before_count - after_count
        logger.info(f"Removed {removed} duplicate rows.")
        return cleaned_df

    def validate_prices(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Removes rows with invalid price structures or negative volume.

        Validation rules:
        - High >= Low
        - High >= Open
        - High >= Close
        - Low <= Open
        - Low <= Close
        - Volume >= 0

        Args:
            data (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: Cleaned DataFrame.
        """
        logger.info("Validating prices and volumes...")
        before_count = len(data)

        # Build mask for valid rows
        valid_mask = (
            (data["High"] >= data["Low"])
            & (data["High"] >= data["Open"])
            & (data["High"] >= data["Close"])
            & (data["Low"] <= data["Open"])
            & (data["Low"] <= data["Close"])
            & (data["Volume"] >= 0)
        )

        cleaned_df = data[valid_mask].copy()
        after_count = len(cleaned_df)
        
        removed = before_count - after_count
        if removed > 0:
            logger.warning(f"Removed {removed} rows with invalid prices or volume.")
        else:
            logger.info("No invalid price/volume rows found.")
        return cleaned_df

    def sort_by_date(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Sorts the DataFrame by Date and resets the index.

        Args:
            data (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: Sorted DataFrame with a reset index.
        """
        logger.info("Sorting data by Date and resetting index...")
        sorted_df = data.sort_values(by="Date", ascending=True)
        sorted_df = sorted_df.reset_index(drop=True)
        return sorted_df

    def generate_quality_report(
        self,
        before_rows: int,
        after_rows: int,
        duplicates_removed: int,
        missing_removed: int,
        invalid_removed: int,
    ) -> Dict[str, Union[int, float]]:
        """
        Generates and logs a dictionary reporting the metrics of data quality.

        Args:
            before_rows (int): Original row count before preprocessing.
            after_rows (int): Cleaned row count after preprocessing.
            duplicates_removed (int): Number of duplicate rows removed.
            missing_removed (int): Number of missing value rows removed.
            invalid_removed (int): Number of invalid price rows removed.

        Returns:
            Dict[str, Union[int, float]]: A dictionary with the metrics.
        """
        retention_rate = (
            (after_rows / before_rows) * 100 if before_rows > 0 else 0.0
        )
        report = {
            "initial_rows": before_rows,
            "final_rows": after_rows,
            "duplicates_removed": duplicates_removed,
            "missing_removed": missing_removed,
            "invalid_removed": invalid_removed,
            "retention_rate_pct": round(retention_rate, 2),
        }

        logger.info("=== Data Quality Report ===")
        logger.info(f"Initial rows: {before_rows}")
        logger.info(f"Duplicates removed: {duplicates_removed}")
        logger.info(f"Missing values rows removed: {missing_removed}")
        logger.info(f"Invalid price rows removed: {invalid_removed}")
        logger.info(f"Final rows: {after_rows}")
        logger.info(f"Data retention rate: {report['retention_rate_pct']}%")
        logger.info("===========================")

        return report

    def save_processed_data(self, data: pd.DataFrame, output_path: str) -> None:
        """
        Saves the processed DataFrame as a CSV file.

        Args:
            data (pd.DataFrame): Cleaned stock data DataFrame.
            output_path (str): Location to save the CSV file.

        Raises:
            ValueError: If data is empty or output_path is invalid.
            IOError: If writing to the file system fails.
        """
        if data is None or data.empty:
            logger.error("Save failed: DataFrame is empty or None.")
            raise ValueError("Cannot save empty or None DataFrame.")
        if not output_path:
            logger.error("Save failed: Output path is empty.")
            raise ValueError("Output path cannot be empty.")

        logger.info(f"Saving processed data to '{output_path}'...")
        try:
            directory = os.path.dirname(output_path)
            if directory:
                os.makedirs(directory, exist_ok=True)

            data.to_csv(output_path, index=False)
            logger.info(f"Processed data saved to '{output_path}' successfully.")
        except Exception as e:
            logger.exception(f"Failed to save processed CSV to '{output_path}': {e}")
            raise IOError(f"Failed to save data to '{output_path}': {e}") from e

    def preprocess(self, file_path: str, output_path: str) -> pd.DataFrame:
        """
        Executes the entire data preprocessing pipeline.

        Args:
            file_path (str): Path to raw CSV data file.
            output_path (str): Path to save the processed CSV data.

        Returns:
            pd.DataFrame: Cleaned and validated DataFrame.

        Raises:
            Exception: If any error occurs during pipeline execution.
        """
        logger.info("Starting data preprocessing pipeline...")
        try:
            # 1. Load data
            df = self.load_data(file_path)
            initial_count = len(df)

            # 2. Validate column structure
            self.validate_columns(df)

            # 3. Convert datatypes
            df = self.convert_datatypes(df)

            # 4. Remove duplicates
            pre_dup = len(df)
            df = self.remove_duplicates(df)
            duplicates_removed = pre_dup - len(df)

            # 5. Clean missing values
            pre_missing = len(df)
            df = self.clean_missing_values(df)
            missing_removed = pre_missing - len(df)

            # 6. Validate price logic
            pre_invalid = len(df)
            df = self.validate_prices(df)
            invalid_removed = pre_invalid - len(df)

            # 7. Sort by Date
            df = self.sort_by_date(df)
            final_count = len(df)

            # 8. Generate and log report
            self.generate_quality_report(
                initial_count,
                final_count,
                duplicates_removed,
                missing_removed,
                invalid_removed,
            )

            # 9. Save output
            self.save_processed_data(df, output_path)

            logger.info("Data preprocessing pipeline completed successfully.")
            return df

        except Exception as e:
            logger.exception(f"Preprocessing pipeline failed for '{file_path}': {e}")
            raise
