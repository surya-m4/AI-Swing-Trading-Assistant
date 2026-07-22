import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

class DatasetLoaderError(Exception):
    """Base exception for DatasetLoader errors."""
    pass

class DatasetLoader:
    """
    A class used to load and preprocess datasets for machine learning models.

    Attributes:
        target_column (str): The name of the target variable column.
        required_columns (List[str]): A list of column names that must be present in the dataset.
    """

    def __init__(self, target_column: str, required_columns: Optional[List[str]] = None):
        """
        Initializes the DatasetLoader.

        Args:
            target_column (str): The name of the column containing the target variable.
            required_columns (Optional[List[str]]): A list of required column names.
                If provided, the loader will validate their presence.
        """
        self.target_column = target_column
        self.required_columns = required_columns or []
        
        # Ensure the target column is in the required columns list if we are validating
        if self.required_columns and self.target_column not in self.required_columns:
            self.required_columns.append(self.target_column)

    def load_and_prepare(
        self,
        file_path: Union[str, Path],
        test_size: float = 0.2,
        random_state: int = 42,
        handle_missing: str = 'drop'
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Loads a CSV file, validates it, preprocesses it, and splits it into train/test sets.

        Args:
            file_path (Union[str, Path]): Path to the CSV file.
            test_size (float): The proportion of the dataset to include in the test split.
            random_state (int): Controls the shuffling applied to the data before applying the split.
            handle_missing (str): Strategy to handle missing values ('drop' or 'fill_mean').
                Defaults to 'drop'.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: 
                X_train, X_test, y_train, y_test

        Raises:
            DatasetLoaderError: If the file is missing, empty, or missing required columns.
        """
        logger.info(f"Starting to load dataset from {file_path}")
        df = self._load_data(file_path)
        
        logger.info("Validating dataset structure.")
        self._validate_data(df)
        
        logger.info("Preprocessing dataset (duplicates, missing values).")
        df = self._preprocess_data(df, handle_missing)
        
        logger.info("Splitting dataset into train and test sets.")
        return self._split_data(df, test_size, random_state)

    def _load_data(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """Loads the dataset from a CSV file."""
        path = Path(file_path)
        if not path.exists():
            logger.error(f"File not found: {path}")
            raise DatasetLoaderError(f"Dataset file not found at {path}")
        
        try:
            df = pd.read_csv(path)
        except Exception as e:
            logger.error(f"Failed to read CSV file: {e}")
            raise DatasetLoaderError(f"Error reading CSV file: {e}")
        
        if df.empty:
            logger.error("Dataset is empty.")
            raise DatasetLoaderError("The loaded dataset is empty.")
            
        logger.info(f"Successfully loaded dataset with shape: {df.shape}")
        return df

    def _validate_data(self, df: pd.DataFrame) -> None:
        """Validates that required columns exist in the DataFrame."""
        if self.required_columns:
            missing_cols = [col for col in self.required_columns if col not in df.columns]
            if missing_cols:
                logger.error(f"Missing required columns: {missing_cols}")
                raise DatasetLoaderError(f"Dataset is missing required columns: {missing_cols}")
        
        if self.target_column not in df.columns:
            logger.error(f"Target column '{self.target_column}' not found.")
            raise DatasetLoaderError(f"Target column '{self.target_column}' is missing from the dataset.")

    def _preprocess_data(self, df: pd.DataFrame, handle_missing: str) -> pd.DataFrame:
        """Removes duplicates and handles missing values."""
        initial_rows = len(df)
        df = df.drop_duplicates()
        duplicates_removed = initial_rows - len(df)
        if duplicates_removed > 0:
            logger.info(f"Removed {duplicates_removed} duplicate rows.")

        missing_counts = df.isnull().sum().sum()
        if missing_counts > 0:
            logger.info(f"Found {missing_counts} missing values. Handling using strategy: '{handle_missing}'")
            if handle_missing == 'drop':
                df = df.dropna()
                logger.info(f"Dropped rows with missing values. New shape: {df.shape}")
            elif handle_missing == 'fill_mean':
                # Only fill numeric columns with mean
                numeric_cols = df.select_dtypes(include=['number']).columns
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
                # Drop remaining NaNs (e.g., in categorical columns)
                df = df.dropna()
                logger.info("Filled numeric missing values with mean and dropped remaining.")
            else:
                logger.warning(f"Unknown missing value strategy: '{handle_missing}'. Dropping rows instead.")
                df = df.dropna()
        
        if df.empty:
            logger.error("Dataset became empty after preprocessing.")
            raise DatasetLoaderError("Dataset is empty after preprocessing (removing duplicates/missing values).")

        return df

    def _split_data(
        self, 
        df: pd.DataFrame, 
        test_size: float, 
        random_state: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Separates features/target and performs train_test_split."""
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]
        
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            logger.info(f"Split data into train ({len(X_train)} rows) and test ({len(X_test)} rows)")
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logger.error(f"Failed to split data: {e}")
            raise DatasetLoaderError(f"Error during train_test_split: {e}")
