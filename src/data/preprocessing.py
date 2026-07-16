import pandas as pd
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Class to handle data preprocessing steps."""

    def load_data(self, path: str) -> pd.DataFrame:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        return pd.read_csv(path)

    def validate_columns(self, df: pd.DataFrame):
        required_cols = {"Date", "Open", "High", "Low", "Close", "Volume"}
        if not required_cols.issubset(set(df.columns)):
            raise ValueError(f"Missing required columns. Expected {required_cols}, found {list(df.columns)}")

    def convert_datatypes(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        try:
            df["Date"] = pd.to_datetime(df["Date"])
            for col in ["Open", "High", "Low", "Close", "Volume"]:
                df[col] = pd.to_numeric(df[col])
        except Exception as e:
            raise ValueError(f"Error converting datatypes: {e}")
        return df

    def clean_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.dropna()

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop_duplicates()

    def validate_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        # High must be >= Open and Close, Low must be <= Open and Close
        # Volume must be >= 0
        valid_mask = (
            (df["High"] >= df["Open"]) &
            (df["High"] >= df["Close"]) &
            (df["Low"] <= df["Open"]) &
            (df["Low"] <= df["Close"]) &
            (df["Volume"] >= 0)
        )
        return df[valid_mask].copy()

    def sort_by_date(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.sort_values("Date").reset_index(drop=True)

    def save_processed_data(self, df: pd.DataFrame, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)

    def preprocess(self, raw_path: str, processed_path: str) -> pd.DataFrame:
        df = self.load_data(raw_path)
        self.validate_columns(df)
        df = self.convert_datatypes(df)
        df = self.clean_missing_values(df)
        df = self.remove_duplicates(df)
        df = self.validate_prices(df)
        df = self.sort_by_date(df)
        self.save_processed_data(df, processed_path)
        return df
