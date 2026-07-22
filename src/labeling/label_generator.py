"""
Label generation module for the AI-Powered Swing Trading Assistant.
"""

import logging
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class LabelGeneratorError(Exception):
    """Base exception for LabelGenerator errors."""
    pass


class LabelGenerator:
    """Class for generating trading labels based on future returns."""

    def __init__(
        self,
        prediction_horizon: int = 5,
        buy_threshold: float = 0.05,
        sell_threshold: float = -0.05
    ):
        """Initialize the LabelGenerator with configurable parameters.

        Args:
            prediction_horizon (int): Number of trading days to look ahead. Defaults to 5.
            buy_threshold (float): Minimum future return to classify as BUY. Defaults to 0.05 (5%).
            sell_threshold (float): Maximum future return to classify as SELL. Defaults to -0.05 (-5%).

        Raises:
            LabelGeneratorError: If prediction_horizon is less than 1, or if buy_threshold <= sell_threshold.
        """
        if prediction_horizon < 1:
            error_msg = f"prediction_horizon must be >= 1. Got {prediction_horizon}."
            logger.error(error_msg)
            raise LabelGeneratorError(error_msg)
            
        if buy_threshold <= sell_threshold:
            error_msg = (
                f"buy_threshold ({buy_threshold}) must be strictly greater "
                f"than sell_threshold ({sell_threshold})."
            )
            logger.error(error_msg)
            raise LabelGeneratorError(error_msg)

        self.prediction_horizon = prediction_horizon
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

    def _validate_columns(self, df: pd.DataFrame) -> None:
        """Validate if the required 'Close' column exists.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Raises:
            LabelGeneratorError: If 'Close' column is missing.
        """
        if "Close" not in df.columns:
            error_msg = "Missing required column: 'Close'"
            logger.error(error_msg)
            raise LabelGeneratorError(error_msg)

    def _validate_labels(self, df: pd.DataFrame) -> None:
        """Validate that only BUY, SELL, and HOLD labels exist.

        Args:
            df (pd.DataFrame): DataFrame containing the 'Label' column.

        Raises:
            LabelGeneratorError: If invalid labels are found.
        """
        valid_labels = {"BUY", "SELL", "HOLD"}
        unique_labels = set(df["Label"].unique())
        invalid_labels = unique_labels - valid_labels
        
        if invalid_labels:
            error_msg = f"Found invalid labels: {invalid_labels}. Allowed labels are {valid_labels}."
            logger.error(error_msg)
            raise LabelGeneratorError(error_msg)

    def generate_labels(self, df: pd.DataFrame, output_dir: Union[str, Path, None] = None) -> pd.DataFrame:
        """Generate BUY/HOLD/SELL labels based on future returns and optionally save.

        Args:
            df (pd.DataFrame): Input DataFrame containing engineered features and 'Close' price.
            output_dir (Union[str, Path, None]): Directory to save the labeled dataset. Defaults to None.
                If provided, saves the dataset as 'labeled_dataset.csv'.

        Returns:
            pd.DataFrame: DataFrame with 'Future_Close', 'Future_Return', and 'Label' columns added,
                          and rows with NaN values resulting from the forward shift removed.
        """
        if df.empty:
            logger.warning("Empty DataFrame provided to label generator.")
            return df
            
        self._validate_columns(df)
        
        # Work on a copy to prevent SettingWithCopyWarning
        df = df.copy()

        logger.info(
            f"Generating labels with prediction_horizon={self.prediction_horizon}, "
            f"buy_threshold={self.buy_threshold}, sell_threshold={self.sell_threshold}."
        )

        # Calculate future close and future return
        df["Future_Close"] = df["Close"].shift(-self.prediction_horizon)
        df["Future_Return"] = (df["Future_Close"] - df["Close"]) / df["Close"]

        # Generate labels using numpy select for vectorized conditional assignment
        conditions = [
            (df["Future_Return"] > self.buy_threshold),
            (df["Future_Return"] < self.sell_threshold)
        ]
        choices = ["BUY", "SELL"]
        
        # Default is HOLD
        df["Label"] = np.select(conditions, choices, default="HOLD")

        # Remove rows containing NaN values created by shifting (the last `prediction_horizon` rows)
        initial_len = len(df)
        df = df.dropna(subset=["Future_Close", "Future_Return"]).reset_index(drop=True)
        dropped_len = initial_len - len(df)
        if dropped_len > 0:
            logger.info(f"Removed {dropped_len} rows with missing future values.")
        
        # Validate that only BUY, SELL, and HOLD labels exist
        self._validate_labels(df)

        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            file_path = output_path / "labeled_dataset.csv"
            df.to_csv(file_path, index=False)
            logger.info(f"Saved labeled dataset to {file_path}")

        logger.info("Label generation completed successfully.")
        
        return df
