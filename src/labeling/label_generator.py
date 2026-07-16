"""
Label generation module for the AI-Powered Swing Trading Assistant.
"""

import logging
import numpy as np
import pandas as pd

# Configure logger
logger = logging.getLogger(__name__)


class LabelGenerator:
    """Class for generating trading labels based on future returns."""

    def __init__(
        self,
        forecast_horizon: int = 5,
        buy_threshold: float = 0.03,
        sell_threshold: float = -0.03
    ):
        """Initialize the LabelGenerator with configurable parameters.

        Args:
            forecast_horizon (int): Number of trading days to look ahead. Defaults to 5.
            buy_threshold (float): Minimum future return to classify as BUY (1). Defaults to 0.03 (3%).
            sell_threshold (float): Maximum future return to classify as SELL (-1). Defaults to -0.03 (-3%).

        Raises:
            ValueError: If forecast_horizon is less than 1, or if buy_threshold <= sell_threshold.
        """
        if forecast_horizon < 1:
            error_msg = f"forecast_horizon must be >= 1. Got {forecast_horizon}."
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        if buy_threshold <= sell_threshold:
            error_msg = (
                f"buy_threshold ({buy_threshold}) must be strictly greater "
                f"than sell_threshold ({sell_threshold})."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        self.forecast_horizon = forecast_horizon
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

    def _validate_columns(self, df: pd.DataFrame) -> None:
        """Validate if the required 'Close' column exists.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Raises:
            ValueError: If 'Close' column is missing.
        """
        if "Close" not in df.columns:
            error_msg = "Missing required column: 'Close'"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def generate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate BUY/HOLD/SELL labels based on future returns.

        Args:
            df (pd.DataFrame): Input DataFrame containing engineered features and 'Close' price.

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
            f"Generating labels with forecast_horizon={self.forecast_horizon}, "
            f"buy_threshold={self.buy_threshold}, sell_threshold={self.sell_threshold}."
        )

        # Calculate future close and future return
        df["Future_Close"] = df["Close"].shift(-self.forecast_horizon)
        df["Future_Return"] = (df["Future_Close"] - df["Close"]) / df["Close"]

        # Generate labels using numpy select for vectorized conditional assignment
        conditions = [
            (df["Future_Return"] >= self.buy_threshold),
            (df["Future_Return"] <= self.sell_threshold)
        ]
        # 1 for BUY, -1 for SELL
        choices = [1, -1] 
        
        # Default is 0 (HOLD)
        df["Label"] = np.select(conditions, choices, default=0)

        # Remove rows containing NaN values created by shifting (the last `forecast_horizon` rows)
        df = df.dropna(subset=["Future_Close", "Future_Return"]).reset_index(drop=True)
        
        # Cast Label to integer just to be completely sure it isn't float
        df["Label"] = df["Label"].astype(int)

        logger.info("Label generation completed successfully.")
        
        return df
