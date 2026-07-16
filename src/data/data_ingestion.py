"""Data ingestion utilities for the Swing Trading Assistant.

Provides functions to load raw market data from CSV, JSON, or external APIs.
"""

import pandas as pd
import os
from typing import Union


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

# Additional loaders (e.g., JSON, API) can be added here.
