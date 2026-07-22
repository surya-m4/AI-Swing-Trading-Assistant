import pandas as pd
import pytest
from pathlib import Path
import numpy as np

from src.labeling.label_generator import LabelGenerator, LabelGeneratorError


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Fixture providing a sample DataFrame with 'Close' prices."""
    # Prices designed to trigger BUY, SELL, and HOLD
    # Horizon = 2 for testing simplicity
    # Day 0: Close 100
    # Day 1: Close 100
    # Day 2: Close 106 -> Return for Day 0 = 6% (BUY)
    # Day 3: Close 94  -> Return for Day 1 = -6% (SELL)
    # Day 4: Close 108 -> Return for Day 2 = ~1.88% (HOLD)
    # Day 5: Close 100
    # Day 6: Close 100
    
    return pd.DataFrame({
        "Close": [100.0, 100.0, 106.0, 94.0, 108.0, 100.0, 100.0]
    })


@pytest.fixture
def label_generator() -> LabelGenerator:
    """Fixture providing a LabelGenerator instance with prediction_horizon=2."""
    return LabelGenerator(prediction_horizon=2, buy_threshold=0.05, sell_threshold=-0.05)


def test_label_generation_logic(sample_df: pd.DataFrame, label_generator: LabelGenerator):
    """Test future return calculation and BUY/SELL/HOLD generation."""
    df_labeled = label_generator.generate_labels(sample_df)
    
    # After dropping last 2 rows (because of prediction_horizon=2), 
    # we should have 5 rows left.
    assert len(df_labeled) == 5
    
    # Day 0: (106 - 100) / 100 = 0.06 -> BUY
    assert df_labeled.loc[0, "Future_Return"] == pytest.approx(0.06)
    assert df_labeled.loc[0, "Label"] == "BUY"
    
    # Day 1: (94 - 100) / 100 = -0.06 -> SELL
    assert df_labeled.loc[1, "Future_Return"] == pytest.approx(-0.06)
    assert df_labeled.loc[1, "Label"] == "SELL"
    
    # Day 2: (108 - 106) / 106 = ~0.0188 -> HOLD
    assert df_labeled.loc[2, "Future_Return"] == pytest.approx((108 - 106) / 106)
    assert df_labeled.loc[2, "Label"] == "HOLD"


def test_missing_value_handling(sample_df: pd.DataFrame, label_generator: LabelGenerator):
    """Test that rows with missing future values are dropped."""
    initial_len = len(sample_df)
    df_labeled = label_generator.generate_labels(sample_df)
    
    # The last `prediction_horizon` rows should be dropped
    assert len(df_labeled) == initial_len - label_generator.prediction_horizon
    assert not df_labeled["Future_Return"].isnull().any()
    assert not df_labeled["Future_Close"].isnull().any()


def test_invalid_input_handling_missing_column():
    """Test exception when 'Close' column is missing."""
    df = pd.DataFrame({"Price": [10, 11, 12]}) # 'Close' is missing
    generator = LabelGenerator()
    with pytest.raises(LabelGeneratorError, match="Missing required column: 'Close'"):
        generator.generate_labels(df)


def test_invalid_input_handling_bad_params():
    """Test exception for invalid initialization parameters."""
    with pytest.raises(LabelGeneratorError, match="prediction_horizon must be >= 1"):
        LabelGenerator(prediction_horizon=0)
        
    with pytest.raises(LabelGeneratorError, match="must be strictly greater than"):
        LabelGenerator(buy_threshold=0.03, sell_threshold=0.03)


def test_output_validation(sample_df: pd.DataFrame, label_generator: LabelGenerator, monkeypatch: pytest.MonkeyPatch):
    """Test validation of generated labels."""
    # We can mock the numpy.select output to generate an invalid label and ensure it raises an error
    def mock_select(*args, **kwargs):
        return np.array(["INVALID"] * len(sample_df))
        
    monkeypatch.setattr(np, "select", mock_select)
    
    with pytest.raises(LabelGeneratorError, match="Found invalid labels"):
        label_generator.generate_labels(sample_df)


def test_empty_dataframe(label_generator: LabelGenerator):
    """Test passing an empty DataFrame."""
    df = pd.DataFrame()
    df_labeled = label_generator.generate_labels(df)
    assert df_labeled.empty


def test_saving_dataset(sample_df: pd.DataFrame, label_generator: LabelGenerator, tmp_path: Path):
    """Test saving labeled dataset to output directory."""
    label_generator.generate_labels(sample_df, output_dir=tmp_path)
    
    output_file = tmp_path / "labeled_dataset.csv"
    assert output_file.exists()
    
    # Verify contents of the saved file
    saved_df = pd.read_csv(output_file)
    assert "Label" in saved_df.columns
    assert "Future_Return" in saved_df.columns
    assert len(saved_df) == len(sample_df) - label_generator.prediction_horizon
