import pandas as pd
import pytest
from pathlib import Path

from src.training.dataset_loader import DatasetLoader, DatasetLoaderError


@pytest.fixture
def valid_csv(tmp_path: Path) -> Path:
    """Fixture to create a valid CSV file for testing."""
    df = pd.DataFrame({
        'feature1': [1.0, 2.0, 3.0, 4.0, 5.0],
        'feature2': [5.0, 4.0, 3.0, 2.0, 1.0],
        'target': [0, 1, 0, 1, 0]
    })
    file_path = tmp_path / "valid.csv"
    df.to_csv(file_path, index=False)
    return file_path


@pytest.fixture
def duplicates_csv(tmp_path: Path) -> Path:
    """Fixture to create a CSV with duplicate rows."""
    df = pd.DataFrame({
        'feature1': [1.0, 1.0, 2.0],
        'feature2': [5.0, 5.0, 4.0],
        'target': [0, 0, 1]
    })
    file_path = tmp_path / "duplicates.csv"
    df.to_csv(file_path, index=False)
    return file_path


@pytest.fixture
def missing_values_csv(tmp_path: Path) -> Path:
    """Fixture to create a CSV with missing values."""
    df = pd.DataFrame({
        'feature1': [1.0, None, 3.0, 4.0],
        'feature2': [5.0, 4.0, None, 2.0],
        'target': [0, 1, 0, 1]
    })
    file_path = tmp_path / "missing.csv"
    df.to_csv(file_path, index=False)
    return file_path


@pytest.fixture
def empty_csv(tmp_path: Path) -> Path:
    """Fixture to create an empty CSV."""
    df = pd.DataFrame()
    file_path = tmp_path / "empty.csv"
    df.to_csv(file_path, index=False)
    return file_path


def test_successful_load_and_split(valid_csv: Path):
    """Test standard successful loading and splitting."""
    loader = DatasetLoader(target_column='target', required_columns=['feature1', 'feature2'])
    X_train, X_test, y_train, y_test = loader.load_and_prepare(
        file_path=valid_csv, test_size=0.2, random_state=42
    )

    # Total rows = 5. test_size=0.2 means 1 test row, 4 train rows.
    assert len(X_train) == 4
    assert len(X_test) == 1
    assert len(y_train) == 4
    assert len(y_test) == 1

    # Check columns
    assert 'target' not in X_train.columns
    assert 'feature1' in X_train.columns
    assert 'feature2' in X_train.columns


def test_missing_file():
    """Test exception when file is missing."""
    loader = DatasetLoader(target_column='target')
    with pytest.raises(DatasetLoaderError, match="Dataset file not found"):
        loader.load_and_prepare(file_path="nonexistent_file.csv")


def test_empty_dataset(empty_csv: Path):
    """Test exception on empty dataset."""
    loader = DatasetLoader(target_column='target')
    with pytest.raises(DatasetLoaderError, match="No columns to parse from file"):
        loader.load_and_prepare(file_path=empty_csv)


def test_missing_required_columns(valid_csv: Path):
    """Test exception when required columns are missing."""
    # valid_csv doesn't have 'missing_feature'
    loader = DatasetLoader(target_column='target', required_columns=['feature1', 'missing_feature'])
    with pytest.raises(DatasetLoaderError, match="missing required columns"):
        loader.load_and_prepare(file_path=valid_csv)


def test_missing_target_column(valid_csv: Path):
    """Test exception when target column is missing."""
    # We don't set it in required_columns to bypass the first check,
    # but the target column check itself should fail.
    # Actually, DatasetLoader forces target_column into required_columns if required_columns is provided.
    # Let's test with no required_columns.
    loader = DatasetLoader(target_column='wrong_target')
    with pytest.raises(DatasetLoaderError, match="is missing from the dataset"):
        loader.load_and_prepare(file_path=valid_csv)


def test_duplicates_removal(duplicates_csv: Path):
    """Test that duplicate rows are removed."""
    loader = DatasetLoader(target_column='target')
    X_train, X_test, y_train, y_test = loader.load_and_prepare(
        file_path=duplicates_csv, test_size=0.5
    )
    
    # Original has 3 rows, 1 is duplicate -> 2 rows remain.
    # 2 rows total: train=1, test=1 with test_size=0.5
    assert len(X_train) + len(X_test) == 2


def test_missing_values_drop(missing_values_csv: Path):
    """Test dropping missing values."""
    loader = DatasetLoader(target_column='target')
    X_train, X_test, y_train, y_test = loader.load_and_prepare(
        file_path=missing_values_csv, test_size=0.5, handle_missing='drop'
    )
    
    # Original has 4 rows, 2 have NaNs -> 2 rows remain.
    assert len(X_train) + len(X_test) == 2


def test_missing_values_fill_mean(missing_values_csv: Path):
    """Test filling missing values with mean."""
    loader = DatasetLoader(target_column='target')
    X_train, X_test, y_train, y_test = loader.load_and_prepare(
        file_path=missing_values_csv, test_size=0.25, handle_missing='fill_mean'
    )
    
    # Original has 4 rows. Mean imputation shouldn't drop any unless all NaNs.
    # So 4 rows remain.
    assert len(X_train) + len(X_test) == 4
    
    # Combine back to check values
    X_full = pd.concat([X_train, X_test])
    # The means for feature1 (1,3,4 -> mean 2.666) and feature2 (5,4,2 -> mean 3.666)
    # Check that there are no NaNs left
    assert not X_full.isnull().values.any()
