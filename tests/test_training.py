import os
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from src.training.model_factory import ModelFactory
from src.training.trainer import Trainer
from src.training.train_pipeline import TrainPipeline
from src.mlflow_tracking.experiment_tracker import ExperimentTracker

@pytest.fixture
def sample_data():
    df = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [5, 4, 3, 2, 1],
        'label': ['BUY', 'SELL', 'HOLD', 'BUY', 'SELL']
    })
    return df

@pytest.fixture
def trainer(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    models_dir = tmp_path / "models"
    artifacts_dir = tmp_path / "artifacts"
    
    return Trainer(data_dir=str(data_dir), models_dir=str(models_dir), artifacts_dir=str(artifacts_dir))

def test_model_factory_get_model():
    # Test valid models
    rf = ModelFactory.get_model('random_forest')
    assert rf.__class__.__name__ == 'RandomForestClassifier'
    
    xgb = ModelFactory.get_model('xgboost')
    assert xgb.__class__.__name__ == 'XGBClassifier'
    
    lgb = ModelFactory.get_model('lightgbm')
    assert lgb.__class__.__name__ == 'LGBMClassifier'
    
    cat = ModelFactory.get_model('catboost')
    assert cat.__class__.__name__ == 'CatBoostClassifier'
    
    # Test invalid model
    with pytest.raises(ValueError):
        ModelFactory.get_model('invalid_model')

def test_trainer_load_data_success(trainer, sample_data, tmp_path):
    # Setup mock file
    filename = "test_data.csv"
    filepath = tmp_path / "data" / filename
    sample_data.to_csv(filepath, index=False)
    
    df = trainer.load_data(filename)
    assert not df.empty
    assert len(df) == 5
    assert 'label' in df.columns

def test_trainer_load_data_not_found(trainer):
    with pytest.raises(FileNotFoundError):
        trainer.load_data("non_existent.csv")

def test_trainer_load_data_empty(trainer, tmp_path):
    filename = "empty.csv"
    filepath = tmp_path / "data" / filename
    pd.DataFrame().to_csv(filepath, index=False)
    
    with pytest.raises(ValueError):
        trainer.load_data(filename)

def test_trainer_prepare_data(trainer, sample_data):
    X_train, X_test, y_train, y_test = trainer.prepare_data(sample_data, target_col='label', test_size=0.4, random_state=42)
    
    assert len(X_train) == 3
    assert len(X_test) == 2
    assert len(y_train) == 3
    assert len(y_test) == 2
    # Ensure encoded
    assert set(np.unique(y_train)).issubset({0, 1, 2})

def test_trainer_prepare_data_no_target(trainer, sample_data):
    df_invalid = sample_data.drop(columns=['label'])
    with pytest.raises(ValueError):
        trainer.prepare_data(df_invalid, target_col='label')

def test_trainer_train_and_evaluate_model(trainer, sample_data):
    X_train, X_test, y_train, y_test = trainer.prepare_data(sample_data, target_col='label', test_size=0.4, random_state=42)
    
    model = ModelFactory.get_model('random_forest', params={'n_estimators': 10})
    trained_model = trainer.train_model('random_forest', model, X_train, y_train)
    
    assert trained_model is not None
    
    # Evaluate
    mock_tracker = MagicMock()
    metrics = trainer.evaluate_model('random_forest', trained_model, X_test, y_test, tracker=mock_tracker)
    
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1_score' in metrics
    
    mock_tracker.log_metrics.assert_called_once()
    assert mock_tracker.log_artifact.call_count == 2 # cm_path and report_path

def test_trainer_save_model(trainer):
    model = ModelFactory.get_model('random_forest', params={'n_estimators': 10})
    model_path = trainer.save_model(model, 'random_forest')
    
    assert os.path.exists(model_path)
    assert os.path.exists(os.path.join(trainer.models_dir, "label_encoder.pkl"))

@patch('src.training.train_pipeline.Trainer')
@patch('src.training.train_pipeline.ExperimentTracker')
def test_train_pipeline_run(mock_tracker_class, mock_trainer_class, tmp_path):
    mock_trainer = mock_trainer_class.return_value
    mock_tracker = mock_tracker_class.return_value.__enter__.return_value
    
    # Setup mock returns
    mock_df = pd.DataFrame({'f1': [1,2], 'label': [0,1]})
    mock_trainer.load_data.return_value = mock_df
    mock_trainer.prepare_data.return_value = (pd.DataFrame(), pd.DataFrame(), np.array([]), np.array([]))
    mock_trainer.train_model.return_value = MagicMock()
    mock_trainer.evaluate_model.return_value = {'accuracy': 0.9}
    mock_trainer.save_model.return_value = "path/to/model"
    mock_trainer.artifacts_dir = str(tmp_path)
    
    pipeline = TrainPipeline(data_filename="dummy.csv", models_to_train=['random_forest'])
    results = pipeline.run()
    
    assert 'random_forest' in results
    assert results['random_forest']['accuracy'] == 0.9
    
    mock_trainer.load_data.assert_called_once_with("dummy.csv")
    mock_trainer.train_model.assert_called_once()
    mock_trainer.evaluate_model.assert_called_once()
