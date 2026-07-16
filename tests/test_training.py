import os
import tempfile
import pytest
import pandas as pd
import numpy as np

from src.models.model_factory import ModelFactory
from src.models.trainer import Trainer
from src.models.save_model import ModelSaver
from src.models.train import load_and_validate_data, prepare_data

@pytest.fixture
def dummy_data():
    """Fixture to create a dummy dataframe for testing."""
    np.random.seed(42)
    data = {
        'Feature1': np.random.rand(100),
        'Feature2': np.random.rand(100),
        'Target': np.random.randint(0, 2, 100)
    }
    return pd.DataFrame(data)

@pytest.fixture
def temp_dir():
    """Fixture to create a temporary directory for saving models."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir

# --- Data Validation Tests ---

def test_load_and_validate_data_success(dummy_data, temp_dir):
    filepath = os.path.join(temp_dir, 'dummy.csv')
    dummy_data.to_csv(filepath, index=False)
    
    df = load_and_validate_data(filepath, 'Target')
    assert not df.empty
    assert 'Target' in df.columns

def test_load_and_validate_data_missing_file():
    with pytest.raises(FileNotFoundError):
        load_and_validate_data('non_existent_file.csv', 'Target')

def test_load_and_validate_data_empty_dataset(temp_dir):
    filepath = os.path.join(temp_dir, 'empty.csv')
    pd.DataFrame().to_csv(filepath, index=False)
    
    with pytest.raises(ValueError, match="The dataset is empty."):
        load_and_validate_data(filepath, 'Target')

def test_load_and_validate_data_missing_label(dummy_data, temp_dir):
    df_missing_label = dummy_data.drop(columns=['Target'])
    filepath = os.path.join(temp_dir, 'missing_label.csv')
    df_missing_label.to_csv(filepath, index=False)
    
    with pytest.raises(ValueError, match="Label column 'Target' not found"):
        load_and_validate_data(filepath, 'Target')

def test_prepare_data(dummy_data):
    X_train, X_test, y_train, y_test = prepare_data(dummy_data, 'Target')
    assert X_train.shape[0] == 80
    assert X_test.shape[0] == 20
    assert 'Target' not in X_train.columns
    assert y_train.name == 'Target'

# --- ModelFactory Tests ---

def test_model_factory_valid_models():
    lr_model = ModelFactory.get_model('logistic_regression')
    assert lr_model is not None
    
    rf_model = ModelFactory.get_model('random_forest')
    assert rf_model is not None
    
    xgb_model = ModelFactory.get_model('xgboost')
    assert xgb_model is not None

def test_model_factory_invalid_model():
    with pytest.raises(ValueError, match="Unsupported model name"):
        ModelFactory.get_model('invalid_model')

# --- Trainer Tests ---

def test_trainer_train_and_evaluate(dummy_data):
    X_train, X_test, y_train, y_test = prepare_data(dummy_data, 'Target')
    model = ModelFactory.get_model('logistic_regression')
    
    trainer = Trainer()
    metrics = trainer.train_and_evaluate('logistic_regression', model, X_train, y_train, X_test, y_test)
    
    assert 'accuracy' in metrics
    assert 'f1_score' in metrics
    assert trainer.results['logistic_regression']['metrics'] == metrics

def test_trainer_get_best_model(dummy_data):
    X_train, X_test, y_train, y_test = prepare_data(dummy_data, 'Target')
    trainer = Trainer()
    
    # Train dummy models with deterministic outcomes if possible or just use real
    model1 = ModelFactory.get_model('logistic_regression')
    model2 = ModelFactory.get_model('random_forest')
    
    trainer.train_and_evaluate('lr', model1, X_train, y_train, X_test, y_test)
    trainer.train_and_evaluate('rf', model2, X_train, y_train, X_test, y_test)
    
    best_name, best_model, best_metrics = trainer.get_best_model(metric='f1_score')
    
    assert best_name in ['lr', 'rf']
    assert best_metrics is not None

def test_trainer_get_best_model_empty():
    trainer = Trainer()
    with pytest.raises(ValueError, match="No models have been trained yet."):
        trainer.get_best_model()

# --- ModelSaver Tests ---

def test_save_model(dummy_data, temp_dir):
    X_train, _, y_train, _ = prepare_data(dummy_data, 'Target')
    model = ModelFactory.get_model('logistic_regression')
    model.fit(X_train, y_train)
    
    saver = ModelSaver(output_dir=temp_dir)
    filepath = saver.save_model(model, 'test_model.pkl')
    
    assert os.path.exists(filepath)

def test_save_metrics(temp_dir):
    saver = ModelSaver(output_dir=temp_dir)
    metrics = {'accuracy': 0.9, 'f1_score': 0.85}
    filepath = saver.save_metrics(metrics, 'test_metrics.json')
    
    assert os.path.exists(filepath)

def test_save_classification_report(temp_dir):
    saver = ModelSaver(output_dir=temp_dir)
    report = "Dummy Classification Report"
    filepath = saver.save_classification_report(report, 'test_report.txt')
    
    assert os.path.exists(filepath)
    with open(filepath, 'r') as f:
        assert f.read() == report
