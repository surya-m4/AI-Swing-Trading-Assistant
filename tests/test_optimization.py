import os
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from sklearn.ensemble import RandomForestClassifier
from src.optimization.search_spaces import SearchSpaces
from src.optimization.optimizer import Optimizer
from src.optimization.tuning_pipeline import TuningPipeline
from src.mlflow_tracking.experiment_tracker import ExperimentTracker

@pytest.fixture
def sample_data():
    df = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'feature2': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        'label': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    })
    X = df.drop(columns=['label'])
    y = df['label'].values
    return X, y

def test_search_spaces_grid():
    space = SearchSpaces.get_grid_search_space('random_forest')
    assert isinstance(space, dict)
    assert 'n_estimators' in space

def test_search_spaces_optuna():
    space_func = SearchSpaces.get_optuna_search_space('xgboost')
    assert callable(space_func)
    
    mock_trial = MagicMock()
    mock_trial.suggest_int.return_value = 100
    mock_trial.suggest_float.return_value = 0.1
    params = space_func(mock_trial)
    assert isinstance(params, dict)
    assert 'n_estimators' in params

def test_search_spaces_invalid():
    with pytest.raises(ValueError):
        SearchSpaces.get_grid_search_space('invalid_model')
    with pytest.raises(ValueError):
        SearchSpaces.get_optuna_search_space('invalid_model')

def test_optimizer_grid_search(sample_data):
    X, y = sample_data
    model = RandomForestClassifier(random_state=42)
    search_space = {'n_estimators': [5, 10]}
    
    optimizer = Optimizer('random_forest', model, search_space, optimization_type='grid', cv=2)
    best_model, best_params = optimizer.optimize(X, y)
    
    assert best_model is not None
    assert 'n_estimators' in best_params

def test_optimizer_random_search(sample_data):
    X, y = sample_data
    model = RandomForestClassifier(random_state=42)
    search_space = {'n_estimators': [5, 10]}
    
    optimizer = Optimizer('random_forest', model, search_space, optimization_type='random', cv=2, n_trials=2)
    best_model, best_params = optimizer.optimize(X, y)
    
    assert best_model is not None
    assert 'n_estimators' in best_params

def test_optimizer_optuna(sample_data):
    X, y = sample_data
    model = RandomForestClassifier(random_state=42)
    
    def dummy_space(trial):
        return {'n_estimators': trial.suggest_int('n_estimators', 5, 10)}
    
    optimizer = Optimizer('random_forest', model, dummy_space, optimization_type='optuna', cv=2, n_trials=2)
    best_model, best_params = optimizer.optimize(X, y)
    
    assert best_model is not None
    assert 'n_estimators' in best_params

def test_optimizer_invalid_type(sample_data):
    X, y = sample_data
    model = RandomForestClassifier()
    optimizer = Optimizer('rf', model, {}, optimization_type='invalid')
    with pytest.raises(ValueError):
        optimizer.optimize(X, y)

def test_optimizer_invalid_search_space(sample_data):
    X, y = sample_data
    model = RandomForestClassifier()
    
    optimizer_grid = Optimizer('rf', model, lambda t: {}, optimization_type='grid')
    with pytest.raises(TypeError):
        optimizer_grid.optimize(X, y)
        
    optimizer_optuna = Optimizer('rf', model, {}, optimization_type='optuna')
    with pytest.raises(TypeError):
        optimizer_optuna.optimize(X, y)

@patch('src.optimization.tuning_pipeline.Trainer')
@patch('src.optimization.tuning_pipeline.ExperimentTracker')
@patch('src.optimization.tuning_pipeline.ModelFactory')
def test_tuning_pipeline_run(mock_model_factory, mock_tracker_class, mock_trainer_class, tmp_path):
    # Setup Mocks
    mock_trainer = mock_trainer_class.return_value
    mock_tracker = mock_tracker_class.return_value.__enter__.return_value
    mock_tracker.active_run = MagicMock()
    
    mock_trainer.load_data.return_value = pd.DataFrame()
    X = pd.DataFrame({'f1': list(range(10)), 'f2': list(range(10, 20))})
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    mock_trainer.prepare_data.return_value = (X, X, y, y)
    mock_trainer.evaluate_model.return_value = {'f1_weighted': 0.8}
    mock_trainer.save_model.return_value = "dummy/path.pkl"
    mock_trainer.artifacts_dir = str(tmp_path)
    
    mock_model_factory.get_model.return_value = RandomForestClassifier()
    
    # Init Pipeline
    pipeline = TuningPipeline(data_filename="dummy.csv", models_to_tune=['random_forest'], optimization_type='grid')
    
    # Override search space explicitly or let it use the real one
    with patch('src.optimization.search_spaces.SearchSpaces.get_grid_search_space') as mock_ss:
        mock_ss.return_value = {'n_estimators': [5]}
        results = pipeline.run()
        
    assert 'random_forest' in results
    assert 'baseline_metrics' in results['random_forest']
    assert 'tuned_metrics' in results['random_forest']
    assert 'best_parameters' in results['random_forest']
    
    # Assert saving
    assert os.path.exists(os.path.join(tmp_path, "optimization_results.json"))
    assert os.path.exists(os.path.join(tmp_path, "random_forest_best_params.json"))
    
    # Assert mlflow logging
    assert mock_tracker.log_parameters.call_count >= 2
    assert mock_tracker.log_artifact.call_count >= 2
