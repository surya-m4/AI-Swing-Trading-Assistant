"""
Unit tests for the MLflow tracking module.
"""

import os
import tempfile
import pytest

# Allow MLflow to use file store backend
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import mlflow
from unittest.mock import patch, MagicMock

from src.mlflow_tracking import ExperimentTracker, MLflowManager, ModelRegistry


import pathlib

@pytest.fixture
def temp_mlflow_uri():
    """Fixture to provide a temporary MLflow tracking URI."""
    with tempfile.TemporaryDirectory() as tmpdir:
        uri = pathlib.Path(tmpdir).as_uri()
        yield uri


@pytest.fixture
def tracker(temp_mlflow_uri):
    """Fixture to provide an ExperimentTracker instance with a temp URI."""
    return ExperimentTracker(tracking_uri=temp_mlflow_uri, experiment_name="Test Experiment")


def test_experiment_creation(tracker):
    """Test if experiment is created and configured correctly."""
    assert tracker.manager.tracking_uri.startswith("file://")
    assert tracker.manager.experiment_name == "Test Experiment"
    
    experiment_id = tracker.manager.get_experiment_id()
    assert experiment_id is not None
    assert experiment_id != ""
    
    experiment = mlflow.get_experiment(experiment_id)
    assert experiment.name == "Test Experiment"


def test_start_and_end_experiment(tracker):
    """Test starting and ending an experiment run."""
    run = tracker.start_experiment(run_name="Test Run")
    assert run is not None
    assert tracker.active_run is not None
    assert run.info.run_name == "Test Run"
    
    tracker.end_experiment()
    assert tracker.active_run is None


def test_parameter_logging(tracker):
    """Test logging parameters."""
    with tracker:
        params = {"learning_rate": 0.01, "batch_size": 32}
        tracker.log_parameters(params)
        
        # Verify parameters are logged
        run = mlflow.get_run(tracker.active_run.info.run_id)
        assert run.data.params["learning_rate"] == "0.01"
        assert run.data.params["batch_size"] == "32"


def test_metric_logging(tracker):
    """Test logging metrics (Accuracy, Precision, Recall, F1 Score, ROC-AUC)."""
    with tracker:
        metrics = {
            "Accuracy": 0.95,
            "Precision": 0.94,
            "Recall": 0.96,
            "F1 Score": 0.95,
            "ROC-AUC": 0.98
        }
        tracker.log_metrics(metrics, step=1)
        
        # Verify metrics are logged
        run = mlflow.get_run(tracker.active_run.info.run_id)
        assert run.data.metrics["Accuracy"] == 0.95
        assert run.data.metrics["ROC-AUC"] == 0.98


def test_artifact_logging(tracker):
    """Test logging an artifact file."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".json") as f:
        f.write('{"test": "data"}')
        temp_file_path = f.name
        
    try:
        with tracker:
            tracker.log_artifact(temp_file_path)
            
            from mlflow.utils.file_utils import local_file_uri_to_path
            
            # Since MLflow logs asynchronously/locally, we check if the file was created in the artifact URI
            run = mlflow.get_run(tracker.active_run.info.run_id)
            artifact_uri = run.info.artifact_uri
            
            if artifact_uri.startswith("file:"):
                artifact_path = local_file_uri_to_path(artifact_uri)
            else:
                artifact_path = artifact_uri
            
            logged_file_path = os.path.join(artifact_path, os.path.basename(temp_file_path))
            assert os.path.exists(logged_file_path)
    finally:
        os.remove(temp_file_path)


def test_invalid_path_artifact_logging(tracker):
    """Test logging an invalid artifact path raises an error."""
    with tracker:
        with pytest.raises(FileNotFoundError):
            tracker.log_artifact("non_existent_path.pkl")


@patch('mlflow.register_model')
def test_model_registration(mock_register_model, tracker):
    """Test model registration."""
    # Setup mock return value
    mock_model_version = MagicMock()
    mock_model_version.version = "1"
    mock_register_model.return_value = mock_model_version
    
    with tracker:
        run_id = tracker.active_run.info.run_id
        version = tracker.register_model(model_name="TestModel")
        
        assert version == "1"
        mock_register_model.assert_called_once_with(
            model_uri=f"runs:/{run_id}/model",
            name="TestModel"
        )


@patch('mlflow.pyfunc.load_model')
def test_missing_model(mock_load_model):
    """Test loading a missing model handles exceptions properly."""
    registry = ModelRegistry()
    
    # Make the mock raise an exception to simulate missing model
    mock_load_model.side_effect = Exception("Model not found")
    
    model = registry.load_latest_model(model_name="NonExistentModel", stage="Production")
    assert model is None


def test_compare_runs(tracker):
    """Test compare_runs method."""
    # Create two runs with different accuracies
    with tracker:
        tracker.log_metrics({"accuracy": 0.80})
    
    with tracker:
        tracker.log_metrics({"accuracy": 0.95})
        
    best_run = tracker.compare_runs(metric_name="accuracy")
    assert best_run is not None
    assert best_run.data.metrics["accuracy"] == 0.95
