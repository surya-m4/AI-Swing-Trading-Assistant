import pytest
import os
import json
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from src.api.app import app
from src.api.dependencies import get_predictor
from src.api.predictor import ModelPredictor

client = TestClient(app)

@pytest.fixture
def mock_predictor():
    mock_pred = MagicMock(spec=ModelPredictor)
    mock_pred.model_name = "test_model"
    mock_pred.predict.return_value = ("BUY", 0.95)
    return mock_pred

@pytest.fixture(autouse=True)
def override_dependency(mock_predictor):
    app.dependency_overrides[get_predictor] = lambda: mock_predictor
    yield
    app.dependency_overrides.clear()

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the AI Swing Trading Assistant API"}

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict_success(mock_predictor):
    payload = {
        "features": {
            "Open": 100,
            "Close": 105
        }
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert data["action"] == "BUY"
    assert data["confidence_score"] == 0.95
    assert data["model_name"] == "test_model"
    assert "timestamp" in data
    
    mock_predictor.predict.assert_called_once_with(payload["features"])

def test_predict_empty_features():
    payload = {"features": {}}
    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    assert "Features cannot be empty" in response.json()["detail"]

def test_predict_model_not_loaded(mock_predictor):
    mock_predictor.predict.side_effect = RuntimeError("Model is not loaded.")
    payload = {"features": {"Open": 100}}
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 503
    assert "Model is not loaded." in response.json()["detail"]

def test_predict_invalid_schema():
    payload = {"invalid_field": {}}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422 # FastAPI validation error

@patch('os.path.exists')
@patch('builtins.open')
def test_model_info_success(mock_open, mock_exists, mock_predictor):
    # Setup mocks to pretend files exist
    mock_exists.return_value = True
    
    # We need to mock open to return valid json for params and metrics
    mock_file = MagicMock()
    # first call is params, second is metrics
    mock_file.read.side_effect = [
        json.dumps({"n_estimators": 100}),
        json.dumps({"test_model": {"accuracy": 0.8, "precision": 0.85, "recall": 0.75, "f1_score": 0.8}})
    ]
    mock_open.return_value.__enter__.return_value = mock_file
    
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "test_model"
    assert data["accuracy"] == 0.8
    assert data["hyperparameters"]["n_estimators"] == 100

@patch('os.path.exists')
def test_model_info_missing_files(mock_exists, mock_predictor):
    # Files do not exist
    mock_exists.return_value = False
    
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    
    assert data["model_name"] == "test_model"
    assert data["accuracy"] == 0.0
    assert data["hyperparameters"] == {}

@patch('os.path.exists')
@patch('builtins.open')
def test_metrics_success(mock_open, mock_exists):
    mock_exists.return_value = True
    
    mock_file = MagicMock()
    mock_file.read.return_value = json.dumps({"test_model": {"accuracy": 0.9}})
    mock_open.return_value.__enter__.return_value = mock_file
    
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.json() == {"test_model": {"accuracy": 0.9}}

@patch('os.path.exists')
def test_metrics_not_found(mock_exists):
    mock_exists.return_value = False
    
    response = client.get("/metrics")
    assert response.status_code == 404
