"""
Trainer module for training machine learning models.
"""
import os
import logging
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.evaluation.metrics import MetricsCalculator
from src.mlflow_tracking.experiment_tracker import ExperimentTracker

logger = logging.getLogger(__name__)

class Trainer:
    """Class responsible for data preparation, model training, evaluation, and saving."""
    
    def __init__(self, data_dir: str = 'data/labeled', models_dir: str = 'models', artifacts_dir: str = 'artifacts'):
        """
        Initializes the Trainer.
        
        Args:
            data_dir (str): Directory containing labeled datasets.
            models_dir (str): Directory to save trained models.
            artifacts_dir (str): Directory for evaluation artifacts.
        """
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.artifacts_dir = artifacts_dir
        
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.artifacts_dir, exist_ok=True)
        
        self.metrics_calculator = MetricsCalculator(artifacts_dir=self.artifacts_dir)
        self.label_encoder = LabelEncoder()
        
    def load_data(self, filename: str) -> pd.DataFrame:
        """
        Loads a labeled dataset.
        
        Args:
            filename (str): Name of the CSV file in the data_dir.
            
        Returns:
            pd.DataFrame: Loaded dataset.
            
        Raises:
            FileNotFoundError: If the dataset file does not exist.
            ValueError: If the dataset is empty.
        """
        filepath = os.path.join(self.data_dir, filename)
        if not os.path.exists(filepath):
            logger.error(f"Dataset not found: {filepath}")
            raise FileNotFoundError(f"Dataset not found: {filepath}")
            
        logger.info(f"Loading dataset from {filepath}")
        df = pd.read_csv(filepath)
        
        if df.empty:
            logger.error(f"Dataset is empty: {filepath}")
            raise ValueError(f"Dataset is empty: {filepath}")
            
        # Ensure timestamp/date is parsed if available, set as index or drop
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
        return df
        
    def prepare_data(self, df: pd.DataFrame, target_col: str = 'label', test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Splits data into train/test sets and encodes labels.
        
        Args:
            df (pd.DataFrame): The dataset.
            target_col (str): The column containing the labels.
            test_size (float): Proportion of the dataset to include in the test split.
            random_state (int): Random state for reproducibility.
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]: X_train, X_test, y_train, y_test
            
        Raises:
            ValueError: If target_col is not in df.
        """
        if target_col not in df.columns:
            logger.error(f"Target column '{target_col}' not found in dataset.")
            raise ValueError(f"Target column '{target_col}' not found in dataset.")
            
        logger.info(f"Preparing data. Target column: {target_col}")
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Fill missing values if any
        X = X.ffill().bfill()
        
        # Check for any remaining NaNs or infinite values and drop them
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        valid_idx = X.dropna().index
        
        if len(valid_idx) < len(X):
            logger.warning(f"Dropped {len(X) - len(valid_idx)} rows containing NaNs/Inf values.")
            X = X.loc[valid_idx]
            y = y.loc[valid_idx]
            
        # Encode labels (e.g., BUY, SELL, HOLD -> 0, 1, 2)
        y_encoded = self.label_encoder.fit_transform(y)
        logger.info(f"Label classes encoded: {self.label_encoder.classes_}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=random_state, shuffle=False
        )
        
        logger.info(f"Data split - Train size: {len(X_train)}, Test size: {len(X_test)}")
        return X_train, X_test, y_train, y_test
        
    def train_model(self, model_name: str, model: Any, X_train: pd.DataFrame, y_train: np.ndarray) -> Any:
        """
        Trains the given model.
        
        Args:
            model_name (str): Name of the model.
            model (Any): The instantiated machine learning model.
            X_train (pd.DataFrame): Training features.
            y_train (np.ndarray): Training labels.
            
        Returns:
            Any: Trained model.
        """
        logger.info(f"Training model: {model_name}")
        model.fit(X_train, y_train)
        logger.info(f"Model {model_name} training completed.")
        return model
        
    def evaluate_model(self, model_name: str, model: Any, X_test: pd.DataFrame, y_test: np.ndarray, tracker: Optional[ExperimentTracker] = None) -> Dict[str, float]:
        """
        Evaluates the model and logs metrics/artifacts to MLflow.
        
        Args:
            model_name (str): Name of the model.
            model (Any): The trained model.
            X_test (pd.DataFrame): Test features.
            y_test (np.ndarray): Test labels.
            tracker (Optional[ExperimentTracker]): MLflow experiment tracker.
            
        Returns:
            Dict[str, float]: Evaluation metrics.
        """
        logger.info(f"Evaluating model: {model_name}")
        y_pred = model.predict(X_test)
        
        y_prob = None
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)
            # For multi-class, we might just pass probabilities if the metrics support it,
            # or simplify it based on the metric calculator capabilities.
            
        metrics = self.metrics_calculator.calculate_metrics(y_test, y_pred, y_prob)
        logger.info(f"Metrics for {model_name}: {metrics}")
        
        cm_path = self.metrics_calculator.plot_confusion_matrix(y_test, y_pred)
        report_path = self.metrics_calculator.save_evaluation_report(metrics)
        
        if tracker:
            tracker.log_metrics(metrics)
            if cm_path:
                tracker.log_artifact(cm_path)
            if report_path:
                tracker.log_artifact(report_path)
                
        return metrics
        
    def save_model(self, model: Any, model_name: str) -> str:
        """
        Saves the trained model to disk.
        
        Args:
            model (Any): The trained model.
            model_name (str): The name to save the model as.
            
        Returns:
            str: Path to the saved model file.
        """
        model_filename = f"{model_name}.pkl"
        model_path = os.path.join(self.models_dir, model_filename)
        
        logger.info(f"Saving model to {model_path}")
        joblib.dump(model, model_path)
        
        # Save label encoder as well
        le_path = os.path.join(self.models_dir, "label_encoder.pkl")
        joblib.dump(self.label_encoder, le_path)
        
        return model_path

