import logging
from typing import Dict, Tuple, Any

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

logger = logging.getLogger(__name__)

class Trainer:
    """Class to train and evaluate multiple machine learning models."""
    
    def __init__(self):
        """Initializes the Trainer."""
        self.results = {}
        
    def train_and_evaluate(
        self, 
        model_name: str, 
        model: Any, 
        X_train: pd.DataFrame, 
        y_train: pd.Series, 
        X_test: pd.DataFrame, 
        y_test: pd.Series
    ) -> Dict[str, float]:
        """
        Trains the model and evaluates it on the test set.

        Args:
            model_name (str): Name of the model (for tracking).
            model (Any): The instantiated machine learning model.
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training labels.
            X_test (pd.DataFrame): Testing features.
            y_test (pd.Series): Testing labels.

        Returns:
            Dict[str, float]: A dictionary containing evaluation metrics.
        """
        logger.info(f"Training {model_name}...")
        try:
            model.fit(X_train, y_train)
        except Exception as e:
            logger.error(f"Error training {model_name}: {e}")
            raise
            
        logger.info(f"Evaluating {model_name}...")
        try:
            y_pred = model.predict(X_test)
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
            }
            self.results[model_name] = {
                'model': model,
                'metrics': metrics
            }
            return metrics
        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {e}")
            raise
            
    def get_best_model(self, metric: str = 'f1_score') -> Tuple[str, Any, Dict[str, float]]:
        """
        Selects and returns the best model based on a specified metric.

        Args:
            metric (str): The metric to use for selection. Defaults to 'f1_score'.

        Returns:
            Tuple[str, Any, Dict[str, float]]: The name, instance, and metrics of the best model.
            
        Raises:
            ValueError: If no models have been trained or metric is not found.
        """
        if not self.results:
            logger.error("No models have been trained yet.")
            raise ValueError("No models have been trained yet.")
            
        best_model_name = None
        best_model = None
        best_metrics = None
        best_score = -1.0
        
        for name, data in self.results.items():
            metrics = data['metrics']
            if metric not in metrics:
                logger.error(f"Metric '{metric}' not found in results for {name}.")
                raise ValueError(f"Metric '{metric}' not found.")
                
            if metrics[metric] > best_score:
                best_score = metrics[metric]
                best_model_name = name
                best_model = data['model']
                best_metrics = metrics
                
        logger.info(f"Best model selected: {best_model_name} with {metric}: {best_score}")
        return best_model_name, best_model, best_metrics
