import os
import json
import logging
from typing import Any, Dict
import joblib

logger = logging.getLogger(__name__)

class ModelSaver:
    """Class responsible for saving models and their evaluation metrics."""
    
    def __init__(self, output_dir: str = 'models'):
        """
        Initializes the ModelSaver.
        
        Args:
            output_dir (str): Directory where artifacts will be saved.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def save_model(self, model: Any, filename: str = 'best_model.pkl') -> str:
        """
        Saves the trained model to disk.

        Args:
            model (Any): The trained machine learning model.
            filename (str): Name of the file to save the model.

        Returns:
            str: Path where the model was saved.
        """
        filepath = os.path.join(self.output_dir, filename)
        logger.info(f"Saving model to {filepath}...")
        try:
            joblib.dump(model, filepath)
            logger.info("Model saved successfully.")
            return filepath
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise
            
    def save_metrics(self, metrics: Dict[str, float], filename: str = 'model_metrics.json') -> str:
        """
        Saves the evaluation metrics to a JSON file.

        Args:
            metrics (Dict[str, float]): The metrics dictionary.
            filename (str): Name of the file to save metrics.

        Returns:
            str: Path where the metrics were saved.
        """
        filepath = os.path.join(self.output_dir, filename)
        logger.info(f"Saving metrics to {filepath}...")
        try:
            with open(filepath, 'w') as f:
                json.dump(metrics, f, indent=4)
            logger.info("Metrics saved successfully.")
            return filepath
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")
            raise
            
    def save_classification_report(self, report: str, filename: str = 'classification_report.txt') -> str:
        """
        Saves the classification report as a text file.

        Args:
            report (str): The classification report string.
            filename (str): Name of the file to save the report.

        Returns:
            str: Path where the report was saved.
        """
        filepath = os.path.join(self.output_dir, filename)
        logger.info(f"Saving classification report to {filepath}...")
        try:
            with open(filepath, 'w') as f:
                f.write(report)
            logger.info("Classification report saved successfully.")
            return filepath
        except Exception as e:
            logger.error(f"Failed to save classification report: {e}")
            raise
