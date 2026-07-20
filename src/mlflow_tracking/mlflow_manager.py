"""
MLflow Manager module for configuring MLflow and setting up experiments.
"""

import logging
import os

# Allow MLflow to use file store backend (recent versions throw exception by default)
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import mlflow
from typing import Optional

logger = logging.getLogger(__name__)


class MLflowManager:
    """Manages MLflow configuration and experiment setup.
    
    Attributes:
        tracking_uri (str, optional): The URI for the MLflow tracking server.
        experiment_name (str): The name of the MLflow experiment.
    """

    def __init__(self, tracking_uri: Optional[str] = None, experiment_name: str = "AI Swing Trading Assistant"):
        """Initializes the MLflowManager.
        
        Args:
            tracking_uri: The tracking URI for MLflow. Defaults to None (local ./mlruns).
            experiment_name: The name of the experiment to use or create.
        """
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self._setup_mlflow()

    def _setup_mlflow(self) -> None:
        """Configures MLflow tracking URI and ensures the experiment exists."""
        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)
            logger.info(f"Set MLflow tracking URI to {self.tracking_uri}")

        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(self.experiment_name)
            logger.info(f"Created new experiment '{self.experiment_name}' with ID {experiment_id}")
        else:
            experiment_id = experiment.experiment_id
            logger.info(f"Using existing experiment '{self.experiment_name}' with ID {experiment_id}")

        mlflow.set_experiment(experiment_id=experiment_id)

    def get_experiment_id(self) -> str:
        """Retrieves the ID of the current experiment.
        
        Returns:
            The experiment ID as a string, or an empty string if it doesn't exist.
        """
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        return experiment.experiment_id if experiment else ""
