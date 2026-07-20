"""
Experiment Tracker module for logging parameters, metrics, and artifacts to MLflow.
"""

import os
import logging

# Allow MLflow to use file store backend
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import mlflow
from typing import Dict, Any, Optional

from .mlflow_manager import MLflowManager
from .model_registry import ModelRegistry

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Class to track experiments using MLflow.
    
    Attributes:
        manager (MLflowManager): The MLflow manager to handle setup.
        registry (ModelRegistry): The Model Registry handler.
        active_run (mlflow.ActiveRun): The currently active MLflow run.
    """

    def __init__(self, tracking_uri: Optional[str] = None, experiment_name: str = "AI Swing Trading Assistant"):
        """Initializes the ExperimentTracker.
        
        Args:
            tracking_uri: Optional MLflow tracking URI.
            experiment_name: The name of the experiment.
        """
        self.manager = MLflowManager(tracking_uri=tracking_uri, experiment_name=experiment_name)
        self.registry = ModelRegistry()
        self.active_run = None

    def start_experiment(self, run_name: Optional[str] = None) -> mlflow.ActiveRun:
        """Starts an MLflow run for the current experiment.
        
        Args:
            run_name: Optional name for the MLflow run.
            
        Returns:
            The started MLflow active run.
        """
        if self.active_run:
            logger.warning("An experiment run is already active. Ending it first.")
            self.end_experiment()

        experiment_id = self.manager.get_experiment_id()
        self.active_run = mlflow.start_run(experiment_id=experiment_id, run_name=run_name)
        logger.info(f"Started MLflow run '{run_name}' with ID: {self.active_run.info.run_id}")
        return self.active_run

    def end_experiment(self) -> None:
        """Ends the currently active MLflow run."""
        if self.active_run:
            logger.info(f"Ending MLflow run with ID: {self.active_run.info.run_id}")
            mlflow.end_run()
            self.active_run = None
        else:
            logger.warning("No active experiment run to end.")

    def log_parameters(self, params: Dict[str, Any]) -> None:
        """Logs parameters to the current MLflow run.
        
        Args:
            params: Dictionary of parameters to log.
        """
        if not self.active_run:
            logger.error("No active run. Call start_experiment() first.")
            return

        logger.info("Logging parameters to MLflow.")
        mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Logs multiple metrics to the current MLflow run.
        
        Expected metrics: Accuracy, Precision, Recall, F1 Score, ROC-AUC.
        
        Args:
            metrics: Dictionary of metrics to log.
            step: Optional step at which the metrics are logged.
        """
        if not self.active_run:
            logger.error("No active run. Call start_experiment() first.")
            return

        logger.info(f"Logging metrics to MLflow: {list(metrics.keys())}")
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """Logs a single artifact (file or directory) to MLflow.
        
        Expected artifacts: best_model.pkl, confusion_matrix.png, 
        feature_importance.png, evaluation_report.json, classification_report.txt.
        
        Args:
            local_path: Local path to the file or directory to log.
            artifact_path: Optional path in MLflow's artifact store to save it under.
        """
        if not self.active_run:
            logger.error("No active run. Call start_experiment() first.")
            return

        if not os.path.exists(local_path):
            logger.error(f"Artifact path does not exist: {local_path}")
            raise FileNotFoundError(f"Artifact path does not exist: {local_path}")

        logger.info(f"Logging artifact '{local_path}' to MLflow.")
        mlflow.log_artifact(local_path, artifact_path=artifact_path)

    def register_model(self, model_name: str, artifact_path: str = "model") -> Optional[str]:
        """Registers the model from the current run to the Model Registry.
        
        Args:
            model_name: The name under which to register the model.
            artifact_path: Path in the run's artifacts where the model is logged.
            
        Returns:
            The version of the registered model, or None if it fails.
        """
        if not self.active_run:
            logger.error("No active run to register a model from.")
            return None

        run_id = self.active_run.info.run_id
        return self.registry.register_model(run_id, model_name, artifact_path)

    def load_latest_model(self, model_name: str, stage: str = "Production") -> Optional[Any]:
        """Loads the latest model from the registry for a given stage.
        
        Args:
            model_name: The registered name of the model.
            stage: The stage/alias of the model to load.
            
        Returns:
            The loaded model.
        """
        return self.registry.load_latest_model(model_name, stage)

    def compare_runs(self, metric_name: str = "accuracy") -> Optional[mlflow.entities.Run]:
        """Compares runs within the current experiment based on a specific metric.
        
        Returns the run with the highest value for the given metric.
        
        Args:
            metric_name: The metric to compare runs on (e.g., 'accuracy', 'f1_score').
            
        Returns:
            The MLflow Run object with the best metric, or None if no runs exist.
        """
        experiment_id = self.manager.get_experiment_id()
        if not experiment_id:
            logger.error("No experiment ID found.")
            return None

        # Search runs in the current experiment
        runs = mlflow.search_runs(
            experiment_ids=[experiment_id],
            order_by=[f"metrics.{metric_name} DESC"],
            max_results=1
        )

        if runs.empty:
            logger.warning(f"No runs found in the experiment to compare on '{metric_name}'.")
            return None

        best_run_id = runs.iloc[0]["run_id"]
        logger.info(f"Best run based on '{metric_name}' is {best_run_id}")
        return mlflow.get_run(best_run_id)

    def __enter__(self):
        """Context manager entry point."""
        self.start_experiment()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit point."""
        self.end_experiment()
