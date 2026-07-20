"""
Model Registry module for handling MLflow model registration and loading.
"""

import logging
import os

# Allow MLflow to use file store backend (recent versions throw exception by default)
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import mlflow
from mlflow.tracking import MlflowClient
from typing import Optional, Any

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Handles registering and loading models from the MLflow Model Registry.
    
    Attributes:
        client (MlflowClient): The MLflow client used to interact with the registry.
    """

    def __init__(self):
        """Initializes the ModelRegistry with an MLflowClient."""
        self.client = MlflowClient()

    def register_model(self, run_id: str, model_name: str, artifact_path: str = "model") -> Optional[str]:
        """Registers a model from a specific run to the Model Registry.
        
        Args:
            run_id: The ID of the MLflow run that produced the model.
            model_name: The name under which to register the model.
            artifact_path: The relative path to the model artifact in the run. Defaults to "model".
            
        Returns:
            The version of the registered model as a string, or None if registration fails.
        """
        model_uri = f"runs:/{run_id}/{artifact_path}"
        try:
            logger.info(f"Registering model '{model_name}' from run '{run_id}'")
            model_version = mlflow.register_model(model_uri=model_uri, name=model_name)
            logger.info(f"Successfully registered model '{model_name}' version {model_version.version}")
            return model_version.version
        except Exception as e:
            logger.error(f"Failed to register model '{model_name}': {str(e)}")
            return None

    def load_latest_model(self, model_name: str, stage: str = "Production") -> Optional[Any]:
        """Loads the latest model from the registry for a given stage.
        
        Args:
            model_name: The registered name of the model.
            stage: The stage of the model to load (e.g., "Production", "Staging", "None").
                   Note: MLflow >= 2.9 deprecates stages in favor of aliases.
                   If using a modern MLflow version, we might use aliases instead.
                   Assuming standard older approach for this method for compatibility.
            
        Returns:
            The loaded model object, or None if loading fails.
        """
        model_uri = f"models:/{model_name}/{stage}"
        try:
            logger.info(f"Loading latest model '{model_name}' from stage '{stage}'")
            # Usually sklearn or pyfunc. Using pyfunc as generic loader.
            model = mlflow.pyfunc.load_model(model_uri)
            logger.info(f"Successfully loaded model '{model_name}'")
            return model
        except Exception as e:
            logger.error(f"Failed to load model '{model_name}' at stage '{stage}': {str(e)}")
            
            # If using aliases instead of stages (MLflow > 2.9)
            if "alias" not in str(e).lower():
                try:
                    logger.info(f"Attempting to load model '{model_name}' using alias '{stage}'")
                    model_uri_alias = f"models:/{model_name}@{stage.lower()}"
                    model = mlflow.pyfunc.load_model(model_uri_alias)
                    return model
                except Exception as e2:
                    logger.error(f"Failed to load model using alias: {str(e2)}")

            return None
