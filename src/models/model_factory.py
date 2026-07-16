import logging
from typing import Any

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

class ModelFactory:
    """Factory class to create machine learning models."""
    
    @staticmethod
    def get_model(model_name: str, **kwargs) -> Any:
        """
        Creates and returns a machine learning model based on the provided name.

        Args:
            model_name (str): Name of the model to create ('logistic_regression', 'random_forest', 'xgboost').
            **kwargs: Additional keyword arguments to pass to the model constructor.

        Returns:
            Any: An instance of the requested model.

        Raises:
            ValueError: If the model_name is not supported.
        """
        model_name = model_name.lower()
        if model_name == 'logistic_regression':
            logger.info("Creating Logistic Regression model.")
            return LogisticRegression(random_state=42, **kwargs)
        elif model_name == 'random_forest':
            logger.info("Creating Random Forest model.")
            return RandomForestClassifier(random_state=42, **kwargs)
        elif model_name == 'xgboost':
            logger.info("Creating XGBoost model.")
            return XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', **kwargs)
        else:
            error_msg = f"Unsupported model name: {model_name}"
            logger.error(error_msg)
            raise ValueError(error_msg)
