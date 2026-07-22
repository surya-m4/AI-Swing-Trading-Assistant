"""
Model Factory for instantiating different machine learning models.
"""
import logging
from typing import Dict, Any, Optional

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

logger = logging.getLogger(__name__)

class ModelFactory:
    """Factory class to create and configure machine learning models."""
    
    @staticmethod
    def get_model(model_name: str, params: Optional[Dict[str, Any]] = None):
        """
        Creates and returns a machine learning model instance based on the provided name.
        
        Args:
            model_name (str): The name of the model to instantiate.
                Supported values: 'random_forest', 'xgboost', 'lightgbm', 'catboost'.
            params (Optional[Dict[str, Any]]): Hyperparameters for the model.
                If None, default hyperparameters will be used.
        
        Returns:
            The instantiated machine learning model.
            
        Raises:
            ValueError: If an unsupported model_name is provided.
        """
        model_name = model_name.lower()
        params = params or {}
        
        if model_name == 'random_forest':
            default_params = {
                'n_estimators': 100,
                'random_state': 42,
                'n_jobs': -1
            }
            default_params.update(params)
            logger.info(f"Instantiating RandomForestClassifier with params: {default_params}")
            return RandomForestClassifier(**default_params)
            
        elif model_name == 'xgboost':
            default_params = {
                'n_estimators': 100,
                'random_state': 42,
                'n_jobs': -1,
                'eval_metric': 'mlogloss',
                'use_label_encoder': False
            }
            default_params.update(params)
            logger.info(f"Instantiating XGBClassifier with params: {default_params}")
            return XGBClassifier(**default_params)
            
        elif model_name == 'lightgbm':
            default_params = {
                'n_estimators': 100,
                'random_state': 42,
                'n_jobs': -1,
                'verbose': -1
            }
            default_params.update(params)
            logger.info(f"Instantiating LGBMClassifier with params: {default_params}")
            return LGBMClassifier(**default_params)
            
        elif model_name == 'catboost':
            default_params = {
                'iterations': 100,
                'random_seed': 42,
                'verbose': False
            }
            default_params.update(params)
            logger.info(f"Instantiating CatBoostClassifier with params: {default_params}")
            return CatBoostClassifier(**default_params)
            
        else:
            logger.error(f"Unsupported model name: {model_name}")
            raise ValueError(f"Unsupported model name: {model_name}. "
                             f"Supported models are: 'random_forest', 'xgboost', 'lightgbm', 'catboost'.")

