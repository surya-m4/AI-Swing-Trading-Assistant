"""
Search Spaces module.
Defines the hyperparameter search spaces for various models.
"""
from typing import Dict, Any, Callable
import logging

logger = logging.getLogger(__name__)

class SearchSpaces:
    """Class containing hyperparameter search spaces for supported models."""

    @staticmethod
    def get_grid_search_space(model_name: str) -> Dict[str, list]:
        """Returns the search space for GridSearchCV/RandomizedSearchCV."""
        model_name = model_name.lower()
        if model_name == 'random_forest':
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            }
        elif model_name == 'xgboost':
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 10],
                'learning_rate': [0.01, 0.1, 0.2]
            }
        elif model_name == 'lightgbm':
            return {
                'n_estimators': [50, 100, 200],
                'num_leaves': [31, 50, 100],
                'learning_rate': [0.01, 0.1, 0.2]
            }
        elif model_name == 'catboost':
            return {
                'iterations': [50, 100, 200],
                'depth': [4, 6, 8],
                'learning_rate': [0.01, 0.1, 0.2]
            }
        else:
            raise ValueError(f"Unsupported model name: {model_name}")

    @staticmethod
    def get_optuna_search_space(model_name: str) -> Callable:
        """Returns a callable that takes an Optuna trial and returns parameters."""
        model_name = model_name.lower()
        
        if model_name == 'random_forest':
            def rf_space(trial) -> Dict[str, Any]:
                return {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_categorical('max_depth', [None, 10, 20, 30]),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 15),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
                }
            return rf_space
            
        elif model_name == 'xgboost':
            def xgb_space(trial) -> Dict[str, Any]:
                return {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 12),
                    'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0)
                }
            return xgb_space
            
        elif model_name == 'lightgbm':
            def lgb_space(trial) -> Dict[str, Any]:
                return {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                    'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
                    'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0)
                }
            return lgb_space
            
        elif model_name == 'catboost':
            def cb_space(trial) -> Dict[str, Any]:
                return {
                    'iterations': trial.suggest_int('iterations', 50, 300),
                    'depth': trial.suggest_int('depth', 4, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
                    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-2, 10.0, log=True)
                }
            return cb_space
            
        else:
            raise ValueError(f"Unsupported model name: {model_name}")
