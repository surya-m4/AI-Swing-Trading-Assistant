"""
Optimizer module for tuning model hyperparameters.
"""
import logging
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np

import optuna
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.base import clone

logger = logging.getLogger(__name__)

class Optimizer:
    """Class to handle hyperparameter optimization."""
    
    def __init__(
        self, 
        model_name: str, 
        model: Any, 
        search_space: Any, 
        optimization_type: str = 'optuna', 
        scoring: str = 'f1_weighted', 
        cv: int = 5,
        n_trials: int = 20
    ):
        """
        Initializes the Optimizer.
        
        Args:
            model_name (str): Name of the model.
            model (Any): The instantiated machine learning model.
            search_space (Any): Search space dictionary (for grid/random) or callable (for optuna).
            optimization_type (str): Type of optimization ('grid', 'random', 'optuna').
            scoring (str): Scikit-learn scoring metric.
            cv (int): Number of cross-validation folds.
            n_trials (int): Number of trials for random search or optuna.
        """
        self.model_name = model_name
        self.model = model
        self.search_space = search_space
        self.optimization_type = optimization_type.lower()
        self.scoring = scoring
        self.cv = cv
        self.n_trials = n_trials

    def optimize(self, X_train: pd.DataFrame, y_train: np.ndarray) -> Tuple[Any, Dict[str, Any]]:
        """
        Runs the specified optimization strategy.
        
        Args:
            X_train (pd.DataFrame): Training features.
            y_train (np.ndarray): Training labels.
            
        Returns:
            Tuple[Any, Dict[str, Any]]: The best trained model and its hyperparameters.
        """
        logger.info(f"Starting {self.optimization_type} optimization for {self.model_name}")
        
        if self.optimization_type == 'grid':
            return self._run_grid_search(X_train, y_train)
        elif self.optimization_type == 'random':
            return self._run_random_search(X_train, y_train)
        elif self.optimization_type == 'optuna':
            return self._run_optuna(X_train, y_train)
        else:
            raise ValueError(f"Unsupported optimization type: {self.optimization_type}")

    def _run_grid_search(self, X_train: pd.DataFrame, y_train: np.ndarray) -> Tuple[Any, Dict[str, Any]]:
        """Runs GridSearchCV."""
        if not isinstance(self.search_space, dict):
            raise TypeError("Grid search requires a dictionary search space.")
            
        search = GridSearchCV(
            estimator=self.model,
            param_grid=self.search_space,
            scoring=self.scoring,
            cv=self.cv,
            n_jobs=-1,
            verbose=1
        )
        search.fit(X_train, y_train)
        logger.info(f"Grid search completed. Best score: {search.best_score_}")
        return search.best_estimator_, search.best_params_

    def _run_random_search(self, X_train: pd.DataFrame, y_train: np.ndarray) -> Tuple[Any, Dict[str, Any]]:
        """Runs RandomizedSearchCV."""
        if not isinstance(self.search_space, dict):
            raise TypeError("Random search requires a dictionary search space.")
            
        search = RandomizedSearchCV(
            estimator=self.model,
            param_distributions=self.search_space,
            n_iter=self.n_trials,
            scoring=self.scoring,
            cv=self.cv,
            n_jobs=-1,
            verbose=1,
            random_state=42
        )
        search.fit(X_train, y_train)
        logger.info(f"Random search completed. Best score: {search.best_score_}")
        return search.best_estimator_, search.best_params_

    def _run_optuna(self, X_train: pd.DataFrame, y_train: np.ndarray) -> Tuple[Any, Dict[str, Any]]:
        """Runs Optuna optimization."""
        if not callable(self.search_space):
            raise TypeError("Optuna optimization requires a callable search space.")

        def objective(trial):
            params = self.search_space(trial)
            model = clone(self.model)
            model.set_params(**params)
            
            scores = cross_val_score(
                model, X_train, y_train, scoring=self.scoring, cv=self.cv, n_jobs=-1
            )
            return scores.mean()

        study = optuna.create_study(direction='maximize')
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=self.n_trials)
        
        logger.info(f"Optuna optimization completed. Best score: {study.best_value}")
        best_params = study.best_params
        
        # Train final model on all training data with best params
        best_model = clone(self.model)
        best_model.set_params(**best_params)
        best_model.fit(X_train, y_train)
        
        return best_model, best_params
