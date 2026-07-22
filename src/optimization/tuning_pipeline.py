"""
Tuning Pipeline module.
Orchestrates the model tuning and evaluation process.
"""
import logging
import json
import os
from typing import Dict, Any, List

from src.training.model_factory import ModelFactory
from src.training.trainer import Trainer
from src.optimization.search_spaces import SearchSpaces
from src.optimization.optimizer import Optimizer
from src.mlflow_tracking.experiment_tracker import ExperimentTracker

logger = logging.getLogger(__name__)

class TuningPipeline:
    """Pipeline to execute hyperparameter tuning and model evaluation."""
    
    def __init__(
        self, 
        data_filename: str, 
        models_to_tune: List[str] = None,
        optimization_type: str = 'optuna',
        n_trials: int = 20
    ):
        """
        Initializes the TuningPipeline.
        
        Args:
            data_filename (str): The filename of the labeled dataset in data/labeled/.
            models_to_tune (List[str]): List of model names to tune.
            optimization_type (str): Type of optimization ('grid', 'random', 'optuna').
            n_trials (int): Number of trials for random search or optuna.
        """
        self.data_filename = data_filename
        self.models_to_tune = models_to_tune or ['random_forest', 'xgboost', 'lightgbm', 'catboost']
        self.optimization_type = optimization_type
        self.n_trials = n_trials
        self.trainer = Trainer()
        
    def run(self) -> Dict[str, Dict[str, Any]]:
        """
        Runs the tuning pipeline.
        
        Returns:
            Dict[str, Dict[str, Any]]: A dictionary containing comparison results for each model.
        """
        logger.info("Starting hyperparameter tuning pipeline...")
        
        # Load and prepare data
        df = self.trainer.load_data(self.data_filename)
        X_train, X_test, y_train, y_test = self.trainer.prepare_data(df)
        
        comparison_results = {}
        
        for model_name in self.models_to_tune:
            logger.info(f"--- Tuning model: {model_name} ---")
            
            with ExperimentTracker(experiment_name="AI Swing Trading - Optimization") as tracker:
                # 1. Baseline Model
                baseline_model = ModelFactory.get_model(model_name)
                trained_baseline = self.trainer.train_model(f"{model_name}_baseline", baseline_model, X_train, y_train)
                baseline_metrics = self.trainer.evaluate_model(f"{model_name}_baseline", trained_baseline, X_test, y_test)
                
                # 2. Optimization
                if self.optimization_type == 'optuna':
                    search_space = SearchSpaces.get_optuna_search_space(model_name)
                else:
                    search_space = SearchSpaces.get_grid_search_space(model_name)
                    
                optimizer = Optimizer(
                    model_name=model_name,
                    model=baseline_model,
                    search_space=search_space,
                    optimization_type=self.optimization_type,
                    n_trials=self.n_trials
                )
                
                best_model, best_params = optimizer.optimize(X_train, y_train)
                
                # 3. Evaluate Optimized Model
                tracker.active_run = tracker.start_run(run_name=f"tuned_{model_name}") if not tracker.active_run else tracker.active_run
                tracker.log_parameters({"model_name": model_name, "optimization_type": self.optimization_type})
                tracker.log_parameters(best_params)
                
                tuned_metrics = self.trainer.evaluate_model(f"{model_name}_tuned", best_model, X_test, y_test, tracker=tracker)
                
                # Save optimized model and params
                model_path = self.trainer.save_model(best_model, f"{model_name}_optimized")
                tracker.log_artifact(model_path)
                
                params_path = os.path.join(self.trainer.artifacts_dir, f"{model_name}_best_params.json")
                with open(params_path, 'w') as f:
                    json.dump(best_params, f, indent=4)
                tracker.log_artifact(params_path)
                
                comparison_results[model_name] = {
                    'baseline_metrics': baseline_metrics,
                    'tuned_metrics': tuned_metrics,
                    'best_parameters': best_params
                }
                
        self.save_optimization_results(comparison_results)
        logger.info("Tuning pipeline completed successfully.")
        return comparison_results
        
    def save_optimization_results(self, results: Dict[str, Dict[str, Any]]) -> str:
        """
        Saves the optimization results to a JSON file.
        
        Args:
            results (Dict[str, Dict[str, Any]]): The comparison results.
            
        Returns:
            str: Path to the saved results.
        """
        report_path = os.path.join(self.trainer.artifacts_dir, 'optimization_results.json')
        logger.info(f"Saving optimization results to {report_path}")
        
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=4)
            
        return report_path
