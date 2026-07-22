"""
Train Pipeline module.
Orchestrates the model training and evaluation process.
"""
import logging
import json
import os
from typing import Dict, Any, List

from src.training.model_factory import ModelFactory
from src.training.trainer import Trainer
from src.mlflow_tracking.experiment_tracker import ExperimentTracker

logger = logging.getLogger(__name__)

class TrainPipeline:
    """Pipeline to execute the complete training process for multiple models."""
    
    def __init__(self, data_filename: str, models_to_train: List[str] = None):
        """
        Initializes the TrainPipeline.
        
        Args:
            data_filename (str): The filename of the labeled dataset in data/labeled/.
            models_to_train (List[str]): List of model names to train.
                Defaults to ['random_forest', 'xgboost', 'lightgbm', 'catboost'].
        """
        self.data_filename = data_filename
        self.models_to_train = models_to_train or ['random_forest', 'xgboost', 'lightgbm', 'catboost']
        self.trainer = Trainer()
        
    def run(self) -> Dict[str, Dict[str, float]]:
        """
        Runs the training pipeline.
        
        Returns:
            Dict[str, Dict[str, float]]: A dictionary containing evaluation metrics for each model.
        """
        logger.info("Starting training pipeline...")
        
        # Load and prepare data
        df = self.trainer.load_data(self.data_filename)
        X_train, X_test, y_train, y_test = self.trainer.prepare_data(df)
        
        results = {}
        
        for model_name in self.models_to_train:
            logger.info(f"--- Processing model: {model_name} ---")
            
            # Use a separate context for each model run in MLflow
            with ExperimentTracker(experiment_name="AI Swing Trading - Model Training") as tracker:
                # Instantiate model
                model = ModelFactory.get_model(model_name)
                
                # Train
                trained_model = self.trainer.train_model(model_name, model, X_train, y_train)
                
                # Evaluate and log to MLflow
                tracker.active_run = tracker.start_run(run_name=f"train_{model_name}") if not tracker.active_run else tracker.active_run
                tracker.log_parameters({"model_name": model_name})
                
                # Additional params logging depending on model type could be added here
                if hasattr(model, 'get_params'):
                    try:
                        tracker.log_parameters(model.get_params())
                    except Exception as e:
                        logger.warning(f"Could not log full parameters for {model_name}: {e}")
                
                metrics = self.trainer.evaluate_model(model_name, trained_model, X_test, y_test, tracker=tracker)
                results[model_name] = metrics
                
                # Save model
                model_path = self.trainer.save_model(trained_model, model_name)
                tracker.log_artifact(model_path)
                
        self.generate_comparison_report(results)
        logger.info("Training pipeline completed successfully.")
        return results
        
    def generate_comparison_report(self, results: Dict[str, Dict[str, float]]) -> str:
        """
        Generates and saves a JSON report comparing all models.
        
        Args:
            results (Dict[str, Dict[str, float]]): The evaluation results.
            
        Returns:
            str: Path to the comparison report.
        """
        report_path = os.path.join(self.trainer.artifacts_dir, 'model_comparison_report.json')
        logger.info(f"Generating model comparison report at {report_path}")
        
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=4)
            
        return report_path
