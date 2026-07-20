import os
import argparse
import logging
import joblib
import pandas as pd
from typing import Tuple, Any

from src.evaluation.metrics import MetricsCalculator
from src.evaluation.explainability import ModelExplainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Orchestrator class for Model Evaluation and Explainability."""

    def __init__(self, model_path: str, data_path: str, label_col: str, artifacts_dir: str = 'artifacts'):
        self.model_path = model_path
        self.data_path = data_path
        self.label_col = label_col
        self.artifacts_dir = artifacts_dir
        os.makedirs(self.artifacts_dir, exist_ok=True)

    def load_inputs(self) -> Tuple[Any, pd.DataFrame, pd.Series]:
        """Loads the trained model and dataset."""
        logger.info(f"Loading model from {self.model_path}...")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        model = joblib.load(self.model_path)

        logger.info(f"Loading data from {self.data_path}...")
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        df = pd.read_csv(self.data_path)
        
        if df.empty:
            raise ValueError("The test dataset is empty.")
        if self.label_col not in df.columns:
            raise ValueError(f"Label column '{self.label_col}' not found in dataset.")

        # In a real scenario, you'd apply the exact same splits or just evaluate on the given df assuming it's a test set
        # We assume the input data_path is the test dataset for evaluation purposes
        drop_cols = [self.label_col]
        if 'Date' in df.columns:
            drop_cols.append('Date')
        if 'Ticker' in df.columns:
            drop_cols.append('Ticker')
            
        X = df.drop(columns=[col for col in drop_cols if col in df.columns])
        y = df[self.label_col]
        
        return model, X, y

    def run_evaluation(self):
        """Runs the full evaluation and explainability pipeline."""
        try:
            model, X, y = self.load_inputs()
            
            logger.info("Generating predictions...")
            y_pred = model.predict(X)
            
            # Not all models support predict_proba
            y_prob = None
            if hasattr(model, "predict_proba"):
                try:
                    y_prob = model.predict_proba(X)[:, 1]
                except Exception as e:
                    logger.warning(f"Could not extract probabilities: {e}")

            # 1. Metrics
            metrics_calc = MetricsCalculator(artifacts_dir=self.artifacts_dir)
            metrics = metrics_calc.calculate_metrics(y, y_pred, y_prob)
            metrics_calc.save_evaluation_report(metrics)
            metrics_calc.plot_confusion_matrix(y, y_pred)
            
            if y_prob is not None:
                metrics_calc.plot_roc_curve(y, y_prob)
                metrics_calc.plot_precision_recall_curve(y, y_prob)
                
            # 2. Explainability
            explainer = ModelExplainer(model, artifacts_dir=self.artifacts_dir)
            explainer.extract_feature_importance(X.columns)
            
            shap_values = explainer.compute_shap_values(X)
            
            # Limit X_sample matching shap computation size
            if len(X) > 1000:
                X_sample = X.sample(n=1000, random_state=42)
            else:
                X_sample = X
                
            explainer.generate_summary_plot(shap_values, X_sample)
            explainer.generate_bar_plot(shap_values, X_sample)
            
            logger.info("Evaluation and Explainability pipeline completed successfully.")
            
        except Exception as e:
            logger.error(f"Evaluation pipeline failed: {e}")
            raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained machine learning model.")
    parser.add_argument("--model", type=str, default="models/best_model.pkl", help="Path to trained model.")
    parser.add_argument("--data", type=str, default="data/processed/test_data.csv", help="Path to test dataset.")
    parser.add_argument("--label", type=str, default="Target", help="Name of the label column.")
    parser.add_argument("--output", type=str, default="artifacts", help="Directory to save evaluation artifacts.")
    
    args = parser.parse_args()
    
    evaluator = ModelEvaluator(args.model, args.data, args.label, args.output)
    evaluator.run_evaluation()
