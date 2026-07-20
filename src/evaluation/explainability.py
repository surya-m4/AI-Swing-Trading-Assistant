import os
import logging
from typing import Any
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
import joblib

logger = logging.getLogger(__name__)

class ModelExplainer:
    """Class to handle model explainability and feature importance using SHAP."""

    def __init__(self, model: Any, artifacts_dir: str = 'artifacts'):
        """
        Initializes the ModelExplainer.

        Args:
            model (Any): The trained machine learning model.
            artifacts_dir (str): Directory to save plots and artifacts.
        """
        self.model = model
        self.artifacts_dir = artifacts_dir
        os.makedirs(self.artifacts_dir, exist_ok=True)

    def extract_feature_importance(self, feature_names: pd.Index) -> str:
        """
        Extracts tree-based feature importance if available, saves to CSV and plots it.

        Args:
            feature_names (pd.Index): The names of the features.

        Returns:
            str: Path to the saved feature importance CSV.
        """
        if not hasattr(self.model, 'feature_importances_'):
            logger.info("Model does not have feature_importances_ attribute.")
            return ""
            
        logger.info("Extracting and saving feature importance...")
        importances = self.model.feature_importances_
        df_imp = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        
        csv_path = os.path.join(self.artifacts_dir, 'feature_importance.csv')
        df_imp.to_csv(csv_path, index=False)
        
        # Plot top 20
        plt.figure(figsize=(10, 8))
        top_n = df_imp.head(20)
        plt.barh(top_n['Feature'][::-1], top_n['Importance'][::-1], color='skyblue')
        plt.xlabel('Importance')
        plt.title('Top 20 Feature Importances')
        
        img_path = os.path.join(self.artifacts_dir, 'feature_importance.png')
        plt.savefig(img_path, bbox_inches='tight')
        plt.close()
        
        return csv_path

    def compute_shap_values(self, X: pd.DataFrame) -> Any:
        """
        Computes SHAP values for the given dataset.

        Args:
            X (pd.DataFrame): Dataset features to explain.

        Returns:
            Any: The computed SHAP values object.
        """
        logger.info("Computing SHAP values...")
        # Subsample to avoid massive delays on large test sets
        if len(X) > 1000:
            logger.info("Subsampling dataset for SHAP computation (n=1000).")
            X_sample = X.sample(n=1000, random_state=42)
        else:
            X_sample = X
            
        try:
            # TreeExplainer is fast for trees, Kernel/Linear for others
            # We try TreeExplainer first as our models are likely RF or XGBoost
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer(X_sample)
        except Exception:
            logger.info("TreeExplainer failed, falling back to KernelExplainer...")
            # For linear models or unsupported trees
            background = shap.sample(X, 100)
            # Use predict instead of predict_proba if probabilities are not supported
            predict_fn = getattr(self.model, "predict_proba", self.model.predict)
            explainer = shap.KernelExplainer(predict_fn, background)
            shap_values = explainer(X_sample)
            
        # Save shap values
        save_path = os.path.join(self.artifacts_dir, 'shap_values.pkl')
        joblib.dump(shap_values, save_path)
        logger.info(f"SHAP values saved to {save_path}")
        
        return shap_values

    def generate_summary_plot(self, shap_values: Any, X: pd.DataFrame) -> str:
        """
        Generates and saves a SHAP summary plot.

        Args:
            shap_values (Any): Precomputed SHAP values.
            X (pd.DataFrame): The dataset (subset if sampled) for context.

        Returns:
            str: Path to the saved plot.
        """
        logger.info("Generating SHAP summary plot...")
        try:
            plt.figure()
            
            if hasattr(shap_values, 'values'):
                if len(shap_values.shape) > 2:
                    shap.summary_plot(shap_values[:, :, 1], show=False)
                else:
                    shap.summary_plot(shap_values, show=False)
            else:
                if isinstance(shap_values, list):
                    shap.summary_plot(shap_values[1], X, show=False)
                else:
                    shap.summary_plot(shap_values, X, show=False)
                    
            path = os.path.join(self.artifacts_dir, 'summary_plot.png')
            plt.savefig(path, bbox_inches='tight')
            plt.close()
            return path
        except Exception as e:
            logger.warning(f"Failed to generate SHAP summary plot: {e}")
            plt.close()
            return ""

    def generate_bar_plot(self, shap_values: Any, X: pd.DataFrame) -> str:
        """
        Generates and saves a SHAP bar plot.

        Args:
            shap_values (Any): Precomputed SHAP values.
            X (pd.DataFrame): Dataset context.

        Returns:
            str: Path to the saved plot.
        """
        logger.info("Generating SHAP bar plot...")
        try:
            plt.figure()
            if hasattr(shap_values, 'values'):
                if len(shap_values.shape) > 2:
                    shap.plots.bar(shap_values[:, :, 1], show=False)
                else:
                    shap.plots.bar(shap_values, show=False)
            else:
                if isinstance(shap_values, list):
                    shap.summary_plot(shap_values[1], X, plot_type="bar", show=False)
                else:
                    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
                    
            path = os.path.join(self.artifacts_dir, 'bar_plot.png')
            plt.savefig(path, bbox_inches='tight')
            plt.close()
            return path
        except Exception as e:
            logger.warning(f"Failed to generate SHAP bar plot: {e}")
            plt.close()
            return ""
