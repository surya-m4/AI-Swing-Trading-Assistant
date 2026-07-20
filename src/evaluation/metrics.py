import os
import json
import logging
from typing import Dict, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
)

logger = logging.getLogger(__name__)

class MetricsCalculator:
    """Class to calculate metrics and generate plots."""

    def __init__(self, artifacts_dir: str = 'artifacts'):
        """
        Initializes the MetricsCalculator.

        Args:
            artifacts_dir (str): Directory where plots and reports will be saved.
        """
        self.artifacts_dir = artifacts_dir
        os.makedirs(self.artifacts_dir, exist_ok=True)

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculates core classification metrics.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.
            y_prob (np.ndarray, optional): Predicted probabilities for the positive class.

        Returns:
            Dict[str, float]: Calculated metrics.
        """
        logger.info("Calculating metrics...")
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        }
        
        if y_prob is not None:
            try:
                # Handle binary classification ROC-AUC
                if len(np.unique(y_true)) == 2:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
            except Exception as e:
                logger.warning(f"Could not calculate ROC-AUC: {e}")
                
        return metrics

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> str:
        """
        Plots and saves the confusion matrix.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.

        Returns:
            str: Path to the saved plot.
        """
        logger.info("Generating confusion matrix plot...")
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        
        path = os.path.join(self.artifacts_dir, 'confusion_matrix.png')
        plt.savefig(path, bbox_inches='tight')
        plt.close()
        return path

    def plot_roc_curve(self, y_true: np.ndarray, y_prob: np.ndarray) -> str:
        """
        Plots and saves the ROC curve.

        Args:
            y_true (np.ndarray): True labels.
            y_prob (np.ndarray): Predicted probabilities for the positive class.

        Returns:
            str: Path to the saved plot.
        """
        logger.info("Generating ROC curve plot...")
        try:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc_score = roc_auc_score(y_true, y_prob)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc_score:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc="lower right")
            
            path = os.path.join(self.artifacts_dir, 'roc_curve.png')
            plt.savefig(path, bbox_inches='tight')
            plt.close()
            return path
        except Exception as e:
            logger.warning(f"Failed to generate ROC curve: {e}")
            plt.close()
            return ""

    def plot_precision_recall_curve(self, y_true: np.ndarray, y_prob: np.ndarray) -> str:
        """
        Plots and saves the Precision-Recall curve.

        Args:
            y_true (np.ndarray): True labels.
            y_prob (np.ndarray): Predicted probabilities.

        Returns:
            str: Path to the saved plot.
        """
        logger.info("Generating Precision-Recall curve plot...")
        try:
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, color='b', lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            
            path = os.path.join(self.artifacts_dir, 'precision_recall_curve.png')
            plt.savefig(path, bbox_inches='tight')
            plt.close()
            return path
        except Exception as e:
            logger.warning(f"Failed to generate PR curve: {e}")
            plt.close()
            return ""

    def save_evaluation_report(self, metrics: Dict[str, float]) -> str:
        """
        Saves the metrics to a JSON file.

        Args:
            metrics (Dict[str, float]): Metrics dictionary.

        Returns:
            str: Path to the saved report.
        """
        path = os.path.join(self.artifacts_dir, 'evaluation_report.json')
        logger.info(f"Saving evaluation report to {path}...")
        with open(path, 'w') as f:
            json.dump(metrics, f, indent=4)
        return path
