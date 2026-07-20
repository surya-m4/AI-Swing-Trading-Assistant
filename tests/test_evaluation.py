import os
import json
import tempfile
import unittest
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from src.evaluation.metrics import MetricsCalculator
from src.evaluation.explainability import ModelExplainer
from src.evaluation.evaluator import ModelEvaluator

class TestEvaluation(unittest.TestCase):
    
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.artifacts_dir = self.temp_dir.name
        
        # Synthetic data
        np.random.seed(42)
        self.X = pd.DataFrame(np.random.randn(100, 5), columns=[f'feature_{i}' for i in range(5)])
        self.y = pd.Series(np.random.randint(0, 2, 100), name='Target')
        
        self.y_pred = np.random.randint(0, 2, 100)
        self.y_prob = np.random.rand(100)
        
        # Train a dummy model
        self.model = RandomForestClassifier(random_state=42)
        self.model.fit(self.X, self.y)
        
        self.model_path = os.path.join(self.artifacts_dir, 'best_model.pkl')
        joblib.dump(self.model, self.model_path)
        
        self.data_path = os.path.join(self.artifacts_dir, 'test_data.csv')
        df = self.X.copy()
        df['Target'] = self.y
        df.to_csv(self.data_path, index=False)
        
        self.metrics_calc = MetricsCalculator(artifacts_dir=self.artifacts_dir)
        self.explainer = ModelExplainer(self.model, artifacts_dir=self.artifacts_dir)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_calculate_metrics(self):
        metrics = self.metrics_calc.calculate_metrics(self.y, self.y_pred, self.y_prob)
        self.assertIn('accuracy', metrics)
        self.assertIn('f1_score', metrics)
        self.assertIn('roc_auc', metrics)

    def test_plot_confusion_matrix(self):
        path = self.metrics_calc.plot_confusion_matrix(self.y, self.y_pred)
        self.assertTrue(os.path.exists(path))

    def test_plot_roc_curve(self):
        path = self.metrics_calc.plot_roc_curve(self.y, self.y_prob)
        self.assertTrue(os.path.exists(path))

    def test_plot_precision_recall_curve(self):
        path = self.metrics_calc.plot_precision_recall_curve(self.y, self.y_prob)
        self.assertTrue(os.path.exists(path))

    def test_save_evaluation_report(self):
        metrics = {'accuracy': 0.9}
        path = self.metrics_calc.save_evaluation_report(metrics)
        self.assertTrue(os.path.exists(path))
        with open(path, 'r') as f:
            data = json.load(f)
            self.assertEqual(data['accuracy'], 0.9)

    def test_extract_feature_importance(self):
        path = self.explainer.extract_feature_importance(self.X.columns)
        self.assertTrue(os.path.exists(path))
        self.assertTrue(os.path.exists(os.path.join(self.artifacts_dir, 'feature_importance.png')))

    def test_compute_shap_values(self):
        shap_values = self.explainer.compute_shap_values(self.X)
        self.assertIsNotNone(shap_values)
        self.assertTrue(os.path.exists(os.path.join(self.artifacts_dir, 'shap_values.pkl')))
        
        # Test plots
        path_summary = self.explainer.generate_summary_plot(shap_values, self.X)
        self.assertTrue(os.path.exists(path_summary))
        
        path_bar = self.explainer.generate_bar_plot(shap_values, self.X)
        self.assertTrue(os.path.exists(path_bar))

    def test_evaluator_missing_model(self):
        evaluator = ModelEvaluator(
            model_path='non_existent_model.pkl',
            data_path=self.data_path,
            label_col='Target',
            artifacts_dir=self.artifacts_dir
        )
        with self.assertRaises(FileNotFoundError):
            evaluator.run_evaluation()

    def test_evaluator_missing_data(self):
        evaluator = ModelEvaluator(
            model_path=self.model_path,
            data_path='non_existent_data.csv',
            label_col='Target',
            artifacts_dir=self.artifacts_dir
        )
        with self.assertRaises(FileNotFoundError):
            evaluator.run_evaluation()

    def test_evaluator_missing_label(self):
        # Remove label column
        df = self.X.copy()
        invalid_data_path = os.path.join(self.artifacts_dir, 'invalid_data.csv')
        df.to_csv(invalid_data_path, index=False)
        
        evaluator = ModelEvaluator(
            model_path=self.model_path,
            data_path=invalid_data_path,
            label_col='Target',
            artifacts_dir=self.artifacts_dir
        )
        with self.assertRaises(ValueError):
            evaluator.run_evaluation()

    def test_evaluator_run_evaluation(self):
        evaluator = ModelEvaluator(
            model_path=self.model_path,
            data_path=self.data_path,
            label_col='Target',
            artifacts_dir=self.artifacts_dir
        )
        evaluator.run_evaluation()
        
        # Check that outputs are created
        self.assertTrue(os.path.exists(os.path.join(self.artifacts_dir, 'evaluation_report.json')))
        self.assertTrue(os.path.exists(os.path.join(self.artifacts_dir, 'confusion_matrix.png')))
        self.assertTrue(os.path.exists(os.path.join(self.artifacts_dir, 'roc_curve.png')))
        self.assertTrue(os.path.exists(os.path.join(self.artifacts_dir, 'precision_recall_curve.png')))
        self.assertTrue(os.path.exists(os.path.join(self.artifacts_dir, 'feature_importance.csv')))
        self.assertTrue(os.path.exists(os.path.join(self.artifacts_dir, 'feature_importance.png')))
        self.assertTrue(os.path.exists(os.path.join(self.artifacts_dir, 'shap_values.pkl')))
        self.assertTrue(os.path.exists(os.path.join(self.artifacts_dir, 'summary_plot.png')))
        self.assertTrue(os.path.exists(os.path.join(self.artifacts_dir, 'bar_plot.png')))

if __name__ == '__main__':
    unittest.main()
