"""
Model Predictor logic for the API.
Handles loading the model and making predictions.
"""
import os
import joblib
import logging
from typing import Dict, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class ModelPredictor:
    """Class to load the model and make predictions."""
    
    def __init__(self, models_dir: str = 'models'):
        """
        Initializes the ModelPredictor, loading the best model and label encoder.
        
        Args:
            models_dir (str): Path to the directory containing model files.
        """
        self.models_dir = models_dir
        self.model = None
        self.label_encoder = None
        self.model_name = "AI_Swing_Trading_Model"
        
        self.load_model()
        
    def load_model(self):
        """Loads the best model and label encoder from disk."""
        # Find the best model, typically saved as something like *_optimized.pkl 
        # or best_model.pkl. If multiple exist, we take a default one.
        # For this implementation, we assume we want 'random_forest_optimized.pkl' or similar,
        # but we'll try to find a valid .pkl file in the models dir.
        
        if not os.path.exists(self.models_dir):
            logger.warning(f"Models directory '{self.models_dir}' not found.")
            return

        le_path = os.path.join(self.models_dir, 'label_encoder.pkl')
        if os.path.exists(le_path):
            self.label_encoder = joblib.load(le_path)
            logger.info("Loaded label encoder.")
        else:
            logger.warning("Label encoder not found.")

        # Try to load a specific optimized model or just the first .pkl that isn't the encoder
        potential_models = [f for f in os.listdir(self.models_dir) if f.endswith('.pkl') and f != 'label_encoder.pkl']
        
        if potential_models:
            # Prefer 'optimized' if available
            optimized_models = [f for f in potential_models if 'optimized' in f]
            target_model = optimized_models[0] if optimized_models else potential_models[0]
            
            model_path = os.path.join(self.models_dir, target_model)
            self.model = joblib.load(model_path)
            self.model_name = target_model.replace('.pkl', '')
            logger.info(f"Loaded model: {self.model_name}")
        else:
            logger.warning("No trained models found in the models directory.")
            
    def predict(self, features: Dict[str, float]) -> Tuple[str, float]:
        """
        Generates a prediction for the given features.
        
        Args:
            features (Dict[str, float]): Feature dictionary.
            
        Returns:
            Tuple[str, float]: Action string and confidence score.
            
        Raises:
            RuntimeError: If the model is not loaded.
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded. Cannot make predictions.")
            
        df = pd.DataFrame([features])
        
        prediction = self.model.predict(df)[0]
        
        # Get probability if available
        confidence = 0.0
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(df)[0]
            confidence = float(np.max(proba))
            
        # Decode label
        action = str(prediction)
        if self.label_encoder:
            action = self.label_encoder.inverse_transform([prediction])[0]
            
        return action, confidence
