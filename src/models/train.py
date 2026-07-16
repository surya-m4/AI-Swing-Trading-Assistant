import logging
import argparse
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from src.models.model_factory import ModelFactory
from src.models.trainer import Trainer
from src.models.save_model import ModelSaver

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_and_validate_data(filepath: str, label_col: str) -> pd.DataFrame:
    """
    Loads data from CSV and validates the presence of features and label column.

    Args:
        filepath (str): Path to the dataset.
        label_col (str): The name of the label column.

    Returns:
        pd.DataFrame: The loaded dataset.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        ValueError: If the dataset is empty or missing the label column.
    """
    logger.info(f"Loading data from {filepath}...")
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        raise
    except pd.errors.EmptyDataError:
        error_msg = "The dataset is empty."
        logger.error(error_msg)
        raise ValueError(error_msg)
        
    if df.empty:
        error_msg = "The dataset is empty."
        logger.error(error_msg)
        raise ValueError(error_msg)
        
    if label_col not in df.columns:
        error_msg = f"Label column '{label_col}' not found in the dataset."
        logger.error(error_msg)
        raise ValueError(error_msg)
        
    # Assume all other columns except some typical non-feature columns are features
    # If there are specific non-feature columns (like Date, Ticker), they should be handled
    # For now, just ensure we have more than just the label column
    if len(df.columns) <= 1:
         error_msg = "Dataset contains no features."
         logger.error(error_msg)
         raise ValueError(error_msg)
         
    logger.info(f"Data loaded successfully. Shape: {df.shape}")
    return df

def prepare_data(df: pd.DataFrame, label_col: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits the data into training and testing sets (80/20).

    Args:
        df (pd.DataFrame): The full dataset.
        label_col (str): The name of the label column.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: X_train, X_test, y_train, y_test
    """
    logger.info("Splitting data into 80/20 train/test sets...")
    
    # Drop typical non-predictive columns if they exist just in case
    drop_cols = [label_col]
    if 'Date' in df.columns:
        drop_cols.append('Date')
    if 'Ticker' in df.columns:
        drop_cols.append('Ticker')
        
    X = df.drop(columns=[col for col in drop_cols if col in df.columns])
    y = df[label_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logger.info(f"Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test

def main(data_filepath: str, label_col: str, output_dir: str):
    """Main pipeline for model training."""
    try:
        # 1. Load and validate
        df = load_and_validate_data(data_filepath, label_col)
        
        # 2. Split data
        X_train, X_test, y_train, y_test = prepare_data(df, label_col)
        
        # 3. Initialize models via factory
        models_to_train = ['logistic_regression', 'random_forest', 'xgboost']
        trainer = Trainer()
        
        # 4. Train and evaluate
        for model_name in models_to_train:
            model = ModelFactory.get_model(model_name)
            trainer.train_and_evaluate(model_name, model, X_train, y_train, X_test, y_test)
            
        # 5. Select best model
        best_model_name, best_model, best_metrics = trainer.get_best_model(metric='f1_score')
        
        # 6. Save outputs
        saver = ModelSaver(output_dir=output_dir)
        saver.save_model(best_model)
        saver.save_metrics(best_metrics)
        
        # Generate classification report for best model
        y_pred = best_model.predict(X_test)
        report = classification_report(y_test, y_pred, zero_division=0)
        saver.save_classification_report(report)
        
        logger.info("Training pipeline completed successfully.")
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train machine learning models for Swing Trading Assistant.")
    parser.add_argument("--data", type=str, default="data/processed/labeled_data.csv", help="Path to labeled dataset.")
    parser.add_argument("--label", type=str, default="Target", help="Name of the label column.")
    parser.add_argument("--output", type=str, default="models", help="Directory to save model artifacts.")
    
    args = parser.parse_args()
    main(args.data, args.label, args.output)
