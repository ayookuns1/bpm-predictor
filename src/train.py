import pandas as pd
import numpy as np
from data.load_data import DataLoader
from features.feature_engineering import create_feature_pipeline
from models.train_model import train_complete_pipeline
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main training function"""
    logger.info("Starting BPM prediction model training...")
    
    # Load data
    logger.info("Loading data...")
    loader = DataLoader('../data/raw/train_less.csv')
    data = loader.load_data()
    
    # Separate features and target
    X, y = loader.get_feature_target()
    
    # Feature engineering
    logger.info("Performing feature engineering...")
    feature_pipeline = create_feature_pipeline()
    X_processed = feature_pipeline.fit_transform(X, y)
    
    # Save feature pipeline
    import joblib
    joblib.dump(feature_pipeline, '../models/feature_pipeline.pkl')
    
    # Train models
    logger.info("Training models...")
    trainer, results = train_complete_pipeline(X_processed, y)
    
    logger.info("Training completed successfully!")
    logger.info("Best model saved to ../models/best_model.pkl")
    logger.info("Feature pipeline saved to ../models/feature_pipeline.pkl")
    
    # Print best model results
    best_model_name = min(trainer.models.items(), 
                         key=lambda x: x[1]['best_score'])[0]
    best_metrics = results[best_model_name]
    
    print(f"\n=== BEST MODEL: {best_model_name.upper()} ===")
    print(f"RMSE: {best_metrics['rmse']:.4f}")
    print(f"MAE: {best_metrics['mae']:.4f}")
    print(f"R²: {best_metrics['r2']:.4f}")

if __name__ == "__main__":
    main()