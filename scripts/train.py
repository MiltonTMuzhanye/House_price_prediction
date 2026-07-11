#!/usr/bin/env python
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
from src.house_price_prediction.data.preprocessing import DataPreprocessor
from src.house_price_prediction.training.trainer import ModelTrainer
from src.house_price_prediction.utils.logger import logger
from src.house_price_prediction.utils.helpers import save_artifact
from src.house_price_prediction.utils.config import config

def main():
    logger.info("Starting model training process...")
    
    try:
        # Load processed data
        data_path = config.get('data.processed_path', 'data/processed/house_prices_processed.csv')
        df = pd.read_csv(data_path)
        logger.info(f"Loaded data shape: {df.shape}")
        
        # Prepare data
        target = config.get('features.target', 'PRICE')
        X = df.drop(columns=[target])
        y = df[target]
        
        # Initialize preprocessor and fit transform
        preprocessor = DataPreprocessor()
        X_processed = preprocessor.fit_transform(X, y)
        
        # Save feature columns for inference
        feature_columns = X.columns.tolist()
        save_artifact(feature_columns, 'artifacts/feature_columns.joblib')
        
        # Train models
        trainer = ModelTrainer()
        trainer.initialize_models()
        results = trainer.train_and_evaluate(X_processed, y)
        
        # Print results
        logger.info("\n" + "="*50)
        logger.info("Model Training Results:")
        for model_name, metrics in results.items():
            logger.info(f"\n{model_name}:")
            for metric_name, value in metrics.items():
                logger.info(f"  {metric_name}: {value:.4f}")
        logger.info("="*50)
        
        logger.info("Model training completed successfully")
        
    except Exception as e:
        logger.error(f"Error in model training: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()