import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
from sklearn.model_selection import train_test_split
from src.house_price_prediction.data.preprocessing import DataPreprocessor
from src.house_price_prediction.training.trainer import ModelTrainer
from src.house_price_prediction.utils.logger import logger
from src.house_price_prediction.utils.helpers import save_artifact
from src.house_price_prediction.utils.config import config

def main():
    logger.info("Starting model training process...")
    
    try:
        # Load processed data
        data_path = config.get(
            'data.processed_path', 
            'data/processed/house_prices_processed.csv'

        )

        df = pd.read_csv(data_path)

        logger.info(
            f"Loaded data shape: {df.shape}"
        )
        
        # Prepare data
        target = config.get(
            'features.target', 
            'PRICE'
        )

        if target not in df.columns:
            raise ValueError(
                f"Target column '{target}' not found in dataset."
            )

        y = df[target]
        
        numeric_features = config.get(
            'features.numeric_features',
            ['SQFT', 'BEDROOMS']
        )

        categorical_features = config.get(
            'features.categorical_features',
            ['LOCATION', 'REGION', 'TITLED', 'LEASE', 'FOOTINGS']
        )

        model_features = (
            numeric_features +
            categorical_features
        )

        missing_features = [
            feature
            for feature in model_features
            if feature not in df.columns
        ]

        if missing_features:
            raise ValueError(
                f"Missing model features: {missing_features}"
            )

        X = df[model_features].copy()

        logger.info(
            f"Using model features: {model_features}"
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=config.get(
                'preprocessing.test_size',
                0.2
            ),
            random_state=config.get(
                'preprocessing.random_state',
                42
            )
        )

        logger.info(
            f"Training set shape: {X_train.shape}"
        )

        logger.info(
            f"Test set shape: {X_test.shape}"
        )

        preprocessor = DataPreprocessor()

        X_train_processed = preprocessor.fit_transform(
            X_train
        )

        X_test_processed = preprocessor.transform(
            X_test
        )

        feature_names = preprocessor.get_feature_names()

        save_artifact(
            feature_names,
            'artifacts/feature_columns.joblib'
        )

        logger.info(
            f"Transformed feature count: {len(feature_names)}"
        )

        trainer = ModelTrainer()

        trainer.initialize_models()

        results = trainer.train_and_evaluate(
            X_train_processed,
            y_train,
            X_test_processed,
            y_test
        )

        logger.info("\n" + "=" * 60)
        logger.info("MODEL TRAINING RESULTS")
        logger.info("=" * 60)

        for model_name, metrics in results.items():

            logger.info(f"\n{model_name}:")

            for metric_name, value in metrics.items():

                logger.info(
                    f"  {metric_name}: {value:.4f}"
                )

        logger.info("=" * 60)

        logger.info(
            "Model training completed successfully"
        )

    except Exception as e:

        logger.error(
            f"Error in model training: {str(e)}"
        )

        raise

if __name__ == "__main__":
    main()