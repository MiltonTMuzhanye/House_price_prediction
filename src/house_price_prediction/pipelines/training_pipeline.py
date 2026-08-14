"""
End-to-end training pipeline for house price prediction.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

from ..data.ingestion import DataIngestion
from ..data.validation import DataValidator
from ..data.preprocessing import DataPreprocessor
from ..features.engineering import FeatureEngineer
from ..features.selection import FeatureSelector
from ..training.trainer import ModelTrainer
from ..training.hyperparameter_tuning import HyperparameterTuner
from ..evaluation.metrics import MetricsCalculator
from ..evaluation.validation import ModelValidator
from ..utils.logger import logger
from ..utils.config import config
from ..utils.helpers import save_artifact, load_artifact

class TrainingPipeline:
    """Complete training pipeline orchestrator."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = config
        self.data_ingestion = DataIngestion()
        self.data_validator = DataValidator()
        self.data_preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineer()
        self.feature_selector = FeatureSelector()
        self.model_trainer = ModelTrainer()
        self.model_validator = ModelValidator()
        self.metrics_calculator = MetricsCalculator()
        
        self.pipeline_results = {}
    
    def run_pipeline(self) -> Dict[str, Any]:
        """Execute the complete training pipeline."""
        logger.info("="*60)
        logger.info("Starting Training Pipeline")
        logger.info("="*60)
        
        # Step 1: Data Ingestion
        logger.info("\n Step 1: Data Ingestion")
        df_raw = self.data_ingestion.load_data()
        
        # Step 2: Data Validation
        logger.info("\n Step 2: Data Validation")
        if not self.data_validator.validate_schema(df_raw):
            raise ValueError("Schema validation failed")
        quality_report = self.data_validator.validate_data_quality(df_raw)
        self.pipeline_results['quality_report'] = quality_report
        
        # Step 3: Data Cleaning
        logger.info("\n Step 3: Data Cleaning")
        df_clean = self.data_validator.clean_data(df_raw)
        
        # Step 4: Data Preprocessing
        logger.info("\n Step 4: Data Preprocessing")
        df_preprocessed = self.data_preprocessor.map_categorical(df_clean)
        
        # Step 5: Feature Engineering
        logger.info("\n Step 5: Feature Engineering")
        df_engineered = self.feature_engineer.create_all_features(df_preprocessed)
        
        # Step 6: Prepare Features and Target
        logger.info("\n Step 6: Prepare Data")
        target_col = self.config.get('features.target', 'PRICE')
        X = df_engineered.drop(columns=[target_col])
        y = df_engineered[target_col]
        
        # Step 7: Feature Selection
        logger.info("\n Step 7: Feature Selection")
        X_selected = X
        if self.config.get('features.selection.enabled', False):
            X_selected = X[self.feature_selector.select_features(X, y)]
        self.pipeline_results['selected_features'] = X_selected.columns.tolist()
        
        # Step 8: Preprocessing Pipeline
        logger.info("\n Step 8: Create Preprocessing Pipeline")
        X_processed = self.data_preprocessor.fit_transform(X_selected)
        
        # Save feature columns
        save_artifact(X_selected.columns.tolist(), 'artifacts/feature_columns.joblib')
        
        # Step 9: Model Training
        logger.info("\n Step 9: Model Training")
        self.model_trainer.initialize_models()
        results = self.model_trainer.train_and_evaluate(X_processed, y)
        self.pipeline_results['model_results'] = results
        
        # Step 10: Model Validation
        logger.info("\n Step 10: Model Validation")
        best_model = self.model_trainer.best_model
        if best_model:
            cv_results = self.model_validator.cross_validate(
                best_model.model, X_processed, y
            )
            self.pipeline_results['cv_results'] = cv_results
        
        # Step 11: Save Pipeline Results
        logger.info("\n Step 11: Save Pipeline Results")
        save_artifact(self.pipeline_results, 'artifacts/pipeline_results.joblib')
        
        logger.info("\n Training Pipeline Completed Successfully!")
        logger.info("="*60)
        
        return self.pipeline_results
    
    def get_best_model(self):
        """Get the best trained model."""
        return self.model_trainer.best_model
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get summary of pipeline execution."""
        summary = {
            'data_shape': self.pipeline_results.get('quality_report', {}).get('total_rows', 0),
            'features_selected': len(self.pipeline_results.get('selected_features', [])),
            'best_model': self.pipeline_results.get('model_results', {}).get('best_model', 'Unknown'),
            'best_r2': self.pipeline_results.get('model_results', {})
                           .get(self.pipeline_results.get('model_results', {})
                           .get('best_model', ''), {})
                           .get('r2_score', 0)
        }
        return summary

# Convenience function
def train_pipeline():
    """Run the training pipeline."""
    pipeline = TrainingPipeline()
    return pipeline.run_pipeline()