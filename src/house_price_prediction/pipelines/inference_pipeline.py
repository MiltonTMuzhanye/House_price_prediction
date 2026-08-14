"""
End-to-end inference pipeline for house price prediction.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Union, Optional
from pathlib import Path

from ..data.preprocessing import DataPreprocessor
from ..features.engineering import FeatureEngineer
from ..utils.logger import logger
from ..utils.helpers import load_artifact
from ..utils.config import config

class InferencePipeline:
    """Complete inference pipeline orchestrator."""
    
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.feature_columns = None
        self.feature_engineer = FeatureEngineer()
        self._load_artifacts()
    
    def _load_artifacts(self):
        """Load all required artifacts."""
        logger.info("Loading inference artifacts...")
        
        # Load best model
        best_model_info = load_artifact('artifacts/best_model_info.joblib')
        model_name = best_model_info.get('best_model', 'xgboost')
        self.model = load_artifact(f'artifacts/trained_models/{model_name}.joblib')
        
        # Load preprocessor
        self.preprocessor = load_artifact('artifacts/preprocessor.joblib')
        
        # Load feature columns
        self.feature_columns = load_artifact('artifacts/feature_columns.joblib')
        
        logger.info(f"Loaded model: {model_name}")
        logger.info(f"Loaded {len(self.feature_columns)} feature columns")
    
    def preprocess_input(self, features: Dict[str, Any]) -> pd.DataFrame:
        """Preprocess single input."""
        df = pd.DataFrame([features])
        
        # Map categorical variables
        location_map = {1: 'Urban', 2: 'Suburban', 3: 'Rural'}
        region_map = {1: 'Northeast', 2: 'Midwest', 3: 'South', 4: 'West'}
        titled_map = {1: 'Vehicle', 2: 'Land-Home', 3: 'Other'}
        
        if 'LOCATION' in df.columns:
            df['LOCATION'] = df['LOCATION'].map(location_map)
        if 'REGION' in df.columns:
            df['REGION'] = df['REGION'].map(region_map)
        if 'TITLED' in df.columns:
            df['TITLED'] = df['TITLED'].map(titled_map)
        if 'LEASE' in df.columns:
            df['LEASE'] = df['LEASE'].replace({2: 0, 1: 1})
        
        # Feature engineering
        df_engineered = self.feature_engineer.create_all_features(df)
        
        # Ensure all feature columns are present
        for col in self.feature_columns:
            if col not in df_engineered.columns:
                df_engineered[col] = 0
        
        # Select only required features
        df_selected = df_engineered[self.feature_columns]
        
        return df_selected
    
    def predict_single(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Make prediction for single input."""
        try:
            logger.info("Making single prediction...")
            
            # Preprocess
            X_processed = self.preprocess_input(features)
            
            # Transform using preprocessor
            X_transformed = self.preprocessor.transform(X_processed)
            
            # Make prediction
            prediction = self.model.predict(X_transformed)[0]
            
            result = {
                'predicted_price': float(prediction),
                'predicted_price_formatted': f"${float(prediction):,.2f}",
                'features_used': self.feature_columns
            }
            
            logger.info(f"Prediction: ${prediction:,.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise
    
    def predict_batch(self, features_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Make predictions for batch inputs."""
        try:
            logger.info(f"Making batch predictions for {len(features_list)} samples...")
            
            results = []
            for features in features_list:
                result = self.predict_single(features)
                results.append(result)
            
            logger.info(f"Batch prediction completed for {len(results)} samples")
            return results
            
        except Exception as e:
            logger.error(f"Batch prediction error: {str(e)}")
            raise
    
    def predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Make predictions for DataFrame input."""
        try:
            logger.info(f"Making predictions for DataFrame with {len(df)} rows...")
            
            results = []
            for _, row in df.iterrows():
                features = row.to_dict()
                prediction = self.predict_single(features)
                results.append(prediction['predicted_price'])
            
            df_results = df.copy()
            df_results['predicted_price'] = results
            df_results['predicted_price_formatted'] = df_results['predicted_price'].apply(
                lambda x: f"${x:,.2f}"
            )
            
            return df_results
            
        except Exception as e:
            logger.error(f"DataFrame prediction error: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        try:
            best_model_info = load_artifact('artifacts/best_model_info.joblib')
            
            return {
                'model_name': best_model_info.get('best_model', 'Unknown'),
                'metrics': best_model_info.get('metrics', {}),
                'features': self.feature_columns,
                'n_features': len(self.feature_columns)
            }
        except Exception as e:
            logger.error(f"Error getting model info: {str(e)}")
            return {'error': str(e)}

# Convenience function
def create_inference_pipeline():
    """Create an inference pipeline instance."""
    return InferencePipeline()