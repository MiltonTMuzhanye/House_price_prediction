import pandas as pd
import numpy as np
from typing import Dict, Any
from src.house_price_prediction.data.preprocessing import DataPreprocessor
from src.house_price_prediction.utils.helpers import load_artifact
from src.house_price_prediction.utils.logger import logger

class InferencePreprocessor:
    """Preprocess input data for inference"""
    
    def __init__(self):
        self.preprocessor = None
        self.feature_columns = None
        self.load_preprocessor()
    
    def load_preprocessor(self):
        """Load saved preprocessor"""
        try:
            self.preprocessor = load_artifact('artifacts/preprocessor.joblib')
            self.feature_columns = load_artifact('artifacts/feature_columns.joblib')
            logger.info("Loaded preprocessor and feature columns")
        except Exception as e:
            logger.error(f"Error loading preprocessor: {str(e)}")
            raise
    
    def preprocess_input(self, input_data: Dict[str, Any]) -> np.ndarray:
        """Preprocess single input"""
        # Convert to DataFrame
        df = pd.DataFrame([input_data])
        
        # Map categorical values
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
        
        # Calculate PRICE_PER_SQFT (if PRICE is provided for comparison)
        if 'PRICE' in df.columns and 'SQFT' in df.columns:
            df['PRICE_PER_SQFT'] = df['PRICE'] / df['SQFT']
        elif 'SQFT' in df.columns:
            # For prediction, use median price per sqft from training data
            # This is a placeholder - in production, you'd use the actual trained value
            df['PRICE_PER_SQFT'] = df['SQFT'] * 150  # Placeholder
        
        # Engineer features
        if 'BEDROOMS' in df.columns and 'SQFT' in df.columns:
            df['BEDROOMS_PER_SQFT'] = df['BEDROOMS'] / df['SQFT'] * 1000
        if 'PRICE' in df.columns:
            df['LOG_PRICE'] = np.log1p(df['PRICE'])
        if 'SQFT' in df.columns:
            df['LOG_SQFT'] = np.log1p(df['SQFT'])
        
        # Transform using preprocessor
        X_processed = self.preprocessor.transform(df)
        
        return X_processed
    
    def preprocess_batch(self, input_data: pd.DataFrame) -> np.ndarray:
        """Preprocess batch input"""
        # Apply same transformations as single input
        df = input_data.copy()
        
        # Map categorical values
        location_map = {1: 'Urban', 2: 'Suburban', 3: 'Rural'}
        region_map = {1: 'Northeast', 2: 'Midwest', 3: 'South', 4: 'West'}
        titled_map = {1: 'Vehicle', 2: 'Land-Home', 3: 'Other'}
        
        df['LOCATION'] = df['LOCATION'].map(location_map)
        df['REGION'] = df['REGION'].map(region_map)
        df['TITLED'] = df['TITLED'].map(titled_map)
        df['LEASE'] = df['LEASE'].replace({2: 0, 1: 1})
        
        # Engineering features
        df['PRICE_PER_SQFT'] = df['PRICE'] / df['SQFT']
        df['BEDROOMS_PER_SQFT'] = df['BEDROOMS'] / df['SQFT'] * 1000
        df['LOG_PRICE'] = np.log1p(df['PRICE'])
        df['LOG_SQFT'] = np.log1p(df['SQFT'])
        
        # Transform using preprocessor
        X_processed = self.preprocessor.transform(df)
        
        return X_processed