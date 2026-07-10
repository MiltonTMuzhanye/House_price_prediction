import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from ..utils.logger import logger
from ..utils.config import config
from ..utils.helpers import save_artifact, load_artifact

class DataPreprocessor:
    """Handles data preprocessing and feature engineering"""
    
    def __init__(self):
        self.numeric_features = config.get('features.numeric_features', ['SQFT', 'BEDROOMS', 'PRICE_PER_SQFT'])
        self.categorical_features = config.get('features.categorical_features', 
                                              ['LOCATION', 'REGION', 'TITLED', 'LEASE', 'FOOTINGS'])
        self.target = config.get('features.target', 'PRICE')
        
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(drop='first', sparse_output=False)
        
        # Mapping dictionaries
        self.location_map = {1: 'Urban', 2: 'Suburban', 3: 'Rural'}
        self.region_map = {1: 'Northeast', 2: 'Midwest', 3: 'South', 4: 'West'}
        self.titled_map = {1: 'Vehicle', 2: 'Land-Home', 3: 'Other'}
        
    def map_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map categorical variables to readable labels"""
        df_mapped = df.copy()
        
        if 'LOCATION' in df_mapped.columns:
            df_mapped['LOCATION'] = df_mapped['LOCATION'].map(self.location_map)
        if 'REGION' in df_mapped.columns:
            df_mapped['REGION'] = df_mapped['REGION'].map(self.region_map)
        if 'TITLED' in df_mapped.columns:
            df_mapped['TITLED'] = df_mapped['TITLED'].map(self.titled_map)
        if 'LEASE' in df_mapped.columns:
            df_mapped['LEASE'] = df_mapped['LEASE'].replace({2: 0, 1: 1})
        
        return df_mapped
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create new features"""
        df_engineered = df.copy()
        
        # Price per square foot
        df_engineered['PRICE_PER_SQFT'] = df_engineered['PRICE'] / df_engineered['SQFT']
        
        # Additional engineered features
        df_engineered['BEDROOMS_PER_SQFT'] = df_engineered['BEDROOMS'] / df_engineered['SQFT'] * 1000
        df_engineered['LOG_PRICE'] = np.log1p(df_engineered['PRICE'])
        df_engineered['LOG_SQFT'] = np.log1p(df_engineered['SQFT'])
        
        logger.info(f"Engineered features: {df_engineered.columns.tolist()}")
        return df_engineered
    
    def create_preprocessing_pipeline(self):
        """Create scikit-learn preprocessing pipeline"""
        # Define transformers
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(drop='first', sparse_output=False))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ])
        
        return preprocessor
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None):
        """Fit preprocessor and transform data"""
        preprocessor = self.create_preprocessing_pipeline()
        X_processed = preprocessor.fit_transform(X)
        
        # Save the preprocessor
        save_artifact(preprocessor, 'artifacts/preprocessor.joblib')
        
        return X_processed
    
    def transform(self, X: pd.DataFrame):
        """Transform new data using saved preprocessor"""
        preprocessor = load_artifact('artifacts/preprocessor.joblib')
        return preprocessor.transform(X)