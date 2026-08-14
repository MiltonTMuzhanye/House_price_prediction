"""
Feature engineering module for creating new features from raw data.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from ..utils.logger import logger
from ..utils.config import config

class FeatureEngineer:
    """Creates new features from existing data."""
    
    def __init__(self):
        self.features_config = config.get('features', {})
        self.engineered_features = []
    
    def create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create price-related features."""
        df_engineered = df.copy()
        
        if 'PRICE' in df.columns and 'SQFT' in df.columns:
            # Price per square foot
            df_engineered['PRICE_PER_SQFT'] = df['PRICE'] / df['SQFT']
            self.engineered_features.append('PRICE_PER_SQFT')
            
            # Price category (binned)
            df_engineered['PRICE_CATEGORY'] = pd.qcut(
                df['PRICE'], 
                q=4, 
                labels=['Low', 'Medium-Low', 'Medium-High', 'High']
            )
            self.engineered_features.append('PRICE_CATEGORY')
        
        if 'PRICE' in df.columns:
            # Log transform for price
            df_engineered['LOG_PRICE'] = np.log1p(df['PRICE'])
            self.engineered_features.append('LOG_PRICE')
        
        return df_engineered
    
    def create_size_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create size-related features."""
        df_engineered = df.copy()
        
        if 'SQFT' in df.columns:
            # Log transform for square footage
            df_engineered['LOG_SQFT'] = np.log1p(df['SQFT'])
            self.engineered_features.append('LOG_SQFT')
            
            # Size category
            df_engineered['SIZE_CATEGORY'] = pd.cut(
                df['SQFT'],
                bins=[0, 1000, 2000, 3000, 5000],
                labels=['Small', 'Medium', 'Large', 'Very Large']
            )
            self.engineered_features.append('SIZE_CATEGORY')
            
            # Square footage per bedroom
            if 'BEDROOMS' in df.columns:
                df_engineered['SQFT_PER_BEDROOM'] = df['SQFT'] / df['BEDROOMS']
                self.engineered_features.append('SQFT_PER_BEDROOM')
        
        return df_engineered
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between variables."""
        df_engineered = df.copy()
        
        if 'LOCATION' in df.columns and 'REGION' in df.columns:
            # Location-Region interaction
            df_engineered['LOCATION_REGION'] = df['LOCATION'].astype(str) + '_' + df['REGION'].astype(str)
            self.engineered_features.append('LOCATION_REGION')
        
        if 'SQFT' in df.columns and 'BEDROOMS' in df.columns:
            # Size ratio features
            df_engineered['AVG_ROOM_SIZE'] = df['SQFT'] / df['BEDROOMS']
            self.engineered_features.append('AVG_ROOM_SIZE')
            
            # Large property indicator
            df_engineered['LARGE_PROPERTY'] = ((df['SQFT'] > 2000) & (df['BEDROOMS'] > 3)).astype(int)
            self.engineered_features.append('LARGE_PROPERTY')
        
        return df_engineered
    
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create all engineered features."""
        logger.info("Starting feature engineering...")
        
        df_engineered = df.copy()
        
        # Apply all feature engineering methods
        df_engineered = self.create_price_features(df_engineered)
        df_engineered = self.create_size_features(df_engineered)
        df_engineered = self.create_interaction_features(df_engineered)
        
        logger.info(f"Created {len(self.engineered_features)} new features: {self.engineered_features}")
        logger.info(f"Final shape: {df_engineered.shape}")
        
        return df_engineered
    
    def get_engineered_features(self) -> List[str]:
        """Get list of engineered feature names."""
        return self.engineered_features