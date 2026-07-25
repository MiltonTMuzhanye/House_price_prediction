import pandas as pd
import numpy as np
from typing import Dict, Any
from ..utils.logger import logger
from ..utils.config import config

class DataValidator:
    """Validates data quality and schema"""
    
    def __init__(self):
        self.price_min, self.price_max = config.get('preprocessing.price_range', [10000, 1000000])
        self.sqft_min, self.sqft_max = config.get('preprocessing.sqft_range', [100, 5000])
    
    def validate_schema(self, df: pd.DataFrame) -> bool:
        """Validate that data has expected columns"""
        expected_columns = ['PRICE', 'SQFT', 'BEDROOMS', 'LOCATION', 'REGION', 
                           'TITLED', 'LEASE', 'FOOTINGS', 'SECTIONS']
        
        missing_cols = [col for col in expected_columns if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing columns: {missing_cols}")
            return False
        
        logger.info("Schema validation passed")
        return True
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check data quality metrics"""
        quality_report = {
            'total_rows': len(df),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicates': df.duplicated().sum(),
            'price_outliers': sum((df['PRICE'] < self.price_min) | (df['PRICE'] > self.price_max)),
            'sqft_outliers': sum((df['SQFT'] < self.sqft_min) | (df['SQFT'] > self.sqft_max))
        }
        
        logger.info(f"Data quality report: {quality_report}")
        return quality_report
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean data by handling outliers and missing values"""
        df_clean = df.copy()
        
        # Remove duplicate/redundant columns (j-prefixed)
        df_clean = df_clean.drop(columns=df_clean.filter(regex='^j').columns)
        
        # Handle missing values for numeric columns
        num_cols = ['PRICE', 'SQFT', 'BEDROOMS']
        for col in num_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
        # Filter outliers
        df_clean = df_clean[
            (df_clean['PRICE'] > self.price_min) & 
            (df_clean['PRICE'] < self.price_max) &
            (df_clean['SQFT'] > self.sqft_min) & 
            (df_clean['SQFT'] < self.sqft_max)
        ]
        
        logger.info(f"Cleaned data shape: {df_clean.shape}")
        return df_clean