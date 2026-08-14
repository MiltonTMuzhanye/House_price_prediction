"""
Custom transformers for feature engineering and preprocessing.
"""
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from typing import List, Optional, Dict, Union
from ..utils.logger import logger
from ..utils.config import config

class OutlierTransformer(BaseEstimator, TransformerMixin):
    """Transformer to handle outliers in numeric features."""
    
    def __init__(self, method: str = 'iqr', threshold: float = 1.5):
        self.method = method
        self.threshold = threshold
        self.bounds = {}
    
    def fit(self, X: pd.DataFrame, y=None):
        """Calculate outlier bounds."""
        logger.info(f"Fitting OutlierTransformer with method={self.method}")
        
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                if self.method == 'iqr':
                    Q1 = X[col].quantile(0.25)
                    Q3 = X[col].quantile(0.75)
                    IQR = Q3 - Q1
                    self.bounds[col] = {
                        'lower': Q1 - self.threshold * IQR,
                        'upper': Q3 + self.threshold * IQR
                    }
                elif self.method == 'zscore':
                    mean = X[col].mean()
                    std = X[col].std()
                    self.bounds[col] = {
                        'lower': mean - self.threshold * std,
                        'upper': mean + self.threshold * std
                    }
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply outlier handling."""
        X_transformed = X.copy()
        
        for col, bounds in self.bounds.items():
            if col in X_transformed.columns:
                # Clip values to bounds
                X_transformed[col] = X_transformed[col].clip(
                    lower=bounds['lower'],
                    upper=bounds['upper']
                )
        
        logger.info(f"OutlierTransformer applied to {len(self.bounds)} columns")
        return X_transformed

class FeatureScaler(BaseEstimator, TransformerMixin):
    """Flexible feature scaler wrapper."""
    
    def __init__(self, scaler_type: str = 'standard'):
        self.scaler_type = scaler_type
        self.scaler = None
    
    def fit(self, X: pd.DataFrame, y=None):
        """Fit the scaler."""
        logger.info(f"Fitting FeatureScaler with scaler_type={self.scaler_type}")
        
        if self.scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif self.scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        elif self.scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unsupported scaler type: {self.scaler_type}")
        
        self.scaler.fit(X)
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply scaling."""
        X_scaled = self.scaler.transform(X)
        return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

class CategoryEncoder(BaseEstimator, TransformerMixin):
    """Flexible categorical encoder."""
    
    def __init__(self, encoding_type: str = 'onehot', max_categories: int = 10):
        self.encoding_type = encoding_type
        self.max_categories = max_categories
        self.encoders = {}
        self.categories = {}
    
    def fit(self, X: pd.DataFrame, y=None):
        """Fit category encoders."""
        logger.info(f"Fitting CategoryEncoder with encoding_type={self.encoding_type}")
        
        for col in X.columns:
            if pd.api.types.is_categorical_dtype(X[col]) or X[col].dtype == 'object':
                # Get categories
                categories = X[col].value_counts().head(self.max_categories).index.tolist()
                self.categories[col] = categories
                
                if self.encoding_type == 'onehot':
                    # One-hot encoding will be applied in transform
                    pass
                elif self.encoding_type == 'label':
                    # Create label mapping
                    self.encoders[col] = {cat: i for i, cat in enumerate(categories)}
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply encoding."""
        X_transformed = X.copy()
        
        for col in X_transformed.columns:
            if col in self.categories:
                if self.encoding_type == 'label':
                    # Apply label encoding
                    X_transformed[col] = X_transformed[col].map(self.encoders[col])
                    X_transformed[col] = X_transformed[col].fillna(0).astype(int)
                elif self.encoding_type == 'onehot':
                    # Apply one-hot encoding
                    dummies = pd.get_dummies(X_transformed[col], prefix=col, drop_first=True)
                    X_transformed = pd.concat([X_transformed, dummies], axis=1)
                    X_transformed = X_transformed.drop(columns=[col])
        
        logger.info(f"CategoryEncoder applied to {len(self.categories)} columns")
        return X_transformed

class CreateDummies(BaseEstimator, TransformerMixin):
    """Create dummy variables for categorical columns."""
    
    def __init__(self, columns: Optional[List[str]] = None, drop_first: bool = True):
        self.columns = columns
        self.drop_first = drop_first
    
    def fit(self, X: pd.DataFrame, y=None):
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create dummy variables."""
        if self.columns is None:
            # Use all object/categorical columns
            self.columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if not self.columns:
            return X
        
        X_transformed = X.copy()
        
        for col in self.columns:
            if col in X_transformed.columns:
                dummies = pd.get_dummies(
                    X_transformed[col], 
                    prefix=col, 
                    drop_first=self.drop_first,
                    dummy_na=False
                )
                X_transformed = pd.concat([X_transformed, dummies], axis=1)
                X_transformed = X_transformed.drop(columns=[col])
        
        logger.info(f"CreateDummies applied to {len(self.columns)} columns")
        return X_transformed