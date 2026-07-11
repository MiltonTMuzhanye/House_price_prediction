import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error, 
    mean_squared_error, 
    r2_score,
    explained_variance_score,
    mean_absolute_percentage_error
)
from typing import Dict, Any
from ..utils.logger import logger

class MetricsCalculator:
    """Calculate comprehensive evaluation metrics"""
    
    @staticmethod
    def calculate_all_metrics(y_true, y_pred) -> Dict[str, float]:
        """Calculate all evaluation metrics"""
        metrics = {
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mse': mean_squared_error(y_true, y_pred),
            'r2_score': r2_score(y_true, y_pred),
            'explained_variance': explained_variance_score(y_true, y_pred),
            'mape': mean_absolute_percentage_error(y_true, y_pred) * 100
        }
        
        logger.info(f"Calculated metrics: {metrics}")
        return metrics
    
    @staticmethod
    def calculate_residuals(y_true, y_pred) -> np.ndarray:
        """Calculate prediction residuals"""
        return y_true - y_pred
    
    @staticmethod
    def calculate_confidence_interval(y_true, y_pred, confidence=0.95) -> Dict[str, float]:
        """Calculate confidence interval for predictions"""
        residuals = y_true - y_pred
        se = np.std(residuals) / np.sqrt(len(residuals))
        z_score = 1.96  # 95% confidence
        
        return {
            'lower_bound': np.mean(y_pred) - z_score * se,
            'upper_bound': np.mean(y_pred) + z_score * se
        }
    
    @staticmethod
    def get_metrics_df(metrics_dict: Dict[str, Dict]) -> pd.DataFrame:
        """Convert metrics dict to DataFrame"""
        return pd.DataFrame(metrics_dict).T.round(4)