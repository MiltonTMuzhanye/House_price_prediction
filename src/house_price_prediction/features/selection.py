"""
Feature selection module for selecting the most important features.
"""
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Dict
from sklearn.feature_selection import (
    SelectKBest, 
    f_regression, 
    mutual_info_regression,
    RFE,
    SelectFromModel
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from ..utils.logger import logger
from ..utils.config import config

class FeatureSelector:
    """Selects the most important features for model training."""
    
    def __init__(self, n_features: Optional[int] = None):
        self.n_features = n_features
        self.selected_features = None
        self.feature_scores = {}
    
    def select_by_correlation(self, X: pd.DataFrame, y: pd.Series, threshold: float = 0.1) -> List[str]:
        """Select features based on correlation with target."""
        logger.info(f"Selecting features by correlation (threshold={threshold})...")
        
        correlations = X.corrwith(y).abs()
        selected = correlations[correlations > threshold].index.tolist()
        
        self.feature_scores['correlation'] = correlations.to_dict()
        logger.info(f"Selected {len(selected)} features by correlation")
        
        return selected
    
    def select_by_mutual_info(self, X: pd.DataFrame, y: pd.Series, n_features: Optional[int] = None) -> List[str]:
        """Select features using mutual information."""
        logger.info("Selecting features by mutual information...")
        
        if n_features is None:
            n_features = min(len(X.columns), 20) if self.n_features is None else self.n_features
        
        selector = SelectKBest(mutual_info_regression, k=n_features)
        selector.fit(X, y)
        
        selected_indices = selector.get_support(indices=True)
        selected = X.columns[selected_indices].tolist()
        
        self.feature_scores['mutual_info'] = dict(zip(X.columns, selector.scores_))
        logger.info(f"Selected {len(selected)} features by mutual information")
        
        return selected
    
    def select_by_rfe(self, X: pd.DataFrame, y: pd.Series, n_features: Optional[int] = None) -> List[str]:
        """Select features using Recursive Feature Elimination."""
        logger.info("Selecting features by RFE...")
        
        if n_features is None:
            n_features = min(len(X.columns), 20) if self.n_features is None else self.n_features
        
        estimator = RandomForestRegressor(n_estimators=100, random_state=42)
        selector = RFE(estimator, n_features_to_select=n_features)
        selector.fit(X, y)
        
        selected = X.columns[selector.support_].tolist()
        
        # Get feature rankings
        rankings = dict(zip(X.columns, selector.ranking_))
        self.feature_scores['rfe'] = rankings
        logger.info(f"Selected {len(selected)} features by RFE")
        
        return selected
    
    def select_by_lasso(self, X: pd.DataFrame, y: pd.Series, alpha: float = 0.01) -> List[str]:
        """Select features using Lasso regularization."""
        logger.info(f"Selecting features by Lasso (alpha={alpha})...")
        
        lasso = Lasso(alpha=alpha, random_state=42)
        lasso.fit(X, y)
        
        # Get features with non-zero coefficients
        selected = X.columns[lasso.coef_ != 0].tolist()
        
        self.feature_scores['lasso'] = dict(zip(X.columns, lasso.coef_))
        logger.info(f"Selected {len(selected)} features by Lasso")
        
        return selected
    
    def select_by_importance(self, X: pd.DataFrame, y: pd.Series, threshold: float = 0.01) -> List[str]:
        """Select features using Random Forest feature importance."""
        logger.info(f"Selecting features by importance (threshold={threshold})...")
        
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        importances = pd.Series(rf.feature_importances_, index=X.columns)
        selected = importances[importances > threshold].index.tolist()
        
        self.feature_scores['importance'] = importances.to_dict()
        logger.info(f"Selected {len(selected)} features by importance")
        
        return selected
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, methods: List[str] = None) -> List[str]:
        """Apply multiple feature selection methods and combine results."""
        if methods is None:
            methods = ['correlation', 'mutual_info', 'importance']
        
        logger.info(f"Applying feature selection methods: {methods}")
        
        all_selected = []
        
        if 'correlation' in methods:
            all_selected.extend(self.select_by_correlation(X, y))
        
        if 'mutual_info' in methods:
            all_selected.extend(self.select_by_mutual_info(X, y))
        
        if 'rfe' in methods:
            all_selected.extend(self.select_by_rfe(X, y))
        
        if 'lasso' in methods:
            all_selected.extend(self.select_by_lasso(X, y))
        
        if 'importance' in methods:
            all_selected.extend(self.select_by_importance(X, y))
        
        # Get consensus features (appear in most methods)
        from collections import Counter
        feature_counts = Counter(all_selected)
        
        # Select features that appear in at least 2 methods
        consensus_threshold = min(2, len(methods) // 2 + 1)
        self.selected_features = [f for f, count in feature_counts.items() 
                                 if count >= consensus_threshold]
        
        logger.info(f"Final selection: {len(self.selected_features)} features from {len(methods)} methods")
        
        return self.selected_features
    
    def get_feature_scores(self) -> Dict:
        """Get feature importance scores from all methods."""
        return self.feature_scores