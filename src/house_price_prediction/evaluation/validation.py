import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from ..utils.logger import logger
from ..utils.config import config

class ModelValidator:
    """Performs advanced model validation"""
    
    def __init__(self):
        self.cv_folds = config.get('training.cv_folds', 5)
        self.scoring = config.get('training.scoring', 'r2')
    
    def cross_validate(self, model, X, y):
        """Perform cross-validation"""
        logger.info(f"Performing {self.cv_folds}-fold cross-validation")
        
        cv_scores = cross_val_score(
            model, X, y, 
            cv=self.cv_folds, 
            scoring=self.scoring,
            n_jobs=-1
        )
        
        return {
            'scores': cv_scores,
            'mean': np.mean(cv_scores),
            'std': np.std(cv_scores)
        }
    
    def time_series_cv(self, model, X, y, n_splits=5):
        """Time series cross-validation for temporal data"""
        kf = KFold(n_splits=n_splits, shuffle=False)
        scores = []
        
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            scores.append(r2_score(y_test, y_pred))
        
        return {
            'scores': scores,
            'mean': np.mean(scores),
            'std': np.std(scores)
        }