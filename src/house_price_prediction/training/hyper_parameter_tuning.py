from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import numpy as np
from ..utils.logger import logger
from ..utils.config import config

class HyperparameterTuner:
    """Performs hyperparameter tuning for models"""
    
    def __init__(self, model, param_grid, cv=3, scoring='r2', n_iter=20):
        self.model = model
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.n_iter = n_iter
        self.best_model = None
        self.best_params = None
    
    def grid_search(self, X_train, y_train):
        """Perform grid search"""
        logger.info(f"Performing GridSearchCV for {self.model.__class__.__name__}")
        grid_search = GridSearchCV(
            self.model, 
            self.param_grid, 
            cv=self.cv, 
            scoring=self.scoring,
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        self.best_model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        
        logger.info(f"Best parameters: {self.best_params}")
        logger.info(f"Best score: {grid_search.best_score_:.4f}")
        
        return self.best_model, self.best_params
    
    def random_search(self, X_train, y_train):
        """Perform random search"""
        logger.info(f"Performing RandomizedSearchCV for {self.model.__class__.__name__}")
        random_search = RandomizedSearchCV(
            self.model,
            self.param_grid,
            n_iter=self.n_iter,
            cv=self.cv,
            scoring=self.scoring,
            random_state=42,
            n_jobs=-1
        )
        random_search.fit(X_train, y_train)
        
        self.best_model = random_search.best_estimator_
        self.best_params = random_search.best_params_
        
        logger.info(f"Best parameters: {self.best_params}")
        logger.info(f"Best score: {random_search.best_score_:.4f}")
        
        return self.best_model, self.best_params