from xgboost import XGBRegressor
from .base_model import BaseModel
from ..utils.logger import logger

class XGBoostModel(BaseModel):
    """XGBoost model implementation"""
    
    def __init__(self, **kwargs):
        super().__init__()
        self.model = XGBRegressor(
            n_estimators=kwargs.get('n_estimators', 300),
            max_depth=kwargs.get('max_depth', 7),
            learning_rate=kwargs.get('learning_rate', 0.1),
            subsample=kwargs.get('subsample', 0.8),
            colsample_bytree=kwargs.get('colsample_bytree', 0.8),
            random_state=kwargs.get('random_state', 42),
            n_jobs=-1
        )
        self.model_name = "XGBoost"
    
    def train(self, X_train, y_train):
        """Train XGBoost model"""
        logger.info("Training XGBoost model...")
        self.model.fit(X_train, y_train)
        return self.model