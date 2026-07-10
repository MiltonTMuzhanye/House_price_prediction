import lightgbm as lgb
from .base_model import BaseModel
from ..utils.logger import logger

class LightGBMModel(BaseModel):
    """LightGBM model implementation"""
    
    def __init__(self, **kwargs):
        super().__init__()
        self.model = lgb.LGBMRegressor(
            n_estimators=kwargs.get('n_estimators', 300),
            num_leaves=kwargs.get('num_leaves', 50),
            learning_rate=kwargs.get('learning_rate', 0.1),
            feature_fraction=kwargs.get('feature_fraction', 0.8),
            random_state=kwargs.get('random_state', 42),
            n_jobs=-1
        )
        self.model_name = "LightGBM"
    
    def train(self, X_train, y_train):
        """Train LightGBM model"""
        logger.info("Training LightGBM model...")
        self.model.fit(X_train, y_train)
        return self.model