from catboost import CatBoostRegressor
from .base_model import BaseModel
from ..utils.logger import logger

class CatBoostModel(BaseModel):
    """CatBoost model implementation"""
    
    def __init__(self, **kwargs):
        super().__init__()
        self.model = CatBoostRegressor(
            iterations=kwargs.get('iterations', 300),
            depth=kwargs.get('depth', 7),
            learning_rate=kwargs.get('learning_rate', 0.1),
            random_seed=kwargs.get('random_state', 42),
            verbose=False
        )
        self.model_name = "CatBoost"
    
    def train(self, X_train, y_train):
        """Train CatBoost model"""
        logger.info("Training CatBoost model...")
        self.model.fit(X_train, y_train)
        return self.model