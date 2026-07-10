from sklearn.ensemble import RandomForestRegressor
from .base_model import BaseModel
from ..utils.logger import logger

class RandomForestModel(BaseModel):
    """Random Forest model implementation"""
    
    def __init__(self, **kwargs):
        super().__init__()
        self.model = RandomForestRegressor(
            n_estimators=kwargs.get('n_estimators', 200),
            max_depth=kwargs.get('max_depth', 20),
            min_samples_split=kwargs.get('min_samples_split', 5),
            min_samples_leaf=kwargs.get('min_samples_leaf', 2),
            random_state=kwargs.get('random_state', 42),
            n_jobs=-1
        )
        self.model_name = "RandomForest"
    
    def train(self, X_train, y_train):
        """Train Random Forest model"""
        logger.info("Training Random Forest model...")
        self.model.fit(X_train, y_train)
        return self.model