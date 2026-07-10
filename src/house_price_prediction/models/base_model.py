from abc import ABC, abstractmethod
import numpy as np
from ..utils.logger import logger
from ..utils.helpers import save_artifact

class BaseModel(ABC):
    """Base class for all models"""
    
    def __init__(self):
        self.model = None
        self.model_name = None
        self.feature_importance = None
    
    @abstractmethod
    def train(self, X_train, y_train):
        """Train the model"""
        pass
    
    def predict(self, X_test):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict(X_test)
    
    def get_feature_importance(self, feature_names):
        """Get feature importance"""
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            self.feature_importance = np.abs(self.model.coef_)
        else:
            logger.warning("Model doesn't have feature importance attribute")
            self.feature_importance = None
        
        return self.feature_importance
    
    def save_model(self, path: str = None):
        """Save trained model"""
        if path is None:
            path = f"artifacts/trained_models/{self.model_name.lower()}.joblib"
        save_artifact(self.model, path)
        logger.info(f"Saved {self.model_name} model to {path}")