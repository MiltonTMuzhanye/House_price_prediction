from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from .base_model import BaseModel
from ..utils.logger import logger

class LinearRegressionModel(BaseModel):
    """Linear Regression baseline model"""
    
    def __init__(self, **kwargs):
        super().__init__()
        self.model = LinearRegression(**kwargs)
        self.model_name = "LinearRegression"
    
    def train(self, X_train, y_train):
        """Train linear regression model"""
        logger.info("Training Linear Regression model...")
        self.model.fit(X_train, y_train)
        return self.model

class DummyBaseline(BaseModel):
    """Dummy model for baseline comparison"""
    
    def __init__(self, strategy='mean', **kwargs):
        super().__init__()
        self.model = DummyRegressor(strategy=strategy, **kwargs)
        self.model_name = "Dummy"
    
    def train(self, X_train, y_train):
        """Train dummy model"""
        logger.info(f"Training Dummy model with strategy: {self.model.strategy}")
        self.model.fit(X_train, y_train)
        return self.model
