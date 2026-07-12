import pytest
import numpy as np
from sklearn.datasets import make_regression
from src.house_price_prediction.models.xgboost_model import XGBoostModel
from src.house_price_prediction.models.random_forest import RandomForestModel
from src.house_price_prediction.models.lightgbm_model import LightGBMModel

class TestModels:
    
    @pytest.fixture
    def sample_data(self):
        X, y = make_regression(n_samples=100, n_features=5, noise=0.1)
        return X, y
    
    def test_xgboost_model(self, sample_data):
        X, y = sample_data
        model = XGBoostModel()
        model.train(X, y)
        
        predictions = model.predict(X)
        assert len(predictions) == len(y)
        assert isinstance(predictions, np.ndarray)
    
    def test_random_forest_model(self, sample_data):
        X, y = sample_data
        model = RandomForestModel()
        model.train(X, y)
        
        predictions = model.predict(X)
        assert len(predictions) == len(y)
        
    def test_lightgbm_model(self, sample_data):
        X, y = sample_data
        model = LightGBMModel()
        model.train(X, y)
        
        predictions = model.predict(X)
        assert len(predictions) == len(y)