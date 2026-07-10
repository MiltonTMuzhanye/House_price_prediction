import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
from ..utils.logger import logger
from ..utils.config import config
from ..utils.helpers import save_artifact
from ..models.random_forest import RandomForestModel
from ..models.xgboost_model import XGBoostModel
from ..models.lightgbm_model import LightGBMModel
from ..models.catboost_model import CatBoostModel
from ..models.baseline import LinearRegressionModel

class ModelTrainer:
    """Handles model training and evaluation"""
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.results = {}
        self.config = config
        
    def initialize_models(self):
        """Initialize all models with their parameters"""
        models_config = self.config.get('model', {})
        
        self.models = {
            'linear_regression': LinearRegressionModel(),
            'random_forest': RandomForestModel(**models_config.get('random_forest_params', {})),
            'xgboost': XGBoostModel(**models_config.get('xgboost_params', {})),
            'lightgbm': LightGBMModel(**models_config.get('lightgbm_params', {})),
            'catboost': CatBoostModel(**models_config.get('catboost_params', {}))
        }
        
        # Filter models based on config
        models_to_train = models_config.get('models_to_train', list(self.models.keys()))
        self.models = {k: v for k, v in self.models.items() if k in models_to_train}
    
    def train_and_evaluate(self, X, y, test_size=0.2, cv_folds=5):
        """Train and evaluate all models"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Initialize MLflow
        mlflow.set_experiment(config.get('training.experiment_name', 'house_price_prediction'))
        
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            
            with mlflow.start_run(run_name=name):
                # Train model
                model.train(X_train, y_train)
                predictions = model.predict(X_test)
                
                # Evaluate
                metrics = self.evaluate(y_test, predictions)
                self.results[name] = metrics
                
                # Cross validation
                cv_scores = cross_val_score(
                    model.model, X, y, cv=cv_folds, 
                    scoring='r2', n_jobs=-1
                )
                self.results[name]['cv_mean'] = np.mean(cv_scores)
                self.results[name]['cv_std'] = np.std(cv_scores)
                
                # Log metrics to MLflow
                for metric_name, value in metrics.items():
                    mlflow.log_metric(metric_name, value)
                
                # Log parameters
                if hasattr(model.model, 'get_params'):
                    params = model.model.get_params()
                    mlflow.log_params(params)
                
                # Save model
                model.save_model()
        
        # Select best model
        self.select_best_model()
        
        return self.results
    
    def evaluate(self, y_true, y_pred):
        """Calculate evaluation metrics"""
        return {
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2_score': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }
    
    def select_best_model(self):
        """Select best model based on R2 score"""
        if not self.results:
            raise ValueError("No models trained yet")
        
        best_model_name = max(self.results, key=lambda x: self.results[x]['r2_score'])
        self.best_model = self.models[best_model_name]
        logger.info(f"Best model: {best_model_name} with R2: {self.results[best_model_name]['r2_score']:.4f}")
        
        # Save best model info
        save_artifact({
            'best_model': best_model_name,
            'metrics': self.results[best_model_name]
        }, 'artifacts/best_model_info.joblib')
        
        return self.best_model