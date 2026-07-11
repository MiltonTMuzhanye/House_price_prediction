import numpy as np
import pandas as pd
from typing import Dict, Any, List, Union
from src.house_price_prediction.utils.helpers import load_artifact
from src.house_price_prediction.utils.logger import logger
from .preprocess import InferencePreprocessor

class HousePricePredictor:
    """Handles house price predictions"""
    
    def __init__(self):
        self.model = None
        self.preprocessor = InferencePreprocessor()
        self.load_model()
    
    def load_model(self):
        """Load trained model"""
        try:
            # Load best model info
            best_model_info = load_artifact('artifacts/best_model_info.joblib')
            model_name = best_model_info['best_model']
            
            # Load the actual model
            self.model = load_artifact(f'artifacts/trained_models/{model_name}.joblib')
            logger.info(f"Loaded model: {model_name}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def predict_single(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make prediction for single input"""
        try:
            # Preprocess input
            X_processed = self.preprocessor.preprocess_input(input_data)
            
            # Make prediction
            prediction = self.model.predict(X_processed)[0]
            
            return {
                'predicted_price': float(prediction),
                'predicted_price_formatted': f"${float(prediction):,.2f}",
                'input_features': input_data
            }
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise
    
    def predict_batch(self, input_data: Union[List[Dict], pd.DataFrame]) -> List[Dict]:
        """Make predictions for batch inputs"""
        try:
            # Convert to DataFrame if list of dicts
            if isinstance(input_data, list):
                df = pd.DataFrame(input_data)
            else:
                df = input_data
            
            # Preprocess batch
            X_processed = self.preprocessor.preprocess_batch(df)
            
            # Make predictions
            predictions = self.model.predict(X_processed)
            
            # Format results
            results = []
            for i, pred in enumerate(predictions):
                results.append({
                    'predicted_price': float(pred),
                    'predicted_price_formatted': f"${float(pred):,.2f}",
                    'input_features': df.iloc[i].to_dict()
                })
            
            return results
        except Exception as e:
            logger.error(f"Error making batch predictions: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        try:
            best_model_info = load_artifact('artifacts/best_model_info.joblib')
            
            return {
                'model_name': best_model_info['best_model'],
                'metrics': best_model_info['metrics'],
                'features_used': self.preprocessor.feature_columns
            }
        except Exception as e:
            logger.error(f"Error getting model info: {str(e)}")
            return {'error': str(e)}