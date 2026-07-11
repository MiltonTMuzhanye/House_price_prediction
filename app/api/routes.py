from fastapi import APIRouter, HTTPException
from typing import List
import numpy as np
from .schemas import (
    PredictionRequest, 
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ModelInfoResponse,
    ErrorResponse
)
from app.inference.predictor import HousePricePredictor
from src.house_price_prediction.utils.logger import logger

router = APIRouter()
predictor = HousePricePredictor()

@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make a single house price prediction"""
    try:
        input_dict = request.features.dict()
        result = predictor.predict_single(input_dict)
        
        return PredictionResponse(
            predicted_price=result['predicted_price'],
            predicted_price_formatted=result['predicted_price_formatted'],
            input_features=request.features
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Make batch house price predictions"""
    try:
        input_dicts = [feature.dict() for feature in request.features]
        results = predictor.predict_batch(input_dicts)
        
        prices = [r['predicted_price'] for r in results]
        
        return BatchPredictionResponse(
            predictions=results,
            total_predictions=len(results),
            avg_price=float(np.mean(prices)),
            min_price=float(np.min(prices)),
            max_price=float(np.max(prices))
        )
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get information about the currently loaded model"""
    try:
        info = predictor.get_model_info()
        return ModelInfoResponse(
            model_name=info.get('model_name', 'Unknown'),
            version="1.0.0",
            metrics=info.get('metrics', {}),
            features_used=info.get('features_used', [])
        )
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": predictor.model is not None}