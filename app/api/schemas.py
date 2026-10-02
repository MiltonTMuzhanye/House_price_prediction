from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

class HouseFeatures(BaseModel):
    """Input schema for house features"""
    SQFT: int = Field(..., ge=100, le=5000, description="Square footage")
    BEDROOMS: int = Field(..., ge=1, le=10, description="Number of bedrooms")
    LOCATION: int = Field(..., description="Source dataset LOCATION code")
    REGION: int = Field(..., description="Source dataset REGION code")
    TITLED: int = Field(..., description="Source dataset TITLED code")
    LEASE: int = Field(..., description="Source dataset LEASE code")
    FOOTINGS: int = Field(..., description="Source dataset FOOTINGS code")

class PredictionRequest(BaseModel):
    """Request schema for single prediction"""
    features: HouseFeatures

class BatchPredictionRequest(BaseModel):
    """Request schema for batch prediction"""
    features: List[HouseFeatures]

class PredictionResponse(BaseModel):
    """Response schema for prediction"""
    predicted_price: float
    predicted_price_formatted: str
    input_features: HouseFeatures
    timestamp: datetime = Field(default_factory=datetime.now)
    
class BatchPredictionResponse(BaseModel):
    """Response schema for batch prediction"""
    predictions: List[PredictionResponse]
    total_predictions: int
    avg_price: float
    min_price: float
    max_price: float
    timestamp: datetime = Field(default_factory=datetime.now)

class ModelInfoResponse(BaseModel):
    """Response schema for model info"""
    model_name: str
    version: str
    metrics: Dict[str, float]
    features_used: Optional[List[str]]
    last_updated: datetime = Field(default_factory=datetime.now)

class ErrorResponse(BaseModel):
    """Error response schema"""
    error: str
    timestamp: datetime = Field(default_factory=datetime.now)