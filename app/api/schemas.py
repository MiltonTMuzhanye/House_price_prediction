from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

class HouseFeatures(BaseModel):
    """Input schema for house features"""
    SQFT: int = Field(..., ge=100, le=5000, description="Square footage")
    BEDROOMS: int = Field(..., ge=1, le=10, description="Number of bedrooms")
    LOCATION: int = Field(..., ge=1, le=3, description="1: Urban, 2: Suburban, 3: Rural")
    REGION: int = Field(..., ge=1, le=4, description="1: Northeast, 2: Midwest, 3: South, 4: West")
    TITLED: int = Field(..., ge=1, le=3, description="1: Vehicle, 2: Land-Home, 3: Other")
    LEASE: int = Field(..., ge=0, le=1, description="0: No lease, 1: Lease")
    FOOTINGS: int = Field(..., ge=1, le=9, description="Footing type")

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