from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from datetime import datetime

class PredictionRequest(BaseModel):
    image: str = Field(..., description="Base64 encoded image")

class BatchPredictionRequest(BaseModel):
    images: List[str] = Field(..., description="List of base64 encoded images")

class PredictionResponse(BaseModel):
    success: bool
    predictions: Dict[str, float]
    model_info: Optional[Dict] = None
    message: Optional[str] = None

class BatchPredictionResponse(BaseModel):
    success: bool
    predictions: List[Dict[str, float]]
    model_info: Optional[Dict] = None
    message: Optional[str] = None

class ModelInfo(BaseModel):
    name: str
    version: str
    accuracy: float
    classes: List[str]
    input_shape: List[int]
    trained_at: Optional[datetime] = None
    model_size: Optional[int] = None

class ModelInfoResponse(BaseModel):
    success: bool
    model: ModelInfo
    message: Optional[str] = None

class RetrainingRequest(BaseModel):
    admin_id: int = Field(..., description="ID of the administrator requesting retraining")
    notes: Optional[str] = Field(None, description="Optional notes about the retraining")

class RetrainingResponse(BaseModel):
    success: bool
    message: str
    task_id: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    service: str = "chickfit-ml-service"
    version: Optional[str] = None
    model_loaded: bool = False

class ErrorResponse(BaseModel):
    success: bool = False
    message: str
    error_code: Optional[str] = None