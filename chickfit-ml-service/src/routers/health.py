from datetime import datetime
from fastapi import APIRouter
from ..models.schemas import HealthResponse
from ..models.cnn_model import cnn_model

router = APIRouter()

@router.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if cnn_model.is_loaded else "degraded",
        timestamp=datetime.now(),
        service="chickfit-ml-service",
        version="1.0.0",
        model_loaded=cnn_model.is_loaded
    )