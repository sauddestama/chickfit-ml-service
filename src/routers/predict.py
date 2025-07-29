import base64
import io
import logging
from PIL import Image
import numpy as np
from fastapi import APIRouter, HTTPException

from ..models.schemas import (
    PredictionRequest, 
    PredictionResponse, 
    BatchPredictionRequest, 
    BatchPredictionResponse,
    ModelInfoResponse,
    ErrorResponse
)
from ..models.cnn_model import cnn_model
from ..utils.image_utils import validate_and_preprocess_image
from ..config.settings import settings

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/predict", response_model=PredictionResponse)
async def predict_image(request: PredictionRequest):
    """Make prediction on a single image"""
    try:
        # Check if model is loaded
        if not cnn_model.is_loaded:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Service unavailable."
            )
        
        # Decode base64 image
        try:
            image_data = base64.b64decode(request.image)
            image = Image.open(io.BytesIO(image_data))
        except Exception as e:
            logger.error(f"Error decoding image: {e}")
            raise HTTPException(
                status_code=400,
                detail="Invalid image data. Please provide a valid base64 encoded image."
            )
        
        # Validate and preprocess image
        try:
            image_array = validate_and_preprocess_image(image)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise HTTPException(
                status_code=500,
                detail="Error processing image"
            )
        
        # Make prediction
        try:
            predictions = cnn_model.predict(image_array)
            model_info = cnn_model.get_model_info()
            
            return PredictionResponse(
                success=True,
                predictions=predictions,
                model_info=model_info
            )
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            raise HTTPException(
                status_code=500,
                detail="Error making prediction"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in predict_image: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during prediction"
        )

@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch_images(request: BatchPredictionRequest):
    """Make predictions on multiple images"""
    try:
        # Check if model is loaded
        if not cnn_model.is_loaded:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Service unavailable."
            )
        
        # Limit batch size
        max_batch_size = 10
        if len(request.images) > max_batch_size:
            raise HTTPException(
                status_code=400,
                detail=f"Batch size too large. Maximum {max_batch_size} images allowed."
            )
        
        # Process all images
        image_arrays = []
        for i, image_b64 in enumerate(request.images):
            try:
                # Decode base64 image
                image_data = base64.b64decode(image_b64)
                image = Image.open(io.BytesIO(image_data))
                
                # Validate and preprocess image
                image_array = validate_and_preprocess_image(image)
                image_arrays.append(image_array)
                
            except Exception as e:
                logger.error(f"Error processing image {i}: {e}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Error processing image {i+1}: Invalid image data"
                )
        
        # Make batch predictions
        try:
            predictions = cnn_model.predict_batch(image_arrays)
            model_info = cnn_model.get_model_info()
            
            return BatchPredictionResponse(
                success=True,
                predictions=predictions,
                model_info=model_info
            )
            
        except Exception as e:
            logger.error(f"Error making batch predictions: {e}")
            raise HTTPException(
                status_code=500,
                detail="Error making batch predictions"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in predict_batch_images: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during batch prediction"
        )

@router.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get information about the current model"""
    try:
        if not cnn_model.is_loaded:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded"
            )
        
        model_info = cnn_model.get_model_info()
        
        return ModelInfoResponse(
            success=True,
            model=model_info
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(
            status_code=500,
            detail="Error retrieving model information"
        )