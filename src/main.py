import os
import logging
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config.settings import settings
from .models.cnn_model import cnn_model
from .routers import predict, training, health
from .models.schemas import ErrorResponse

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    logger.info("Starting ChickFit ML Service...")
    
    # Load the CNN model
    try:
        success = cnn_model.load_model()
        if success:
            logger.info("CNN model loaded successfully")
        else:
            logger.error("Failed to load CNN model")
    except Exception as e:
        logger.error(f"Error during model loading: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down ChickFit ML Service...")

# Create FastAPI application
app = FastAPI(
    title="ChickFit ML Service",
    description="CNN-based poultry disease diagnosis service for ChickFit application",
    version="1.0.0",
    docs_url="/docs" if settings.environment == "development" else None,
    redoc_url="/redoc" if settings.environment == "development" else None,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, prefix="/health", tags=["Health"])
app.include_router(predict.router, prefix="", tags=["Prediction"])
app.include_router(training.router, prefix="", tags=["Training"])

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            message="Internal server error",
            error_code="INTERNAL_ERROR"
        ).dict()
    )

# Root endpoint
@app.get("/")
async def root():
    return {
        "service": "ChickFit ML Service",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": cnn_model.is_loaded
    }

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=port,
        reload=False
    )