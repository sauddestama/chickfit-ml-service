import os
from typing import List
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Google Cloud Configuration
    google_cloud_project_id: str = "chickfit-project"
    google_cloud_bucket: str = "chickfit-storage"
    google_application_credentials: str = "./service-account-key.json"
    
    # Model Configuration
    model_path: str = "models/"
    default_model: str = "chickfit_model_v1.h5"
    image_size: int = 224
    batch_size: int = 32
    classes: List[str] = ["Coccidiosis", "ND", "Sehat"]
    
    # Server Configuration
    port: int = 8000
    environment: str = "development"
    log_level: str = "INFO"
    
    # Processing Configuration
    max_image_size: int = 10 * 1024 * 1024  # 10MB
    allowed_image_formats: List[str] = ["JPEG", "JPG", "PNG"]
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Create global settings instance
settings = Settings()