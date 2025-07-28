import os
from google.cloud import storage
from .settings import settings

class GCSClient:
    def __init__(self):
        # Initialize Google Cloud Storage client
        self.client = storage.Client(project=settings.google_cloud_project_id)
        self.bucket = self.client.bucket(settings.google_cloud_bucket)
    
    def download_model(self, model_name: str, local_path: str) -> bool:
        """Download model from Google Cloud Storage"""
        try:
            blob = self.bucket.blob(f"models/{model_name}")
            blob.download_to_filename(local_path)
            return True
        except Exception as e:
            print(f"Error downloading model {model_name}: {e}")
            return False
    
    def upload_model(self, local_path: str, model_name: str) -> bool:
        """Upload model to Google Cloud Storage"""
        try:
            blob = self.bucket.blob(f"models/{model_name}")
            blob.upload_from_filename(local_path)
            return True
        except Exception as e:
            print(f"Error uploading model {model_name}: {e}")
            return False
    
    def list_models(self) -> list:
        """List all models in the models/ folder"""
        try:
            blobs = self.bucket.list_blobs(prefix="models/")
            return [blob.name for blob in blobs if blob.name.endswith(('.h5', '.pkl'))]
        except Exception as e:
            print(f"Error listing models: {e}")
            return []
    
    def download_training_data(self, folder_path: str) -> bool:
        """Download training data from Google Cloud Storage"""
        try:
            blobs = self.bucket.list_blobs(prefix=f"training-data/{folder_path}")
            for blob in blobs:
                if not blob.name.endswith('/'):  # Skip directory entries
                    local_path = os.path.join("training_data", blob.name.replace('training-data/', ''))
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)
                    blob.download_to_filename(local_path)
            return True
        except Exception as e:
            print(f"Error downloading training data: {e}")
            return False
    
    def upload_file(self, local_path: str, gcs_path: str) -> bool:
        """Upload any file to Google Cloud Storage"""
        try:
            blob = self.bucket.blob(gcs_path)
            blob.upload_from_filename(local_path)
            return True
        except Exception as e:
            print(f"Error uploading file to {gcs_path}: {e}")
            return False
    
    def file_exists(self, gcs_path: str) -> bool:
        """Check if file exists in Google Cloud Storage"""
        try:
            blob = self.bucket.blob(gcs_path)
            return blob.exists()
        except Exception as e:
            print(f"Error checking file existence {gcs_path}: {e}")
            return False

# Create global GCS client instance
gcs_client = GCSClient()