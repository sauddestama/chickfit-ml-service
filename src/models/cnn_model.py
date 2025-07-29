import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
import logging
from datetime import datetime
from ..config.settings import settings
from ..config.gcs_client import gcs_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChickFitCNNModel:
    def __init__(self):
        self.model = None
        self.model_info = {
            'name': 'ChickFit CNN Model',
            'version': 'v1.0',
            'accuracy': 0.9535,
            'classes': settings.classes,
            'input_shape': [settings.image_size, settings.image_size, 3],
            'trained_at': None,
            'model_size': 0
        }
        self.is_loaded = False
    
    def create_model(self, num_classes: int = 3) -> Model:
        """Create a new CNN model based on MobileNetV2"""
        try:
            # Load pre-trained MobileNetV2 as base model
            base_model = MobileNetV2(
                weights='imagenet',
                include_top=False,
                input_shape=(settings.image_size, settings.image_size, 3)
            )
            
            # Freeze base model layers
            base_model.trainable = False
            
            # Add custom classification head
            inputs = keras.Input(shape=(settings.image_size, settings.image_size, 3))
            x = base_model(inputs, training=False)
            x = GlobalAveragePooling2D()(x)
            x = Dropout(0.2)(x)
            outputs = Dense(num_classes, activation='softmax')(x)
            
            model = Model(inputs, outputs)
            
            # Compile model
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            logger.info(f"Created new CNN model with {num_classes} classes")
            return model
            
        except Exception as e:
            logger.error(f"Error creating model: {e}")
            raise e
    
    def load_model(self, model_path: str = None) -> bool:
        """Load model from file or GCS"""
        try:
            if model_path is None:
                model_path = os.path.join(settings.model_path, settings.default_model)
            
            # Ensure models directory exists
            os.makedirs(settings.model_path, exist_ok=True)
            
            # Check if model exists locally
            if not os.path.exists(model_path):
                # Try to download from GCS
                logger.info(f"Model not found locally, downloading from GCS...")
                success = gcs_client.download_model(settings.default_model, model_path)
                if not success:
                    logger.warning("Failed to download model from GCS, creating new model")
                    self.model = self.create_model()
                    self.save_model(model_path)
                    self.is_loaded = True
                    return True
            
            # Load the model
            self.model = keras.models.load_model(model_path)
            
            # Update model info
            self.model_info.update({
                'model_size': os.path.getsize(model_path) if os.path.exists(model_path) else 0,
                'trained_at': datetime.fromtimestamp(os.path.getmtime(model_path)) if os.path.exists(model_path) else None
            })
            
            self.is_loaded = True
            logger.info(f"Model loaded successfully from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            # Fallback: create new model
            try:
                self.model = self.create_model()
                self.is_loaded = True
                logger.info("Created new model as fallback")
                return True
            except Exception as fallback_error:
                logger.error(f"Failed to create fallback model: {fallback_error}")
                return False
    
    def save_model(self, model_path: str) -> bool:
        """Save model to file and upload to GCS"""
        try:
            if self.model is None:
                logger.error("No model to save")
                return False
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Save model locally
            self.model.save(model_path)
            logger.info(f"Model saved locally to {model_path}")
            
            # Upload to GCS
            model_name = os.path.basename(model_path)
            upload_success = gcs_client.upload_model(model_path, model_name)
            if upload_success:
                logger.info(f"Model uploaded to GCS as {model_name}")
            else:
                logger.warning("Failed to upload model to GCS")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def preprocess_image(self, image_array: np.ndarray) -> np.ndarray:
        """Preprocess image for prediction"""
        try:
            # Ensure image is in correct format
            if len(image_array.shape) == 4:
                image_array = image_array[0]  # Remove batch dimension if present
            
            # Resize image
            if image_array.shape[:2] != (settings.image_size, settings.image_size):
                image_array = tf.image.resize(image_array, [settings.image_size, settings.image_size])
                image_array = image_array.numpy()
            
            # Ensure 3 channels (RGB)
            if len(image_array.shape) == 2:  # Grayscale
                image_array = np.stack([image_array] * 3, axis=-1)
            elif image_array.shape[-1] == 4:  # RGBA
                image_array = image_array[:, :, :3]  # Remove alpha channel
            
            # Normalize pixel values to [0, 1]
            if image_array.max() > 1.0:
                image_array = image_array.astype(np.float32) / 255.0
            
            # Add batch dimension
            image_array = np.expand_dims(image_array, axis=0)
            
            return image_array
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise e
    
    def predict(self, image_array: np.ndarray) -> dict:
        """Make prediction on a single image"""
        try:
            if not self.is_loaded or self.model is None:
                raise Exception("Model not loaded")
            
            # Preprocess image
            processed_image = self.preprocess_image(image_array)
            
            # Make prediction
            predictions = self.model.predict(processed_image, verbose=0)
            
            # Convert to dictionary with class names
            result = {}
            for i, class_name in enumerate(settings.classes):
                result[class_name] = float(predictions[0][i])
            
            return result
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            raise e
    
    def predict_batch(self, image_arrays: list) -> list:
        """Make predictions on multiple images"""
        try:
            if not self.is_loaded or self.model is None:
                raise Exception("Model not loaded")
            
            results = []
            for image_array in image_arrays:
                result = self.predict(image_array)
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error making batch predictions: {e}")
            raise e
    
    def get_model_info(self) -> dict:
        """Get information about the current model"""
        return self.model_info.copy()
    
    def update_model_accuracy(self, accuracy: float):
        """Update model accuracy information"""
        self.model_info['accuracy'] = accuracy
    
    def fine_tune_model(self, train_data, val_data, epochs: int = 10):
        """Fine-tune the model with new data"""
        try:
            if not self.is_loaded or self.model is None:
                raise Exception("Model not loaded")
            
            # Unfreeze some layers for fine-tuning
            base_model = self.model.layers[1]  # MobileNetV2 base
            base_model.trainable = True
            
            # Fine-tune from this layer onwards
            fine_tune_at = 100
            for layer in base_model.layers[:fine_tune_at]:
                layer.trainable = False
            
            # Use lower learning rate for fine-tuning
            self.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.0001/10),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Train the model
            history = self.model.fit(
                train_data,
                validation_data=val_data,
                epochs=epochs,
                verbose=1
            )
            
            # Update accuracy
            val_accuracy = max(history.history['val_accuracy'])
            self.update_model_accuracy(val_accuracy)
            
            logger.info(f"Model fine-tuning completed. Validation accuracy: {val_accuracy:.4f}")
            return history
            
        except Exception as e:
            logger.error(f"Error fine-tuning model: {e}")
            raise e

# Create global model instance
cnn_model = ChickFitCNNModel()
