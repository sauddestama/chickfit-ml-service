import os
import logging
import numpy as np
from datetime import datetime
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import shutil

from ..config.settings import settings
from ..config.gcs_client import gcs_client
from ..models.cnn_model import cnn_model

logger = logging.getLogger(__name__)

async def trigger_model_retraining(admin_id: int, notes: str = "", progress_callback=None):
    """Trigger model retraining with new verified data"""
    try:
        logger.info(f"Starting model retraining requested by admin {admin_id}")
        
        if progress_callback:
            progress_callback(5)
        
        # Step 1: Download new training data from GCS
        logger.info("Downloading new training data...")
        training_data_path = "training_data"
        os.makedirs(training_data_path, exist_ok=True)
        
        # Download new samples that have been verified by veterinarians
        success = gcs_client.download_training_data("new-samples")
        if not success:
            return {"success": False, "error": "Failed to download training data"}
        
        if progress_callback:
            progress_callback(20)
        
        # Step 2: Prepare data generators
        logger.info("Preparing training data...")
        train_generator, val_generator, sample_count = prepare_training_data(training_data_path)
        
        if train_generator is None:
            return {"success": False, "error": "No training data available"}
        
        if progress_callback:
            progress_callback(40)
        
        # Step 3: Fine-tune the existing model
        logger.info("Starting model fine-tuning...")
        try:
            # Calculate training epochs based on data size
            epochs = min(20, max(5, sample_count // 50))
            
            history = cnn_model.fine_tune_model(
                train_data=train_generator,
                val_data=val_generator,
                epochs=epochs
            )
            
            if progress_callback:
                progress_callback(80)
            
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            return {"success": False, "error": f"Training failed: {str(e)}"}
        
        # Step 4: Save the updated model
        logger.info("Saving updated model...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"chickfit_model_retrained_{timestamp}.h5"
        model_path = os.path.join(settings.model_path, model_filename)
        
        save_success = cnn_model.save_model(model_path)
        if not save_success:
            return {"success": False, "error": "Failed to save updated model"}
        
        if progress_callback:
            progress_callback(90)
        
        # Step 5: Clean up training data
        try:
            shutil.rmtree(training_data_path)
        except Exception as e:
            logger.warning(f"Failed to clean up training data: {e}")
        
        # Step 6: Get final model accuracy
        val_accuracy = max(history.history['val_accuracy']) if history else 0.0
        
        if progress_callback:
            progress_callback(100)
        
        logger.info(f"Model retraining completed successfully. Validation accuracy: {val_accuracy:.4f}")
        
        return {
            "success": True,
            "message": "Model retraining completed successfully",
            "accuracy": val_accuracy,
            "model_filename": model_filename,
            "epochs_trained": epochs if 'epochs' in locals() else 0
        }
        
    except Exception as e:
        logger.error(f"Error in model retraining: {e}")
        return {"success": False, "error": str(e)}

def prepare_training_data(data_path: str):
    """Prepare training and validation data generators"""
    try:
        # Check if training data exists
        new_samples_path = os.path.join(data_path, "new-samples")
        if not os.path.exists(new_samples_path):
            logger.error("No new training samples found")
            return None, None, 0
        
        # Count total samples
        total_samples = 0
        class_counts = {}
        
        for class_name in settings.classes:
            class_path = os.path.join(new_samples_path, class_name)
            if os.path.exists(class_path):
                count = len([f for f in os.listdir(class_path) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                class_counts[class_name] = count
                total_samples += count
        
        logger.info(f"Found {total_samples} total samples: {class_counts}")
        
        if total_samples < 10:
            logger.error("Insufficient training data (minimum 10 samples required)")
            return None, None, 0
        
        # Create data generators with augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=0.2  # 20% for validation
        )
        
        train_generator = train_datagen.flow_from_directory(
            new_samples_path,
            target_size=(settings.image_size, settings.image_size),
            batch_size=min(settings.batch_size, total_samples // 4),
            class_mode='categorical',
            classes=settings.classes,
            subset='training'
        )
        
        val_generator = train_datagen.flow_from_directory(
            new_samples_path,
            target_size=(settings.image_size, settings.image_size),
            batch_size=min(settings.batch_size, total_samples // 4),
            class_mode='categorical',
            classes=settings.classes,
            subset='validation'
        )
        
        return train_generator, val_generator, total_samples
        
    except Exception as e:
        logger.error(f"Error preparing training data: {e}")
        return None, None, 0

def evaluate_model_performance(model, test_data):
    """Evaluate model performance on test data"""
    try:
        # Get predictions
        predictions = model.predict(test_data)
        
        # Calculate accuracy
        test_loss, test_accuracy = model.evaluate(test_data, verbose=0)
        
        # Calculate per-class metrics if needed
        y_true = test_data.classes
        y_pred = np.argmax(predictions, axis=1)
        
        from sklearn.metrics import classification_report, confusion_matrix
        
        report = classification_report(y_true, y_pred, target_names=settings.classes)
        confusion_mat = confusion_matrix(y_true, y_pred)
        
        return {
            'accuracy': test_accuracy,
            'loss': test_loss,
            'classification_report': report,
            'confusion_matrix': confusion_mat.tolist()
        }
        
    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        return {'accuracy': 0.0, 'loss': float('inf')}

def backup_current_model():
    """Create a backup of the current model before retraining"""
    try:
        current_model_path = os.path.join(settings.model_path, settings.default_model)
        if os.path.exists(current_model_path):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"chickfit_model_backup_{timestamp}.h5"
            backup_path = os.path.join(settings.model_path, backup_filename)
            
            shutil.copy2(current_model_path, backup_path)
            
            # Upload backup to GCS
            gcs_client.upload_file(backup_path, f"backups/{backup_filename}")
            
            logger.info(f"Model backup created: {backup_filename}")
            return backup_filename
        
        return None
        
    except Exception as e:
        logger.error(f"Error creating model backup: {e}")
        return None