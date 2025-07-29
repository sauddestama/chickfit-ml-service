import numpy as np
from PIL import Image
import logging
from ..config.settings import settings

logger = logging.getLogger(__name__)

def validate_and_preprocess_image(image: Image.Image) -> np.ndarray:
    """Validate and preprocess image for model prediction"""
    try:
        # Check image format
        if image.format not in settings.allowed_image_formats and image.format is not None:
            raise ValueError(f"Unsupported image format: {image.format}")
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Check image size constraints
        min_size = 32
        max_size = 4096
        width, height = image.size
        
        if width < min_size or height < min_size:
            raise ValueError(f"Image too small. Minimum size: {min_size}x{min_size}px")
        
        if width > max_size or height > max_size:
            raise ValueError(f"Image too large. Maximum size: {max_size}x{max_size}px")
        
        # Resize image to model input size
        image = image.resize((settings.image_size, settings.image_size), Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Normalize to [0, 1]
        if image_array.max() > 1.0:
            image_array = image_array.astype(np.float32) / 255.0
        
        # Ensure correct shape
        if len(image_array.shape) != 3 or image_array.shape[-1] != 3:
            raise ValueError("Image must be RGB with 3 channels")
        
        return image_array
        
    except Exception as e:
        logger.error(f"Error validating/preprocessing image: {e}")
        raise

def calculate_image_stats(image_array: np.ndarray) -> dict:
    """Calculate basic statistics of an image"""
    try:
        stats = {
            'mean': float(np.mean(image_array)),
            'std': float(np.std(image_array)),
            'min': float(np.min(image_array)),
            'max': float(np.max(image_array)),
            'shape': image_array.shape
        }
        
        # Calculate per-channel statistics
        if len(image_array.shape) == 3:
            stats['channel_means'] = [float(np.mean(image_array[:, :, i])) for i in range(image_array.shape[2])]
            stats['channel_stds'] = [float(np.std(image_array[:, :, i])) for i in range(image_array.shape[2])]
        
        return stats
        
    except Exception as e:
        logger.error(f"Error calculating image stats: {e}")
        return {}

def check_image_quality(image_array: np.ndarray) -> dict:
    """Check basic image quality metrics"""
    try:
        quality_info = {
            'is_blurry': False,
            'is_too_dark': False,
            'is_too_bright': False,
            'has_good_contrast': True,
            'quality_score': 1.0
        }
        
        # Convert to grayscale for analysis
        if len(image_array.shape) == 3:
            gray = np.mean(image_array, axis=2)
        else:
            gray = image_array
        
        # Check brightness
        mean_brightness = np.mean(gray)
        if mean_brightness < 0.1:
            quality_info['is_too_dark'] = True
            quality_info['quality_score'] *= 0.7
        elif mean_brightness > 0.9:
            quality_info['is_too_bright'] = True
            quality_info['quality_score'] *= 0.8
        
        # Check contrast
        contrast = np.std(gray)
        if contrast < 0.05:
            quality_info['has_good_contrast'] = False
            quality_info['quality_score'] *= 0.6
        
        # Simple blur detection using variance of Laplacian
        # This is a basic implementation - more sophisticated methods exist
        try:
            from scipy import ndimage
            laplacian_var = ndimage.variance(ndimage.laplace(gray))
            if laplacian_var < 100:  # Threshold may need tuning
                quality_info['is_blurry'] = True
                quality_info['quality_score'] *= 0.5
        except ImportError:
            # scipy not available, skip blur detection
            pass
        
        return quality_info
        
    except Exception as e:
        logger.error(f"Error checking image quality: {e}")
        return {'quality_score': 1.0}

def augment_image(image_array: np.ndarray, augmentation_type: str = 'basic') -> np.ndarray:
    """Apply basic image augmentation"""
    try:
        augmented = image_array.copy()
        
        if augmentation_type == 'horizontal_flip':
            augmented = np.fliplr(augmented)
        elif augmentation_type == 'brightness':
            # Slight brightness adjustment
            factor = np.random.uniform(0.8, 1.2)
            augmented = np.clip(augmented * factor, 0, 1)
        elif augmentation_type == 'contrast':
            # Slight contrast adjustment
            factor = np.random.uniform(0.8, 1.2)
            mean = np.mean(augmented)
            augmented = np.clip((augmented - mean) * factor + mean, 0, 1)
        elif augmentation_type == 'rotation':
            # Small rotation (requires scipy)
            try:
                from scipy import ndimage
                angle = np.random.uniform(-10, 10)
                augmented = ndimage.rotate(augmented, angle, reshape=False, mode='nearest')
            except ImportError:
                pass  # Skip rotation if scipy not available
        
        return augmented
        
    except Exception as e:
        logger.error(f"Error augmenting image: {e}")
        return image_array

def create_image_thumbnail(image: Image.Image, size: tuple = (128, 128)) -> Image.Image:
    """Create a thumbnail of the image"""
    try:
        thumbnail = image.copy()
        thumbnail.thumbnail(size, Image.Resampling.LANCZOS)
        return thumbnail
    except Exception as e:
        logger.error(f"Error creating thumbnail: {e}")
        return image