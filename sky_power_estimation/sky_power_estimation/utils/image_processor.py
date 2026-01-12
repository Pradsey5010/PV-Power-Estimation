"""
Image Processor

Handles preprocessing of sky images including cloud detection,
opacity estimation, and feature extraction using OpenCV.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import warnings

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    warnings.warn("OpenCV not available. Image processing features will be limited.")

from PIL import Image


class ImageProcessor:
    """
    Processor for sky images.
    
    Provides methods for:
    - Image loading and preprocessing
    - Cloud detection and segmentation
    - Opacity estimation
    - Feature extraction for machine learning
    
    Args:
        image_size: Target size for processed images (height, width)
        normalize: Whether to normalize pixel values
    """
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        normalize: bool = True,
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225)
    ):
        self.image_size = image_size
        self.normalize = normalize
        self.mean = np.array(mean)
        self.std = np.array(std)
    
    def load_image(
        self,
        path: Union[str, Path],
        as_rgb: bool = True
    ) -> np.ndarray:
        """
        Load an image from file.
        
        Args:
            path: Path to image file
            as_rgb: Whether to convert to RGB (default: True)
            
        Returns:
            Image as numpy array
        """
        if CV2_AVAILABLE:
            img = cv2.imread(str(path))
            if img is None:
                raise ValueError(f"Could not load image: {path}")
            if as_rgb:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = np.array(Image.open(path))
            if not as_rgb and len(img.shape) == 3:
                img = img[..., ::-1]  # RGB to BGR
        
        return img
    
    def preprocess(
        self,
        image: np.ndarray,
        resize: bool = True
    ) -> np.ndarray:
        """
        Preprocess an image for model input.
        
        Args:
            image: Input image array (H, W, C)
            resize: Whether to resize to target size
            
        Returns:
            Preprocessed image (C, H, W) normalized
        """
        # Resize if needed
        if resize:
            if CV2_AVAILABLE:
                image = cv2.resize(
                    image,
                    (self.image_size[1], self.image_size[0]),
                    interpolation=cv2.INTER_LINEAR
                )
            else:
                pil_img = Image.fromarray(image)
                pil_img = pil_img.resize(
                    (self.image_size[1], self.image_size[0]),
                    Image.BILINEAR
                )
                image = np.array(pil_img)
        
        # Convert to float and normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Apply ImageNet normalization
        if self.normalize:
            image = (image - self.mean) / self.std
        
        # Convert to CHW format
        image = np.transpose(image, (2, 0, 1))
        
        return image.astype(np.float32)
    
    def extract_cloud_features(
        self,
        image: np.ndarray
    ) -> Dict[str, float]:
        """
        Extract cloud-related features from sky image.
        
        Args:
            image: Sky image (RGB, HWC format)
            
        Returns:
            Dictionary of cloud features:
            - cloud_cover: Estimated cloud cover percentage
            - brightness: Average brightness
            - blue_ratio: Blue channel ratio (indicator of clear sky)
            - contrast: Image contrast
            - entropy: Image entropy
        """
        if not CV2_AVAILABLE:
            return self._extract_features_fallback(image)
        
        # Convert to different color spaces
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Cloud detection using color thresholding
        # Clouds tend to be bright and low in saturation
        brightness = hsv[:, :, 2]
        saturation = hsv[:, :, 1]
        
        # Cloud mask: high brightness, low saturation
        cloud_mask = (brightness > 150) & (saturation < 50)
        cloud_cover = np.sum(cloud_mask) / cloud_mask.size * 100
        
        # Average brightness
        avg_brightness = np.mean(brightness) / 255.0
        
        # Blue ratio (clear sky indicator)
        blue_ratio = np.mean(image[:, :, 2]) / (np.mean(image) + 1e-6)
        
        # Contrast
        contrast = np.std(gray) / (np.mean(gray) + 1e-6)
        
        # Entropy (texture complexity)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist / hist.sum()
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log2(hist))
        
        return {
            "cloud_cover": cloud_cover,
            "brightness": avg_brightness,
            "blue_ratio": blue_ratio,
            "contrast": contrast,
            "entropy": entropy
        }
    
    def _extract_features_fallback(
        self,
        image: np.ndarray
    ) -> Dict[str, float]:
        """Fallback feature extraction without OpenCV."""
        # Simple brightness
        brightness = np.mean(image) / 255.0
        
        # Blue ratio
        blue_ratio = np.mean(image[:, :, 2]) / (np.mean(image) + 1e-6)
        
        # Rough cloud estimate based on brightness
        cloud_cover = min(100, max(0, (brightness - 0.3) * 200))
        
        return {
            "cloud_cover": cloud_cover,
            "brightness": brightness,
            "blue_ratio": blue_ratio,
            "contrast": np.std(image) / (np.mean(image) + 1e-6),
            "entropy": 5.0  # Default value
        }
    
    def segment_clouds(
        self,
        image: np.ndarray,
        method: str = "threshold"
    ) -> np.ndarray:
        """
        Segment clouds from sky image.
        
        Args:
            image: Sky image (RGB)
            method: Segmentation method ('threshold', 'adaptive', 'otsu')
            
        Returns:
            Binary mask where 1 indicates cloud pixels
        """
        if not CV2_AVAILABLE:
            # Fallback: simple brightness thresholding
            gray = np.mean(image, axis=2)
            return (gray > 180).astype(np.uint8)
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        if method == "threshold":
            _, mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        
        elif method == "adaptive":
            mask = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11, 2
            )
        
        elif method == "otsu":
            _, mask = cv2.threshold(
                gray, 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return mask // 255
    
    def estimate_opacity(
        self,
        image: np.ndarray,
        cloud_mask: Optional[np.ndarray] = None
    ) -> float:
        """
        Estimate cloud opacity from sky image.
        
        Args:
            image: Sky image
            cloud_mask: Optional pre-computed cloud mask
            
        Returns:
            Opacity value between 0 (transparent) and 1 (opaque)
        """
        if cloud_mask is None:
            cloud_mask = self.segment_clouds(image)
        
        if not CV2_AVAILABLE:
            # Simple estimate based on cloud brightness
            if cloud_mask.sum() == 0:
                return 0.0
            cloud_pixels = image[cloud_mask == 1]
            return 1.0 - (np.std(cloud_pixels) / 127.5)
        
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Get brightness and saturation of cloud regions
        cloud_pixels = hsv[cloud_mask == 1]
        
        if len(cloud_pixels) == 0:
            return 0.0
        
        # Opacity based on brightness uniformity and saturation
        brightness_std = np.std(cloud_pixels[:, 2])
        saturation_mean = np.mean(cloud_pixels[:, 1])
        
        # Lower std and saturation = more opaque clouds
        opacity = 1.0 - (brightness_std / 127.5) - (saturation_mean / 255.0)
        
        return np.clip(opacity, 0, 1)
    
    def compute_sun_visibility(
        self,
        image: np.ndarray,
        sun_position: Optional[Tuple[int, int]] = None
    ) -> float:
        """
        Estimate sun visibility from sky image.
        
        Args:
            image: Sky image
            sun_position: Optional (x, y) position of sun in image
            
        Returns:
            Visibility score 0-1 (1 = fully visible)
        """
        if not CV2_AVAILABLE:
            # Fallback: check for bright region
            max_brightness = np.max(image)
            return min(1.0, max_brightness / 255.0)
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Find brightest region
        _, max_val, _, max_loc = cv2.minMaxLoc(gray)
        
        if sun_position is None:
            sun_position = max_loc
        
        # Create region around sun position
        x, y = sun_position
        h, w = image.shape[:2]
        
        # Region size (10% of image)
        region_size = int(min(h, w) * 0.1)
        
        # Extract region (with bounds checking)
        x1 = max(0, x - region_size)
        x2 = min(w, x + region_size)
        y1 = max(0, y - region_size)
        y2 = min(h, y + region_size)
        
        region = gray[y1:y2, x1:x2]
        
        # Visibility based on brightness and uniformity of sun region
        brightness = np.mean(region) / 255.0
        uniformity = 1.0 - (np.std(region) / 127.5)
        
        visibility = 0.7 * brightness + 0.3 * uniformity
        
        return np.clip(visibility, 0, 1)
    
    def enhance_image(
        self,
        image: np.ndarray,
        contrast: float = 1.2,
        brightness: float = 0
    ) -> np.ndarray:
        """
        Enhance sky image for better feature extraction.
        
        Args:
            image: Input image
            contrast: Contrast factor (1.0 = no change)
            brightness: Brightness adjustment
            
        Returns:
            Enhanced image
        """
        if not CV2_AVAILABLE:
            enhanced = image.astype(np.float32) * contrast + brightness
            return np.clip(enhanced, 0, 255).astype(np.uint8)
        
        # Apply CLAHE for adaptive contrast enhancement
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Adjust contrast and brightness
        enhanced = enhanced.astype(np.float32) * contrast + brightness
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
        
        return enhanced
    
    def batch_preprocess(
        self,
        images: List[np.ndarray]
    ) -> np.ndarray:
        """
        Preprocess a batch of images.
        
        Args:
            images: List of images
            
        Returns:
            Stacked preprocessed images [N, C, H, W]
        """
        processed = [self.preprocess(img) for img in images]
        return np.stack(processed, axis=0)
    
    def get_feature_array(
        self,
        image: np.ndarray
    ) -> np.ndarray:
        """
        Get cloud features as numpy array.
        
        Args:
            image: Sky image
            
        Returns:
            Feature array [cloud_cover, brightness, blue_ratio, contrast, entropy]
        """
        features = self.extract_cloud_features(image)
        return np.array([
            features["cloud_cover"],
            features["brightness"],
            features["blue_ratio"],
            features["contrast"],
            features["entropy"]
        ], dtype=np.float32)


class SkyImageAugmentor:
    """
    Augmentation for sky images during training.
    """
    
    def __init__(
        self,
        horizontal_flip: bool = True,
        vertical_flip: bool = False,
        rotation_range: int = 15,
        brightness_range: float = 0.2,
        contrast_range: float = 0.2,
        saturation_range: float = 0.2,
        hue_range: float = 0.1
    ):
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rotation_range = rotation_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.hue_range = hue_range
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Apply random augmentations to image."""
        
        # Random horizontal flip
        if self.horizontal_flip and np.random.random() > 0.5:
            image = np.fliplr(image)
        
        # Random vertical flip
        if self.vertical_flip and np.random.random() > 0.5:
            image = np.flipud(image)
        
        # Random rotation
        if self.rotation_range > 0 and CV2_AVAILABLE:
            angle = np.random.uniform(-self.rotation_range, self.rotation_range)
            h, w = image.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
            image = cv2.warpAffine(image, M, (w, h))
        
        # Random brightness
        if self.brightness_range > 0:
            factor = 1 + np.random.uniform(-self.brightness_range, self.brightness_range)
            image = np.clip(image * factor, 0, 255).astype(np.uint8)
        
        # Random contrast
        if self.contrast_range > 0:
            factor = 1 + np.random.uniform(-self.contrast_range, self.contrast_range)
            mean = np.mean(image)
            image = np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)
        
        return image


if __name__ == "__main__":
    # Test image processor
    print("Testing ImageProcessor...")
    
    processor = ImageProcessor(image_size=(224, 224))
    
    # Create a synthetic sky image
    np.random.seed(42)
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Blue sky gradient
    for i in range(480):
        blue_val = int(200 - i * 0.2)
        test_image[i, :, 2] = blue_val  # Blue channel
        test_image[i, :, 1] = int(blue_val * 0.6)  # Green
        test_image[i, :, 0] = int(blue_val * 0.3)  # Red
    
    # Add some "clouds" (white regions)
    test_image[100:200, 200:400] = [230, 230, 235]
    test_image[250:300, 100:250] = [220, 220, 225]
    
    print(f"Original image shape: {test_image.shape}")
    
    # Preprocess
    preprocessed = processor.preprocess(test_image)
    print(f"Preprocessed shape: {preprocessed.shape}")
    print(f"Preprocessed range: [{preprocessed.min():.3f}, {preprocessed.max():.3f}]")
    
    # Extract features
    features = processor.extract_cloud_features(test_image)
    print(f"\nCloud features:")
    for key, value in features.items():
        print(f"  {key}: {value:.3f}")
    
    # Segment clouds
    cloud_mask = processor.segment_clouds(test_image)
    print(f"\nCloud mask shape: {cloud_mask.shape}")
    print(f"Cloud coverage from mask: {cloud_mask.mean() * 100:.1f}%")
    
    # Estimate opacity
    opacity = processor.estimate_opacity(test_image, cloud_mask)
    print(f"Cloud opacity: {opacity:.3f}")
    
    # Sun visibility
    visibility = processor.compute_sun_visibility(test_image)
    print(f"Sun visibility: {visibility:.3f}")
    
    # Test augmentor
    print("\nTesting SkyImageAugmentor...")
    augmentor = SkyImageAugmentor()
    augmented = augmentor(test_image.copy())
    print(f"Augmented shape: {augmented.shape}")
