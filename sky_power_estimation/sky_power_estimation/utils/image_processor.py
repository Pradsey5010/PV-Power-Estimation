"""Image Processor with cloud detection using OpenCV."""

import numpy as np
from typing import Dict, Tuple, Optional
from PIL import Image
import warnings

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    warnings.warn("OpenCV not available. Using basic image processing.")


class ImageProcessor:
    """Processor for sky images with cloud detection."""
    
    def __init__(self, image_size: Tuple[int, int] = (224, 224), normalize: bool = True,
                 mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
                 std: Tuple[float, ...] = (0.229, 0.224, 0.225)):
        self.image_size = image_size
        self.normalize = normalize
        self.mean = np.array(mean)
        self.std = np.array(std)
    
    def load_image(self, path: str, as_rgb: bool = True) -> np.ndarray:
        if CV2_AVAILABLE:
            img = cv2.imread(str(path))
            if as_rgb:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = np.array(Image.open(path))
        return img
    
    def preprocess(self, image: np.ndarray, resize: bool = True) -> np.ndarray:
        if resize:
            if CV2_AVAILABLE:
                image = cv2.resize(image, (self.image_size[1], self.image_size[0]))
            else:
                pil_img = Image.fromarray(image)
                pil_img = pil_img.resize((self.image_size[1], self.image_size[0]))
                image = np.array(pil_img)
        
        image = image.astype(np.float32) / 255.0
        
        if self.normalize:
            image = (image - self.mean) / self.std
        
        return np.transpose(image, (2, 0, 1)).astype(np.float32)
    
    def extract_cloud_features(self, image: np.ndarray) -> Dict[str, float]:
        if not CV2_AVAILABLE:
            brightness = np.mean(image) / 255.0
            blue_ratio = np.mean(image[:, :, 2]) / (np.mean(image) + 1e-6)
            return {
                "cloud_cover": min(100, max(0, (brightness - 0.3) * 200)),
                "brightness": brightness,
                "blue_ratio": blue_ratio,
                "contrast": np.std(image) / (np.mean(image) + 1e-6),
                "entropy": 5.0,
                "opacity": 0.5
            }
        
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        brightness = hsv[:, :, 2]
        saturation = hsv[:, :, 1]
        
        cloud_mask = (brightness > 150) & (saturation < 50)
        cloud_cover = np.sum(cloud_mask) / cloud_mask.size * 100
        
        avg_brightness = np.mean(brightness) / 255.0
        blue_ratio = np.mean(image[:, :, 2]) / (np.mean(image) + 1e-6)
        contrast = np.std(gray) / (np.mean(gray) + 1e-6)
        
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist / hist.sum()
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log2(hist))
        
        opacity = 1.0 - (np.std(brightness) / 127.5)
        
        return {
            "cloud_cover": cloud_cover,
            "brightness": avg_brightness,
            "blue_ratio": blue_ratio,
            "contrast": contrast,
            "entropy": float(entropy),
            "opacity": float(np.clip(opacity, 0, 1))
        }
    
    def segment_clouds(self, image: np.ndarray, method: str = "threshold") -> np.ndarray:
        if not CV2_AVAILABLE:
            gray = np.mean(image, axis=2)
            return (gray > 180).astype(np.uint8)
        
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        if method == "threshold":
            _, mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        elif method == "adaptive":
            mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)
        elif method == "otsu":
            _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        
        return mask // 255
    
    def estimate_opacity(self, image: np.ndarray, cloud_mask: Optional[np.ndarray] = None) -> float:
        if cloud_mask is None:
            cloud_mask = self.segment_clouds(image)
        
        if not CV2_AVAILABLE or cloud_mask.sum() == 0:
            return 0.0
        
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        cloud_pixels = hsv[cloud_mask == 1]
        
        if len(cloud_pixels) == 0:
            return 0.0
        
        brightness_std = np.std(cloud_pixels[:, 2])
        saturation_mean = np.mean(cloud_pixels[:, 1])
        
        opacity = 1.0 - (brightness_std / 127.5) - (saturation_mean / 255.0)
        return float(np.clip(opacity, 0, 1))
    
    def get_feature_array(self, image: np.ndarray) -> np.ndarray:
        features = self.extract_cloud_features(image)
        return np.array([
            features["cloud_cover"], features["brightness"],
            features["blue_ratio"], features["contrast"],
            features["entropy"], features["opacity"]
        ], dtype=np.float32)


def generate_synthetic_sky_image(width: int = 640, height: int = 480,
                                 cloud_cover: float = 0.3) -> np.ndarray:
    """Generate a synthetic sky image for testing."""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Blue sky gradient
    for i in range(height):
        blue_val = int(200 - i * 0.3)
        image[i, :, 2] = max(100, min(255, blue_val))
        image[i, :, 1] = max(80, min(200, int(blue_val * 0.7)))
        image[i, :, 0] = max(50, min(150, int(blue_val * 0.4)))
    
    # Add clouds
    num_clouds = int(cloud_cover * 15)
    for _ in range(num_clouds):
        cx = np.random.randint(0, width)
        cy = np.random.randint(0, height // 2)
        radius = np.random.randint(30, 80)
        
        y, x = np.ogrid[:height, :width]
        mask = (x - cx) ** 2 + (y - cy) ** 2 < radius ** 2
        
        cloud_color = np.random.randint(210, 250)
        image[mask] = [cloud_color, cloud_color, min(255, cloud_color + 5)]
    
    return image
