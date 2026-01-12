"""
Image Transforms Module

Data augmentation and preprocessing transforms for sky images.
"""

import numpy as np
from typing import Dict, Tuple, Optional, List, Any
import warnings

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    warnings.warn("Albumentations not available. Using basic transforms.")

import torch
from PIL import Image


class BasicTransform:
    """Basic transform without albumentations."""
    
    def __init__(
        self,
        image_size: int = 224,
        normalize: bool = True,
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225)
    ):
        self.image_size = image_size
        self.normalize = normalize
        self.mean = np.array(mean)
        self.std = np.array(std)
    
    def __call__(self, image: np.ndarray) -> torch.Tensor:
        # Resize
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image
        
        pil_image = pil_image.resize(
            (self.image_size, self.image_size),
            Image.BILINEAR
        )
        image = np.array(pil_image)
        
        # Convert to float
        image = image.astype(np.float32) / 255.0
        
        # Normalize
        if self.normalize:
            image = (image - self.mean) / self.std
        
        # To tensor (CHW)
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        
        return image


def get_train_transforms(
    image_size: int = 224,
    augmentation_config: Optional[Dict[str, Any]] = None
) -> Any:
    """
    Get training transforms with augmentation.
    
    Args:
        image_size: Target image size
        augmentation_config: Augmentation configuration
        
    Returns:
        Transform pipeline
    """
    if not ALBUMENTATIONS_AVAILABLE:
        return BasicTransform(image_size=image_size)
    
    config = augmentation_config or {}
    
    transforms = [
        # Resize
        A.Resize(image_size, image_size),
        
        # Geometric transforms
        A.HorizontalFlip(p=0.5 if config.get("horizontal_flip", True) else 0),
        A.VerticalFlip(p=0.5 if config.get("vertical_flip", False) else 0),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=config.get("rotation", 15),
            p=0.5
        ),
        
        # Color transforms
        A.ColorJitter(
            brightness=config.get("brightness", 0.2),
            contrast=config.get("contrast", 0.2),
            saturation=config.get("saturation", 0.2),
            hue=config.get("hue", 0.1),
            p=0.5
        ),
        
        # Blur
        A.GaussianBlur(
            blur_limit=(3, 7),
            p=0.3 if config.get("gaussian_blur", True) else 0
        ),
        
        # Random crop and resize
        A.RandomResizedCrop(
            size=(image_size, image_size),
            scale=(0.8, 1.0),
            ratio=(0.9, 1.1),
            p=0.3 if config.get("random_crop", True) else 0
        ),
        
        # Normalize and convert
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
        ToTensorV2()
    ]
    
    return A.Compose(transforms)


def get_val_transforms(
    image_size: int = 224,
    augmentation_config: Optional[Dict[str, Any]] = None
) -> Any:
    """
    Get validation/test transforms (no augmentation).
    
    Args:
        image_size: Target image size
        augmentation_config: Configuration (for center crop option)
        
    Returns:
        Transform pipeline
    """
    if not ALBUMENTATIONS_AVAILABLE:
        return BasicTransform(image_size=image_size)
    
    config = augmentation_config or {}
    
    transforms = [
        A.Resize(image_size, image_size),
    ]
    
    if config.get("center_crop", True):
        # Resize larger then center crop
        transforms = [
            A.Resize(int(image_size * 1.1), int(image_size * 1.1)),
            A.CenterCrop(image_size, image_size)
        ]
    
    transforms.extend([
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
        ToTensorV2()
    ])
    
    return A.Compose(transforms)


def get_tta_transforms(image_size: int = 224) -> List[Any]:
    """
    Get Test-Time Augmentation transforms.
    
    Returns multiple transforms for TTA ensemble.
    
    Args:
        image_size: Target image size
        
    Returns:
        List of transform pipelines
    """
    if not ALBUMENTATIONS_AVAILABLE:
        return [BasicTransform(image_size=image_size)]
    
    base_transform = A.Compose([
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
        ToTensorV2()
    ])
    
    tta_transforms = [
        # Original
        A.Compose([
            A.Resize(image_size, image_size),
            base_transform
        ]),
        # Horizontal flip
        A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=1.0),
            base_transform
        ]),
        # Slight rotation left
        A.Compose([
            A.Resize(image_size, image_size),
            A.Rotate(limit=(5, 5), p=1.0),
            base_transform
        ]),
        # Slight rotation right
        A.Compose([
            A.Resize(image_size, image_size),
            A.Rotate(limit=(-5, -5), p=1.0),
            base_transform
        ]),
        # Brighter
        A.Compose([
            A.Resize(image_size, image_size),
            A.ColorJitter(brightness=0.1, p=1.0),
            base_transform
        ]),
    ]
    
    return tta_transforms


class SkyImageTransform:
    """
    Wrapper class for sky image transforms with additional processing.
    """
    
    def __init__(
        self,
        transform: Any,
        extract_cloud_features: bool = False
    ):
        self.transform = transform
        self.extract_cloud_features = extract_cloud_features
        
        if extract_cloud_features:
            from ..utils.image_processor import ImageProcessor
            self.image_processor = ImageProcessor()
    
    def __call__(
        self,
        image: np.ndarray
    ) -> Dict[str, torch.Tensor]:
        """
        Apply transforms and optionally extract cloud features.
        
        Args:
            image: Input image (HWC, RGB)
            
        Returns:
            Dictionary with 'image' tensor and optional 'cloud_features'
        """
        result = {}
        
        # Extract cloud features before transforms (on original image)
        if self.extract_cloud_features:
            cloud_feats = self.image_processor.get_feature_array(image)
            result["cloud_features"] = torch.from_numpy(cloud_feats)
        
        # Apply transforms
        if ALBUMENTATIONS_AVAILABLE:
            transformed = self.transform(image=image)
            result["image"] = transformed["image"]
        else:
            result["image"] = self.transform(image)
        
        return result


if __name__ == "__main__":
    # Test transforms
    print("Testing Transforms...")
    
    # Create test image
    np.random.seed(42)
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Test train transforms
    train_transform = get_train_transforms(image_size=224)
    
    if ALBUMENTATIONS_AVAILABLE:
        result = train_transform(image=test_image)
        print(f"Train transform output shape: {result['image'].shape}")
    else:
        result = train_transform(test_image)
        print(f"Train transform output shape: {result.shape}")
    
    # Test val transforms
    val_transform = get_val_transforms(image_size=224)
    
    if ALBUMENTATIONS_AVAILABLE:
        result = val_transform(image=test_image)
        print(f"Val transform output shape: {result['image'].shape}")
    else:
        result = val_transform(test_image)
        print(f"Val transform output shape: {result.shape}")
    
    # Test TTA
    tta_transforms = get_tta_transforms(image_size=224)
    print(f"Number of TTA transforms: {len(tta_transforms)}")
