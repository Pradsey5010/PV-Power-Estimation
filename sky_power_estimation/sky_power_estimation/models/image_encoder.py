"""
Image Encoder Module

CNN-based encoder for extracting features from sky images.
Supports ResNet, MobileNet, and EfficientNet backbones.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional, Literal
import timm


class ImageEncoder(nn.Module):
    """
    CNN-based image encoder for sky images.
    
    Extracts visual features from sky images to capture cloud cover,
    opacity, and other atmospheric conditions.
    
    Args:
        backbone: CNN backbone architecture
        pretrained: Whether to use pretrained weights
        freeze_backbone: Whether to freeze backbone weights
        output_dim: Output feature dimension
        dropout: Dropout rate for the projection layer
    """
    
    SUPPORTED_BACKBONES = [
        "resnet18", "resnet34", "resnet50", "resnet101",
        "mobilenet_v3_small", "mobilenet_v3_large",
        "efficientnet_b0", "efficientnet_b1", "efficientnet_b2",
        "convnext_tiny", "convnext_small"
    ]
    
    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        output_dim: int = 512,
        dropout: float = 0.2
    ):
        super().__init__()
        
        if backbone not in self.SUPPORTED_BACKBONES:
            raise ValueError(
                f"Backbone '{backbone}' not supported. "
                f"Choose from: {self.SUPPORTED_BACKBONES}"
            )
        
        self.backbone_name = backbone
        self.output_dim = output_dim
        
        # Initialize backbone and get feature dimension
        self.backbone, backbone_dim = self._create_backbone(backbone, pretrained)
        
        # Freeze backbone if specified
        if freeze_backbone:
            self._freeze_backbone()
        
        # Projection layer to desired output dimension
        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        # Cloud feature extraction head (auxiliary task)
        self.cloud_head = nn.Sequential(
            nn.Linear(backbone_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 3)  # Cloud cover percentage, opacity, type
        )
    
    def _create_backbone(self, backbone: str, pretrained: bool):
        """Create the CNN backbone and return it with its feature dimension."""
        
        if backbone.startswith("resnet"):
            return self._create_resnet(backbone, pretrained)
        elif backbone.startswith("mobilenet"):
            return self._create_mobilenet(backbone, pretrained)
        elif backbone.startswith("efficientnet"):
            return self._create_efficientnet(backbone, pretrained)
        elif backbone.startswith("convnext"):
            return self._create_convnext(backbone, pretrained)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
    
    def _create_resnet(self, backbone: str, pretrained: bool):
        """Create ResNet backbone."""
        weights = "IMAGENET1K_V1" if pretrained else None
        
        if backbone == "resnet18":
            model = models.resnet18(weights=weights)
            feature_dim = 512
        elif backbone == "resnet34":
            model = models.resnet34(weights=weights)
            feature_dim = 512
        elif backbone == "resnet50":
            model = models.resnet50(weights=weights)
            feature_dim = 2048
        elif backbone == "resnet101":
            model = models.resnet101(weights=weights)
            feature_dim = 2048
        else:
            raise ValueError(f"Unknown ResNet variant: {backbone}")
        
        # Remove the final fully connected layer
        model = nn.Sequential(*list(model.children())[:-1])
        
        return model, feature_dim
    
    def _create_mobilenet(self, backbone: str, pretrained: bool):
        """Create MobileNetV3 backbone."""
        weights = "IMAGENET1K_V1" if pretrained else None
        
        if backbone == "mobilenet_v3_small":
            model = models.mobilenet_v3_small(weights=weights)
            feature_dim = 576
        elif backbone == "mobilenet_v3_large":
            model = models.mobilenet_v3_large(weights=weights)
            feature_dim = 960
        else:
            raise ValueError(f"Unknown MobileNet variant: {backbone}")
        
        # Remove classifier
        model = nn.Sequential(
            model.features,
            model.avgpool
        )
        
        return model, feature_dim
    
    def _create_efficientnet(self, backbone: str, pretrained: bool):
        """Create EfficientNet backbone using timm."""
        model = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,  # Remove classifier
            global_pool='avg'
        )
        feature_dim = model.num_features
        
        return model, feature_dim
    
    def _create_convnext(self, backbone: str, pretrained: bool):
        """Create ConvNeXt backbone using timm."""
        model = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
            global_pool='avg'
        )
        feature_dim = model.num_features
        
        return model, feature_dim
    
    def _freeze_backbone(self):
        """Freeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self, num_layers: Optional[int] = None):
        """
        Unfreeze backbone parameters.
        
        Args:
            num_layers: If specified, only unfreeze the last N layers.
                       If None, unfreeze all layers.
        """
        if num_layers is None:
            for param in self.backbone.parameters():
                param.requires_grad = True
        else:
            # Get all named parameters
            params = list(self.backbone.named_parameters())
            
            # Unfreeze last N layers
            for name, param in params[-num_layers:]:
                param.requires_grad = True
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract raw features from backbone.
        
        Args:
            x: Input images [batch_size, 3, H, W]
            
        Returns:
            features: Raw backbone features [batch_size, backbone_dim]
        """
        features = self.backbone(x)
        
        # Flatten if needed
        if features.dim() > 2:
            features = features.flatten(1)
        
        return features
    
    def forward(
        self,
        x: torch.Tensor,
        return_cloud_features: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through the image encoder.
        
        Args:
            x: Input sky images [batch_size, 3, H, W]
            return_cloud_features: Whether to return cloud prediction as auxiliary output
            
        Returns:
            features: Encoded image features [batch_size, output_dim]
            cloud_features: (optional) Cloud predictions [batch_size, 3]
        """
        # Extract backbone features
        backbone_features = self.extract_features(x)
        
        # Project to output dimension
        features = self.projection(backbone_features)
        
        if return_cloud_features:
            cloud_features = self.cloud_head(backbone_features)
            return features, cloud_features
        
        return features


class MultiScaleImageEncoder(nn.Module):
    """
    Multi-scale image encoder that extracts features at multiple resolutions.
    
    Useful for capturing both local cloud details and global sky patterns.
    """
    
    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained: bool = True,
        output_dim: int = 512,
        scales: list = [1.0, 0.75, 0.5]
    ):
        super().__init__()
        
        self.scales = scales
        self.output_dim = output_dim
        
        # Single encoder for all scales (shared weights)
        self.encoder = ImageEncoder(
            backbone=backbone,
            pretrained=pretrained,
            output_dim=output_dim
        )
        
        # Fusion layer for multi-scale features
        self.fusion = nn.Sequential(
            nn.Linear(output_dim * len(scales), output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.GELU(),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract multi-scale features.
        
        Args:
            x: Input images [batch_size, 3, H, W]
            
        Returns:
            features: Fused multi-scale features [batch_size, output_dim]
        """
        batch_size = x.size(0)
        scale_features = []
        
        for scale in self.scales:
            if scale != 1.0:
                # Resize image
                scaled_size = (int(x.size(2) * scale), int(x.size(3) * scale))
                scaled_x = nn.functional.interpolate(
                    x, size=scaled_size, mode='bilinear', align_corners=False
                )
            else:
                scaled_x = x
            
            # Extract features
            features = self.encoder(scaled_x)
            scale_features.append(features)
        
        # Concatenate and fuse
        concat_features = torch.cat(scale_features, dim=1)
        fused_features = self.fusion(concat_features)
        
        return fused_features


if __name__ == "__main__":
    # Test the image encoder
    batch_size = 4
    img_size = 224
    
    # Create random input
    x = torch.randn(batch_size, 3, img_size, img_size)
    
    # Test different backbones
    for backbone in ["resnet50", "mobilenet_v3_small", "efficientnet_b0"]:
        print(f"\nTesting {backbone}...")
        encoder = ImageEncoder(backbone=backbone, pretrained=False)
        
        output = encoder(x)
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
        
        output, cloud = encoder(x, return_cloud_features=True)
        print(f"  Cloud features shape: {cloud.shape}")
    
    # Test multi-scale encoder
    print("\nTesting MultiScaleImageEncoder...")
    ms_encoder = MultiScaleImageEncoder(backbone="resnet50", pretrained=False)
    output = ms_encoder(x)
    print(f"  Output shape: {output.shape}")
