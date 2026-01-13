"""
Image Encoder Module - CNN-based encoder for sky images.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional, List
import timm


class ImageEncoder(nn.Module):
    """CNN-based image encoder for sky images."""
    
    SUPPORTED_BACKBONES = [
        "resnet18", "resnet34", "resnet50", "resnet101",
        "mobilenet_v3_small", "mobilenet_v3_large",
        "efficientnet_b0", "efficientnet_b1", "efficientnet_b2",
        "convnext_tiny", "convnext_small", "vit_tiny_patch16_224"
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
        
        self.backbone_name = backbone
        self.output_dim = output_dim
        
        self.backbone, backbone_dim = self._create_backbone(backbone, pretrained)
        
        if freeze_backbone:
            self._freeze_backbone()
        
        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        self.cloud_head = nn.Sequential(
            nn.Linear(backbone_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 3)
        )
    
    def _create_backbone(self, backbone: str, pretrained: bool):
        if backbone.startswith("resnet"):
            return self._create_resnet(backbone, pretrained)
        elif backbone.startswith("mobilenet"):
            return self._create_mobilenet(backbone, pretrained)
        else:
            return self._create_timm(backbone, pretrained)
    
    def _create_resnet(self, backbone: str, pretrained: bool):
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
        else:
            model = models.resnet101(weights=weights)
            feature_dim = 2048
        
        model = nn.Sequential(*list(model.children())[:-1])
        return model, feature_dim
    
    def _create_mobilenet(self, backbone: str, pretrained: bool):
        weights = "IMAGENET1K_V1" if pretrained else None
        
        if backbone == "mobilenet_v3_small":
            model = models.mobilenet_v3_small(weights=weights)
            feature_dim = 576
        else:
            model = models.mobilenet_v3_large(weights=weights)
            feature_dim = 960
        
        model = nn.Sequential(model.features, model.avgpool)
        return model, feature_dim
    
    def _create_timm(self, backbone: str, pretrained: bool):
        model = timm.create_model(backbone, pretrained=pretrained, num_classes=0, global_pool='avg')
        feature_dim = model.num_features
        return model, feature_dim
    
    def _freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        if features.dim() > 2:
            features = features.flatten(1)
        return features
    
    def forward(self, x: torch.Tensor, return_cloud_features: bool = False):
        backbone_features = self.extract_features(x)
        features = self.projection(backbone_features)
        
        if return_cloud_features:
            cloud_features = self.cloud_head(backbone_features)
            return features, cloud_features
        return features
