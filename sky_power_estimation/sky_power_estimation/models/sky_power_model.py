"""
Sky Power Model - Main multi-modal model for DC power estimation.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List

from .image_encoder import ImageEncoder
from .temporal_encoder import TemporalEncoder
from .fusion import FusionLayer


class WeatherEncoder(nn.Module):
    def __init__(self, weather_dim: int = 9, sun_position_dim: int = 4,
                 output_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        
        total_dim = weather_dim + sun_position_dim
        self.encoder = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU()
        )
        self.output_dim = output_dim
    
    def forward(self, weather: torch.Tensor, sun_position: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([weather, sun_position], dim=-1)
        return self.encoder(combined)


class OutputHead(nn.Module):
    def __init__(self, input_dim: int = 512, hidden_dims: List[int] = [256, 128],
                 dropout: float = 0.3):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.head = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x).squeeze(-1)


class SkyPowerModel(nn.Module):
    """Multi-modal model for DC power estimation."""
    
    def __init__(
        self,
        image_backbone: str = "resnet50",
        image_pretrained: bool = True,
        image_feature_dim: int = 512,
        freeze_backbone: bool = False,
        weather_dim: int = 9,
        sun_position_dim: int = 4,
        weather_feature_dim: int = 128,
        temporal_type: str = "transformer",
        temporal_hidden_dim: int = 256,
        temporal_num_layers: int = 3,
        temporal_num_heads: int = 8,
        temporal_dropout: float = 0.1,
        temporal_bidirectional: bool = True,
        fusion_method: str = "attention",
        fusion_dim: int = 512,
        fusion_dropout: float = 0.2,
        output_hidden_dims: List[int] = [256, 128],
        output_dropout: float = 0.3
    ):
        super().__init__()
        
        self.image_encoder = ImageEncoder(
            backbone=image_backbone, pretrained=image_pretrained,
            freeze_backbone=freeze_backbone, output_dim=image_feature_dim
        )
        
        self.weather_encoder = WeatherEncoder(
            weather_dim=weather_dim, sun_position_dim=sun_position_dim,
            output_dim=weather_feature_dim
        )
        
        temporal_input_dim = image_feature_dim + weather_feature_dim
        self.temporal_encoder = TemporalEncoder(
            encoder_type=temporal_type, input_dim=temporal_input_dim,
            hidden_dim=temporal_hidden_dim, num_layers=temporal_num_layers,
            num_heads=temporal_num_heads, dropout=temporal_dropout,
            bidirectional=temporal_bidirectional
        )
        
        fusion_input_dims = [image_feature_dim, temporal_hidden_dim, weather_feature_dim]
        self.fusion = FusionLayer(
            method=fusion_method, input_dims=fusion_input_dims,
            output_dim=fusion_dim, dropout=fusion_dropout
        )
        
        self.output_head = OutputHead(
            input_dim=fusion_dim, hidden_dims=output_hidden_dims, dropout=output_dropout
        )
        
        self.config = {
            "image_backbone": image_backbone,
            "temporal_type": temporal_type,
            "fusion_method": fusion_method
        }
    
    def encode_image_sequence(self, images: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = images.shape[:2]
        images_flat = images.view(batch_size * seq_len, *images.shape[2:])
        features = self.image_encoder(images_flat)
        return features.view(batch_size, seq_len, -1)
    
    def forward(
        self,
        current_image: torch.Tensor,
        image_sequence: Optional[torch.Tensor] = None,
        current_weather: torch.Tensor = None,
        weather_sequence: Optional[torch.Tensor] = None,
        current_sun_position: torch.Tensor = None,
        sun_position_sequence: Optional[torch.Tensor] = None,
        return_features: bool = False
    ) -> torch.Tensor:
        
        current_image_features = self.image_encoder(current_image)
        current_weather_features = self.weather_encoder(current_weather, current_sun_position)
        
        if image_sequence is not None:
            image_seq_features = self.encode_image_sequence(image_sequence)
            weather_seq_features = self.weather_encoder(weather_sequence, sun_position_sequence)
            temporal_input = torch.cat([image_seq_features, weather_seq_features], dim=-1)
            temporal_features = self.temporal_encoder(temporal_input)
        else:
            temporal_features = torch.cat(
                [current_image_features, current_weather_features], dim=-1
            )
            temporal_features = nn.functional.adaptive_avg_pool1d(
                temporal_features.unsqueeze(1), self.temporal_encoder.output_dim
            ).squeeze(1)
        
        fused_features = self.fusion([
            current_image_features, temporal_features, current_weather_features
        ])
        
        power = self.output_head(fused_features)
        
        if return_features:
            return power, {
                "image_features": current_image_features,
                "weather_features": current_weather_features,
                "temporal_features": temporal_features,
                "fused_features": fused_features
            }
        return power
    
    @classmethod
    def from_config(cls, config: Dict) -> "SkyPowerModel":
        model_cfg = config.get("model", {})
        
        return cls(
            image_backbone=model_cfg.get("image_encoder", {}).get("backbone", "resnet50"),
            image_pretrained=model_cfg.get("image_encoder", {}).get("pretrained", True),
            image_feature_dim=model_cfg.get("image_encoder", {}).get("image_feature_dim", 512),
            temporal_type=model_cfg.get("temporal_encoder", {}).get("type", "transformer"),
            temporal_hidden_dim=model_cfg.get("temporal_encoder", {}).get("hidden_dim", 256),
            temporal_num_layers=model_cfg.get("temporal_encoder", {}).get("num_layers", 3),
            temporal_num_heads=model_cfg.get("temporal_encoder", {}).get("num_heads", 8),
            fusion_method=model_cfg.get("fusion", {}).get("method", "attention"),
            fusion_dim=model_cfg.get("fusion", {}).get("hidden_dim", 512),
            output_hidden_dims=model_cfg.get("output", {}).get("hidden_dims", [256, 128]),
        )
