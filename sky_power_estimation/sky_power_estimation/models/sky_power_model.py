"""
Sky Power Model

Main model for DC power estimation combining sky images,
weather data, and temporal features.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List

from .image_encoder import ImageEncoder, MultiScaleImageEncoder
from .temporal_encoder import TemporalEncoder
from .fusion import FusionLayer


class WeatherEncoder(nn.Module):
    """
    Encoder for weather sensor data and sun position features.
    """
    
    def __init__(
        self,
        weather_dim: int = 9,
        sun_position_dim: int = 4,
        output_dim: int = 128,
        dropout: float = 0.1
    ):
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
    
    def forward(
        self,
        weather: torch.Tensor,
        sun_position: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            weather: Weather features [batch_size, weather_dim] or 
                    [batch_size, seq_len, weather_dim]
            sun_position: Sun position features [batch_size, sun_dim] or
                         [batch_size, seq_len, sun_dim]
                         
        Returns:
            Encoded features
        """
        combined = torch.cat([weather, sun_position], dim=-1)
        return self.encoder(combined)


class OutputHead(nn.Module):
    """
    Output regression head for DC power prediction.
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dims: List[int] = [256, 128],
        dropout: float = 0.3,
        activation: str = "gelu"
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        activation_fn = nn.GELU() if activation == "gelu" else nn.ReLU()
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                activation_fn,
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Final output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.head = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x).squeeze(-1)


class SkyPowerModel(nn.Module):
    """
    Multi-modal model for DC power estimation.
    
    Combines:
    - Sky images (CNN encoder)
    - Weather sensor data
    - Sun position features
    - Temporal sequence modeling
    
    Architecture:
    1. Image Encoder (ResNet/MobileNet/EfficientNet)
    2. Weather + Sun Position Encoder
    3. Temporal Encoder (LSTM/Transformer)
    4. Multi-modal Fusion
    5. Output Regression Head
    
    Args:
        config: Model configuration dictionary
    """
    
    def __init__(
        self,
        # Image encoder config
        image_backbone: str = "resnet50",
        image_pretrained: bool = True,
        image_feature_dim: int = 512,
        freeze_backbone: bool = False,
        use_multiscale: bool = False,
        
        # Weather encoder config
        weather_dim: int = 9,
        sun_position_dim: int = 4,
        weather_feature_dim: int = 128,
        
        # Temporal encoder config
        temporal_type: str = "transformer",
        temporal_hidden_dim: int = 256,
        temporal_num_layers: int = 3,
        temporal_num_heads: int = 8,
        temporal_dropout: float = 0.1,
        temporal_bidirectional: bool = True,
        
        # Fusion config
        fusion_method: str = "attention",
        fusion_dim: int = 512,
        fusion_dropout: float = 0.2,
        
        # Output config
        output_hidden_dims: List[int] = [256, 128],
        output_dropout: float = 0.3,
        output_activation: str = "gelu"
    ):
        super().__init__()
        
        # Image encoder
        if use_multiscale:
            self.image_encoder = MultiScaleImageEncoder(
                backbone=image_backbone,
                pretrained=image_pretrained,
                output_dim=image_feature_dim
            )
        else:
            self.image_encoder = ImageEncoder(
                backbone=image_backbone,
                pretrained=image_pretrained,
                freeze_backbone=freeze_backbone,
                output_dim=image_feature_dim
            )
        
        # Weather encoder
        self.weather_encoder = WeatherEncoder(
            weather_dim=weather_dim,
            sun_position_dim=sun_position_dim,
            output_dim=weather_feature_dim
        )
        
        # Combined feature dim for temporal processing
        temporal_input_dim = image_feature_dim + weather_feature_dim
        
        # Temporal encoder
        self.temporal_encoder = TemporalEncoder(
            encoder_type=temporal_type,
            input_dim=temporal_input_dim,
            hidden_dim=temporal_hidden_dim,
            num_layers=temporal_num_layers,
            num_heads=temporal_num_heads,
            dropout=temporal_dropout,
            bidirectional=temporal_bidirectional
        )
        
        # Fusion layer
        fusion_input_dims = [
            image_feature_dim,      # Current image features
            temporal_hidden_dim,     # Temporal features
            weather_feature_dim      # Current weather features
        ]
        
        self.fusion = FusionLayer(
            method=fusion_method,
            input_dims=fusion_input_dims,
            output_dim=fusion_dim,
            dropout=fusion_dropout
        )
        
        # Output head
        self.output_head = OutputHead(
            input_dim=fusion_dim,
            hidden_dims=output_hidden_dims,
            dropout=output_dropout,
            activation=output_activation
        )
        
        # Store config
        self.config = {
            "image_backbone": image_backbone,
            "temporal_type": temporal_type,
            "fusion_method": fusion_method
        }
    
    def encode_image_sequence(
        self,
        images: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode a sequence of images.
        
        Args:
            images: [batch_size, seq_len, 3, H, W]
            
        Returns:
            Image features [batch_size, seq_len, image_feature_dim]
        """
        batch_size, seq_len = images.shape[:2]
        
        # Flatten batch and sequence
        images_flat = images.view(batch_size * seq_len, *images.shape[2:])
        
        # Encode
        features = self.image_encoder(images_flat)
        
        # Reshape back
        features = features.view(batch_size, seq_len, -1)
        
        return features
    
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
        """
        Forward pass.
        
        Args:
            current_image: Current sky image [batch_size, 3, H, W]
            image_sequence: Historical images [batch_size, seq_len, 3, H, W]
            current_weather: Current weather [batch_size, weather_dim]
            weather_sequence: Historical weather [batch_size, seq_len, weather_dim]
            current_sun_position: Current sun position [batch_size, sun_dim]
            sun_position_sequence: Historical sun positions [batch_size, seq_len, sun_dim]
            return_features: Whether to return intermediate features
            
        Returns:
            power: Predicted DC power [batch_size]
            features: (optional) Dictionary of intermediate features
        """
        batch_size = current_image.size(0)
        
        # Encode current image
        current_image_features = self.image_encoder(current_image)
        
        # Encode current weather + sun position
        current_weather_features = self.weather_encoder(
            current_weather,
            current_sun_position
        )
        
        # Temporal processing (if sequence provided)
        if image_sequence is not None:
            # Encode image sequence
            image_seq_features = self.encode_image_sequence(image_sequence)
            
            # Encode weather sequence
            weather_seq_features = self.weather_encoder(
                weather_sequence,
                sun_position_sequence
            )
            
            # Combine image and weather for temporal encoding
            temporal_input = torch.cat(
                [image_seq_features, weather_seq_features],
                dim=-1
            )
            
            # Temporal encoding
            temporal_features = self.temporal_encoder(temporal_input)
        else:
            # No sequence - use current features as temporal
            temporal_features = torch.cat(
                [current_image_features, current_weather_features],
                dim=-1
            )
            # Project to match temporal encoder output dim
            temporal_features = nn.functional.adaptive_avg_pool1d(
                temporal_features.unsqueeze(1),
                self.temporal_encoder.output_dim
            ).squeeze(1)
        
        # Fusion
        fused_features = self.fusion([
            current_image_features,
            temporal_features,
            current_weather_features
        ])
        
        # Output
        power = self.output_head(fused_features)
        
        if return_features:
            features = {
                "image_features": current_image_features,
                "weather_features": current_weather_features,
                "temporal_features": temporal_features,
                "fused_features": fused_features
            }
            return power, features
        
        return power
    
    def predict_with_uncertainty(
        self,
        current_image: torch.Tensor,
        image_sequence: Optional[torch.Tensor] = None,
        current_weather: torch.Tensor = None,
        weather_sequence: Optional[torch.Tensor] = None,
        current_sun_position: torch.Tensor = None,
        sun_position_sequence: Optional[torch.Tensor] = None,
        num_samples: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict with Monte Carlo dropout uncertainty estimation.
        
        Returns mean prediction and uncertainty (std).
        """
        self.train()  # Enable dropout
        
        predictions = []
        for _ in range(num_samples):
            pred = self.forward(
                current_image=current_image,
                image_sequence=image_sequence,
                current_weather=current_weather,
                weather_sequence=weather_sequence,
                current_sun_position=current_sun_position,
                sun_position_sequence=sun_position_sequence
            )
            predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)
        mean = predictions.mean(dim=0)
        std = predictions.std(dim=0)
        
        self.eval()
        
        return mean, std
    
    @classmethod
    def from_config(cls, config: Dict) -> "SkyPowerModel":
        """Create model from configuration dictionary."""
        model_cfg = config.get("model", {})
        
        return cls(
            # Image encoder
            image_backbone=model_cfg.get("image_encoder", {}).get("backbone", "resnet50"),
            image_pretrained=model_cfg.get("image_encoder", {}).get("pretrained", True),
            image_feature_dim=model_cfg.get("image_encoder", {}).get("image_feature_dim", 512),
            freeze_backbone=model_cfg.get("image_encoder", {}).get("freeze_backbone", False),
            
            # Temporal encoder
            temporal_type=model_cfg.get("temporal_encoder", {}).get("type", "transformer"),
            temporal_hidden_dim=model_cfg.get("temporal_encoder", {}).get("hidden_dim", 256),
            temporal_num_layers=model_cfg.get("temporal_encoder", {}).get("num_layers", 3),
            temporal_num_heads=model_cfg.get("temporal_encoder", {}).get("num_heads", 8),
            temporal_dropout=model_cfg.get("temporal_encoder", {}).get("dropout", 0.1),
            temporal_bidirectional=model_cfg.get("temporal_encoder", {}).get("bidirectional", True),
            
            # Fusion
            fusion_method=model_cfg.get("fusion", {}).get("method", "attention"),
            fusion_dim=model_cfg.get("fusion", {}).get("hidden_dim", 512),
            fusion_dropout=model_cfg.get("fusion", {}).get("dropout", 0.2),
            
            # Output
            output_hidden_dims=model_cfg.get("output", {}).get("hidden_dims", [256, 128]),
            output_dropout=model_cfg.get("output", {}).get("dropout", 0.3),
            output_activation=model_cfg.get("output", {}).get("activation", "gelu")
        )


if __name__ == "__main__":
    # Test the full model
    print("Testing SkyPowerModel...")
    
    batch_size = 4
    seq_len = 12
    img_size = 224
    weather_dim = 9
    sun_dim = 4
    
    # Create model
    model = SkyPowerModel(
        image_backbone="resnet50",
        image_pretrained=False,  # For testing
        temporal_type="transformer"
    )
    
    # Create inputs
    current_image = torch.randn(batch_size, 3, img_size, img_size)
    image_sequence = torch.randn(batch_size, seq_len, 3, img_size, img_size)
    current_weather = torch.randn(batch_size, weather_dim)
    weather_sequence = torch.randn(batch_size, seq_len, weather_dim)
    current_sun_position = torch.randn(batch_size, sun_dim)
    sun_position_sequence = torch.randn(batch_size, seq_len, sun_dim)
    
    # Forward pass
    output = model(
        current_image=current_image,
        image_sequence=image_sequence,
        current_weather=current_weather,
        weather_sequence=weather_sequence,
        current_sun_position=current_sun_position,
        sun_position_sequence=sun_position_sequence
    )
    
    print(f"Input shapes:")
    print(f"  Current image: {current_image.shape}")
    print(f"  Image sequence: {image_sequence.shape}")
    print(f"  Current weather: {current_weather.shape}")
    print(f"  Weather sequence: {weather_sequence.shape}")
    print(f"  Current sun position: {current_sun_position.shape}")
    print(f"  Sun position sequence: {sun_position_sequence.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test without sequence
    print("\nTesting without sequence...")
    output_no_seq = model(
        current_image=current_image,
        current_weather=current_weather,
        current_sun_position=current_sun_position
    )
    print(f"Output shape (no sequence): {output_no_seq.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
