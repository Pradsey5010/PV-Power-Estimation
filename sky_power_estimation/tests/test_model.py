"""
Tests for the Sky Power Estimation model.
"""

import pytest
import torch
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sky_power_estimation.models import (
    SkyPowerModel,
    ImageEncoder,
    TemporalEncoder,
    FusionLayer
)


class TestImageEncoder:
    """Tests for ImageEncoder."""
    
    def test_resnet_encoder(self):
        """Test ResNet backbone."""
        encoder = ImageEncoder(backbone="resnet18", pretrained=False)
        x = torch.randn(2, 3, 224, 224)
        output = encoder(x)
        
        assert output.shape == (2, 512)
    
    def test_mobilenet_encoder(self):
        """Test MobileNet backbone."""
        encoder = ImageEncoder(backbone="mobilenet_v3_small", pretrained=False)
        x = torch.randn(2, 3, 224, 224)
        output = encoder(x)
        
        assert output.shape == (2, 512)
    
    def test_cloud_features(self):
        """Test cloud feature extraction."""
        encoder = ImageEncoder(backbone="resnet18", pretrained=False)
        x = torch.randn(2, 3, 224, 224)
        output, cloud = encoder(x, return_cloud_features=True)
        
        assert output.shape == (2, 512)
        assert cloud.shape == (2, 3)


class TestTemporalEncoder:
    """Tests for TemporalEncoder."""
    
    def test_lstm_encoder(self):
        """Test LSTM encoder."""
        encoder = TemporalEncoder(
            encoder_type="lstm",
            input_dim=128,
            hidden_dim=256
        )
        x = torch.randn(2, 12, 128)
        output = encoder(x)
        
        assert output.shape == (2, 256)
    
    def test_transformer_encoder(self):
        """Test Transformer encoder."""
        encoder = TemporalEncoder(
            encoder_type="transformer",
            input_dim=128,
            hidden_dim=256
        )
        x = torch.randn(2, 12, 128)
        output = encoder(x)
        
        assert output.shape == (2, 256)
    
    def test_return_sequence(self):
        """Test returning full sequence."""
        encoder = TemporalEncoder(
            encoder_type="transformer",
            input_dim=128,
            hidden_dim=256
        )
        x = torch.randn(2, 12, 128)
        output = encoder(x, return_sequence=True)
        
        assert output.shape == (2, 12, 256)


class TestFusionLayer:
    """Tests for FusionLayer."""
    
    def test_concat_fusion(self):
        """Test concatenation fusion."""
        fusion = FusionLayer(
            method="concat",
            input_dims=[256, 128, 64],
            output_dim=512
        )
        features = [
            torch.randn(2, 256),
            torch.randn(2, 128),
            torch.randn(2, 64)
        ]
        output = fusion(features)
        
        assert output.shape == (2, 512)
    
    def test_attention_fusion(self):
        """Test attention fusion."""
        fusion = FusionLayer(
            method="attention",
            input_dims=[256, 128, 64],
            output_dim=512
        )
        features = [
            torch.randn(2, 256),
            torch.randn(2, 128),
            torch.randn(2, 64)
        ]
        output = fusion(features)
        
        assert output.shape == (2, 512)
    
    def test_gated_fusion(self):
        """Test gated fusion."""
        fusion = FusionLayer(
            method="gated",
            input_dims=[256, 128, 64],
            output_dim=512
        )
        features = [
            torch.randn(2, 256),
            torch.randn(2, 128),
            torch.randn(2, 64)
        ]
        output = fusion(features)
        
        assert output.shape == (2, 512)


class TestSkyPowerModel:
    """Tests for the full SkyPowerModel."""
    
    @pytest.fixture
    def model(self):
        """Create test model."""
        return SkyPowerModel(
            image_backbone="resnet18",
            image_pretrained=False,
            temporal_type="lstm"
        )
    
    def test_forward_no_sequence(self, model):
        """Test forward pass without sequences."""
        batch_size = 2
        
        current_image = torch.randn(batch_size, 3, 224, 224)
        current_weather = torch.randn(batch_size, 9)
        current_sun_position = torch.randn(batch_size, 4)
        
        output = model(
            current_image=current_image,
            current_weather=current_weather,
            current_sun_position=current_sun_position
        )
        
        assert output.shape == (batch_size,)
    
    def test_forward_with_sequence(self, model):
        """Test forward pass with sequences."""
        batch_size = 2
        seq_len = 12
        
        current_image = torch.randn(batch_size, 3, 224, 224)
        image_sequence = torch.randn(batch_size, seq_len, 3, 224, 224)
        current_weather = torch.randn(batch_size, 9)
        weather_sequence = torch.randn(batch_size, seq_len, 9)
        current_sun_position = torch.randn(batch_size, 4)
        sun_position_sequence = torch.randn(batch_size, seq_len, 4)
        
        output = model(
            current_image=current_image,
            image_sequence=image_sequence,
            current_weather=current_weather,
            weather_sequence=weather_sequence,
            current_sun_position=current_sun_position,
            sun_position_sequence=sun_position_sequence
        )
        
        assert output.shape == (batch_size,)
    
    def test_return_features(self, model):
        """Test returning intermediate features."""
        batch_size = 2
        
        current_image = torch.randn(batch_size, 3, 224, 224)
        current_weather = torch.randn(batch_size, 9)
        current_sun_position = torch.randn(batch_size, 4)
        
        output, features = model(
            current_image=current_image,
            current_weather=current_weather,
            current_sun_position=current_sun_position,
            return_features=True
        )
        
        assert output.shape == (batch_size,)
        assert "image_features" in features
        assert "weather_features" in features
        assert "fused_features" in features
    
    def test_from_config(self):
        """Test model creation from config."""
        config = {
            "model": {
                "image_encoder": {
                    "backbone": "resnet18",
                    "pretrained": False
                },
                "temporal_encoder": {
                    "type": "transformer",
                    "hidden_dim": 128,
                    "num_layers": 2
                },
                "fusion": {
                    "method": "attention"
                }
            }
        }
        
        model = SkyPowerModel.from_config(config)
        
        # Test forward pass
        x = torch.randn(2, 3, 224, 224)
        w = torch.randn(2, 9)
        s = torch.randn(2, 4)
        
        output = model(
            current_image=x,
            current_weather=w,
            current_sun_position=s
        )
        
        assert output.shape == (2,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
