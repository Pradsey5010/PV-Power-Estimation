"""
Tests for data loading and processing.
"""

import pytest
import torch
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sky_power_estimation.data import SkyPowerDataset, create_dataloaders
from sky_power_estimation.data.dataset import SyntheticSkyDataset
from sky_power_estimation.utils import WeatherProcessor, SunPositionCalculator, ImageProcessor


class TestSyntheticDataset:
    """Tests for SyntheticSkyDataset."""
    
    def test_dataset_creation(self):
        """Test dataset creation."""
        dataset = SyntheticSkyDataset(
            num_samples=50,
            sequence_length=6,
            image_size=224
        )
        
        assert len(dataset) == 50 - 6  # Accounting for sequence length
    
    def test_sample_format(self):
        """Test sample format."""
        dataset = SyntheticSkyDataset(
            num_samples=50,
            sequence_length=6,
            image_size=224
        )
        
        sample = dataset[0]
        
        assert "current_image" in sample
        assert "current_weather" in sample
        assert "current_sun_position" in sample
        assert "target" in sample
        assert "image_sequence" in sample
        assert "weather_sequence" in sample
        assert "sun_position_sequence" in sample
    
    def test_sample_shapes(self):
        """Test sample shapes."""
        dataset = SyntheticSkyDataset(
            num_samples=50,
            sequence_length=6,
            image_size=224
        )
        
        sample = dataset[0]
        
        assert sample["current_image"].shape == (3, 224, 224)
        assert sample["current_weather"].shape == (9,)
        assert sample["current_sun_position"].shape == (4,)
        assert sample["target"].shape == ()
        assert sample["image_sequence"].shape == (6, 3, 224, 224)
        assert sample["weather_sequence"].shape == (6, 9)
        assert sample["sun_position_sequence"].shape == (6, 4)


class TestDataLoader:
    """Tests for dataloader creation."""
    
    def test_create_dataloaders(self):
        """Test dataloader creation."""
        dataloaders = create_dataloaders(
            use_synthetic=True,
            synthetic_samples=100,
            batch_size=8,
            sequence_length=6,
            num_workers=0
        )
        
        assert "train" in dataloaders
        assert "val" in dataloaders
        assert "test" in dataloaders
    
    def test_batch_format(self):
        """Test batch format."""
        dataloaders = create_dataloaders(
            use_synthetic=True,
            synthetic_samples=100,
            batch_size=8,
            sequence_length=6,
            num_workers=0
        )
        
        batch = next(iter(dataloaders["train"]))
        
        assert batch["current_image"].shape[0] == 8
        assert batch["current_image"].shape[1:] == (3, 224, 224)


class TestWeatherProcessor:
    """Tests for WeatherProcessor."""
    
    def test_processor_creation(self):
        """Test processor creation."""
        processor = WeatherProcessor()
        
        assert processor.features is not None
        assert len(processor.features) > 0
    
    def test_transform(self):
        """Test data transformation."""
        import pandas as pd
        
        processor = WeatherProcessor(scaler_type="standard")
        
        # Create dummy data
        data = pd.DataFrame({
            "temperature": [20, 25, 30],
            "humidity": [50, 60, 70],
            "pressure": [1010, 1015, 1020],
            "wind_speed": [5, 10, 15],
            "ghi": [500, 600, 700],
            "dni": [400, 500, 600],
            "dhi": [100, 100, 100]
        })
        
        transformed = processor.fit_transform(data)
        
        assert transformed.shape == (3, 7)
    
    def test_create_sequences(self):
        """Test sequence creation."""
        processor = WeatherProcessor()
        
        data = np.random.randn(100, 9).astype(np.float32)
        sequences = processor.create_sequences(data, sequence_length=12, stride=1)
        
        assert sequences.shape == (89, 12, 9)


class TestSunPositionCalculator:
    """Tests for SunPositionCalculator."""
    
    def test_calculator_creation(self):
        """Test calculator creation."""
        calc = SunPositionCalculator(
            latitude=37.7749,
            longitude=-122.4194
        )
        
        assert calc.latitude == 37.7749
        assert calc.longitude == -122.4194
    
    def test_get_position(self):
        """Test position calculation."""
        import pandas as pd
        
        calc = SunPositionCalculator()
        pos = calc.get_sun_position(pd.Timestamp("2024-06-21 12:00:00"))
        
        assert "zenith" in pos
        assert "azimuth" in pos
        assert "apparent_elevation" in pos
    
    def test_get_features_array(self):
        """Test feature array."""
        import pandas as pd
        
        calc = SunPositionCalculator()
        features = calc.get_features_array(pd.Timestamp("2024-06-21 12:00:00"))
        
        assert features.shape == (4,)


class TestImageProcessor:
    """Tests for ImageProcessor."""
    
    def test_processor_creation(self):
        """Test processor creation."""
        processor = ImageProcessor(image_size=(224, 224))
        
        assert processor.image_size == (224, 224)
    
    def test_preprocess(self):
        """Test image preprocessing."""
        processor = ImageProcessor(image_size=(224, 224))
        
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        processed = processor.preprocess(image)
        
        assert processed.shape == (3, 224, 224)
    
    def test_cloud_features(self):
        """Test cloud feature extraction."""
        processor = ImageProcessor()
        
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        features = processor.extract_cloud_features(image)
        
        assert "cloud_cover" in features
        assert "brightness" in features
        assert "blue_ratio" in features


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
