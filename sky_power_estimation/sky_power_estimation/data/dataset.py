"""
Dataset Module

PyTorch Dataset for sky power estimation.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
import warnings

import torch
from torch.utils.data import Dataset
from PIL import Image

from .transforms import get_train_transforms, get_val_transforms
from ..utils.sun_position import SunPositionCalculator
from ..utils.weather_processor import WeatherProcessor


class SkyPowerDataset(Dataset):
    """
    Dataset for sky image-based DC power estimation.
    
    Supports:
    - Loading sky images and corresponding power measurements
    - Weather sensor data integration
    - Sun position calculation
    - Temporal sequence preparation
    
    Args:
        data_dir: Root directory containing data
        annotations_file: CSV file with annotations (timestamps, power, weather)
        image_dir: Directory containing sky images
        mode: Dataset mode ('train', 'val', 'test')
        sequence_length: Number of historical timesteps
        image_size: Target image size
        transform: Image transform (optional, auto-selected based on mode)
        weather_features: List of weather features to use
        location: Dictionary with latitude, longitude, timezone for sun position
    """
    
    def __init__(
        self,
        data_dir: str,
        annotations_file: str = "annotations.csv",
        image_dir: str = "images",
        mode: str = "train",
        sequence_length: int = 12,
        image_size: int = 224,
        transform: Optional[Any] = None,
        weather_features: Optional[List[str]] = None,
        location: Optional[Dict[str, float]] = None,
        return_sequences: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.image_dir = self.data_dir / image_dir
        self.mode = mode
        self.sequence_length = sequence_length
        self.image_size = image_size
        self.return_sequences = return_sequences
        
        # Load annotations
        annotations_path = self.data_dir / annotations_file
        if annotations_path.exists():
            self.annotations = pd.read_csv(annotations_path)
            self._prepare_annotations()
        else:
            warnings.warn(f"Annotations file not found: {annotations_path}")
            self.annotations = self._create_dummy_annotations()
        
        # Setup transforms
        if transform is None:
            if mode == "train":
                self.transform = get_train_transforms(image_size)
            else:
                self.transform = get_val_transforms(image_size)
        else:
            self.transform = transform
        
        # Weather processor
        self.weather_features = weather_features or [
            "temperature", "humidity", "pressure",
            "wind_speed", "ghi", "dni", "dhi"
        ]
        self.weather_processor = WeatherProcessor(
            features=self.weather_features,
            scaler_type="standard"
        )
        
        # Fit weather processor on available data
        available_weather = [f for f in self.weather_features if f in self.annotations.columns]
        if available_weather:
            self.weather_processor.fit(self.annotations[available_weather])
        
        # Sun position calculator
        location = location or {
            "latitude": 37.7749,
            "longitude": -122.4194,
            "altitude": 10,
            "timezone": "UTC"
        }
        self.sun_calculator = SunPositionCalculator(**location)
        
        # Prepare valid indices (accounting for sequence length)
        self._prepare_indices()
    
    def _prepare_annotations(self):
        """Prepare and validate annotations."""
        # Parse timestamps
        if "timestamp" in self.annotations.columns:
            self.annotations["timestamp"] = pd.to_datetime(self.annotations["timestamp"])
            self.annotations = self.annotations.sort_values("timestamp")
        
        # Ensure required columns
        if "dc_power" not in self.annotations.columns:
            if "power" in self.annotations.columns:
                self.annotations["dc_power"] = self.annotations["power"]
            else:
                warnings.warn("No power column found. Using dummy values.")
                self.annotations["dc_power"] = np.random.uniform(0, 1000, len(self.annotations))
        
        # Image filename
        if "image_file" not in self.annotations.columns:
            if "timestamp" in self.annotations.columns:
                self.annotations["image_file"] = self.annotations["timestamp"].apply(
                    lambda x: f"{x.strftime('%Y%m%d_%H%M%S')}.jpg"
                )
    
    def _create_dummy_annotations(self) -> pd.DataFrame:
        """Create dummy annotations for testing."""
        n_samples = 100
        timestamps = pd.date_range("2024-01-01", periods=n_samples, freq="15min")
        
        return pd.DataFrame({
            "timestamp": timestamps,
            "dc_power": np.random.uniform(0, 1000, n_samples),
            "temperature": np.random.uniform(15, 35, n_samples),
            "humidity": np.random.uniform(30, 80, n_samples),
            "pressure": np.random.uniform(1000, 1020, n_samples),
            "wind_speed": np.random.uniform(0, 10, n_samples),
            "ghi": np.random.uniform(0, 1000, n_samples),
            "dni": np.random.uniform(0, 800, n_samples),
            "dhi": np.random.uniform(0, 300, n_samples),
            "image_file": [f"img_{i:05d}.jpg" for i in range(n_samples)]
        })
    
    def _prepare_indices(self):
        """Prepare valid indices accounting for sequence length."""
        if self.return_sequences:
            # Need sequence_length previous samples
            self.valid_indices = list(range(self.sequence_length, len(self.annotations)))
        else:
            self.valid_indices = list(range(len(self.annotations)))
    
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def _load_image(self, image_file: str) -> np.ndarray:
        """Load an image from file."""
        image_path = self.image_dir / image_file
        
        if image_path.exists():
            image = Image.open(image_path).convert("RGB")
            return np.array(image)
        else:
            # Return random image for testing
            return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    def _get_weather_features(self, idx: int) -> np.ndarray:
        """Get weather features for a given index."""
        row = self.annotations.iloc[idx]
        
        weather_data = {}
        for feature in self.weather_features:
            if feature in row:
                weather_data[feature] = row[feature]
            else:
                weather_data[feature] = 0.0
        
        # Transform using fitted processor
        features = self.weather_processor.transform(weather_data)
        
        return features.flatten()
    
    def _get_sun_position(self, idx: int) -> np.ndarray:
        """Get sun position features for a given index."""
        row = self.annotations.iloc[idx]
        
        if "timestamp" in row:
            timestamp = row["timestamp"]
        else:
            timestamp = pd.Timestamp.now()
        
        features = self.sun_calculator.get_features_array(timestamp)
        
        # Normalize
        features = np.array([
            features[0] / 180,  # zenith
            features[1] / 360,  # azimuth
            (features[2] + 90) / 180,  # elevation
            (features[3] + 20) / 40   # equation of time
        ], dtype=np.float32)
        
        return features
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample."""
        actual_idx = self.valid_indices[idx]
        row = self.annotations.iloc[actual_idx]
        
        # Current data
        current_image = self._load_image(row["image_file"])
        current_weather = self._get_weather_features(actual_idx)
        current_sun_position = self._get_sun_position(actual_idx)
        target_power = row["dc_power"]
        
        # Apply transform to current image
        if hasattr(self.transform, '__call__'):
            try:
                transformed = self.transform(image=current_image)
                current_image = transformed["image"]
            except:
                current_image = self.transform(current_image)
        
        sample = {
            "current_image": current_image,
            "current_weather": torch.tensor(current_weather, dtype=torch.float32),
            "current_sun_position": torch.tensor(current_sun_position, dtype=torch.float32),
            "target": torch.tensor(target_power, dtype=torch.float32)
        }
        
        # Sequence data
        if self.return_sequences:
            seq_images = []
            seq_weather = []
            seq_sun = []
            
            for i in range(self.sequence_length):
                seq_idx = actual_idx - self.sequence_length + i
                seq_row = self.annotations.iloc[seq_idx]
                
                # Image
                img = self._load_image(seq_row["image_file"])
                if hasattr(self.transform, '__call__'):
                    try:
                        transformed = self.transform(image=img)
                        img = transformed["image"]
                    except:
                        img = self.transform(img)
                seq_images.append(img)
                
                # Weather
                weather = self._get_weather_features(seq_idx)
                seq_weather.append(weather)
                
                # Sun position
                sun = self._get_sun_position(seq_idx)
                seq_sun.append(sun)
            
            sample["image_sequence"] = torch.stack(seq_images)
            sample["weather_sequence"] = torch.tensor(np.stack(seq_weather), dtype=torch.float32)
            sample["sun_position_sequence"] = torch.tensor(np.stack(seq_sun), dtype=torch.float32)
        
        return sample


class SyntheticSkyDataset(Dataset):
    """
    Synthetic dataset for testing and development.
    
    Generates random sky images and simulated power output.
    """
    
    def __init__(
        self,
        num_samples: int = 1000,
        sequence_length: int = 12,
        image_size: int = 224,
        mode: str = "train"
    ):
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.image_size = image_size
        self.mode = mode
        
        # Generate synthetic data
        np.random.seed(42 if mode == "train" else 123)
        self._generate_data()
        
        # Transforms
        if mode == "train":
            self.transform = get_train_transforms(image_size)
        else:
            self.transform = get_val_transforms(image_size)
    
    def _generate_data(self):
        """Generate synthetic data."""
        # Time features
        hours = np.random.uniform(6, 18, self.num_samples)  # Daylight hours
        
        # Weather features
        self.weather = np.stack([
            np.random.uniform(15, 35, self.num_samples),  # temperature
            np.random.uniform(30, 80, self.num_samples),  # humidity
            np.random.uniform(1000, 1020, self.num_samples),  # pressure
            np.random.uniform(0, 10, self.num_samples),   # wind_speed
            np.random.uniform(0, 360, self.num_samples),  # wind_direction
            np.random.uniform(5, 25, self.num_samples),   # dew_point
            np.random.uniform(0, 1000, self.num_samples), # ghi
            np.random.uniform(0, 800, self.num_samples),  # dni
            np.random.uniform(0, 300, self.num_samples),  # dhi
        ], axis=1).astype(np.float32)
        
        # Sun position (simplified)
        self.sun_position = np.stack([
            90 - np.abs(hours - 12) * 7.5,  # zenith
            hours * 15,  # azimuth
            np.abs(hours - 12) * 7.5,  # elevation
            np.zeros(self.num_samples)  # equation of time
        ], axis=1).astype(np.float32)
        
        # Normalize sun position
        self.sun_position[:, 0] /= 90
        self.sun_position[:, 1] /= 360
        self.sun_position[:, 2] /= 90
        
        # Power output (correlated with irradiance and sun position)
        ghi = self.weather[:, 6]
        elevation = self.sun_position[:, 2] * 90
        cloud_factor = np.random.uniform(0.5, 1.0, self.num_samples)
        
        self.power = ghi * np.sin(np.radians(elevation)) * cloud_factor
        self.power = np.clip(self.power, 0, 1000).astype(np.float32)
        
        # Normalize weather
        self.weather = (self.weather - self.weather.mean(axis=0)) / (self.weather.std(axis=0) + 1e-8)
    
    def _generate_sky_image(self, cloud_factor: float) -> np.ndarray:
        """Generate a synthetic sky image."""
        h, w = self.image_size, self.image_size
        
        # Blue sky gradient
        image = np.zeros((h, w, 3), dtype=np.uint8)
        for i in range(h):
            blue_val = int(200 - i * 0.3)
            image[i, :, 2] = blue_val
            image[i, :, 1] = int(blue_val * 0.6)
            image[i, :, 0] = int(blue_val * 0.3)
        
        # Add clouds based on factor
        if cloud_factor < 0.8:
            num_clouds = int((1 - cloud_factor) * 10)
            for _ in range(num_clouds):
                cx = np.random.randint(0, w)
                cy = np.random.randint(0, h // 2)
                radius = np.random.randint(20, 60)
                
                y, x = np.ogrid[:h, :w]
                mask = (x - cx) ** 2 + (y - cy) ** 2 < radius ** 2
                
                cloud_color = np.random.randint(200, 250)
                image[mask] = [cloud_color, cloud_color, min(255, cloud_color + 5)]
        
        return image
    
    def __len__(self) -> int:
        return self.num_samples - self.sequence_length
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        actual_idx = idx + self.sequence_length
        
        # Current data
        cloud_factor = self.power[actual_idx] / 1000
        current_image = self._generate_sky_image(cloud_factor)
        
        # Apply transform
        try:
            transformed = self.transform(image=current_image)
            current_image = transformed["image"]
        except:
            current_image = self.transform(current_image)
        
        sample = {
            "current_image": current_image,
            "current_weather": torch.tensor(self.weather[actual_idx], dtype=torch.float32),
            "current_sun_position": torch.tensor(self.sun_position[actual_idx], dtype=torch.float32),
            "target": torch.tensor(self.power[actual_idx], dtype=torch.float32)
        }
        
        # Sequence data
        seq_images = []
        for i in range(self.sequence_length):
            seq_idx = actual_idx - self.sequence_length + i
            cf = self.power[seq_idx] / 1000
            img = self._generate_sky_image(cf)
            
            try:
                transformed = self.transform(image=img)
                img = transformed["image"]
            except:
                img = self.transform(img)
            
            seq_images.append(img)
        
        sample["image_sequence"] = torch.stack(seq_images)
        sample["weather_sequence"] = torch.tensor(
            self.weather[actual_idx - self.sequence_length:actual_idx],
            dtype=torch.float32
        )
        sample["sun_position_sequence"] = torch.tensor(
            self.sun_position[actual_idx - self.sequence_length:actual_idx],
            dtype=torch.float32
        )
        
        return sample


if __name__ == "__main__":
    # Test datasets
    print("Testing SyntheticSkyDataset...")
    
    dataset = SyntheticSkyDataset(
        num_samples=100,
        sequence_length=12,
        image_size=224,
        mode="train"
    )
    
    print(f"Dataset length: {len(dataset)}")
    
    sample = dataset[0]
    print(f"\nSample keys: {list(sample.keys())}")
    print(f"Current image shape: {sample['current_image'].shape}")
    print(f"Current weather shape: {sample['current_weather'].shape}")
    print(f"Current sun position shape: {sample['current_sun_position'].shape}")
    print(f"Target: {sample['target']}")
    print(f"Image sequence shape: {sample['image_sequence'].shape}")
    print(f"Weather sequence shape: {sample['weather_sequence'].shape}")
    print(f"Sun position sequence shape: {sample['sun_position_sequence'].shape}")
