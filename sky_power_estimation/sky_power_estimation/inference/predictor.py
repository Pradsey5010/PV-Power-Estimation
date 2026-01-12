"""
Predictor Module

Inference utilities for DC power prediction.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
from collections import deque

import torch
import torch.nn as nn
from PIL import Image

from ..models import SkyPowerModel
from ..utils.sun_position import SunPositionCalculator
from ..utils.weather_processor import WeatherProcessor
from ..utils.image_processor import ImageProcessor
from ..data.transforms import get_val_transforms


class Predictor:
    """
    Predictor class for DC power estimation inference.
    
    Provides methods for:
    - Single image prediction
    - Batch prediction
    - Sequence-aware prediction
    - Uncertainty estimation
    
    Args:
        model_path: Path to saved model checkpoint
        config: Model configuration (optional, loaded from checkpoint)
        device: Device to run inference on
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        model: Optional[nn.Module] = None,
        config: Optional[Dict] = None,
        device: Optional[torch.device] = None,
        image_size: int = 224,
        location: Optional[Dict] = None
    ):
        # Device
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Load model
        if model is not None:
            self.model = model.to(self.device)
            self.config = config or {}
        elif model_path is not None:
            self.model, self.config = self._load_model(model_path)
        else:
            raise ValueError("Either model or model_path must be provided")
        
        self.model.eval()
        
        # Image processing
        self.image_size = image_size
        self.transform = get_val_transforms(image_size)
        self.image_processor = ImageProcessor(image_size=(image_size, image_size))
        
        # Weather processor
        self.weather_processor = WeatherProcessor()
        
        # Sun position calculator
        location = location or {
            "latitude": 37.7749,
            "longitude": -122.4194,
            "altitude": 10,
            "timezone": "UTC"
        }
        self.sun_calculator = SunPositionCalculator(**location)
    
    def _load_model(self, model_path: str) -> Tuple[nn.Module, Dict]:
        """Load model from checkpoint."""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        config = checkpoint.get("config", {})
        
        # Create model
        model = SkyPowerModel.from_config(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(self.device)
        
        return model, config
    
    def _preprocess_image(
        self,
        image: Union[str, Path, np.ndarray, Image.Image]
    ) -> torch.Tensor:
        """Preprocess image for model input."""
        # Load image if path
        if isinstance(image, (str, Path)):
            image = np.array(Image.open(image).convert("RGB"))
        elif isinstance(image, Image.Image):
            image = np.array(image.convert("RGB"))
        
        # Apply transforms
        try:
            transformed = self.transform(image=image)
            image_tensor = transformed["image"]
        except:
            image_tensor = self.transform(image)
        
        return image_tensor.to(self.device)
    
    def _preprocess_weather(
        self,
        weather: Dict[str, float]
    ) -> torch.Tensor:
        """Preprocess weather data."""
        features = []
        for key in self.weather_processor.features:
            features.append(weather.get(key, 0.0))
        
        return torch.tensor(features, dtype=torch.float32, device=self.device)
    
    def _get_sun_position(
        self,
        timestamp: Optional[datetime] = None
    ) -> torch.Tensor:
        """Get sun position features."""
        timestamp = timestamp or datetime.now()
        features = self.sun_calculator.get_features_array(timestamp)
        
        # Normalize
        normalized = np.array([
            features[0] / 180,  # zenith
            features[1] / 360,  # azimuth
            (features[2] + 90) / 180,  # elevation
            (features[3] + 20) / 40   # equation of time
        ], dtype=np.float32)
        
        return torch.tensor(normalized, device=self.device)
    
    @torch.no_grad()
    def predict(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        weather: Optional[Dict[str, float]] = None,
        timestamp: Optional[datetime] = None,
        return_features: bool = False
    ) -> Union[float, Tuple[float, Dict]]:
        """
        Predict DC power for a single image.
        
        Args:
            image: Sky image (path, numpy array, or PIL Image)
            weather: Weather data dictionary
            timestamp: Timestamp for sun position
            return_features: Whether to return intermediate features
            
        Returns:
            Predicted power (and optionally features)
        """
        # Preprocess
        image_tensor = self._preprocess_image(image).unsqueeze(0)
        
        if weather is not None:
            weather_tensor = self._preprocess_weather(weather).unsqueeze(0)
        else:
            weather_tensor = torch.zeros(1, len(self.weather_processor.features), device=self.device)
        
        sun_tensor = self._get_sun_position(timestamp).unsqueeze(0)
        
        # Forward pass
        output = self.model(
            current_image=image_tensor,
            current_weather=weather_tensor,
            current_sun_position=sun_tensor,
            return_features=return_features
        )
        
        if return_features:
            power, features = output
            return power.item(), {k: v.cpu().numpy() for k, v in features.items()}
        
        return output.item()
    
    @torch.no_grad()
    def predict_batch(
        self,
        images: List[Union[str, Path, np.ndarray]],
        weather_list: Optional[List[Dict]] = None,
        timestamps: Optional[List[datetime]] = None
    ) -> np.ndarray:
        """
        Predict DC power for a batch of images.
        
        Args:
            images: List of sky images
            weather_list: List of weather data dictionaries
            timestamps: List of timestamps
            
        Returns:
            Array of predicted power values
        """
        # Preprocess all images
        image_tensors = torch.stack([
            self._preprocess_image(img) for img in images
        ])
        
        # Weather
        if weather_list is not None:
            weather_tensors = torch.stack([
                self._preprocess_weather(w) for w in weather_list
            ])
        else:
            weather_tensors = torch.zeros(
                len(images), 
                len(self.weather_processor.features),
                device=self.device
            )
        
        # Sun position
        if timestamps is not None:
            sun_tensors = torch.stack([
                self._get_sun_position(ts) for ts in timestamps
            ])
        else:
            sun_tensors = torch.stack([
                self._get_sun_position() for _ in range(len(images))
            ])
        
        # Forward pass
        predictions = self.model(
            current_image=image_tensors,
            current_weather=weather_tensors,
            current_sun_position=sun_tensors
        )
        
        return predictions.cpu().numpy()
    
    @torch.no_grad()
    def predict_with_sequence(
        self,
        current_image: Union[str, Path, np.ndarray],
        image_sequence: List[Union[str, Path, np.ndarray]],
        current_weather: Dict[str, float],
        weather_sequence: List[Dict[str, float]],
        current_timestamp: datetime,
        timestamp_sequence: List[datetime]
    ) -> float:
        """
        Predict using image and weather sequences.
        
        Args:
            current_image: Current sky image
            image_sequence: Historical images
            current_weather: Current weather
            weather_sequence: Historical weather
            current_timestamp: Current timestamp
            timestamp_sequence: Historical timestamps
            
        Returns:
            Predicted power
        """
        # Current inputs
        current_img = self._preprocess_image(current_image).unsqueeze(0)
        current_wx = self._preprocess_weather(current_weather).unsqueeze(0)
        current_sun = self._get_sun_position(current_timestamp).unsqueeze(0)
        
        # Sequence inputs
        img_seq = torch.stack([
            self._preprocess_image(img) for img in image_sequence
        ]).unsqueeze(0)
        
        wx_seq = torch.stack([
            self._preprocess_weather(w) for w in weather_sequence
        ]).unsqueeze(0)
        
        sun_seq = torch.stack([
            self._get_sun_position(ts) for ts in timestamp_sequence
        ]).unsqueeze(0)
        
        # Forward pass
        prediction = self.model(
            current_image=current_img,
            image_sequence=img_seq,
            current_weather=current_wx,
            weather_sequence=wx_seq,
            current_sun_position=current_sun,
            sun_position_sequence=sun_seq
        )
        
        return prediction.item()
    
    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        image: Union[str, Path, np.ndarray],
        weather: Optional[Dict[str, float]] = None,
        timestamp: Optional[datetime] = None,
        num_samples: int = 20
    ) -> Tuple[float, float]:
        """
        Predict with uncertainty estimation using MC Dropout.
        
        Args:
            image: Sky image
            weather: Weather data
            timestamp: Timestamp
            num_samples: Number of Monte Carlo samples
            
        Returns:
            Tuple of (mean prediction, std uncertainty)
        """
        # Preprocess
        image_tensor = self._preprocess_image(image).unsqueeze(0)
        
        if weather is not None:
            weather_tensor = self._preprocess_weather(weather).unsqueeze(0)
        else:
            weather_tensor = torch.zeros(1, len(self.weather_processor.features), device=self.device)
        
        sun_tensor = self._get_sun_position(timestamp).unsqueeze(0)
        
        # Enable dropout for MC sampling
        self.model.train()
        
        predictions = []
        for _ in range(num_samples):
            pred = self.model(
                current_image=image_tensor,
                current_weather=weather_tensor,
                current_sun_position=sun_tensor
            )
            predictions.append(pred.item())
        
        self.model.eval()
        
        mean = np.mean(predictions)
        std = np.std(predictions)
        
        return mean, std
    
    def get_cloud_features(
        self,
        image: Union[str, Path, np.ndarray]
    ) -> Dict[str, float]:
        """
        Extract cloud features from image.
        
        Args:
            image: Sky image
            
        Returns:
            Dictionary of cloud features
        """
        if isinstance(image, (str, Path)):
            image = np.array(Image.open(image).convert("RGB"))
        
        return self.image_processor.extract_cloud_features(image)


class RealTimePredictor:
    """
    Real-time predictor with sequence buffering.
    
    Maintains history of recent observations for sequence-aware predictions.
    """
    
    def __init__(
        self,
        model_path: str,
        sequence_length: int = 12,
        **kwargs
    ):
        self.predictor = Predictor(model_path=model_path, **kwargs)
        self.sequence_length = sequence_length
        
        # Buffers
        self.image_buffer = deque(maxlen=sequence_length)
        self.weather_buffer = deque(maxlen=sequence_length)
        self.timestamp_buffer = deque(maxlen=sequence_length)
    
    def update(
        self,
        image: Union[str, Path, np.ndarray],
        weather: Dict[str, float],
        timestamp: Optional[datetime] = None
    ):
        """
        Update buffers with new observation.
        
        Args:
            image: New sky image
            weather: New weather data
            timestamp: Observation timestamp
        """
        timestamp = timestamp or datetime.now()
        
        self.image_buffer.append(image)
        self.weather_buffer.append(weather)
        self.timestamp_buffer.append(timestamp)
    
    def predict(self) -> Optional[float]:
        """
        Make prediction using current buffer.
        
        Returns:
            Predicted power, or None if buffer not full
        """
        if len(self.image_buffer) < self.sequence_length:
            # Not enough history - use simple prediction
            if len(self.image_buffer) > 0:
                return self.predictor.predict(
                    image=self.image_buffer[-1],
                    weather=self.weather_buffer[-1],
                    timestamp=self.timestamp_buffer[-1]
                )
            return None
        
        # Use full sequence
        return self.predictor.predict_with_sequence(
            current_image=self.image_buffer[-1],
            image_sequence=list(self.image_buffer)[:-1],
            current_weather=self.weather_buffer[-1],
            weather_sequence=list(self.weather_buffer)[:-1],
            current_timestamp=self.timestamp_buffer[-1],
            timestamp_sequence=list(self.timestamp_buffer)[:-1]
        )
    
    def predict_with_update(
        self,
        image: Union[str, Path, np.ndarray],
        weather: Dict[str, float],
        timestamp: Optional[datetime] = None
    ) -> float:
        """
        Update buffer and make prediction.
        
        Args:
            image: New sky image
            weather: New weather data
            timestamp: Observation timestamp
            
        Returns:
            Predicted power
        """
        self.update(image, weather, timestamp)
        return self.predict()
    
    def reset(self):
        """Reset buffers."""
        self.image_buffer.clear()
        self.weather_buffer.clear()
        self.timestamp_buffer.clear()


if __name__ == "__main__":
    # Test predictor
    print("Testing Predictor...")
    
    from ..models import SkyPowerModel
    
    # Create model
    model = SkyPowerModel(
        image_backbone="resnet18",
        image_pretrained=False
    )
    
    # Create predictor
    predictor = Predictor(model=model)
    
    # Test with synthetic image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    test_weather = {
        "temperature": 25.0,
        "humidity": 60.0,
        "ghi": 800.0,
        "dni": 600.0,
        "dhi": 200.0
    }
    
    # Simple prediction
    power = predictor.predict(
        image=test_image,
        weather=test_weather,
        timestamp=datetime.now()
    )
    print(f"Predicted power: {power:.2f} W")
    
    # Prediction with uncertainty
    mean, std = predictor.predict_with_uncertainty(
        image=test_image,
        weather=test_weather,
        num_samples=10
    )
    print(f"Power with uncertainty: {mean:.2f} ± {std:.2f} W")
    
    # Cloud features
    features = predictor.get_cloud_features(test_image)
    print(f"Cloud features: {features}")
