"""Weather Data Processor."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class WeatherProcessor:
    """Processor for weather sensor data."""
    
    DEFAULT_FEATURES = [
        "temperature", "humidity", "pressure", "wind_speed",
        "wind_direction", "dew_point", "ghi", "dni", "dhi"
    ]
    
    FEATURE_RANGES = {
        "temperature": (-40, 50), "humidity": (0, 100), "pressure": (950, 1050),
        "wind_speed": (0, 50), "wind_direction": (0, 360), "dew_point": (-40, 40),
        "ghi": (0, 1500), "dni": (0, 1200), "dhi": (0, 500)
    }
    
    def __init__(self, features: Optional[List[str]] = None, scaler_type: str = "standard"):
        self.features = features or self.DEFAULT_FEATURES
        self.scaler_type = scaler_type
        
        if scaler_type == "standard":
            self.scaler = StandardScaler()
        elif scaler_type == "minmax":
            self.scaler = MinMaxScaler()
        else:
            self.scaler = None
        
        self.fitted = False
        self.feature_stats = {}
    
    def fit(self, data: pd.DataFrame) -> "WeatherProcessor":
        available_features = [f for f in self.features if f in data.columns]
        self.features = available_features
        
        for feature in self.features:
            self.feature_stats[feature] = {
                "mean": data[feature].mean(),
                "std": data[feature].std(),
                "min": data[feature].min(),
                "max": data[feature].max()
            }
        
        if self.scaler is not None:
            feature_data = np.nan_to_num(data[self.features].values)
            self.scaler.fit(feature_data)
        
        self.fitted = True
        return self
    
    def transform(self, data: Union[pd.DataFrame, np.ndarray, Dict]) -> np.ndarray:
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        elif isinstance(data, np.ndarray):
            data = pd.DataFrame(data, columns=self.features)
        
        feature_data = data[self.features].fillna(0).values
        
        if self.scaler is not None and self.fitted:
            feature_data = self.scaler.transform(feature_data)
        
        return feature_data.astype(np.float32)
    
    def fit_transform(self, data: pd.DataFrame) -> np.ndarray:
        self.fit(data)
        return self.transform(data)
    
    def normalize_by_range(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if isinstance(data, pd.DataFrame):
            values = data[self.features].values
        else:
            values = data
        
        normalized = np.zeros_like(values, dtype=np.float32)
        
        for i, feature in enumerate(self.features):
            if feature in self.FEATURE_RANGES:
                min_val, max_val = self.FEATURE_RANGES[feature]
                normalized[:, i] = np.clip((values[:, i] - min_val) / (max_val - min_val), 0, 1)
        
        return normalized
    
    def get_feature_dim(self) -> int:
        return len(self.features)
