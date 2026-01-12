"""
Weather Data Processor

Handles preprocessing and normalization of weather sensor data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings


class WeatherProcessor:
    """
    Processor for weather sensor data.
    
    Handles:
    - Missing value imputation
    - Normalization/Standardization
    - Feature engineering
    - Sequence preparation
    
    Args:
        features: List of weather feature names to process
        scaler_type: Type of scaler ('standard', 'minmax', 'none')
    """
    
    # Default weather features
    DEFAULT_FEATURES = [
        "temperature",
        "humidity", 
        "pressure",
        "wind_speed",
        "wind_direction",
        "dew_point",
        "ghi",
        "dni",
        "dhi"
    ]
    
    # Feature ranges for MinMax scaling
    FEATURE_RANGES = {
        "temperature": (-40, 50),      # Celsius
        "humidity": (0, 100),           # Percentage
        "pressure": (950, 1050),        # hPa
        "wind_speed": (0, 50),          # m/s
        "wind_direction": (0, 360),     # degrees
        "dew_point": (-40, 40),         # Celsius
        "ghi": (0, 1500),               # W/m²
        "dni": (0, 1200),               # W/m²
        "dhi": (0, 500)                 # W/m²
    }
    
    def __init__(
        self,
        features: Optional[List[str]] = None,
        scaler_type: str = "standard"
    ):
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
        """
        Fit the processor on training data.
        
        Args:
            data: DataFrame with weather features
            
        Returns:
            self
        """
        # Select available features
        available_features = [f for f in self.features if f in data.columns]
        self.features = available_features
        
        if not self.features:
            raise ValueError("No valid features found in data")
        
        # Compute feature statistics
        for feature in self.features:
            self.feature_stats[feature] = {
                "mean": data[feature].mean(),
                "std": data[feature].std(),
                "min": data[feature].min(),
                "max": data[feature].max(),
                "median": data[feature].median()
            }
        
        # Fit scaler
        if self.scaler is not None:
            feature_data = data[self.features].values
            # Handle NaN before fitting
            feature_data = np.nan_to_num(
                feature_data,
                nan=0,
                posinf=0,
                neginf=0
            )
            self.scaler.fit(feature_data)
        
        self.fitted = True
        return self
    
    def transform(
        self,
        data: Union[pd.DataFrame, np.ndarray, Dict],
        handle_missing: str = "mean"
    ) -> np.ndarray:
        """
        Transform weather data.
        
        Args:
            data: Input data (DataFrame, array, or dict)
            handle_missing: How to handle missing values ('mean', 'median', 'zero', 'interpolate')
            
        Returns:
            Transformed numpy array
        """
        # Convert to DataFrame if needed
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        elif isinstance(data, np.ndarray):
            data = pd.DataFrame(data, columns=self.features)
        
        # Select features
        feature_data = data[self.features].copy()
        
        # Handle missing values
        feature_data = self._handle_missing(feature_data, method=handle_missing)
        
        # Convert to array
        values = feature_data.values
        
        # Scale
        if self.scaler is not None and self.fitted:
            values = self.scaler.transform(values)
        
        return values.astype(np.float32)
    
    def fit_transform(
        self,
        data: pd.DataFrame,
        handle_missing: str = "mean"
    ) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(data)
        return self.transform(data, handle_missing)
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform scaled data back to original scale."""
        if self.scaler is not None and self.fitted:
            return self.scaler.inverse_transform(data)
        return data
    
    def _handle_missing(
        self,
        data: pd.DataFrame,
        method: str = "mean"
    ) -> pd.DataFrame:
        """Handle missing values in data."""
        if method == "mean":
            for feature in self.features:
                if feature in data.columns and data[feature].isna().any():
                    fill_value = self.feature_stats.get(feature, {}).get("mean", 0)
                    data[feature] = data[feature].fillna(fill_value)
        
        elif method == "median":
            for feature in self.features:
                if feature in data.columns and data[feature].isna().any():
                    fill_value = self.feature_stats.get(feature, {}).get("median", 0)
                    data[feature] = data[feature].fillna(fill_value)
        
        elif method == "zero":
            data = data.fillna(0)
        
        elif method == "interpolate":
            data = data.interpolate(method="linear", limit_direction="both")
            data = data.fillna(0)  # Fill any remaining NaN
        
        return data
    
    def create_sequences(
        self,
        data: np.ndarray,
        sequence_length: int,
        stride: int = 1
    ) -> np.ndarray:
        """
        Create sequences from weather data for temporal modeling.
        
        Args:
            data: Weather data array [num_samples, num_features]
            sequence_length: Length of each sequence
            stride: Stride between sequences
            
        Returns:
            Sequences array [num_sequences, sequence_length, num_features]
        """
        num_samples = len(data)
        num_sequences = (num_samples - sequence_length) // stride + 1
        
        sequences = np.zeros(
            (num_sequences, sequence_length, data.shape[1]),
            dtype=np.float32
        )
        
        for i in range(num_sequences):
            start_idx = i * stride
            end_idx = start_idx + sequence_length
            sequences[i] = data[start_idx:end_idx]
        
        return sequences
    
    def add_derived_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived features from base weather data.
        
        Args:
            data: DataFrame with base weather features
            
        Returns:
            DataFrame with additional derived features
        """
        result = data.copy()
        
        # Temperature-related
        if "temperature" in data.columns and "humidity" in data.columns:
            # Heat index (simplified)
            T = data["temperature"]
            H = data["humidity"]
            result["heat_index"] = T + 0.5 * (T - 14.4 + 0.12 * H)
        
        if "temperature" in data.columns and "dew_point" in data.columns:
            # Dew point depression
            result["dp_depression"] = data["temperature"] - data["dew_point"]
        
        # Wind features
        if "wind_speed" in data.columns and "wind_direction" in data.columns:
            # Wind components (u, v)
            wind_rad = np.radians(data["wind_direction"])
            result["wind_u"] = data["wind_speed"] * np.sin(wind_rad)
            result["wind_v"] = data["wind_speed"] * np.cos(wind_rad)
        
        # Irradiance features
        if "ghi" in data.columns and "dni" in data.columns:
            # Clearness index (if GHI > 0)
            result["clearness_ratio"] = np.where(
                data["ghi"] > 0,
                data["dni"] / data["ghi"],
                0
            )
        
        if all(col in data.columns for col in ["ghi", "dni", "dhi"]):
            # Diffuse fraction
            result["diffuse_fraction"] = np.where(
                data["ghi"] > 0,
                data["dhi"] / data["ghi"],
                0
            )
        
        return result
    
    def normalize_by_range(
        self,
        data: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """
        Normalize data using predefined feature ranges.
        
        Args:
            data: Weather data
            
        Returns:
            Normalized array with values in [0, 1]
        """
        if isinstance(data, pd.DataFrame):
            values = data[self.features].values
        else:
            values = data
        
        normalized = np.zeros_like(values, dtype=np.float32)
        
        for i, feature in enumerate(self.features):
            if feature in self.FEATURE_RANGES:
                min_val, max_val = self.FEATURE_RANGES[feature]
                normalized[:, i] = (values[:, i] - min_val) / (max_val - min_val)
                normalized[:, i] = np.clip(normalized[:, i], 0, 1)
            else:
                # Use standard normalization for unknown features
                normalized[:, i] = (values[:, i] - values[:, i].mean()) / (values[:, i].std() + 1e-8)
        
        return normalized
    
    def get_feature_dim(self) -> int:
        """Get the number of features."""
        return len(self.features)
    
    def save_stats(self, filepath: str):
        """Save processor statistics to file."""
        import json
        
        stats = {
            "features": self.features,
            "scaler_type": self.scaler_type,
            "feature_stats": self.feature_stats,
            "fitted": self.fitted
        }
        
        if self.scaler is not None and self.fitted:
            if hasattr(self.scaler, "mean_"):
                stats["scaler_mean"] = self.scaler.mean_.tolist()
                stats["scaler_scale"] = self.scaler.scale_.tolist()
            elif hasattr(self.scaler, "min_"):
                stats["scaler_min"] = self.scaler.min_.tolist()
                stats["scaler_scale"] = self.scaler.scale_.tolist()
        
        with open(filepath, "w") as f:
            json.dump(stats, f, indent=2)
    
    def load_stats(self, filepath: str):
        """Load processor statistics from file."""
        import json
        
        with open(filepath, "r") as f:
            stats = json.load(f)
        
        self.features = stats["features"]
        self.scaler_type = stats["scaler_type"]
        self.feature_stats = stats["feature_stats"]
        self.fitted = stats["fitted"]
        
        if "scaler_mean" in stats:
            self.scaler = StandardScaler()
            self.scaler.mean_ = np.array(stats["scaler_mean"])
            self.scaler.scale_ = np.array(stats["scaler_scale"])
        elif "scaler_min" in stats:
            self.scaler = MinMaxScaler()
            self.scaler.min_ = np.array(stats["scaler_min"])
            self.scaler.scale_ = np.array(stats["scaler_scale"])


class IrradianceCalculator:
    """
    Calculator for irradiance-related features.
    """
    
    @staticmethod
    def estimate_ghi(
        dni: float,
        dhi: float,
        zenith: float
    ) -> float:
        """
        Estimate Global Horizontal Irradiance.
        
        GHI = DNI * cos(zenith) + DHI
        
        Args:
            dni: Direct Normal Irradiance
            dhi: Diffuse Horizontal Irradiance
            zenith: Solar zenith angle in degrees
            
        Returns:
            Estimated GHI
        """
        zenith_rad = np.radians(zenith)
        return dni * np.cos(zenith_rad) + dhi
    
    @staticmethod
    def estimate_clearness_index(
        ghi: float,
        extraterrestrial: float
    ) -> float:
        """
        Calculate clearness index (Kt).
        
        Args:
            ghi: Global Horizontal Irradiance
            extraterrestrial: Extraterrestrial irradiance
            
        Returns:
            Clearness index
        """
        if extraterrestrial <= 0:
            return 0
        return np.clip(ghi / extraterrestrial, 0, 1)


if __name__ == "__main__":
    # Test weather processor
    print("Testing WeatherProcessor...")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 100
    
    data = pd.DataFrame({
        "temperature": np.random.uniform(10, 35, n_samples),
        "humidity": np.random.uniform(30, 90, n_samples),
        "pressure": np.random.uniform(1000, 1025, n_samples),
        "wind_speed": np.random.uniform(0, 15, n_samples),
        "wind_direction": np.random.uniform(0, 360, n_samples),
        "dew_point": np.random.uniform(5, 25, n_samples),
        "ghi": np.random.uniform(0, 1000, n_samples),
        "dni": np.random.uniform(0, 800, n_samples),
        "dhi": np.random.uniform(0, 300, n_samples)
    })
    
    # Add some missing values
    data.loc[10:15, "temperature"] = np.nan
    data.loc[20:22, "ghi"] = np.nan
    
    # Create processor
    processor = WeatherProcessor(scaler_type="standard")
    
    # Fit and transform
    transformed = processor.fit_transform(data)
    print(f"Transformed shape: {transformed.shape}")
    print(f"Transformed mean: {transformed.mean(axis=0)}")
    print(f"Transformed std: {transformed.std(axis=0)}")
    
    # Create sequences
    sequences = processor.create_sequences(transformed, sequence_length=12)
    print(f"Sequences shape: {sequences.shape}")
    
    # Add derived features
    derived = processor.add_derived_features(data)
    print(f"Derived features: {derived.columns.tolist()}")
    
    # Normalize by range
    normalized = processor.normalize_by_range(data)
    print(f"Normalized range: [{normalized.min():.3f}, {normalized.max():.3f}]")
