"""
Sun Position Calculator

Uses pvlib to compute sun position features for a given location and time.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Union, Tuple
import warnings

try:
    import pvlib
    from pvlib.location import Location
    from pvlib.solarposition import get_solarposition
    PVLIB_AVAILABLE = True
except ImportError:
    PVLIB_AVAILABLE = False
    warnings.warn("pvlib not available. Sun position calculations will use fallback.")


class SunPositionCalculator:
    """
    Calculator for sun position and solar radiation features.
    
    Uses pvlib library for accurate solar position calculations
    based on geographic location and timestamp.
    
    Args:
        latitude: Location latitude in degrees
        longitude: Location longitude in degrees  
        altitude: Location altitude in meters
        timezone: Timezone string (e.g., 'US/Pacific')
    """
    
    def __init__(
        self,
        latitude: float = 37.7749,
        longitude: float = -122.4194,
        altitude: float = 10,
        timezone: str = "UTC"
    ):
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
        self.timezone = timezone
        
        if PVLIB_AVAILABLE:
            self.location = Location(
                latitude=latitude,
                longitude=longitude,
                altitude=altitude,
                tz=timezone
            )
    
    def get_sun_position(
        self,
        timestamp: Union[datetime, pd.Timestamp, str]
    ) -> Dict[str, float]:
        """
        Get sun position for a single timestamp.
        
        Args:
            timestamp: Time for calculation
            
        Returns:
            Dictionary with sun position features:
            - zenith: Solar zenith angle (degrees from vertical)
            - azimuth: Solar azimuth angle (degrees from north)
            - apparent_elevation: Apparent solar elevation (degrees)
            - equation_of_time: Equation of time (minutes)
        """
        if isinstance(timestamp, str):
            timestamp = pd.Timestamp(timestamp)
        elif isinstance(timestamp, datetime):
            timestamp = pd.Timestamp(timestamp)
        
        if PVLIB_AVAILABLE:
            return self._get_position_pvlib(timestamp)
        else:
            return self._get_position_fallback(timestamp)
    
    def _get_position_pvlib(
        self,
        timestamp: pd.Timestamp
    ) -> Dict[str, float]:
        """Get sun position using pvlib."""
        times = pd.DatetimeIndex([timestamp])
        
        if times.tz is None:
            times = times.tz_localize(self.timezone)
        
        solar_pos = get_solarposition(
            times,
            self.latitude,
            self.longitude,
            altitude=self.altitude
        )
        
        return {
            "zenith": solar_pos["zenith"].iloc[0],
            "azimuth": solar_pos["azimuth"].iloc[0],
            "apparent_elevation": solar_pos["apparent_elevation"].iloc[0],
            "equation_of_time": solar_pos["equation_of_time"].iloc[0]
        }
    
    def _get_position_fallback(
        self,
        timestamp: pd.Timestamp
    ) -> Dict[str, float]:
        """Simple fallback sun position calculation."""
        # Day of year
        day_of_year = timestamp.dayofyear
        
        # Hour angle
        hour = timestamp.hour + timestamp.minute / 60
        hour_angle = 15 * (hour - 12)  # degrees
        
        # Declination (approximate)
        declination = -23.45 * np.cos(np.radians(360 / 365 * (day_of_year + 10)))
        
        # Solar elevation
        lat_rad = np.radians(self.latitude)
        dec_rad = np.radians(declination)
        hour_rad = np.radians(hour_angle)
        
        elevation = np.degrees(np.arcsin(
            np.sin(lat_rad) * np.sin(dec_rad) +
            np.cos(lat_rad) * np.cos(dec_rad) * np.cos(hour_rad)
        ))
        
        zenith = 90 - elevation
        
        # Azimuth (approximate)
        azimuth = np.degrees(np.arccos(
            (np.sin(dec_rad) - np.sin(lat_rad) * np.sin(np.radians(elevation))) /
            (np.cos(lat_rad) * np.cos(np.radians(elevation)))
        ))
        
        if hour_angle > 0:
            azimuth = 360 - azimuth
        
        return {
            "zenith": zenith,
            "azimuth": azimuth,
            "apparent_elevation": elevation,
            "equation_of_time": 0.0
        }
    
    def get_sun_position_batch(
        self,
        timestamps: Union[List, pd.DatetimeIndex]
    ) -> pd.DataFrame:
        """
        Get sun position for multiple timestamps.
        
        Args:
            timestamps: List of timestamps
            
        Returns:
            DataFrame with sun position features
        """
        if not isinstance(timestamps, pd.DatetimeIndex):
            timestamps = pd.DatetimeIndex(timestamps)
        
        if PVLIB_AVAILABLE:
            if timestamps.tz is None:
                timestamps = timestamps.tz_localize(self.timezone)
            
            solar_pos = get_solarposition(
                timestamps,
                self.latitude,
                self.longitude,
                altitude=self.altitude
            )
            
            return solar_pos[["zenith", "azimuth", "apparent_elevation", "equation_of_time"]]
        else:
            positions = [self._get_position_fallback(ts) for ts in timestamps]
            return pd.DataFrame(positions, index=timestamps)
    
    def get_clear_sky_irradiance(
        self,
        timestamp: Union[datetime, pd.Timestamp]
    ) -> Dict[str, float]:
        """
        Get clear sky irradiance estimates.
        
        Args:
            timestamp: Time for calculation
            
        Returns:
            Dictionary with irradiance values:
            - ghi: Global Horizontal Irradiance
            - dni: Direct Normal Irradiance
            - dhi: Diffuse Horizontal Irradiance
        """
        if not PVLIB_AVAILABLE:
            return {"ghi": 0.0, "dni": 0.0, "dhi": 0.0}
        
        if isinstance(timestamp, str):
            timestamp = pd.Timestamp(timestamp)
        
        times = pd.DatetimeIndex([timestamp])
        if times.tz is None:
            times = times.tz_localize(self.timezone)
        
        clear_sky = self.location.get_clearsky(times)
        
        return {
            "ghi": clear_sky["ghi"].iloc[0],
            "dni": clear_sky["dni"].iloc[0],
            "dhi": clear_sky["dhi"].iloc[0]
        }
    
    def get_features_array(
        self,
        timestamp: Union[datetime, pd.Timestamp]
    ) -> np.ndarray:
        """
        Get sun position features as numpy array.
        
        Args:
            timestamp: Time for calculation
            
        Returns:
            Array of [zenith, azimuth, apparent_elevation, equation_of_time]
        """
        pos = self.get_sun_position(timestamp)
        return np.array([
            pos["zenith"],
            pos["azimuth"],
            pos["apparent_elevation"],
            pos["equation_of_time"]
        ])
    
    def is_daytime(
        self,
        timestamp: Union[datetime, pd.Timestamp],
        elevation_threshold: float = 0
    ) -> bool:
        """
        Check if it's daytime (sun above horizon).
        
        Args:
            timestamp: Time to check
            elevation_threshold: Minimum elevation to consider daytime
            
        Returns:
            True if daytime, False otherwise
        """
        pos = self.get_sun_position(timestamp)
        return pos["apparent_elevation"] > elevation_threshold
    
    def get_daylight_hours(
        self,
        date: Union[datetime, pd.Timestamp, str]
    ) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """
        Get sunrise and sunset times for a given date.
        
        Args:
            date: Date for calculation
            
        Returns:
            Tuple of (sunrise, sunset) timestamps
        """
        if not PVLIB_AVAILABLE:
            # Fallback: assume 6 AM to 6 PM
            if isinstance(date, str):
                date = pd.Timestamp(date)
            sunrise = date.replace(hour=6, minute=0)
            sunset = date.replace(hour=18, minute=0)
            return sunrise, sunset
        
        if isinstance(date, str):
            date = pd.Timestamp(date)
        
        # Create time range for the day
        start = date.replace(hour=0, minute=0, second=0)
        end = date.replace(hour=23, minute=59, second=59)
        times = pd.date_range(start, end, freq='1min')
        
        if times.tz is None:
            times = times.tz_localize(self.timezone)
        
        # Get solar positions
        solar_pos = get_solarposition(
            times,
            self.latitude,
            self.longitude
        )
        
        # Find sunrise and sunset
        above_horizon = solar_pos["apparent_elevation"] > 0
        
        if not above_horizon.any():
            # No daylight
            return None, None
        
        sunrise_idx = above_horizon.idxmax()
        sunset_idx = above_horizon[::-1].idxmax()
        
        return sunrise_idx, sunset_idx


class SunPositionNormalizer:
    """
    Normalizer for sun position features.
    """
    
    def __init__(self):
        # Known ranges for sun position features
        self.ranges = {
            "zenith": (0, 180),
            "azimuth": (0, 360),
            "apparent_elevation": (-90, 90),
            "equation_of_time": (-20, 20)
        }
    
    def normalize(self, features: Dict[str, float]) -> Dict[str, float]:
        """Normalize features to [0, 1] range."""
        normalized = {}
        for key, value in features.items():
            if key in self.ranges:
                min_val, max_val = self.ranges[key]
                normalized[key] = (value - min_val) / (max_val - min_val)
            else:
                normalized[key] = value
        return normalized
    
    def normalize_array(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize feature array.
        
        Args:
            features: Array of [zenith, azimuth, elevation, equation_of_time]
            
        Returns:
            Normalized array
        """
        normalized = features.copy()
        for i, (key, (min_val, max_val)) in enumerate(self.ranges.items()):
            normalized[..., i] = (features[..., i] - min_val) / (max_val - min_val)
        return normalized


if __name__ == "__main__":
    # Test sun position calculator
    print("Testing SunPositionCalculator...")
    
    calc = SunPositionCalculator(
        latitude=37.7749,
        longitude=-122.4194,
        timezone="US/Pacific"
    )
    
    # Test single timestamp
    timestamp = pd.Timestamp("2024-06-21 12:00:00")
    pos = calc.get_sun_position(timestamp)
    print(f"\nSun position at {timestamp}:")
    for key, value in pos.items():
        print(f"  {key}: {value:.2f}")
    
    # Test batch
    timestamps = pd.date_range("2024-06-21 06:00", periods=12, freq="1h")
    batch_pos = calc.get_sun_position_batch(timestamps)
    print(f"\nBatch sun positions:")
    print(batch_pos.head())
    
    # Test daytime check
    print(f"\nIs daytime at noon: {calc.is_daytime(timestamp)}")
    
    # Test feature array
    features = calc.get_features_array(timestamp)
    print(f"\nFeature array: {features}")
    
    # Test normalizer
    normalizer = SunPositionNormalizer()
    norm_features = normalizer.normalize(pos)
    print(f"\nNormalized features:")
    for key, value in norm_features.items():
        print(f"  {key}: {value:.4f}")
