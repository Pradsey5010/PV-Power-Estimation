"""Sun Position Calculator using pvlib."""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Optional, Union
import warnings

try:
    import pvlib
    from pvlib.location import Location
    from pvlib.solarposition import get_solarposition
    PVLIB_AVAILABLE = True
except ImportError:
    PVLIB_AVAILABLE = False
    warnings.warn("pvlib not available. Using fallback calculations.")


class SunPositionCalculator:
    """Calculator for sun position and solar radiation features."""
    
    def __init__(self, latitude: float = 37.7749, longitude: float = -122.4194,
                 altitude: float = 10, timezone: str = "UTC"):
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
        self.timezone = timezone
        
        if PVLIB_AVAILABLE:
            self.location = Location(latitude=latitude, longitude=longitude,
                                     altitude=altitude, tz=timezone)
    
    def get_sun_position(self, timestamp: Union[datetime, pd.Timestamp, str]) -> Dict[str, float]:
        if isinstance(timestamp, str):
            timestamp = pd.Timestamp(timestamp)
        elif isinstance(timestamp, datetime):
            timestamp = pd.Timestamp(timestamp)
        
        if PVLIB_AVAILABLE:
            return self._get_position_pvlib(timestamp)
        return self._get_position_fallback(timestamp)
    
    def _get_position_pvlib(self, timestamp: pd.Timestamp) -> Dict[str, float]:
        times = pd.DatetimeIndex([timestamp])
        if times.tz is None:
            times = times.tz_localize(self.timezone)
        
        solar_pos = get_solarposition(times, self.latitude, self.longitude, altitude=self.altitude)
        
        return {
            "zenith": solar_pos["zenith"].iloc[0],
            "azimuth": solar_pos["azimuth"].iloc[0],
            "apparent_elevation": solar_pos["apparent_elevation"].iloc[0],
            "equation_of_time": solar_pos["equation_of_time"].iloc[0]
        }
    
    def _get_position_fallback(self, timestamp: pd.Timestamp) -> Dict[str, float]:
        day_of_year = timestamp.dayofyear
        hour = timestamp.hour + timestamp.minute / 60
        hour_angle = 15 * (hour - 12)
        
        declination = -23.45 * np.cos(np.radians(360 / 365 * (day_of_year + 10)))
        
        lat_rad = np.radians(self.latitude)
        dec_rad = np.radians(declination)
        hour_rad = np.radians(hour_angle)
        
        elevation = np.degrees(np.arcsin(
            np.sin(lat_rad) * np.sin(dec_rad) +
            np.cos(lat_rad) * np.cos(dec_rad) * np.cos(hour_rad)
        ))
        
        zenith = 90 - elevation
        azimuth = 180 + hour_angle if hour_angle < 0 else 180 - hour_angle
        
        return {
            "zenith": zenith,
            "azimuth": azimuth,
            "apparent_elevation": elevation,
            "equation_of_time": 0.0
        }
    
    def get_features_array(self, timestamp: Union[datetime, pd.Timestamp]) -> np.ndarray:
        pos = self.get_sun_position(timestamp)
        return np.array([pos["zenith"], pos["azimuth"], 
                        pos["apparent_elevation"], pos["equation_of_time"]])
    
    def is_daytime(self, timestamp: Union[datetime, pd.Timestamp]) -> bool:
        pos = self.get_sun_position(timestamp)
        return pos["apparent_elevation"] > 0
    
    def get_clear_sky_irradiance(self, timestamp: Union[datetime, pd.Timestamp]) -> Dict[str, float]:
        if not PVLIB_AVAILABLE:
            pos = self.get_sun_position(timestamp)
            elevation = max(0, pos["apparent_elevation"])
            ghi = 1000 * np.sin(np.radians(elevation))
            return {"ghi": ghi, "dni": ghi * 0.8, "dhi": ghi * 0.2}
        
        times = pd.DatetimeIndex([pd.Timestamp(timestamp)])
        if times.tz is None:
            times = times.tz_localize(self.timezone)
        
        clear_sky = self.location.get_clearsky(times)
        return {
            "ghi": clear_sky["ghi"].iloc[0],
            "dni": clear_sky["dni"].iloc[0],
            "dhi": clear_sky["dhi"].iloc[0]
        }
