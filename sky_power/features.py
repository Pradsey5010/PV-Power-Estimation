from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureSpec:
    sensor_cols: list[str]
    add_time_features: bool = True
    add_sun_features: bool = True


def _cyclical_time_features(ts: pd.DatetimeIndex) -> pd.DataFrame:
    # hour-of-day and day-of-year cyclic encodings
    hour = ts.hour.values.astype(np.float32)
    doy = ts.dayofyear.values.astype(np.float32)
    hour_rad = 2.0 * np.pi * hour / 24.0
    doy_rad = 2.0 * np.pi * doy / 365.25
    return pd.DataFrame(
        {
            "hour_sin": np.sin(hour_rad),
            "hour_cos": np.cos(hour_rad),
            "doy_sin": np.sin(doy_rad),
            "doy_cos": np.cos(doy_rad),
        },
        index=ts,
    )


def add_solar_position_features(
    df: pd.DataFrame,
    timestamp_col: str,
    latitude: float,
    longitude: float,
    tz: str,
) -> pd.DataFrame:
    """
    Adds pvlib-derived sun position features:
    - solar_zenith, solar_azimuth
    - apparent_zenith, apparent_elevation
    - cos_zenith (handy for learning irradiance->power)
    """
    try:
        from pvlib.solarposition import get_solarposition
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "pvlib is required for solar position features. Install pvlib."
        ) from e

    out = df.copy()
    ts = pd.to_datetime(out[timestamp_col], utc=True)
    ts = ts.tz_convert(tz)

    sp = get_solarposition(ts, latitude=latitude, longitude=longitude)
    out["solar_zenith"] = sp["zenith"].astype(np.float32)
    out["solar_azimuth"] = sp["azimuth"].astype(np.float32)
    out["apparent_zenith"] = sp["apparent_zenith"].astype(np.float32)
    out["apparent_elevation"] = sp["apparent_elevation"].astype(np.float32)
    out["cos_zenith"] = np.cos(np.deg2rad(out["solar_zenith"].values)).astype(np.float32)
    return out


def build_feature_frame(
    df: pd.DataFrame,
    timestamp_col: str,
    sensor_cols: Iterable[str],
    latitude: float,
    longitude: float,
    tz: str,
    add_time_features: bool = True,
    add_sun_features: bool = True,
) -> pd.DataFrame:
    out = df.copy()

    if add_sun_features:
        out = add_solar_position_features(
            out,
            timestamp_col=timestamp_col,
            latitude=latitude,
            longitude=longitude,
            tz=tz,
        )

    features = []
    sensor_cols = list(sensor_cols)
    if len(sensor_cols) == 0:
        raise ValueError("sensor_cols must contain at least one sensor column.")
    features.append(out[sensor_cols].astype(np.float32))

    ts = pd.to_datetime(out[timestamp_col], utc=True).tz_convert(tz)
    if add_time_features:
        features.append(_cyclical_time_features(ts))

    if add_sun_features:
        features.append(
            out[
                [
                    "solar_zenith",
                    "solar_azimuth",
                    "apparent_zenith",
                    "apparent_elevation",
                    "cos_zenith",
                ]
            ].astype(np.float32)
        )

    feat = pd.concat(features, axis=1)
    feat = feat.replace([np.inf, -np.inf], np.nan).fillna(method="ffill").fillna(0.0)
    return feat

