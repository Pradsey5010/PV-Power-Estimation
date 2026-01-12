from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from sky_power.features import FeatureSpec, build_feature_frame


@dataclass(frozen=True)
class Sample:
    images: torch.Tensor  # (T, 3, H, W)
    features: torch.Tensor  # (T, F)
    target: torch.Tensor  # (T,)  (sequence-to-sequence) OR (1,) depending on mode


def _imread_rgb(path: str) -> np.ndarray:
    import cv2

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def _resize(img: np.ndarray, size: int) -> np.ndarray:
    import cv2

    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


class SequenceSkyPowerDataset(Dataset):
    """
    Expects a CSV with at least:
    - timestamp column (ISO8601 recommended; will be parsed by pandas)
    - image path column (relative to images_dir or absolute)
    - target column (dc_power)
    - one or more sensor columns (weather/irradiance/etc.)

    Returns sliding windows of length seq_len with stride.
    """

    def __init__(
        self,
        csv_path: str,
        images_dir: str,
        feature_spec: FeatureSpec,
        timestamp_col: str = "timestamp",
        image_col: str = "image_path",
        target_col: str = "dc_power",
        latitude: float = 0.0,
        longitude: float = 0.0,
        tz: str = "UTC",
        seq_len: int = 8,
        stride: int = 1,
        image_size: int = 224,
        return_sequence_target: bool = False,
    ) -> None:
        self.csv_path = str(csv_path)
        self.images_dir = str(images_dir)
        self.feature_spec = feature_spec
        self.timestamp_col = timestamp_col
        self.image_col = image_col
        self.target_col = target_col
        self.latitude = float(latitude)
        self.longitude = float(longitude)
        self.tz = tz
        self.seq_len = int(seq_len)
        self.stride = int(stride)
        self.image_size = int(image_size)
        self.return_sequence_target = bool(return_sequence_target)

        df = pd.read_csv(self.csv_path)
        required = {self.timestamp_col, self.image_col, self.target_col}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"CSV missing required columns: {sorted(missing)}")

        # stable order
        df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col], utc=True)
        df = df.sort_values(self.timestamp_col).reset_index(drop=True)

        self._df = df
        self._features = build_feature_frame(
            df=df,
            timestamp_col=self.timestamp_col,
            sensor_cols=self.feature_spec.sensor_cols,
            latitude=self.latitude,
            longitude=self.longitude,
            tz=self.tz,
            add_time_features=self.feature_spec.add_time_features,
            add_sun_features=self.feature_spec.add_sun_features,
        ).to_numpy(dtype=np.float32)
        self._targets = df[self.target_col].to_numpy(dtype=np.float32)
        self._image_paths = df[self.image_col].astype(str).tolist()

        n = len(df)
        if n < self.seq_len:
            raise ValueError(f"Need at least seq_len={self.seq_len} rows, got {n}.")

        self._starts = list(range(0, n - self.seq_len + 1, self.stride))

    def __len__(self) -> int:
        return len(self._starts)

    def _resolve_image_path(self, p: str) -> str:
        path = Path(p)
        if path.is_absolute():
            return str(path)
        return str(Path(self.images_dir) / p)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        s = self._starts[idx]
        e = s + self.seq_len

        feats = torch.from_numpy(self._features[s:e])  # (T,F)
        y = torch.from_numpy(self._targets[s:e])  # (T,)

        # images: (T,3,H,W)
        imgs = []
        for p in self._image_paths[s:e]:
            ip = self._resolve_image_path(p)
            img = _imread_rgb(ip)
            img = _resize(img, self.image_size)
            img = (img.astype(np.float32) / 255.0).transpose(2, 0, 1)  # CHW
            imgs.append(torch.from_numpy(img))
        images = torch.stack(imgs, dim=0)

        if not self.return_sequence_target:
            y = y[-1:].clone()  # predict final timestep power by default

        return {
            "images": images,
            "features": feats,
            "target": y,
        }


def infer_sensor_columns(
    csv_path: str,
    timestamp_col: str,
    image_col: str,
    target_col: str,
    exclude: Optional[list[str]] = None,
) -> list[str]:
    df = pd.read_csv(csv_path, nrows=1)
    exclude_set = {timestamp_col, image_col, target_col}
    if exclude:
        exclude_set |= set(exclude)
    cols = [c for c in df.columns if c not in exclude_set]
    return cols

