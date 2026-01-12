from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DataConfig:
    csv_path: str
    images_dir: str
    image_key: str = "image_path"
    timestamp_key: str = "timestamp"
    target_key: str = "dc_power"
    latitude: float = 0.0
    longitude: float = 0.0
    tz: str = "UTC"
    seq_len: int = 8
    stride: int = 1
    image_size: int = 224


@dataclass(frozen=True)
class ModelConfig:
    image_backbone: str = "resnet18"  # resnet18 | mobilenet_v3_small
    pretrained: bool = True
    image_embed_dim: int = 256
    sensor_embed_dim: int = 64
    fusion_dim: int = 256
    sequence_model: str = "lstm"  # lstm | transformer
    lstm_hidden: int = 256
    lstm_layers: int = 2
    transformer_layers: int = 2
    transformer_heads: int = 4
    dropout: float = 0.1


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42
    epochs: int = 10
    batch_size: int = 16
    lr: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = 2
    device: str = "auto"  # auto | cpu | cuda
    log_every: int = 25
    val_split: float = 0.2
    shuffle_train: bool = True
    output_dir: str = "artifacts"

