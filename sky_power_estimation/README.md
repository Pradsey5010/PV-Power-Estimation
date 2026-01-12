# Sky Image + Weather-Based DC Power Estimation

A multi-modal deep learning system for predicting DC power output from solar installations using sky images, weather sensor data, and sun position information.

![Architecture](https://img.shields.io/badge/Architecture-Multi--Modal-blue)
![Framework](https://img.shields.io/badge/Framework-PyTorch-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## Overview

This project implements a state-of-the-art approach to solar power forecasting by combining:

- **Sky Images**: CNN-based analysis of cloud cover, opacity, and atmospheric conditions
- **Weather Data**: Temperature, humidity, pressure, wind, and irradiance measurements
- **Sun Position**: Accurate solar geometry calculations using pvlib
- **Temporal Modeling**: LSTM/Transformer for capturing time-dependent patterns

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Sky Power Estimation Model                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │
│  │  Sky Images  │  │Weather Data  │  │ Sun Position │               │
│  │   (RGB)      │  │  (Sensors)   │  │  (pvlib)     │               │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘               │
│         │                 │                 │                        │
│         ▼                 ▼                 ▼                        │
│  ┌──────────────┐  ┌──────────────────────────┐                     │
│  │ CNN Encoder  │  │   Weather + Sun Encoder  │                     │
│  │ (ResNet/     │  │      (MLP + Norm)        │                     │
│  │  MobileNet)  │  └──────────┬───────────────┘                     │
│  └──────┬───────┘             │                                      │
│         │                     │                                      │
│         └─────────┬───────────┘                                      │
│                   ▼                                                  │
│         ┌──────────────────┐                                         │
│         │ Temporal Encoder │                                         │
│         │ (Transformer/    │                                         │
│         │      LSTM)       │                                         │
│         └────────┬─────────┘                                         │
│                  ▼                                                   │
│         ┌──────────────────┐                                         │
│         │  Fusion Layer    │                                         │
│         │ (Attention/      │                                         │
│         │  Gated/Concat)   │                                         │
│         └────────┬─────────┘                                         │
│                  ▼                                                   │
│         ┌──────────────────┐                                         │
│         │   Output Head    │                                         │
│         │  (DC Power W)    │                                         │
│         └──────────────────┘                                         │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

## Features

### Image Encoders
- **ResNet** (18, 34, 50, 101): Deep residual networks
- **MobileNetV3** (Small, Large): Efficient mobile architectures
- **EfficientNet** (B0, B1, B2): Compound scaling networks
- **ConvNeXt**: Modern ConvNet architecture

### Temporal Encoders
- **Transformer**: Self-attention for long-range dependencies
- **LSTM**: Bidirectional LSTM with attention aggregation
- **GRU**: Efficient gated recurrent units

### Fusion Methods
- **Attention Fusion**: Cross-attention for dynamic modality weighting
- **Gated Fusion**: Learned gates for information flow control
- **Bilinear Fusion**: Multiplicative interactions
- **Concatenation**: Simple feature concatenation

### Additional Features
- Mixed precision training (AMP)
- Multi-scale image encoding
- Monte Carlo dropout uncertainty estimation
- Cloud feature extraction (OpenCV)
- Real-time prediction with buffering

## Installation

### Requirements
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU support)

### Install Dependencies

```bash
cd sky_power_estimation
pip install -r requirements.txt
```

### Install Package (Development Mode)

```bash
pip install -e .
```

## Quick Start

### Training with Synthetic Data

```bash
python scripts/train.py --use-synthetic --epochs 50 --batch-size 16
```

### Training with Real Data

```bash
python scripts/train.py \
    --data-dir /path/to/data \
    --config configs/config.yaml \
    --epochs 100 \
    --batch-size 32 \
    --backbone resnet50 \
    --temporal-encoder transformer
```

### Inference on Images

```bash
python scripts/predict.py \
    --model checkpoints/best_model.pt \
    --temperature 25 \
    --humidity 60 \
    /path/to/sky_images/
```

### Evaluation

```bash
python scripts/evaluate.py \
    --model checkpoints/best_model.pt \
    --data-dir /path/to/test_data \
    --output predictions.csv
```

## Data Format

### Directory Structure

```
data/
├── images/
│   ├── 20240101_120000.jpg
│   ├── 20240101_120500.jpg
│   └── ...
└── annotations.csv
```

### Annotations CSV Format

| Column | Description | Unit |
|--------|-------------|------|
| timestamp | ISO format datetime | - |
| dc_power | DC power output | W |
| temperature | Ambient temperature | °C |
| humidity | Relative humidity | % |
| pressure | Atmospheric pressure | hPa |
| wind_speed | Wind speed | m/s |
| wind_direction | Wind direction | degrees |
| ghi | Global Horizontal Irradiance | W/m² |
| dni | Direct Normal Irradiance | W/m² |
| dhi | Diffuse Horizontal Irradiance | W/m² |
| image_file | Image filename | - |

## Configuration

### Model Configuration (config.yaml)

```yaml
model:
  image_encoder:
    backbone: resnet50
    pretrained: true
    freeze_backbone: false
    image_feature_dim: 512
  
  temporal_encoder:
    type: transformer
    hidden_dim: 256
    num_layers: 3
    num_heads: 8
    dropout: 0.1
  
  fusion:
    method: attention
    hidden_dim: 512
    dropout: 0.2

training:
  batch_size: 32
  epochs: 100
  learning_rate: 0.001
  weight_decay: 0.0001
  scheduler:
    type: cosine
    warmup_epochs: 5
```

## Python API

### Training

```python
from sky_power_estimation.models import SkyPowerModel
from sky_power_estimation.data import create_dataloaders
from sky_power_estimation.training import Trainer

# Create model
model = SkyPowerModel(
    image_backbone="resnet50",
    temporal_type="transformer",
    fusion_method="attention"
)

# Create dataloaders
dataloaders = create_dataloaders(
    data_dir="./data",
    batch_size=32,
    sequence_length=12
)

# Train
trainer = Trainer(
    model=model,
    train_loader=dataloaders["train"],
    val_loader=dataloaders["val"],
    config=config
)

results = trainer.train()
```

### Inference

```python
from sky_power_estimation.inference import Predictor

# Load predictor
predictor = Predictor(
    model_path="checkpoints/best_model.pt",
    location={"latitude": 37.77, "longitude": -122.42}
)

# Single prediction
power = predictor.predict(
    image="sky_image.jpg",
    weather={"temperature": 25, "humidity": 60, "ghi": 800},
    timestamp=datetime.now()
)

# With uncertainty
mean, std = predictor.predict_with_uncertainty(
    image="sky_image.jpg",
    weather=weather,
    num_samples=20
)
print(f"Power: {mean:.1f} ± {std:.1f} W")
```

### Real-Time Predictions

```python
from sky_power_estimation.inference import RealTimePredictor

# Create real-time predictor
predictor = RealTimePredictor(
    model_path="model.pt",
    sequence_length=12
)

# Continuous prediction loop
while True:
    image = capture_sky_image()
    weather = read_weather_sensors()
    
    power = predictor.predict_with_update(
        image=image,
        weather=weather
    )
    
    print(f"Current power: {power:.1f} W")
```

## Sky Image Processing

### Cloud Feature Extraction

```python
from sky_power_estimation.utils import ImageProcessor

processor = ImageProcessor()

features = processor.extract_cloud_features(image)
# Returns: cloud_cover, brightness, blue_ratio, contrast, entropy

cloud_mask = processor.segment_clouds(image, method="otsu")
opacity = processor.estimate_opacity(image, cloud_mask)
```

## Sun Position Calculation

```python
from sky_power_estimation.utils import SunPositionCalculator

calc = SunPositionCalculator(
    latitude=37.77,
    longitude=-122.42,
    timezone="US/Pacific"
)

# Get sun position
pos = calc.get_sun_position(datetime.now())
print(f"Zenith: {pos['zenith']:.1f}°")
print(f"Azimuth: {pos['azimuth']:.1f}°")

# Get clear sky irradiance
irr = calc.get_clear_sky_irradiance(datetime.now())
print(f"Clear sky GHI: {irr['ghi']:.1f} W/m²")
```

## Metrics

The system reports comprehensive evaluation metrics:

| Metric | Description |
|--------|-------------|
| RMSE | Root Mean Squared Error |
| MAE | Mean Absolute Error |
| MAPE | Mean Absolute Percentage Error |
| R² | Coefficient of Determination |
| nRMSE | Normalized RMSE |
| MBE | Mean Bias Error |
| Skill Score | Improvement over persistence |

## Model Checkpoints

Checkpoints include:
- Model state dictionary
- Optimizer state
- Scheduler state
- Training configuration
- Best validation metrics

## Datasets

Compatible with common sky image datasets:

1. **NREL Sky Image Archive**
2. **SURFRAD Network Data**
3. **ARM Climate Research Facility**
4. **Custom fisheye camera datasets**

## References

- [Solar Forecasting with Deep Learning](https://arxiv.org/abs/2003.12875)
- [Cloud Segmentation from Sky Images](https://arxiv.org/abs/1904.03565)
- [pvlib: Python Library for Solar Energy](https://pvlib-python.readthedocs.io/)

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests.

## Citation

```bibtex
@software{sky_power_estimation,
  title={Sky Image + Weather-Based DC Power Estimation},
  author={Sky Power Estimation Team},
  year={2024},
  url={https://github.com/example/sky-power-estimation}
}
```
