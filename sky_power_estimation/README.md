# 🌤️ PV Power Estimation

**Multi-Modal Deep Learning for Solar Power Prediction**

An interactive dashboard and deep learning system for predicting DC power output from solar PV installations.

## 🚀 Quick Start

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Launch Dashboard

```bash
# Option 1: Using the launcher script
python run_dashboard.py

# Option 2: Directly with Streamlit
streamlit run dashboard/app.py
```

Then open your browser to: **http://localhost:8501**

## ✨ Features

### Core Features
- 🖼️ **Multi-Modal Learning**: Combines images, weather, and sun position
- 🧠 **Flexible Backbones**: 10+ CNN architectures (ResNet, MobileNet, EfficientNet, ViT)
- ⏱️ **Temporal Modeling**: LSTM/Transformer for sequence patterns
- 🔗 **Attention Fusion**: Dynamic modality weighting
- 📈 **Uncertainty Estimation**: Monte Carlo dropout for confidence intervals

### Training Features
- ⚡ **Mixed Precision**: AMP for faster training
- 📉 **LR Scheduling**: Cosine, Step, Plateau, OneCycle
- 🛑 **Early Stopping**: Prevent overfitting
- 📊 **TensorBoard**: Experiment tracking

### Inference Features
- 🔮 **Real-Time Prediction**: Efficient inference pipeline
- 📦 **Batch Processing**: Process multiple images
- 🔄 **Sequence Buffering**: Automatic history management

### Data Processing
- ☀️ **Sun Position**: pvlib integration
- 🌡️ **Weather Normalization**: StandardScaler/MinMax
- 🖼️ **Image Augmentation**: Albumentations
- ☁️ **Cloud Detection**: OpenCV-based analysis

## 🏗️ Project Structure

```
sky_power_estimation/
├── dashboard/
│   └── app.py              # Streamlit dashboard
├── sky_power_estimation/
│   ├── models/             # Neural network models
│   ├── utils/              # Utilities (sun, weather, image)
│   └── ...
├── requirements.txt
├── run_dashboard.py        # Dashboard launcher
└── README.md
```

## 📊 Dashboard Tabs

| Tab | Description |
|-----|-------------|
| 🏠 **Home** | Overview, features, architecture diagram |
| 🔮 **Prediction** | Run inference on sky images |
| 🎯 **Training** | Training simulation with live metrics |
| 📊 **Analytics** | Power generation analytics |
| ☁️ **Cloud Analysis** | Cloud detection and segmentation |
| ☀️ **Sun Position** | Solar geometry calculator |

## 🛠️ Tech Stack

- **Deep Learning**: PyTorch, timm
- **Dashboard**: Streamlit, Plotly
- **Computer Vision**: OpenCV, Pillow
- **Solar**: pvlib
- **Scientific**: NumPy, Pandas, scikit-learn

## 📄 License

MIT License
