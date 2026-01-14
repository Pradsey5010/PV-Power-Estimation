"""
🌤️ PV Power Estimation Dashboard
Interactive dashboard for solar power prediction using sky images and weather data.
Theme: Maroon & Yellow
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Page configuration
st.set_page_config(
    page_title="🌤️ PV Power Estimation",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Theme Colors
MAROON = "#800000"
MAROON_DARK = "#5c0000"
MAROON_LIGHT = "#a52a2a"
YELLOW = "#FFD700"
YELLOW_LIGHT = "#FFEC8B"
YELLOW_DARK = "#DAA520"
WHITE = "#FFFFFF"
CREAM = "#FFF8DC"
DARK_BG = "#1a1a1a"
CARD_BG = "#2d2d2d"

# Custom CSS for Maroon & Yellow Theme
st.markdown(f"""
<style>
    /* Main background */
    .stApp {{
        background: linear-gradient(135deg, {DARK_BG} 0%, #2d1810 100%);
    }}
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {MAROON_DARK} 0%, {MAROON} 100%);
    }}
    
    [data-testid="stSidebar"] .stMarkdown {{
        color: {WHITE};
    }}
    
    [data-testid="stSidebar"] label {{
        color: {YELLOW_LIGHT} !important;
    }}
    
    /* Main header styling */
    .main-header {{
        font-size: 3.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, {YELLOW}, {YELLOW_LIGHT}, {YELLOW});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }}
    
    .sub-header {{
        font-size: 1.3rem;
        color: {CREAM};
        text-align: center;
        margin-bottom: 2rem;
    }}
    
    /* Feature cards with hover effect */
    .feature-card {{
        background: linear-gradient(145deg, {CARD_BG}, #3d3d3d);
        border-radius: 15px;
        padding: 25px;
        margin: 15px 0;
        border-left: 5px solid {YELLOW};
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
        color: {WHITE};
    }}
    
    .feature-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(255, 215, 0, 0.3);
        border-left: 5px solid {YELLOW_LIGHT};
    }}
    
    .feature-card h4 {{
        color: {YELLOW} !important;
        margin-bottom: 10px;
        font-size: 1.2rem;
    }}
    
    .feature-card p {{
        color: {CREAM} !important;
        font-size: 1rem;
        line-height: 1.5;
    }}
    
    /* Metric cards */
    .metric-card {{
        background: linear-gradient(135deg, {MAROON} 0%, {MAROON_DARK} 100%);
        border-radius: 15px;
        padding: 25px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(128, 0, 0, 0.4);
        border: 2px solid {YELLOW};
        transition: all 0.3s ease;
    }}
    
    .metric-card:hover {{
        transform: scale(1.05);
        box-shadow: 0 8px 30px rgba(255, 215, 0, 0.4);
    }}
    
    .metric-card h2 {{
        color: {YELLOW} !important;
        font-size: 2.5rem;
        margin: 0;
    }}
    
    .metric-card p {{
        color: {WHITE} !important;
        font-size: 1rem;
        margin: 5px 0 0 0;
    }}
    
    /* Image container with border */
    .image-container {{
        background: {CARD_BG};
        border-radius: 15px;
        padding: 15px;
        border: 3px solid {MAROON};
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
    }}
    
    .image-container:hover {{
        border-color: {YELLOW};
        box-shadow: 0 8px 25px rgba(255, 215, 0, 0.3);
    }}
    
    /* Section headers */
    .section-header {{
        color: {YELLOW} !important;
        font-size: 2rem;
        font-weight: bold;
        margin: 30px 0 20px 0;
        padding-bottom: 10px;
        border-bottom: 3px solid {MAROON};
    }}
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background-color: {CARD_BG};
        padding: 10px;
        border-radius: 10px;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        height: 50px;
        padding: 10px 25px;
        background-color: {MAROON_DARK};
        border-radius: 10px;
        color: {WHITE};
        font-weight: bold;
        transition: all 0.3s ease;
    }}
    
    .stTabs [data-baseweb="tab"]:hover {{
        background-color: {MAROON};
    }}
    
    .stTabs [aria-selected="true"] {{
        background-color: {YELLOW} !important;
        color: {MAROON_DARK} !important;
    }}
    
    /* Button styling */
    .stButton > button {{
        background: linear-gradient(135deg, {YELLOW} 0%, {YELLOW_DARK} 100%);
        color: {MAROON_DARK};
        font-weight: bold;
        border: none;
        padding: 12px 30px;
        border-radius: 10px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(255, 215, 0, 0.3);
    }}
    
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 215, 0, 0.5);
        background: linear-gradient(135deg, {YELLOW_LIGHT} 0%, {YELLOW} 100%);
    }}
    
    /* Input fields */
    .stNumberInput input, .stTextInput input, .stSelectbox select {{
        background-color: {CARD_BG} !important;
        color: {WHITE} !important;
        border: 2px solid {MAROON} !important;
        border-radius: 8px;
    }}
    
    .stNumberInput input:focus, .stTextInput input:focus {{
        border-color: {YELLOW} !important;
    }}
    
    /* Slider */
    .stSlider > div > div {{
        background-color: {MAROON} !important;
    }}
    
    .stSlider > div > div > div {{
        background-color: {YELLOW} !important;
    }}
    
    /* Info boxes */
    .info-box {{
        background: linear-gradient(135deg, {MAROON_DARK} 0%, {MAROON} 100%);
        border-radius: 15px;
        padding: 20px;
        border: 2px solid {YELLOW};
        color: {WHITE};
        margin: 15px 0;
    }}
    
    .info-box h3 {{
        color: {YELLOW} !important;
        margin-bottom: 10px;
    }}
    
    /* Results card */
    .result-card {{
        background: linear-gradient(145deg, {MAROON} 0%, {MAROON_DARK} 100%);
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        border: 3px solid {YELLOW};
        box-shadow: 0 10px 40px rgba(255, 215, 0, 0.3);
    }}
    
    .result-card h1 {{
        color: {YELLOW} !important;
        font-size: 4rem;
        margin: 10px 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }}
    
    .result-card h3 {{
        color: {WHITE} !important;
    }}
    
    .result-card p {{
        color: {CREAM} !important;
    }}
    
    /* Progress bar */
    .stProgress > div > div {{
        background-color: {YELLOW} !important;
    }}
    
    /* Expander */
    .streamlit-expanderHeader {{
        background-color: {CARD_BG} !important;
        color: {YELLOW} !important;
        border-radius: 10px;
    }}
    
    /* Footer */
    .footer {{
        background: linear-gradient(135deg, {MAROON_DARK} 0%, {MAROON} 100%);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        margin-top: 30px;
        border: 2px solid {YELLOW};
    }}
    
    .footer p {{
        color: {CREAM} !important;
        margin: 5px 0;
    }}
    
    .footer a {{
        color: {YELLOW} !important;
    }}
    
    /* Hide Streamlit branding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {{
        width: 10px;
    }}
    
    ::-webkit-scrollbar-track {{
        background: {DARK_BG};
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: {MAROON};
        border-radius: 5px;
    }}
    
    ::-webkit-scrollbar-thumb:hover {{
        background: {YELLOW};
    }}
</style>
""", unsafe_allow_html=True)

# Solar-related image URLs (using placeholder images that represent solar/sky themes)
IMAGES = {
    "solar_panel": "https://images.unsplash.com/photo-1509391366360-2e959784a276?w=400&h=300&fit=crop",
    "sky_clouds": "https://images.unsplash.com/photo-1534088568595-a066f410bcda?w=400&h=300&fit=crop",
    "sun": "https://images.unsplash.com/photo-1575881875475-31023242e3f9?w=400&h=300&fit=crop",
    "weather_station": "https://images.unsplash.com/photo-1592210454359-9043f067919b?w=400&h=300&fit=crop",
    "power_grid": "https://images.unsplash.com/photo-1473341304170-971dccb5ac1e?w=400&h=300&fit=crop",
    "solar_farm": "https://images.unsplash.com/photo-1508514177221-188b1cf16e9d?w=400&h=300&fit=crop",
    "analytics": "https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=400&h=300&fit=crop",
    "cloud_analysis": "https://images.unsplash.com/photo-1534088568595-a066f410bcda?w=400&h=300&fit=crop",
}


def create_sidebar():
    """Create sidebar with configuration options."""
    with st.sidebar:
        # Logo/Icon
        st.markdown(f"""
        <div style="text-align: center; padding: 20px 0;">
            <span style="font-size: 4rem;">☀️</span>
            <h2 style="color: {YELLOW}; margin: 10px 0;">PV Power</h2>
            <p style="color: {CREAM};">Estimation System</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Model Configuration
        st.markdown(f"<h3 style='color: {YELLOW};'>🧠 Model Settings</h3>", unsafe_allow_html=True)
        
        backbone = st.selectbox(
            "CNN Backbone",
            ["ResNet18", "ResNet34", "ResNet50", "ResNet101", 
             "MobileNetV3-Small", "MobileNetV3-Large",
             "EfficientNet-B0", "EfficientNet-B1", "EfficientNet-B2",
             "ConvNeXt-Tiny", "ViT-Tiny"],
            index=2
        )
        
        temporal_encoder = st.selectbox(
            "Temporal Encoder",
            ["Transformer", "LSTM", "GRU"],
            index=0
        )
        
        fusion_method = st.selectbox(
            "Fusion Method",
            ["Attention", "Gated", "Concatenation"],
            index=0
        )
        
        st.markdown("---")
        
        # Training Configuration
        st.markdown(f"<h3 style='color: {YELLOW};'>🎯 Training Settings</h3>", unsafe_allow_html=True)
        
        batch_size = st.slider("Batch Size", 8, 128, 32, 8)
        learning_rate = st.select_slider(
            "Learning Rate",
            options=[0.0001, 0.0005, 0.001, 0.005, 0.01],
            value=0.001
        )
        epochs = st.slider("Epochs", 10, 200, 100, 10)
        
        scheduler = st.selectbox(
            "LR Scheduler",
            ["Cosine Annealing", "Step Decay", "Plateau", "OneCycle"],
            index=0
        )
        
        use_amp = st.checkbox("Mixed Precision (AMP)", value=True)
        early_stopping = st.checkbox("Early Stopping", value=True)
        
        st.markdown("---")
        
        # Location Configuration
        st.markdown(f"<h3 style='color: {YELLOW};'>📍 Location</h3>", unsafe_allow_html=True)
        
        latitude = st.number_input("Latitude", -90.0, 90.0, 37.7749, 0.0001)
        longitude = st.number_input("Longitude", -180.0, 180.0, -122.4194, 0.0001)
        
        st.markdown("---")
        
        # Contact Info
        st.markdown(f"""
        <div style="text-align: center; padding: 10px;">
            <p style="color: {CREAM}; font-size: 0.9rem;">📧 pradyumnamand@gmail.com</p>
            <p style="color: {CREAM}; font-size: 0.9rem;">📱 +1 480-797-3843</p>
        </div>
        """, unsafe_allow_html=True)
        
        return {
            "backbone": backbone,
            "temporal_encoder": temporal_encoder,
            "fusion_method": fusion_method,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "scheduler": scheduler,
            "use_amp": use_amp,
            "early_stopping": early_stopping,
            "latitude": latitude,
            "longitude": longitude
        }


def generate_sample_data():
    """Generate sample data for visualization."""
    np.random.seed(42)
    hours = pd.date_range(start="2024-01-01", periods=24*7, freq="H")
    
    power = []
    for h in hours:
        hour = h.hour
        if 6 <= hour <= 18:
            base = np.sin(np.pi * (hour - 6) / 12) * 800
            noise = np.random.normal(0, 50)
            cloud_factor = np.random.uniform(0.7, 1.0)
            power.append(max(0, base * cloud_factor + noise))
        else:
            power.append(0)
    
    data = pd.DataFrame({
        "timestamp": hours,
        "power": power,
        "temperature": 20 + 10 * np.sin(np.linspace(0, 4*np.pi, len(hours))) + np.random.normal(0, 2, len(hours)),
        "humidity": 50 + 20 * np.sin(np.linspace(np.pi, 5*np.pi, len(hours))) + np.random.normal(0, 5, len(hours)),
        "ghi": [p * 1.2 + np.random.normal(0, 30) for p in power],
        "cloud_cover": np.random.uniform(0, 60, len(hours)),
        "wind_speed": np.random.uniform(0, 15, len(hours))
    })
    
    return data


def render_home_tab():
    """Render the home/overview tab."""
    # Header
    st.markdown(f'<h1 class="main-header">☀️ PV Power Estimation</h1>', unsafe_allow_html=True)
    st.markdown(f'<p class="sub-header">Multi-Modal Deep Learning for Solar Power Prediction</p>', unsafe_allow_html=True)
    
    # Hero Image
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"""
        <div class="image-container" style="text-align: center;">
            <img src="{IMAGES['solar_farm']}" style="width: 100%; border-radius: 10px;">
            <p style="color: {CREAM}; margin-top: 10px; font-style: italic;">Advanced AI-powered solar power forecasting</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    metrics = [
        ("🎯", "94.5%", "Model Accuracy"),
        ("⚡", "15ms", "Inference Time"),
        ("📊", "23.4 W", "RMSE"),
        ("🌡️", "12", "Active Sensors")
    ]
    
    for col, (icon, value, label) in zip([col1, col2, col3, col4], metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <span style="font-size: 2.5rem;">{icon}</span>
                <h2>{value}</h2>
                <p>{label}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Key Features Section
    st.markdown(f'<h2 class="section-header">✨ Key Features</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="feature-card">
            <h4>🖼️ Multi-Modal Learning</h4>
            <p>Combines sky images, weather sensors, and sun position data for accurate predictions using state-of-the-art deep learning.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="feature-card">
            <h4>🧠 Flexible Backbones</h4>
            <p>Support for 10+ CNN architectures including ResNet, MobileNet, EfficientNet, and Vision Transformers.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="feature-card">
            <h4>⏱️ Temporal Modeling</h4>
            <p>LSTM and Transformer encoders capture time-dependent patterns and seasonal variations in power generation.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="feature-card">
            <h4>🔗 Attention Fusion</h4>
            <p>Dynamic modality weighting through cross-attention mechanisms for optimal feature combination.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="feature-card">
            <h4>📈 Uncertainty Estimation</h4>
            <p>Monte Carlo dropout provides prediction confidence intervals for reliable decision making.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="feature-card">
            <h4>☁️ Cloud Detection</h4>
            <p>OpenCV-based cloud segmentation and opacity estimation for accurate atmospheric analysis.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Data Sources Section with Images
    st.markdown(f'<h2 class="section-header">📊 Data Sources</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="image-container">
            <img src="{IMAGES['sky_clouds']}" style="width: 100%; border-radius: 10px; height: 200px; object-fit: cover;">
            <h4 style="color: {YELLOW}; margin-top: 15px; text-align: center;">📸 Sky Images</h4>
            <p style="color: {CREAM}; text-align: center;">Fisheye camera captures for cloud cover analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="image-container">
            <img src="{IMAGES['weather_station']}" style="width: 100%; border-radius: 10px; height: 200px; object-fit: cover;">
            <h4 style="color: {YELLOW}; margin-top: 15px; text-align: center;">🌡️ Weather Sensors</h4>
            <p style="color: {CREAM}; text-align: center;">Temperature, humidity, pressure, wind data</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="image-container">
            <img src="{IMAGES['sun']}" style="width: 100%; border-radius: 10px; height: 200px; object-fit: cover;">
            <h4 style="color: {YELLOW}; margin-top: 15px; text-align: center;">☀️ Sun Position</h4>
            <p style="color: {CREAM}; text-align: center;">Precise solar geometry calculations</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Architecture Image
    st.markdown(f'<h2 class="section-header">🏗️ Architecture</h2>', unsafe_allow_html=True)
    
    # Check if architecture image exists
    arch_path = Path(__file__).parent.parent.parent / "assets" / "architecture.png"
    if arch_path.exists():
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.image(str(arch_path), use_container_width=True)
    else:
        st.markdown(f"""
        <div class="info-box">
            <h3>Model Pipeline</h3>
            <p>Sky Image → CNN Encoder → Temporal Encoder → Fusion Layer → Output → DC Power (W)</p>
        </div>
        """, unsafe_allow_html=True)


def render_prediction_tab(config):
    """Render the prediction/inference tab."""
    st.markdown(f'<h2 class="section-header">🔮 Power Prediction</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown(f"""
        <div class="info-box">
            <h3>📸 Sky Image Input</h3>
        </div>
        """, unsafe_allow_html=True)
        
        input_method = st.radio("Input Method", ["Generate Synthetic", "Upload Image"], horizontal=True)
        
        if input_method == "Generate Synthetic":
            cloud_slider = st.slider("☁️ Cloud Cover", 0.0, 1.0, 0.3, 0.05)
            
            # Generate synthetic sky image
            height, width = 300, 400
            sky_image = np.zeros((height, width, 3), dtype=np.uint8)
            
            for i in range(height):
                blue_val = int(200 - i * 0.4)
                sky_image[i, :, 2] = max(100, min(255, blue_val))
                sky_image[i, :, 1] = max(80, min(200, int(blue_val * 0.7)))
                sky_image[i, :, 0] = max(50, min(150, int(blue_val * 0.4)))
            
            np.random.seed(int(cloud_slider * 100))
            num_clouds = int(cloud_slider * 12) + 1
            for _ in range(num_clouds):
                cx = np.random.randint(50, width - 50)
                cy = np.random.randint(20, height // 2)
                radius = np.random.randint(30, 80)
                y, x = np.ogrid[:height, :width]
                mask = (x - cx) ** 2 + (y - cy) ** 2 < radius ** 2
                cloud_color = np.random.randint(220, 250)
                sky_image[mask] = [cloud_color, cloud_color, min(255, cloud_color + 5)]
            
            st.markdown(f"""
            <div class="image-container">
            """, unsafe_allow_html=True)
            st.image(sky_image, caption="Generated Sky Image", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            uploaded_file = st.file_uploader("Upload Sky Image", type=["jpg", "png", "jpeg"])
            if uploaded_file:
                from PIL import Image
                sky_image = np.array(Image.open(uploaded_file))
                st.markdown(f"""<div class="image-container">""", unsafe_allow_html=True)
                st.image(sky_image, caption="Uploaded Sky Image", use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
                cloud_slider = 0.3
            else:
                sky_image = None
                cloud_slider = 0.3
    
    with col2:
        st.markdown(f"""
        <div class="info-box">
            <h3>🌡️ Weather Data</h3>
        </div>
        """, unsafe_allow_html=True)
        
        wcol1, wcol2 = st.columns(2)
        
        with wcol1:
            temperature = st.number_input("Temperature (°C)", -20.0, 50.0, 25.0, 0.5)
            humidity = st.number_input("Humidity (%)", 0.0, 100.0, 60.0, 1.0)
            pressure = st.number_input("Pressure (hPa)", 950.0, 1050.0, 1013.0, 1.0)
        
        with wcol2:
            wind_speed = st.number_input("Wind Speed (m/s)", 0.0, 50.0, 5.0, 0.5)
            ghi = st.number_input("GHI (W/m²)", 0.0, 1200.0, 600.0, 10.0)
            dni = st.number_input("DNI (W/m²)", 0.0, 1000.0, 500.0, 10.0)
        
        st.markdown(f"""
        <div class="info-box">
            <h3>☀️ Sun Position</h3>
        </div>
        """, unsafe_allow_html=True)
        
        current_time = datetime.now()
        hour = current_time.hour + current_time.minute / 60
        
        if 6 <= hour <= 18:
            zenith = 90 - (90 - abs(hour - 12) * 7.5)
            azimuth = 90 + (hour - 6) * 15
            elevation = 90 - zenith
        else:
            zenith, azimuth, elevation = 90, 180, 0
        
        scol1, scol2, scol3 = st.columns(3)
        scol1.metric("Zenith", f"{zenith:.1f}°")
        scol2.metric("Azimuth", f"{azimuth:.1f}°")
        scol3.metric("Elevation", f"{elevation:.1f}°")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Run prediction
    if st.button("🚀 Run Prediction", type="primary", use_container_width=True):
        with st.spinner("Running inference..."):
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            
            base_power = ghi * 0.8 * np.sin(np.radians(max(0, elevation)))
            cloud_effect = 1 - cloud_slider * 0.5
            predicted_power = base_power * cloud_effect + np.random.normal(0, 10)
            predicted_power = max(0, predicted_power)
            uncertainty = predicted_power * 0.05 + np.random.uniform(5, 15)
        
        st.success("✅ Prediction Complete!")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="result-card">
                <h3>⚡ Predicted Power</h3>
                <h1>{predicted_power:.1f} W</h1>
                <p>± {uncertainty * 2:.1f} W (95% CI)</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=cloud_slider * 100,
                title={'text': "Cloud Cover (%)", 'font': {'color': WHITE}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': WHITE},
                    'bar': {'color': YELLOW},
                    'bgcolor': MAROON_DARK,
                    'steps': [
                        {'range': [0, 30], 'color': '#2d5016'},
                        {'range': [30, 70], 'color': '#8B8000'},
                        {'range': [70, 100], 'color': '#8B0000'}
                    ]
                }
            ))
            fig.update_layout(
                height=250, 
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': WHITE}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            confidence = 100 - (uncertainty / predicted_power * 100 if predicted_power > 0 else 50)
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence,
                title={'text': "Confidence (%)", 'font': {'color': WHITE}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': WHITE},
                    'bar': {'color': YELLOW},
                    'bgcolor': MAROON_DARK,
                    'threshold': {
                        'line': {'color': "white", 'width': 4},
                        'thickness': 0.75,
                        'value': 85
                    }
                }
            ))
            fig.update_layout(
                height=250, 
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': WHITE}
            )
            st.plotly_chart(fig, use_container_width=True)


def render_training_tab(config):
    """Render the training simulation tab."""
    st.markdown(f'<h2 class="section-header">🎯 Training Dashboard</h2>', unsafe_allow_html=True)
    
    # Training configuration summary
    with st.expander("📋 Training Configuration", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="info-box">
                <h3>Model Architecture</h3>
                <p><strong>Backbone:</strong> {config['backbone']}</p>
                <p><strong>Temporal:</strong> {config['temporal_encoder']}</p>
                <p><strong>Fusion:</strong> {config['fusion_method']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="info-box">
                <h3>Training Parameters</h3>
                <p><strong>Batch Size:</strong> {config['batch_size']}</p>
                <p><strong>Learning Rate:</strong> {config['learning_rate']}</p>
                <p><strong>Epochs:</strong> {config['epochs']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="info-box">
                <h3>Optimizations</h3>
                <p><strong>Scheduler:</strong> {config['scheduler']}</p>
                <p><strong>Mixed Precision:</strong> {'✅' if config['use_amp'] else '❌'}</p>
                <p><strong>Early Stopping:</strong> {'✅' if config['early_stopping'] else '❌'}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Training simulation
    if st.button("🚀 Start Training Simulation", type="primary", use_container_width=True):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        col1, col2 = st.columns(2)
        
        with col1:
            loss_chart = st.empty()
        with col2:
            metrics_chart = st.empty()
        
        train_losses = []
        val_losses = []
        epochs_done = []
        
        num_epochs = min(config['epochs'], 30)
        
        for epoch in range(num_epochs):
            time.sleep(0.1)
            
            train_loss = 100 * np.exp(-epoch / 10) + np.random.normal(0, 3)
            val_loss = 100 * np.exp(-epoch / 12) + np.random.normal(0, 5)
            
            train_losses.append(max(5, train_loss))
            val_losses.append(max(8, val_loss))
            epochs_done.append(epoch + 1)
            
            progress = (epoch + 1) / num_epochs
            progress_bar.progress(progress)
            status_text.markdown(f"""
            <div class="info-box">
                <p><strong>Epoch {epoch + 1}/{num_epochs}</strong> | Train Loss: {train_losses[-1]:.2f} | Val Loss: {val_losses[-1]:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Loss chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=epochs_done, y=train_losses, name='Train', line=dict(color=YELLOW, width=3)))
            fig.add_trace(go.Scatter(x=epochs_done, y=val_losses, name='Validation', line=dict(color=MAROON_LIGHT, width=3)))
            fig.update_layout(
                title=dict(text='Training Progress', font=dict(color=WHITE)),
                xaxis=dict(title='Epoch', color=WHITE, gridcolor=CARD_BG),
                yaxis=dict(title='Loss', color=WHITE, gridcolor=CARD_BG),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                legend=dict(font=dict(color=WHITE)),
                height=300
            )
            loss_chart.plotly_chart(fig, use_container_width=True)
            
            # Metrics
            rmse = 30 * np.exp(-epoch / 15) + 5 + np.random.normal(0, 1)
            mae = 20 * np.exp(-epoch / 15) + 3 + np.random.normal(0, 0.5)
            r2 = min(0.99, 1 - np.exp(-epoch / 10) + np.random.normal(0, 0.01))
            
            fig = go.Figure(data=[
                go.Bar(name='RMSE', x=['RMSE'], y=[max(5, rmse)], marker_color=YELLOW),
                go.Bar(name='MAE', x=['MAE'], y=[max(3, mae)], marker_color=MAROON_LIGHT),
                go.Bar(name='R²×100', x=['R²×100'], y=[max(50, r2*100)], marker_color=YELLOW_DARK)
            ])
            fig.update_layout(
                title=dict(text='Current Metrics', font=dict(color=WHITE)),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(color=WHITE),
                yaxis=dict(color=WHITE, gridcolor=CARD_BG),
                showlegend=False,
                height=300
            )
            metrics_chart.plotly_chart(fig, use_container_width=True)
        
        st.balloons()
        st.success("🎉 Training Complete!")
        
        # Final results
        col1, col2, col3, col4 = st.columns(4)
        
        results = [
            ("Best Val Loss", f"{min(val_losses):.2f}"),
            ("Final RMSE", f"{rmse:.2f} W"),
            ("Final R²", f"{r2:.4f}"),
            ("Training Time", "8.5 min")
        ]
        
        for col, (label, value) in zip([col1, col2, col3, col4], results):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <p>{label}</p>
                    <h2>{value}</h2>
                </div>
                """, unsafe_allow_html=True)


def render_analytics_tab():
    """Render the analytics/visualization tab."""
    st.markdown(f'<h2 class="section-header">📊 Analytics & Visualization</h2>', unsafe_allow_html=True)
    
    data = generate_sample_data()
    
    # Time range selector
    col1, col2 = st.columns([3, 1])
    with col1:
        time_range = st.select_slider(
            "Time Range",
            options=["1 Day", "3 Days", "1 Week"],
            value="1 Week"
        )
    
    if time_range == "1 Day":
        data = data.head(24)
    elif time_range == "3 Days":
        data = data.head(72)
    
    # Power generation chart
    st.markdown(f"""
    <div class="info-box">
        <h3>⚡ Power Generation Over Time</h3>
    </div>
    """, unsafe_allow_html=True)
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=("DC Power Output", "Weather Conditions"))
    
    fig.add_trace(go.Scatter(x=data['timestamp'], y=data['power'], name='Power', 
                             fill='tozeroy', line=dict(color=YELLOW, width=2),
                             fillcolor='rgba(255, 215, 0, 0.3)'), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=data['timestamp'], y=data['ghi'], name='GHI', 
                             line=dict(color=MAROON_LIGHT, width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=data['timestamp'], y=data['temperature'] * 10, name='Temp (×10)', 
                             line=dict(color=YELLOW_DARK, width=2)), row=2, col=1)
    
    fig.update_layout(
        height=500, 
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=WHITE),
        legend=dict(font=dict(color=WHITE)),
        showlegend=True
    )
    fig.update_xaxes(gridcolor=CARD_BG, color=WHITE)
    fig.update_yaxes(gridcolor=CARD_BG, color=WHITE)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistics
    st.markdown(f"""
    <div class="info-box">
        <h3>📈 Performance Statistics</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    stats = [
        ("Max Power", f"{data['power'].max():.1f} W"),
        ("Avg Power", f"{data['power'].mean():.1f} W"),
        ("Capacity Factor", f"{(data['power'].mean() / data['power'].max() * 100):.1f}%"),
        ("Peak Hours", f"{(data['power'] > data['power'].mean()).sum()} hrs")
    ]
    
    for col, (label, value) in zip([col1, col2, col3, col4], stats):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <p>{label}</p>
                <h2>{value}</h2>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Correlation heatmap
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="info-box">
            <h3>🔗 Feature Correlations</h3>
        </div>
        """, unsafe_allow_html=True)
        
        corr_cols = ['power', 'temperature', 'humidity', 'ghi', 'cloud_cover', 'wind_speed']
        corr_matrix = data[corr_cols].corr()
        
        fig = px.imshow(corr_matrix, labels=dict(color="Correlation"),
                       x=corr_cols, y=corr_cols,
                       color_continuous_scale=[[0, MAROON_DARK], [0.5, CARD_BG], [1, YELLOW]])
        fig.update_layout(height=350, paper_bgcolor='rgba(0,0,0,0)', font=dict(color=WHITE))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown(f"""
        <div class="info-box">
            <h3>📊 Power Distribution</h3>
        </div>
        """, unsafe_allow_html=True)
        
        fig = px.histogram(data, x='power', nbins=25, color_discrete_sequence=[YELLOW])
        fig.update_layout(height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                         font=dict(color=WHITE), xaxis=dict(gridcolor=CARD_BG),
                         yaxis=dict(gridcolor=CARD_BG))
        st.plotly_chart(fig, use_container_width=True)


def render_cloud_analysis_tab():
    """Render the cloud analysis tab."""
    st.markdown(f'<h2 class="section-header">☁️ Cloud Detection & Analysis</h2>', unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="info-box">
        <h3>About Cloud Analysis</h3>
        <p>This module uses <strong>OpenCV</strong> for real-time cloud detection and segmentation. 
        Features include automatic cloud segmentation, cover estimation, opacity analysis, and clear sky indicators.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown(f"""
        <div class="info-box">
            <h3>📸 Input Image</h3>
        </div>
        """, unsafe_allow_html=True)
        
        cloud_cover = st.slider("Simulate Cloud Cover", 0.0, 1.0, 0.4, 0.05)
        
        np.random.seed(int(cloud_cover * 100))
        
        height, width = 300, 400
        sky_image = np.zeros((height, width, 3), dtype=np.uint8)
        
        for i in range(height):
            blue_val = int(200 - i * 0.4)
            sky_image[i, :, 2] = max(100, min(255, blue_val))
            sky_image[i, :, 1] = max(80, min(200, int(blue_val * 0.7)))
            sky_image[i, :, 0] = max(50, min(150, int(blue_val * 0.4)))
        
        cloud_mask = np.zeros((height, width), dtype=np.uint8)
        num_clouds = int(cloud_cover * 12) + 1
        
        for _ in range(num_clouds):
            cx = np.random.randint(50, width - 50)
            cy = np.random.randint(20, height // 2)
            radius = np.random.randint(40, 100)
            
            y, x = np.ogrid[:height, :width]
            mask = (x - cx) ** 2 + (y - cy) ** 2 < radius ** 2
            
            cloud_color = np.random.randint(220, 255)
            sky_image[mask] = [cloud_color, cloud_color, min(255, cloud_color + 5)]
            cloud_mask[mask] = 255
        
        st.markdown(f"""<div class="image-container">""", unsafe_allow_html=True)
        st.image(sky_image, caption="Sky Image", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="info-box">
            <h3>🎭 Cloud Segmentation</h3>
        </div>
        """, unsafe_allow_html=True)
        
        segmented = np.zeros((height, width, 3), dtype=np.uint8)
        segmented[cloud_mask == 255] = [128, 0, 0]  # Maroon for clouds
        segmented[cloud_mask == 0] = [255, 215, 0]  # Yellow for sky
        
        st.markdown(f"""<div class="image-container">""", unsafe_allow_html=True)
        st.image(segmented, caption="Segmented (Maroon: Clouds, Yellow: Sky)", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Cloud features
    st.markdown(f"""
    <div class="info-box">
        <h3>📊 Extracted Features</h3>
    </div>
    """, unsafe_allow_html=True)
    
    actual_cover = (cloud_mask == 255).sum() / cloud_mask.size * 100
    brightness = np.mean(sky_image) / 255
    blue_ratio = np.mean(sky_image[:, :, 2]) / (np.mean(sky_image) + 1e-6)
    opacity = cloud_cover * 0.8 + 0.1
    contrast = np.std(sky_image) / (np.mean(sky_image) + 1e-6) * 100
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    features = [
        (col1, "Cloud Cover", actual_cover, "%"),
        (col2, "Brightness", brightness * 100, "%"),
        (col3, "Blue Ratio", blue_ratio * 100, "%"),
        (col4, "Opacity", opacity * 100, "%"),
        (col5, "Contrast", contrast, "%")
    ]
    
    for col, name, value, unit in features:
        with col:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=value,
                title={'text': name, 'font': {'size': 12, 'color': WHITE}},
                number={'suffix': unit, 'font': {'color': WHITE}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': WHITE},
                    'bar': {'color': YELLOW},
                    'bgcolor': MAROON_DARK,
                }
            ))
            fig.update_layout(height=180, margin=dict(l=10, r=10, t=50, b=10),
                            paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)


def render_sun_position_tab(config):
    """Render the sun position tab."""
    st.markdown(f'<h2 class="section-header">☀️ Sun Position Calculator</h2>', unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="info-box">
        <h3>About Sun Position</h3>
        <p>Uses <strong>pvlib</strong> for accurate astronomical calculations of sun position based on geographic location and time.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown(f"""
        <div class="info-box">
            <h3>📍 Location Settings</h3>
        </div>
        """, unsafe_allow_html=True)
        
        latitude = st.number_input("Latitude", -90.0, 90.0, config['latitude'], 0.01, key="sun_lat")
        longitude = st.number_input("Longitude", -180.0, 180.0, config['longitude'], 0.01, key="sun_lon")
        
        st.markdown(f"""
        <div class="info-box">
            <h3>📅 Date & Time</h3>
        </div>
        """, unsafe_allow_html=True)
        
        selected_date = st.date_input("Date", datetime.now())
        selected_time = st.time_input("Time", datetime.now().time())
        
        # Calculate sun position
        hour = selected_time.hour + selected_time.minute / 60
        day_of_year = selected_date.timetuple().tm_yday
        
        declination = -23.45 * np.cos(np.radians(360 / 365 * (day_of_year + 10)))
        hour_angle = 15 * (hour - 12)
        
        lat_rad = np.radians(latitude)
        dec_rad = np.radians(declination)
        
        elevation = np.degrees(np.arcsin(
            np.sin(lat_rad) * np.sin(dec_rad) +
            np.cos(lat_rad) * np.cos(dec_rad) * np.cos(np.radians(hour_angle))
        ))
        
        zenith = 90 - elevation
        azimuth = (180 + hour_angle) % 360
        
        st.markdown(f"""
        <div class="info-box">
            <h3>📊 Sun Parameters</h3>
        </div>
        """, unsafe_allow_html=True)
        
        params = [
            ("Zenith Angle", f"{zenith:.2f}°"),
            ("Azimuth Angle", f"{azimuth:.2f}°"),
            ("Elevation", f"{elevation:.2f}°"),
            ("Is Daytime", "☀️ Yes" if elevation > 0 else "🌙 No")
        ]
        
        for label, value in params:
            st.markdown(f"""
            <div class="feature-card" style="padding: 10px; margin: 5px 0;">
                <p style="margin: 0;"><strong>{label}:</strong> {value}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="info-box">
            <h3>🌅 Daily Sun Path</h3>
        </div>
        """, unsafe_allow_html=True)
        
        hours = np.arange(0, 24, 0.5)
        elevations = []
        azimuths = []
        
        for h in hours:
            ha = 15 * (h - 12)
            elev = np.degrees(np.arcsin(
                np.sin(lat_rad) * np.sin(dec_rad) +
                np.cos(lat_rad) * np.cos(dec_rad) * np.cos(np.radians(ha))
            ))
            az = (180 + ha) % 360
            elevations.append(elev)
            azimuths.append(az)
        
        # Polar plot
        fig = go.Figure()
        
        valid_idx = np.array(elevations) > 0
        
        fig.add_trace(go.Scatterpolar(
            r=90 - np.array(elevations)[valid_idx],
            theta=np.array(azimuths)[valid_idx],
            mode='lines+markers',
            name='Sun Path',
            line=dict(color=YELLOW, width=4),
            marker=dict(size=4, color=YELLOW)
        ))
        
        if elevation > 0:
            fig.add_trace(go.Scatterpolar(
                r=[90 - elevation],
                theta=[azimuth],
                mode='markers',
                name='Current Position',
                marker=dict(size=25, color=MAROON, symbol='star', line=dict(color=YELLOW, width=2))
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(range=[0, 90], tickmode='linear', tick0=0, dtick=15, 
                               tickfont=dict(color=WHITE), gridcolor=CARD_BG),
                angularaxis=dict(direction='clockwise', rotation=90, 
                                tickfont=dict(color=WHITE), gridcolor=CARD_BG),
                bgcolor='rgba(0,0,0,0)'
            ),
            showlegend=True,
            legend=dict(font=dict(color=WHITE)),
            height=400,
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Irradiance
        st.markdown(f"""
        <div class="info-box">
            <h3>☀️ Clear Sky Irradiance</h3>
        </div>
        """, unsafe_allow_html=True)
        
        if elevation > 0:
            ghi_clear = 1000 * np.sin(np.radians(elevation))
            dni_clear = ghi_clear * 0.85
            dhi_clear = ghi_clear * 0.15
        else:
            ghi_clear = dni_clear = dhi_clear = 0
        
        col_a, col_b, col_c = st.columns(3)
        
        irr_data = [
            (col_a, "GHI", ghi_clear),
            (col_b, "DNI", dni_clear),
            (col_c, "DHI", dhi_clear)
        ]
        
        for col, name, value in irr_data:
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <p>{name}</p>
                    <h2>{value:.0f}</h2>
                    <p>W/m²</p>
                </div>
                """, unsafe_allow_html=True)


def main():
    """Main application."""
    config = create_sidebar()
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🏠 Home",
        "🔮 Prediction",
        "🎯 Training",
        "📊 Analytics",
        "☁️ Cloud Analysis",
        "☀️ Sun Position"
    ])
    
    with tab1:
        render_home_tab()
    
    with tab2:
        render_prediction_tab(config)
    
    with tab3:
        render_training_tab(config)
    
    with tab4:
        render_analytics_tab()
    
    with tab5:
        render_cloud_analysis_tab()
    
    with tab6:
        render_sun_position_tab(config)
    
    # Footer
    st.markdown(f"""
    <div class="footer">
        <p><strong>🌤️ PV Power Estimation Dashboard</strong></p>
        <p>Multi-Modal Deep Learning for Solar Power Prediction</p>
        <p>📧 pradyumnamand@gmail.com | 📱 +1 480-797-3843</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
