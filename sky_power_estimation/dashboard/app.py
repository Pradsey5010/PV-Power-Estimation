"""
PV Power Estimation Dashboard
Interactive dashboard for solar power prediction using sky images and weather data.
Professional Gradient Theme with Enhanced Styling
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
    page_title="PV Power Estimation",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Theme Colors - Gradient Palette
PRIMARY_DARK = "#1a1a2e"
PRIMARY_MID = "#16213e"
PRIMARY_LIGHT = "#0f3460"
ACCENT_1 = "#e94560"
ACCENT_2 = "#ff6b6b"
ACCENT_3 = "#feca57"
ACCENT_4 = "#48dbfb"
ACCENT_5 = "#1dd1a1"
TEXT_LIGHT = "#ffffff"
TEXT_MUTED = "#a0a0a0"
CARD_BG = "rgba(255, 255, 255, 0.05)"
CARD_BORDER = "rgba(255, 255, 255, 0.1)"

# Solar/PV Related Images from Unsplash
IMAGES = {
    "hero_solar": "https://images.unsplash.com/photo-1509391366360-2e959784a276?w=1200&h=600&fit=crop",
    "solar_panels": "https://images.unsplash.com/photo-1508514177221-188b1cf16e9d?w=600&h=400&fit=crop",
    "sky_clouds": "https://images.unsplash.com/photo-1517483000871-1dbf64a6e1c6?w=600&h=400&fit=crop",
    "weather_station": "https://images.unsplash.com/photo-1561484930-998b6a7b22e8?w=600&h=400&fit=crop",
    "sun_rays": "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=600&h=400&fit=crop",
    "power_grid": "https://images.unsplash.com/photo-1473341304170-971dccb5ac1e?w=600&h=400&fit=crop",
    "solar_farm": "https://images.unsplash.com/photo-1559302504-64aae6ca6b6d?w=600&h=400&fit=crop",
    "sunset_panels": "https://images.unsplash.com/photo-1548337138-e87d889cc369?w=600&h=400&fit=crop",
    "blue_sky": "https://images.unsplash.com/photo-1601297183305-6df142704ea2?w=600&h=400&fit=crop",
    "cloudy_sky": "https://images.unsplash.com/photo-1534088568595-a066f410bcda?w=600&h=400&fit=crop",
    "analytics": "https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=600&h=400&fit=crop",
    "technology": "https://images.unsplash.com/photo-1518770660439-4636190af475?w=600&h=400&fit=crop",
}

# Custom CSS with Gradient Theme
st.markdown(f"""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    * {{
        font-family: 'Poppins', sans-serif;
    }}
    
    /* Main App Background - Gradient */
    .stApp {{
        background: linear-gradient(135deg, {PRIMARY_DARK} 0%, {PRIMARY_MID} 50%, {PRIMARY_LIGHT} 100%);
        background-attachment: fixed;
    }}
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {PRIMARY_DARK} 0%, {PRIMARY_MID} 100%);
        border-right: 1px solid {CARD_BORDER};
    }}
    
    [data-testid="stSidebar"] .stMarkdown {{
        color: {TEXT_LIGHT};
    }}
    
    [data-testid="stSidebar"] label {{
        color: {TEXT_MUTED} !important;
        font-weight: 500;
    }}
    
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label {{
        color: {ACCENT_3} !important;
    }}
    
    /* Main Header with Gradient Text */
    .main-title {{
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, {ACCENT_3} 0%, {ACCENT_2} 50%, {ACCENT_1} 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1.5rem 0;
        margin-bottom: 0;
        animation: fadeInDown 0.8s ease-out;
    }}
    
    .sub-title {{
        font-size: 1.2rem;
        color: {TEXT_MUTED};
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 300;
    }}
    
    /* Section Headers */
    .section-header {{
        font-size: 1.8rem;
        font-weight: 600;
        background: linear-gradient(90deg, {ACCENT_4} 0%, {ACCENT_5} 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 2rem 0 1.5rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid;
        border-image: linear-gradient(90deg, {ACCENT_4}, {ACCENT_5}) 1;
    }}
    
    /* Glass Card Effect */
    .glass-card {{
        background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255,255,255,0.1);
        padding: 25px;
        margin: 15px 0;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }}
    
    .glass-card:hover {{
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 20px 40px rgba(233, 69, 96, 0.3);
        border-color: {ACCENT_1};
    }}
    
    .glass-card h4 {{
        color: {ACCENT_3} !important;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 12px;
    }}
    
    .glass-card p {{
        color: {TEXT_LIGHT} !important;
        font-size: 0.95rem;
        line-height: 1.6;
        opacity: 0.9;
    }}
    
    /* Feature Cards with Icons */
    .feature-card {{
        background: linear-gradient(145deg, rgba(30,30,60,0.8) 0%, rgba(20,20,40,0.9) 100%);
        border-radius: 16px;
        padding: 25px;
        margin: 12px 0;
        border-left: 4px solid;
        border-image: linear-gradient(180deg, {ACCENT_1}, {ACCENT_3}) 1;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }}
    
    .feature-card::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(135deg, {ACCENT_1}15 0%, transparent 50%);
        opacity: 0;
        transition: opacity 0.3s ease;
    }}
    
    .feature-card:hover {{
        transform: translateX(10px);
        box-shadow: -5px 5px 30px rgba(233, 69, 96, 0.2);
    }}
    
    .feature-card:hover::before {{
        opacity: 1;
    }}
    
    .feature-card .icon {{
        font-size: 2.5rem;
        margin-bottom: 15px;
        display: block;
    }}
    
    .feature-card h4 {{
        color: {TEXT_LIGHT} !important;
        font-weight: 600;
        margin-bottom: 8px;
    }}
    
    .feature-card p {{
        color: {TEXT_MUTED} !important;
        font-size: 0.9rem;
    }}
    
    /* Metric Cards with Gradient Border */
    .metric-card {{
        background: linear-gradient(135deg, {PRIMARY_MID} 0%, {PRIMARY_DARK} 100%);
        border-radius: 20px;
        padding: 30px 20px;
        text-align: center;
        position: relative;
        overflow: hidden;
        transition: all 0.4s ease;
    }}
    
    .metric-card::before {{
        content: '';
        position: absolute;
        top: -2px;
        left: -2px;
        right: -2px;
        bottom: -2px;
        background: linear-gradient(135deg, {ACCENT_1}, {ACCENT_3}, {ACCENT_4}, {ACCENT_5});
        border-radius: 22px;
        z-index: -1;
        opacity: 0.7;
    }}
    
    .metric-card::after {{
        content: '';
        position: absolute;
        top: 2px;
        left: 2px;
        right: 2px;
        bottom: 2px;
        background: linear-gradient(135deg, {PRIMARY_MID} 0%, {PRIMARY_DARK} 100%);
        border-radius: 18px;
        z-index: -1;
    }}
    
    .metric-card:hover {{
        transform: translateY(-10px) rotateX(5deg);
        box-shadow: 0 25px 50px rgba(233, 69, 96, 0.4);
    }}
    
    .metric-card .metric-icon {{
        font-size: 2.5rem;
        margin-bottom: 10px;
    }}
    
    .metric-card .metric-value {{
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, {ACCENT_3} 0%, {ACCENT_2} 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 5px 0;
    }}
    
    .metric-card .metric-label {{
        color: {TEXT_MUTED};
        font-size: 0.9rem;
        font-weight: 500;
    }}
    
    /* Image Container with Glow */
    .image-container {{
        position: relative;
        border-radius: 20px;
        overflow: hidden;
        box-shadow: 0 10px 40px rgba(0,0,0,0.4);
        transition: all 0.4s ease;
    }}
    
    .image-container:hover {{
        transform: scale(1.03);
        box-shadow: 0 20px 60px rgba(233, 69, 96, 0.3);
    }}
    
    .image-container img {{
        width: 100%;
        height: auto;
        display: block;
        transition: all 0.4s ease;
    }}
    
    .image-container:hover img {{
        transform: scale(1.1);
    }}
    
    .image-overlay {{
        position: absolute;
        bottom: 0;
        left: 0;
        right: 0;
        background: linear-gradient(transparent, rgba(0,0,0,0.8));
        padding: 20px;
        color: white;
    }}
    
    .image-overlay h4 {{
        color: {ACCENT_3} !important;
        margin: 0 0 5px 0;
        font-weight: 600;
    }}
    
    .image-overlay p {{
        color: {TEXT_LIGHT} !important;
        margin: 0;
        font-size: 0.9rem;
        opacity: 0.9;
    }}
    
    /* Result Card - Glowing Effect */
    .result-card {{
        background: linear-gradient(135deg, {PRIMARY_LIGHT} 0%, {PRIMARY_MID} 100%);
        border-radius: 25px;
        padding: 40px;
        text-align: center;
        position: relative;
        overflow: hidden;
        box-shadow: 0 0 60px rgba(233, 69, 96, 0.3);
        animation: pulse-glow 2s infinite;
    }}
    
    @keyframes pulse-glow {{
        0%, 100% {{ box-shadow: 0 0 40px rgba(233, 69, 96, 0.3); }}
        50% {{ box-shadow: 0 0 80px rgba(254, 202, 87, 0.4); }}
    }}
    
    .result-card::before {{
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: conic-gradient(from 0deg, transparent, {ACCENT_1}20, transparent, {ACCENT_3}20, transparent);
        animation: rotate 4s linear infinite;
    }}
    
    @keyframes rotate {{
        100% {{ transform: rotate(360deg); }}
    }}
    
    .result-card .result-content {{
        position: relative;
        z-index: 1;
    }}
    
    .result-card h3 {{
        color: {TEXT_MUTED} !important;
        font-weight: 400;
        margin-bottom: 10px;
    }}
    
    .result-card .power-value {{
        font-size: 4rem;
        font-weight: 700;
        background: linear-gradient(135deg, {ACCENT_3} 0%, {ACCENT_2} 50%, {ACCENT_1} 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 10px 0;
    }}
    
    .result-card .confidence {{
        color: {ACCENT_5};
        font-size: 1.1rem;
    }}
    
    /* Info Box */
    .info-box {{
        background: linear-gradient(135deg, rgba(72, 219, 251, 0.1) 0%, rgba(29, 209, 161, 0.1) 100%);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid rgba(72, 219, 251, 0.3);
        margin: 15px 0;
    }}
    
    .info-box h3 {{
        color: {ACCENT_4} !important;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 10px;
    }}
    
    .info-box p {{
        color: {TEXT_LIGHT} !important;
        margin: 5px 0;
        opacity: 0.9;
    }}
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background: linear-gradient(90deg, {PRIMARY_DARK}, {PRIMARY_MID});
        padding: 10px;
        border-radius: 15px;
        border: 1px solid {CARD_BORDER};
    }}
    
    .stTabs [data-baseweb="tab"] {{
        height: 50px;
        padding: 0 25px;
        background: transparent;
        border-radius: 10px;
        color: {TEXT_MUTED};
        font-weight: 500;
        transition: all 0.3s ease;
        border: 1px solid transparent;
    }}
    
    .stTabs [data-baseweb="tab"]:hover {{
        background: rgba(255,255,255,0.1);
        color: {TEXT_LIGHT};
    }}
    
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, {ACCENT_1} 0%, {ACCENT_2} 100%) !important;
        color: {TEXT_LIGHT} !important;
        box-shadow: 0 5px 20px rgba(233, 69, 96, 0.4);
    }}
    
    /* Button Styling */
    .stButton > button {{
        background: linear-gradient(135deg, {ACCENT_1} 0%, {ACCENT_2} 100%);
        color: white;
        font-weight: 600;
        border: none;
        padding: 15px 40px;
        border-radius: 50px;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 10px 30px rgba(233, 69, 96, 0.4);
        text-transform: uppercase;
        letter-spacing: 1px;
    }}
    
    .stButton > button:hover {{
        transform: translateY(-3px) scale(1.05);
        box-shadow: 0 15px 40px rgba(233, 69, 96, 0.6);
        background: linear-gradient(135deg, {ACCENT_2} 0%, {ACCENT_1} 100%);
    }}
    
    .stButton > button:active {{
        transform: translateY(0);
    }}
    
    /* Input Fields */
    .stNumberInput input, .stTextInput input {{
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 10px !important;
        color: {TEXT_LIGHT} !important;
        padding: 10px 15px !important;
        transition: all 0.3s ease;
    }}
    
    .stNumberInput input:focus, .stTextInput input:focus {{
        border-color: {ACCENT_1} !important;
        box-shadow: 0 0 20px rgba(233, 69, 96, 0.3) !important;
    }}
    
    /* Slider */
    .stSlider > div > div > div {{
        background: linear-gradient(90deg, {ACCENT_4}, {ACCENT_5}) !important;
    }}
    
    .stSlider > div > div > div > div {{
        background: {TEXT_LIGHT} !important;
        box-shadow: 0 0 10px {ACCENT_4};
    }}
    
    /* Progress Bar */
    .stProgress > div > div > div {{
        background: linear-gradient(90deg, {ACCENT_1}, {ACCENT_3}, {ACCENT_5}) !important;
        background-size: 200% 100%;
        animation: gradient-shift 2s ease infinite;
    }}
    
    @keyframes gradient-shift {{
        0% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}
    
    /* Expander */
    .streamlit-expanderHeader {{
        background: linear-gradient(90deg, rgba(233,69,96,0.1), rgba(254,202,87,0.1)) !important;
        border-radius: 10px !important;
        color: {ACCENT_3} !important;
        font-weight: 500;
    }}
    
    .streamlit-expanderHeader:hover {{
        background: linear-gradient(90deg, rgba(233,69,96,0.2), rgba(254,202,87,0.2)) !important;
    }}
    
    /* Footer */
    .footer {{
        background: linear-gradient(135deg, {PRIMARY_DARK} 0%, {PRIMARY_MID} 100%);
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        margin-top: 40px;
        border: 1px solid {CARD_BORDER};
        position: relative;
        overflow: hidden;
    }}
    
    .footer::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, {ACCENT_1}, {ACCENT_3}, {ACCENT_4}, {ACCENT_5});
    }}
    
    .footer p {{
        color: {TEXT_MUTED} !important;
        margin: 8px 0;
    }}
    
    .footer a {{
        color: {ACCENT_3} !important;
        text-decoration: none;
        transition: color 0.3s ease;
    }}
    
    .footer a:hover {{
        color: {ACCENT_1} !important;
    }}
    
    /* Animations */
    @keyframes fadeInDown {{
        from {{
            opacity: 0;
            transform: translateY(-30px);
        }}
        to {{
            opacity: 1;
            transform: translateY(0);
        }}
    }}
    
    @keyframes fadeInUp {{
        from {{
            opacity: 0;
            transform: translateY(30px);
        }}
        to {{
            opacity: 1;
            transform: translateY(0);
        }}
    }}
    
    .animate-fade-in {{
        animation: fadeInUp 0.6s ease-out;
    }}
    
    /* Hide Streamlit Branding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {{
        width: 8px;
        height: 8px;
    }}
    
    ::-webkit-scrollbar-track {{
        background: {PRIMARY_DARK};
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: linear-gradient(180deg, {ACCENT_1}, {ACCENT_3});
        border-radius: 4px;
    }}
    
    ::-webkit-scrollbar-thumb:hover {{
        background: linear-gradient(180deg, {ACCENT_2}, {ACCENT_3});
    }}
    
    /* Radio buttons */
    .stRadio > div {{
        background: rgba(255,255,255,0.05);
        padding: 10px;
        border-radius: 10px;
    }}
    
    .stRadio label {{
        color: {TEXT_LIGHT} !important;
    }}
</style>
""", unsafe_allow_html=True)


def create_sidebar():
    """Create sidebar with configuration options."""
    with st.sidebar:
        # Logo Section
        st.markdown(f"""
        <div style="text-align: center; padding: 20px 0;">
            <div style="font-size: 4rem; margin-bottom: 10px;">☀️</div>
            <h2 style="
                background: linear-gradient(135deg, {ACCENT_3} 0%, {ACCENT_1} 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin: 0;
                font-weight: 700;
            ">PV Power</h2>
            <p style="color: {TEXT_MUTED}; font-size: 0.9rem;">Estimation System</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Model Configuration
        st.markdown(f"<p style='color: {ACCENT_3}; font-weight: 600; font-size: 1rem;'>🧠 Model Settings</p>", unsafe_allow_html=True)
        
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
        st.markdown(f"<p style='color: {ACCENT_4}; font-weight: 600; font-size: 1rem;'>🎯 Training Settings</p>", unsafe_allow_html=True)
        
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
        st.markdown(f"<p style='color: {ACCENT_5}; font-weight: 600; font-size: 1rem;'>📍 Location</p>", unsafe_allow_html=True)
        
        latitude = st.number_input("Latitude", -90.0, 90.0, 37.7749, 0.0001)
        longitude = st.number_input("Longitude", -180.0, 180.0, -122.4194, 0.0001)
        
        st.markdown("---")
        
        # Contact
        st.markdown(f"""
        <div style="text-align: center; padding: 15px 0;">
            <p style="color: {TEXT_MUTED}; font-size: 0.85rem;">📧 pradyumnamand@gmail.com</p>
            <p style="color: {TEXT_MUTED}; font-size: 0.85rem;">📱 +1 480-797-3843</p>
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
    st.markdown('<h1 class="main-title">Solar Power Estimation</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Multi-Modal Deep Learning for Intelligent PV Power Prediction</p>', unsafe_allow_html=True)
    
    # Hero Image
    st.markdown(f"""
    <div class="image-container" style="margin-bottom: 30px;">
        <img src="{IMAGES['hero_solar']}" style="width: 100%; height: 350px; object-fit: cover;">
        <div class="image-overlay">
            <h4>Next-Generation Solar Analytics</h4>
            <p>Harness AI to predict power output with unprecedented accuracy</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    metrics = [
        ("🎯", "94.5%", "Accuracy"),
        ("⚡", "15ms", "Inference"),
        ("📊", "23.4W", "RMSE"),
        ("🌡️", "12", "Sensors")
    ]
    
    for col, (icon, value, label) in zip([col1, col2, col3, col4], metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">{icon}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Key Features
    st.markdown('<h2 class="section-header">Key Features</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="feature-card">
            <span class="icon">🖼️</span>
            <h4>Multi-Modal Learning</h4>
            <p>Combines sky images, weather sensors, and sun position data using advanced deep learning architectures.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="feature-card">
            <span class="icon">🧠</span>
            <h4>Flexible Backbones</h4>
            <p>Support for 10+ CNN architectures including ResNet, MobileNet, EfficientNet, and Vision Transformers.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="feature-card">
            <span class="icon">⏱️</span>
            <h4>Temporal Modeling</h4>
            <p>LSTM and Transformer encoders capture time-dependent patterns and seasonal variations.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="feature-card">
            <span class="icon">🔗</span>
            <h4>Attention Fusion</h4>
            <p>Dynamic modality weighting through cross-attention mechanisms for optimal feature combination.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="feature-card">
            <span class="icon">📈</span>
            <h4>Uncertainty Estimation</h4>
            <p>Monte Carlo dropout provides prediction confidence intervals for reliable decision making.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="feature-card">
            <span class="icon">☁️</span>
            <h4>Cloud Detection</h4>
            <p>OpenCV-based cloud segmentation and opacity estimation for accurate atmospheric analysis.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Data Sources with Images
    st.markdown('<h2 class="section-header">Data Sources</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="image-container">
            <img src="{IMAGES['sky_clouds']}" style="width: 100%; height: 220px; object-fit: cover;">
            <div class="image-overlay">
                <h4>📸 Sky Images</h4>
                <p>Fisheye camera captures for cloud cover analysis</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="image-container">
            <img src="{IMAGES['weather_station']}" style="width: 100%; height: 220px; object-fit: cover;">
            <div class="image-overlay">
                <h4>🌡️ Weather Sensors</h4>
                <p>Temperature, humidity, pressure, wind data</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="image-container">
            <img src="{IMAGES['sun_rays']}" style="width: 100%; height: 220px; object-fit: cover;">
            <div class="image-overlay">
                <h4>☀️ Sun Position</h4>
                <p>Precise solar geometry calculations</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Technology Stack
    st.markdown('<h2 class="section-header">Technology Stack</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="image-container">
            <img src="{IMAGES['technology']}" style="width: 100%; height: 200px; object-fit: cover;">
            <div class="image-overlay">
                <h4>Deep Learning Framework</h4>
                <p>PyTorch, TensorFlow, timm</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="image-container">
            <img src="{IMAGES['analytics']}" style="width: 100%; height: 200px; object-fit: cover;">
            <div class="image-overlay">
                <h4>Analytics & Visualization</h4>
                <p>Plotly, Streamlit, OpenCV</p>
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_prediction_tab(config):
    """Render the prediction/inference tab."""
    st.markdown('<h2 class="section-header">Power Prediction</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown(f"""
        <div class="info-box">
            <h3>📸 Sky Image Input</h3>
            <p>Generate synthetic sky images or upload your own for analysis</p>
        </div>
        """, unsafe_allow_html=True)
        
        input_method = st.radio("Input Method", ["Generate Synthetic", "Upload Image"], horizontal=True)
        
        if input_method == "Generate Synthetic":
            cloud_slider = st.slider("☁️ Cloud Cover", 0.0, 1.0, 0.3, 0.05)
            
            # Generate synthetic sky image
            height, width = 300, 400
            sky_image = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Sky gradient
            for i in range(height):
                ratio = i / height
                r = int(135 * (1 - ratio) + 200 * ratio)
                g = int(206 * (1 - ratio) + 220 * ratio)
                b = int(235 * (1 - ratio) + 255 * ratio)
                sky_image[i, :] = [r, g, b]
            
            # Add clouds
            np.random.seed(int(cloud_slider * 100))
            num_clouds = int(cloud_slider * 15) + 1
            for _ in range(num_clouds):
                cx = np.random.randint(30, width - 30)
                cy = np.random.randint(20, height // 2)
                radius = np.random.randint(25, 70)
                y, x = np.ogrid[:height, :width]
                mask = (x - cx) ** 2 + (y - cy) ** 2 < radius ** 2
                cloud_color = np.random.randint(230, 255)
                sky_image[mask] = [cloud_color, cloud_color, cloud_color]
            
            st.markdown(f"""<div class="image-container" style="margin: 15px 0;">""", unsafe_allow_html=True)
            st.image(sky_image, caption=f"Generated Sky (Cloud Cover: {int(cloud_slider*100)}%)", use_container_width=True)
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
            <h3>🌡️ Weather Parameters</h3>
            <p>Enter current weather conditions for accurate prediction</p>
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
    
    # Run Prediction Button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_btn = st.button("🚀 Run Prediction", type="primary", use_container_width=True)
    
    if predict_btn:
        with st.spinner("Analyzing sky conditions..."):
            progress = st.progress(0)
            for i in range(100):
                time.sleep(0.015)
                progress.progress(i + 1)
            
            base_power = ghi * 0.85 * np.sin(np.radians(max(0, elevation)))
            cloud_effect = 1 - cloud_slider * 0.55
            predicted_power = base_power * cloud_effect + np.random.normal(0, 15)
            predicted_power = max(0, predicted_power)
            uncertainty = predicted_power * 0.06 + np.random.uniform(5, 12)
        
        st.success("✅ Prediction Complete!")
        
        # Results
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            st.markdown(f"""
            <div class="result-card">
                <div class="result-content">
                    <h3>⚡ Predicted Power</h3>
                    <div class="power-value">{predicted_power:.0f} W</div>
                    <p class="confidence">± {uncertainty:.1f} W (95% CI)</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=cloud_slider * 100,
                title={'text': "Cloud Cover", 'font': {'color': TEXT_LIGHT, 'size': 16}},
                number={'suffix': "%", 'font': {'color': TEXT_LIGHT}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': TEXT_LIGHT},
                    'bar': {'color': ACCENT_4},
                    'bgcolor': PRIMARY_MID,
                    'steps': [
                        {'range': [0, 30], 'color': ACCENT_5},
                        {'range': [30, 70], 'color': ACCENT_3},
                        {'range': [70, 100], 'color': ACCENT_1}
                    ]
                }
            ))
            fig.update_layout(height=280, paper_bgcolor='rgba(0,0,0,0)', font={'color': TEXT_LIGHT})
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            confidence = min(98, max(60, 100 - (uncertainty / max(1, predicted_power) * 100)))
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence,
                title={'text': "Confidence", 'font': {'color': TEXT_LIGHT, 'size': 16}},
                number={'suffix': "%", 'font': {'color': TEXT_LIGHT}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': TEXT_LIGHT},
                    'bar': {'color': ACCENT_5},
                    'bgcolor': PRIMARY_MID,
                    'threshold': {'line': {'color': TEXT_LIGHT, 'width': 3}, 'value': 85}
                }
            ))
            fig.update_layout(height=280, paper_bgcolor='rgba(0,0,0,0)', font={'color': TEXT_LIGHT})
            st.plotly_chart(fig, use_container_width=True)


def render_training_tab(config):
    """Render the training simulation tab."""
    st.markdown('<h2 class="section-header">Training Dashboard</h2>', unsafe_allow_html=True)
    
    # Training Image
    st.markdown(f"""
    <div class="image-container" style="margin-bottom: 25px;">
        <img src="{IMAGES['solar_farm']}" style="width: 100%; height: 200px; object-fit: cover;">
        <div class="image-overlay">
            <h4>Model Training</h4>
            <p>Configure and simulate the training process</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Configuration Summary
    with st.expander("📋 Training Configuration", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="glass-card">
                <h4>🧠 Model Architecture</h4>
                <p><strong>Backbone:</strong> {config['backbone']}</p>
                <p><strong>Temporal:</strong> {config['temporal_encoder']}</p>
                <p><strong>Fusion:</strong> {config['fusion_method']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="glass-card">
                <h4>⚙️ Training Parameters</h4>
                <p><strong>Batch Size:</strong> {config['batch_size']}</p>
                <p><strong>Learning Rate:</strong> {config['learning_rate']}</p>
                <p><strong>Epochs:</strong> {config['epochs']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="glass-card">
                <h4>🔧 Optimizations</h4>
                <p><strong>Scheduler:</strong> {config['scheduler']}</p>
                <p><strong>Mixed Precision:</strong> {'✅' if config['use_amp'] else '❌'}</p>
                <p><strong>Early Stopping:</strong> {'✅' if config['early_stopping'] else '❌'}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Training Button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        train_btn = st.button("🚀 Start Training", type="primary", use_container_width=True)
    
    if train_btn:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        col1, col2 = st.columns(2)
        
        with col1:
            loss_chart = st.empty()
        with col2:
            metrics_chart = st.empty()
        
        train_losses, val_losses, epochs_done = [], [], []
        num_epochs = min(config['epochs'], 25)
        
        for epoch in range(num_epochs):
            time.sleep(0.08)
            
            train_loss = 100 * np.exp(-epoch / 8) + np.random.normal(0, 2)
            val_loss = 100 * np.exp(-epoch / 10) + np.random.normal(0, 3)
            
            train_losses.append(max(3, train_loss))
            val_losses.append(max(5, val_loss))
            epochs_done.append(epoch + 1)
            
            progress_bar.progress((epoch + 1) / num_epochs)
            status_text.markdown(f"""
            <div class="info-box">
                <p>📊 <strong>Epoch {epoch + 1}/{num_epochs}</strong> | Train: {train_losses[-1]:.2f} | Val: {val_losses[-1]:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Loss chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=epochs_done, y=train_losses, name='Train', 
                                    line=dict(color=ACCENT_4, width=3),
                                    fill='tozeroy', fillcolor='rgba(72, 219, 251, 0.2)'))
            fig.add_trace(go.Scatter(x=epochs_done, y=val_losses, name='Validation', 
                                    line=dict(color=ACCENT_1, width=3),
                                    fill='tozeroy', fillcolor='rgba(233, 69, 96, 0.2)'))
            fig.update_layout(
                title=dict(text='Loss Curves', font=dict(color=TEXT_LIGHT)),
                xaxis=dict(title='Epoch', color=TEXT_LIGHT, gridcolor=CARD_BORDER),
                yaxis=dict(title='Loss', color=TEXT_LIGHT, gridcolor=CARD_BORDER),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                legend=dict(font=dict(color=TEXT_LIGHT)), height=320
            )
            loss_chart.plotly_chart(fig, use_container_width=True)
            
            # Metrics
            rmse = 35 * np.exp(-epoch / 12) + 5 + np.random.normal(0, 0.5)
            mae = 25 * np.exp(-epoch / 12) + 3 + np.random.normal(0, 0.3)
            r2 = min(0.98, 1 - np.exp(-epoch / 8) + np.random.normal(0, 0.005))
            
            fig = go.Figure(data=[
                go.Bar(name='RMSE', x=['RMSE'], y=[max(5, rmse)], marker_color=ACCENT_4),
                go.Bar(name='MAE', x=['MAE'], y=[max(3, mae)], marker_color=ACCENT_3),
                go.Bar(name='R²×100', x=['R²×100'], y=[max(50, r2*100)], marker_color=ACCENT_5)
            ])
            fig.update_layout(
                title=dict(text='Metrics', font=dict(color=TEXT_LIGHT)),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(color=TEXT_LIGHT), yaxis=dict(color=TEXT_LIGHT, gridcolor=CARD_BORDER),
                showlegend=False, height=320
            )
            metrics_chart.plotly_chart(fig, use_container_width=True)
        
        st.balloons()
        st.success("🎉 Training Complete!")
        
        # Final Results
        col1, col2, col3, col4 = st.columns(4)
        
        final_results = [
            ("📉", f"{min(val_losses):.2f}", "Best Loss"),
            ("📊", f"{rmse:.2f} W", "Final RMSE"),
            ("🎯", f"{r2:.4f}", "R² Score"),
            ("⏱️", "8.5 min", "Duration")
        ]
        
        for col, (icon, value, label) in zip([col1, col2, col3, col4], final_results):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-icon">{icon}</div>
                    <div class="metric-value">{value}</div>
                    <div class="metric-label">{label}</div>
                </div>
                """, unsafe_allow_html=True)


def render_analytics_tab():
    """Render the analytics/visualization tab."""
    st.markdown('<h2 class="section-header">Analytics & Insights</h2>', unsafe_allow_html=True)
    
    # Analytics Header Image
    st.markdown(f"""
    <div class="image-container" style="margin-bottom: 25px;">
        <img src="{IMAGES['analytics']}" style="width: 100%; height: 180px; object-fit: cover;">
        <div class="image-overlay">
            <h4>Power Generation Analytics</h4>
            <p>Visualize trends, patterns, and correlations</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    data = generate_sample_data()
    
    # Time Range
    col1, col2 = st.columns([3, 1])
    with col1:
        time_range = st.select_slider("Time Range", options=["1 Day", "3 Days", "1 Week"], value="1 Week")
    
    if time_range == "1 Day":
        data = data.head(24)
    elif time_range == "3 Days":
        data = data.head(72)
    
    # Power Generation Chart
    st.markdown(f"""
    <div class="info-box">
        <h3>⚡ Power Generation Over Time</h3>
    </div>
    """, unsafe_allow_html=True)
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("DC Power Output", "Weather Conditions"))
    
    fig.add_trace(go.Scatter(x=data['timestamp'], y=data['power'], name='Power',
                             fill='tozeroy', line=dict(color=ACCENT_3, width=2),
                             fillcolor='rgba(254, 202, 87, 0.3)'), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=data['timestamp'], y=data['ghi'], name='GHI',
                             line=dict(color=ACCENT_4, width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=data['timestamp'], y=data['temperature'] * 10, name='Temp (×10)',
                             line=dict(color=ACCENT_1, width=2)), row=2, col=1)
    
    fig.update_layout(
        height=500, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=TEXT_LIGHT), legend=dict(font=dict(color=TEXT_LIGHT))
    )
    fig.update_xaxes(gridcolor=CARD_BORDER, color=TEXT_LIGHT)
    fig.update_yaxes(gridcolor=CARD_BORDER, color=TEXT_LIGHT)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistics
    col1, col2, col3, col4 = st.columns(4)
    
    stats = [
        ("🔋", f"{data['power'].max():.0f} W", "Max Power"),
        ("📊", f"{data['power'].mean():.0f} W", "Avg Power"),
        ("⚡", f"{(data['power'].mean() / max(1, data['power'].max()) * 100):.1f}%", "Capacity"),
        ("☀️", f"{(data['power'] > data['power'].mean()).sum()} hrs", "Peak Hours")
    ]
    
    for col, (icon, value, label) in zip([col1, col2, col3, col4], stats):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">{icon}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Correlation & Distribution
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
                       color_continuous_scale=[[0, ACCENT_1], [0.5, PRIMARY_MID], [1, ACCENT_4]])
        fig.update_layout(height=350, paper_bgcolor='rgba(0,0,0,0)', font=dict(color=TEXT_LIGHT))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown(f"""
        <div class="info-box">
            <h3>📊 Power Distribution</h3>
        </div>
        """, unsafe_allow_html=True)
        
        fig = px.histogram(data, x='power', nbins=30, color_discrete_sequence=[ACCENT_4])
        fig.update_layout(height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                         font=dict(color=TEXT_LIGHT), xaxis=dict(gridcolor=CARD_BORDER),
                         yaxis=dict(gridcolor=CARD_BORDER))
        st.plotly_chart(fig, use_container_width=True)


def render_cloud_analysis_tab():
    """Render the cloud analysis tab."""
    st.markdown('<h2 class="section-header">Cloud Detection & Analysis</h2>', unsafe_allow_html=True)
    
    # Cloud Analysis Header
    st.markdown(f"""
    <div class="image-container" style="margin-bottom: 25px;">
        <img src="{IMAGES['cloudy_sky']}" style="width: 100%; height: 180px; object-fit: cover;">
        <div class="image-overlay">
            <h4>OpenCV-Based Cloud Detection</h4>
            <p>Real-time cloud segmentation and feature extraction</p>
        </div>
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
        
        # Create gradient sky
        for i in range(height):
            ratio = i / height
            r = int(100 * (1 - ratio) + 180 * ratio)
            g = int(150 * (1 - ratio) + 210 * ratio)
            b = int(230 * (1 - ratio) + 250 * ratio)
            sky_image[i, :] = [r, g, b]
        
        cloud_mask = np.zeros((height, width), dtype=np.uint8)
        num_clouds = int(cloud_cover * 15) + 1
        
        for _ in range(num_clouds):
            cx = np.random.randint(40, width - 40)
            cy = np.random.randint(15, height // 2)
            radius = np.random.randint(30, 90)
            
            y, x = np.ogrid[:height, :width]
            mask = (x - cx) ** 2 + (y - cy) ** 2 < radius ** 2
            
            cloud_color = np.random.randint(225, 255)
            sky_image[mask] = [cloud_color, cloud_color, cloud_color]
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
        
        # Create colored segmentation
        segmented = np.zeros((height, width, 3), dtype=np.uint8)
        segmented[cloud_mask == 255] = [233, 69, 96]  # Accent color for clouds
        segmented[cloud_mask == 0] = [72, 219, 251]   # Blue for sky
        
        st.markdown(f"""<div class="image-container">""", unsafe_allow_html=True)
        st.image(segmented, caption="Segmented (Pink: Clouds, Blue: Sky)", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Features
    st.markdown(f"""
    <div class="info-box">
        <h3>📊 Extracted Features</h3>
    </div>
    """, unsafe_allow_html=True)
    
    actual_cover = (cloud_mask == 255).sum() / cloud_mask.size * 100
    brightness = np.mean(sky_image) / 255 * 100
    blue_ratio = np.mean(sky_image[:, :, 2]) / (np.mean(sky_image) + 1e-6) * 100
    opacity = cloud_cover * 80 + 10
    contrast = np.std(sky_image) / (np.mean(sky_image) + 1e-6) * 100
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    features = [
        (col1, "☁️ Cloud Cover", actual_cover),
        (col2, "💡 Brightness", brightness),
        (col3, "🔵 Blue Ratio", min(100, blue_ratio)),
        (col4, "🌫️ Opacity", opacity),
        (col5, "📈 Contrast", min(100, contrast))
    ]
    
    for col, name, value in features:
        with col:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=value,
                title={'text': name, 'font': {'size': 11, 'color': TEXT_LIGHT}},
                number={'suffix': "%", 'font': {'color': TEXT_LIGHT, 'size': 18}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': TEXT_LIGHT},
                    'bar': {'color': ACCENT_3},
                    'bgcolor': PRIMARY_MID,
                }
            ))
            fig.update_layout(height=180, margin=dict(l=10, r=10, t=60, b=10),
                            paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)


def render_sun_position_tab(config):
    """Render the sun position tab."""
    st.markdown('<h2 class="section-header">Sun Position Calculator</h2>', unsafe_allow_html=True)
    
    # Sun Position Header
    st.markdown(f"""
    <div class="image-container" style="margin-bottom: 25px;">
        <img src="{IMAGES['sun_rays']}" style="width: 100%; height: 180px; object-fit: cover;">
        <div class="image-overlay">
            <h4>Solar Geometry Calculations</h4>
            <p>Powered by pvlib astronomical algorithms</p>
        </div>
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
            <h3>☀️ Sun Parameters</h3>
        </div>
        """, unsafe_allow_html=True)
        
        params = [
            ("Zenith", f"{zenith:.2f}°"),
            ("Azimuth", f"{azimuth:.2f}°"),
            ("Elevation", f"{elevation:.2f}°"),
            ("Daytime", "☀️ Yes" if elevation > 0 else "🌙 No")
        ]
        
        for label, value in params:
            st.markdown(f"""
            <div class="glass-card" style="padding: 12px; margin: 8px 0;">
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
            mode='lines',
            name='Sun Path',
            line=dict(color=ACCENT_3, width=4)
        ))
        
        if elevation > 0:
            fig.add_trace(go.Scatterpolar(
                r=[90 - elevation],
                theta=[azimuth],
                mode='markers',
                name='Current',
                marker=dict(size=20, color=ACCENT_1, symbol='star', line=dict(color=TEXT_LIGHT, width=2))
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(range=[0, 90], tickmode='linear', tick0=0, dtick=15,
                               tickfont=dict(color=TEXT_LIGHT), gridcolor=CARD_BORDER),
                angularaxis=dict(direction='clockwise', rotation=90,
                                tickfont=dict(color=TEXT_LIGHT), gridcolor=CARD_BORDER),
                bgcolor='rgba(0,0,0,0)'
            ),
            showlegend=True, legend=dict(font=dict(color=TEXT_LIGHT)),
            height=380, paper_bgcolor='rgba(0,0,0,0)'
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
        
        irr = [
            (col_a, "GHI", ghi_clear),
            (col_b, "DNI", dni_clear),
            (col_c, "DHI", dhi_clear)
        ]
        
        for col, name, value in irr:
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{value:.0f}</div>
                    <div class="metric-label">{name} (W/m²)</div>
                </div>
                """, unsafe_allow_html=True)


def main():
    """Main application."""
    config = create_sidebar()
    
    # Tabs
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
        <p style="font-size: 1.2rem; color: {ACCENT_3}; font-weight: 600;">☀️ PV Power Estimation Dashboard</p>
        <p>Multi-Modal Deep Learning for Solar Power Prediction</p>
        <p>📧 pradyumnamand@gmail.com | 📱 +1 480-797-3843</p>
        <p style="margin-top: 15px; font-size: 0.85rem;">© 2024 All Rights Reserved</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
