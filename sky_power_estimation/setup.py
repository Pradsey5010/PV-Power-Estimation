#!/usr/bin/env python
"""Setup script for PV Power Estimation package."""

from setuptools import setup, find_packages

setup(
    name="sky_power_estimation",
    version="1.0.0",
    author="PV Power Estimation Team",
    description="Multi-modal deep learning for DC power estimation from sky images",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "timm>=0.9.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "opencv-python>=4.8.0",
        "Pillow>=10.0.0",
        "streamlit>=1.29.0",
        "plotly>=5.18.0",
        "scikit-learn>=1.3.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "full": [
            "pvlib>=0.10.0",
            "albumentations>=1.3.0",
            "tensorboard>=2.14.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "pv-dashboard=run_dashboard:main",
        ],
    },
)
