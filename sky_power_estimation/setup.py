#!/usr/bin/env python
"""
Setup script for Sky Power Estimation package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
if readme_path.exists():
    long_description = readme_path.read_text(encoding="utf-8")
else:
    long_description = "Sky Image + Weather-Based DC Power Estimation"

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    requirements = requirements_path.read_text().strip().split("\n")
    requirements = [r.strip() for r in requirements if r.strip() and not r.startswith("#")]
else:
    requirements = [
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "opencv-python>=4.8.0",
        "Pillow>=10.0.0",
        "pvlib>=0.10.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
        "scikit-learn>=1.3.0",
    ]

setup(
    name="sky_power_estimation",
    version="1.0.0",
    author="Sky Power Estimation Team",
    author_email="team@example.com",
    description="Multi-modal deep learning for DC power estimation from sky images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/sky-power-estimation",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "wandb": ["wandb>=0.15.0"],
        "full": [
            "albumentations>=1.3.0",
            "timm>=0.9.0",
            "tensorboard>=2.14.0",
            "wandb>=0.15.0",
            "ephem>=4.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "sky-power-train=scripts.train:main",
            "sky-power-predict=scripts.predict:main",
            "sky-power-evaluate=scripts.evaluate:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
