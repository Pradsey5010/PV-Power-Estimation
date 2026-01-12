"""
Configuration utilities.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config: Dict[str, Any], config_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def merge_configs(
    base_config: Dict[str, Any],
    override_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Merge two configurations, with override taking precedence.
    
    Args:
        base_config: Base configuration
        override_config: Override configuration
        
    Returns:
        Merged configuration
    """
    result = base_config.copy()
    
    for key, value in override_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result


def get_default_config() -> Dict[str, Any]:
    """Get default configuration."""
    return {
        "data": {
            "image_size": 224,
            "sequence_length": 12,
            "train_split": 0.7,
            "val_split": 0.15,
            "test_split": 0.15,
            "num_workers": 4
        },
        "location": {
            "latitude": 37.7749,
            "longitude": -122.4194,
            "altitude": 10,
            "timezone": "US/Pacific"
        },
        "model": {
            "image_encoder": {
                "backbone": "resnet50",
                "pretrained": True,
                "freeze_backbone": False,
                "image_feature_dim": 512
            },
            "temporal_encoder": {
                "type": "transformer",
                "hidden_dim": 256,
                "num_layers": 3,
                "num_heads": 8,
                "dropout": 0.1,
                "bidirectional": True
            },
            "fusion": {
                "method": "attention",
                "hidden_dim": 512,
                "dropout": 0.2
            },
            "output": {
                "hidden_dims": [256, 128],
                "dropout": 0.3,
                "activation": "gelu"
            }
        },
        "training": {
            "batch_size": 32,
            "epochs": 100,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "scheduler": {
                "type": "cosine",
                "warmup_epochs": 5,
                "min_lr": 0.00001
            },
            "early_stopping": {
                "patience": 15,
                "min_delta": 0.0001
            },
            "gradient_clip": 1.0,
            "mixed_precision": True
        },
        "loss": {
            "type": "mse",
            "huber_delta": 1.0
        },
        "paths": {
            "data_dir": "./data",
            "output_dir": "./outputs",
            "checkpoint_dir": "./checkpoints",
            "log_dir": "./logs"
        }
    }


class Config:
    """Configuration class with attribute access."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert back to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result
    
    def __repr__(self):
        return f"Config({self.to_dict()})"


if __name__ == "__main__":
    # Test config utilities
    print("Testing configuration utilities...")
    
    # Get default config
    config = get_default_config()
    print(f"Default config keys: {list(config.keys())}")
    
    # Create Config object
    cfg = Config(config)
    print(f"\nModel backbone: {cfg.model.image_encoder.backbone}")
    print(f"Learning rate: {cfg.training.learning_rate}")
    print(f"Batch size: {cfg.training.batch_size}")
