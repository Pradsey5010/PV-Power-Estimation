import torch
import pandas as pd
import numpy as np
import os
import cv2
from src.models.fusion_model import FusionModel
from src.data.dataset import SolarDataset, get_dataloader
from src.train import train

def test_full_pipeline():
    print("Setting up dummy data...")
    # 1. Setup Dummy Data
    os.makedirs('test_data/images', exist_ok=True)
    
    # Create 20 dummy images
    for i in range(20):
        # Create a random image
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        cv2.imwrite(f'test_data/images/img_{i}.jpg', img)
        
    # Create dummy CSV
    dates = pd.date_range(start='2023-01-01 08:00', periods=20, freq='10min')
    df = pd.DataFrame({
        'timestamp': dates,
        'temp': np.random.rand(20) * 30,
        'humidity': np.random.rand(20) * 100,
        'irradiance': np.random.rand(20) * 1000,
        'dc_power': np.random.rand(20) * 5000,
        'image_name': [f'img_{i}.jpg' for i in range(20)]
    })
    df.to_csv('test_data/data.csv', index=False)
    
    print("Testing Dataset and DataLoader...")
    # 2. Test Dataset
    dataset = SolarDataset('test_data/data.csv', 'test_data/images', seq_len=5)
    print(f"Dataset length: {len(dataset)}")
    
    loader = get_dataloader('test_data/data.csv', 'test_data/images', batch_size=4, shuffle=True)
    
    # Get one batch
    images, weather_seq, targets = next(iter(loader))
    print(f"Batch shapes - Images: {images.shape}, Weather: {weather_seq.shape}, Targets: {targets.shape}")
    
    print("Testing Model Forward Pass...")
    # 3. Test Model
    # weather input dim is 3 (temp, hum, irr) + 2 (zenith, azimuth) = 5
    model = FusionModel(weather_input_dim=5)
    output = model(images, weather_seq)
    print(f"Model output shape: {output.shape}")
    
    print("Pipeline test passed!")

if __name__ == "__main__":
    test_full_pipeline()
