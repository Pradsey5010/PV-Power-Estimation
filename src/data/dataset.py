import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import cv2
from pvlib.solarposition import get_solarposition

class SolarDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None, seq_len=10):
        """
        Args:
            csv_file (string): Path to the csv file with weather data and labels.
            image_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            seq_len (int): Sequence length for LSTM input.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform
        self.seq_len = seq_len
        
        # Preprocessing: Ensure timestamps are sorted
        self.data_frame['timestamp'] = pd.to_datetime(self.data_frame['timestamp'])
        self.data_frame.sort_values('timestamp', inplace=True)
        self.data_frame.reset_index(drop=True, inplace=True)

    def __len__(self):
        return len(self.data_frame) - self.seq_len

    def __getitem__(self, idx):
        # We need a sequence of weather data ending at the current time for the LSTM
        # And the image corresponding to the current time (last in sequence)
        
        # LSTM Input: sequence of length seq_len
        # Features should include weather data + solar position
        # Assuming columns: ['timestamp', 'temp', 'humidity', 'irradiance', 'dc_power', 'image_name']
        
        # Select sequence window
        start_idx = idx
        end_idx = idx + self.seq_len
        
        sequence_data = self.data_frame.iloc[start_idx:end_idx]
        current_row = sequence_data.iloc[-1]
        
        # Extract Weather Features (normalize/scale these in a real pipeline)
        # Example features: temp, humidity, irradiance, etc.
        # Adding solar position
        
        # Calculate solar position for the sequence (can be pre-calculated)
        lat, lon = 37.7749, -122.4194 # Example: San Francisco
        times = pd.DatetimeIndex(sequence_data['timestamp'])
        solpos = get_solarposition(times, lat, lon)
        
        # Construct feature vector
        # This is a placeholder; real implementation needs specific columns
        # shape: (seq_len, num_features)
        weather_features = sequence_data[['temp', 'humidity', 'irradiance']].values
        solar_features = solpos[['apparent_zenith', 'azimuth']].values
        
        combined_features = np.hstack((weather_features, solar_features)).astype(np.float32)
        
        # Load Image
        img_name = os.path.join(self.image_dir, current_row['image_name'])
        image = cv2.imread(img_name)
        if image is None:
            # Placeholder for missing image
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        if self.transform:
            image = self.transform(image)
        else:
            # Simple transform if none provided
            image = cv2.resize(image, (224, 224))
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        # Target Label (DC Power)
        target = torch.tensor(current_row['dc_power'], dtype=torch.float32)
        
        return image, torch.from_numpy(combined_features), target

def get_dataloader(csv_path, img_dir, batch_size=32, shuffle=True):
    dataset = SolarDataset(csv_file=csv_path, image_dir=img_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

if __name__ == "__main__":
    # Create dummy data for testing
    os.makedirs('data_test', exist_ok=True)
    
    # Dummy CSV
    dates = pd.date_range(start='2023-01-01', periods=100, freq='10min')
    df = pd.DataFrame({
        'timestamp': dates,
        'temp': np.random.rand(100) * 30,
        'humidity': np.random.rand(100) * 100,
        'irradiance': np.random.rand(100) * 1000,
        'dc_power': np.random.rand(100) * 5000,
        'image_name': [f'img_{i}.jpg' for i in range(100)]
    })
    df.to_csv('data_test/data.csv', index=False)
    
    # Test Dataloader
    dataset = SolarDataset('data_test/data.csv', 'data_test')
    print(f"Dataset length: {len(dataset)}")
    
    img, weather, target = dataset[0]
    print(f"Image shape: {img.shape}")
    print(f"Weather shape: {weather.shape}")
    print(f"Target: {target}")
