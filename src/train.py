import torch
import torch.nn as nn
import torch.optim as optim
from src.models.fusion_model import FusionModel
from src.data.dataset import get_dataloader
import os
import argparse

def train(args):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Hyperparameters
    batch_size = args.batch_size
    learning_rate = args.lr
    num_epochs = args.epochs
    
    # Initialize Dataset and DataLoader
    # Assuming data exists at these paths (placeholders)
    csv_path = args.csv_path
    img_dir = args.img_dir
    
    if not os.path.exists(csv_path):
        print(f"Warning: CSV file {csv_path} not found. Using dummy data generation logic for testing if needed.")
        # In real usage, we'd exit or handle this. For this skeleton, we assume data is prepped.

    train_loader = get_dataloader(csv_path, img_dir, batch_size=batch_size, shuffle=True)
    
    # Initialize Model
    # weather_input_dim: temp, humidity, irradiance + apparent_zenith, azimuth = 3 + 2 = 5
    model = FusionModel(weather_input_dim=5).to(device)
    
    # Loss and Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training Loop
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, weather_seq, targets) in enumerate(train_loader):
            images = images.to(device)
            weather_seq = weather_seq.to(device)
            targets = targets.to(device).unsqueeze(1) # Match output shape
            
            # Forward pass
            outputs = model(images, weather_seq)
            loss = criterion(outputs, targets)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}')

    # Save the model
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/model.pth')
    print("Training complete. Model saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Solar Power Prediction Model')
    parser.add_argument('--csv_path', type=str, default='data/train.csv', help='Path to training CSV')
    parser.add_argument('--img_dir', type=str, default='data/images/', help='Path to images directory')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    
    args = parser.parse_args()
    train(args)
