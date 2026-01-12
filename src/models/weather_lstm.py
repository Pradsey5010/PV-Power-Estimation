import torch
import torch.nn as nn

class WeatherLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(WeatherLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2 if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

if __name__ == "__main__":
    # Test the model
    # input_dim: e.g., temperature, humidity, wind_speed, solar_irradiance, sun_azimuth, sun_elevation
    model = WeatherLSTM(input_dim=6, hidden_dim=64, num_layers=2, output_dim=32)
    dummy_input = torch.randn(4, 10, 6) # batch=4, seq_len=10, features=6
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
