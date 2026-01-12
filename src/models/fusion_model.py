import torch
import torch.nn as nn
from src.models.sky_cnn import SkyCNN
from src.models.weather_lstm import WeatherLSTM

class FusionModel(nn.Module):
    def __init__(self, 
                 cnn_backbone='resnet18', 
                 cnn_feature_dim=128,
                 weather_input_dim=6, 
                 lstm_hidden_dim=64, 
                 lstm_num_layers=2, 
                 lstm_output_dim=32,
                 final_output_dim=1): # 1 for DC Power
        super(FusionModel, self).__init__()
        
        # Image Branch
        self.sky_cnn = SkyCNN(backbone_name=cnn_backbone, feature_dim=cnn_feature_dim)
        
        # Weather Sequence Branch
        self.weather_lstm = WeatherLSTM(input_dim=weather_input_dim, 
                                        hidden_dim=lstm_hidden_dim, 
                                        num_layers=lstm_num_layers, 
                                        output_dim=lstm_output_dim)
        
        # Fusion Layer
        fusion_input_dim = cnn_feature_dim + lstm_output_dim
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, final_output_dim)
        )
        
    def forward(self, image, weather_seq):
        # image shape: (batch_size, 3, H, W)
        # weather_seq shape: (batch_size, seq_len, weather_features)
        
        img_features = self.sky_cnn(image)
        weather_features = self.weather_lstm(weather_seq)
        
        # Concatenate features
        fused = torch.cat((img_features, weather_features), dim=1)
        
        # Final prediction
        output = self.fusion_head(fused)
        return output

if __name__ == "__main__":
    # Test
    model = FusionModel()
    dummy_img = torch.randn(2, 3, 224, 224)
    dummy_weather = torch.randn(2, 10, 6)
    output = model(dummy_img, dummy_weather)
    print(f"Prediction shape: {output.shape}")
    print(f"Prediction: {output}")
