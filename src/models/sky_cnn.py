import torch
import torch.nn as nn
import torchvision.models as models

class SkyCNN(nn.Module):
    def __init__(self, backbone_name='resnet18', pretrained=True, feature_dim=128):
        super(SkyCNN, self).__init__()
        
        if backbone_name == 'resnet18':
            # Load pre-trained ResNet18
            self.backbone = models.resnet18(pretrained=pretrained)
            # Remove the classification head (fc layer)
            num_ftrs = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone_name == 'mobilenet_v2':
            self.backbone = models.mobilenet_v2(pretrained=pretrained)
            num_ftrs = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Backbone {backbone_name} not supported")
        
        # Projection head to get desired feature dimension
        self.projection = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, feature_dim)
        )

    def forward(self, x):
        # x shape: (batch_size, 3, H, W)
        features = self.backbone(x)
        out = self.projection(features)
        return out

if __name__ == "__main__":
    # Test the model
    model = SkyCNN(backbone_name='resnet18')
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
