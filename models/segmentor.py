import torch
import torch.nn as nn
import torch.nn.functional as F

class Segmentor(nn.Module):
    def __init__(self, in_channels, num_classes, base_channels=64):
        super(Segmentor, self).__init__()
        
        self.num_classes = num_classes
        
        # Simple convolutional layers for segmentation
        self.seg_layers = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),  
            
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(base_channels, base_channels // 2, 3, padding=1),
            nn.BatchNorm2d(base_channels // 2),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(base_channels // 2, num_classes, 1, padding=0)
        )
        
    def forward(self, anatomy_features):
        logits = self.seg_layers(anatomy_features)
        
        # Apply softmax to get probabilities
        probabilities = F.softmax(logits, dim=1)  # [B, num_class, H, W]
        
        return probabilities
