import torch
import torch.nn as nn
import torch.nn.functional as F


class SPADE(nn.Module):
    """
    SPADE normalization layer
    """
    def __init__(self, in_channels, condition_channels=8):
        super(SPADE, self).__init__()
        
        self.in_channels = in_channels
        self.condition_channels = condition_channels
         
        # Instance normalization (no learnable parameters)
        self.instance_norm = nn.InstanceNorm2d(in_channels, affine=False)
        
        # Conditioning network
        self.condition_conv = nn.Conv2d(condition_channels, 128, 3, padding=1)
        self.params_conv = nn.Conv2d(128, in_channels*2, 3, padding=1)

    def forward(self, x, condition_input):
        normalized = self.instance_norm(x)
        # Generate conditioning parameters
        if condition_input.shape[2:] != x.shape[2:]:
            condition_input = F.interpolate(condition_input, size=x.shape[2:], mode='nearest')
        condition_features = F.relu(self.condition_conv(condition_input))
        gamma, beta = torch.chunk(self.params_conv(condition_features), 2, dim=1)
        return normalized * (1 + gamma) + beta


class SPADEBlock(nn.Module):
    def __init__(self, in_channels, out_channels, condition_channels=128):
        super(SPADEBlock, self).__init__()
        
        self.short_conv = True if in_channels != out_channels else False
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.spade1 = SPADE(out_channels, condition_channels)
       
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1) 
        self.spade2 = SPADE(out_channels, condition_channels)
        
        if self.short_conv:
            self.shortcut_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
            self.shortcut_spade = SPADE(out_channels, condition_channels)
    
    def forward(self, x, condition_input):
        h = self.conv1(x)
        h = self.spade1(h, condition_input)
        h = F.leaky_relu(h, 0.2)
        
        h = self.conv2(h)
        h = self.spade2(h, condition_input)
        h = F.leaky_relu(h, 0.2)
        
        if self.short_conv:
            shortcut = self.shortcut_conv(x)
            shortcut = self.shortcut_spade(shortcut, condition_input)
        else:
            shortcut = x
            
        return h + shortcut