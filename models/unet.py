import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    (convolution => [BN] => ReLU) * 2
    """
    # The spatial dimension would not be changed here, just the channel dimension
    def __init__(self, in_channels, out_channels, norm_type='batch'):
        super(DoubleConv, self).__init__()
    
        if norm_type == 'batch':
            norm_layer = nn.BatchNorm2d
        elif norm_type == 'instance':
            norm_layer = nn.InstanceNorm2d
        else:
            norm_layer = nn.Identity
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            norm_layer(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels, norm_type='batch'):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),  # if the stride is not specified, it equals to the kernel size
            DoubleConv(in_channels, out_channels, norm_type)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True, norm_type='batch'):
        super(Up, self).__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        self.bilinear = bilinear
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, norm_type)
        else:
            x1_ch = in_channels - out_channels
            self.up = nn.ConvTranspose2d(x1_ch, x1_ch, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, norm_type)

    def forward(self, x1, x2):  # upsampling the first feature map then concatenate with the skip connection
        x1 = self.up(x1)
        if x1.shape[2:] != x2.shape[2:]:
            diffY = x2.shape[2] - x1.shape[2]
            diffX = x2.shape[3] - x1.shape[3]

            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    UNet architecture
    """
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 base_channels=64, 
                 downsample_levels=4, 
                 bilinear=False,
                 norm_type='batch'):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear  # use bilinear interpolation for upsampling or use transposed convolution
        self.downsample_levels = downsample_levels
        
        # Calculate channels for each level
        ch_mult = [base_channels * (2 ** i) for i in range(downsample_levels + 1)]  # 64 * [1, 2, 4, 8, 16]
        
        # Encoder (downsampling)
        self.inc = DoubleConv(in_channels, ch_mult[0], norm_type)  # in_channels -> 64
        self.down_layers = nn.ModuleList()
        
        for i in range(downsample_levels):
            self.down_layers.append(Down(ch_mult[i], ch_mult[i+1], norm_type))
            # i=0: bs, 64, 128, 128 -> bs, 128, 64, 64
            # i=1: bs, 128, 64, 64 -> bs, 256, 32, 32
            # i=2: bs, 256, 32, 32 -> bs, 512, 16, 16
            # i=3: bs, 512, 16, 16 -> bs, 1024, 8, 8
        
        # Decoder (upsampling)
        factor = 2 if bilinear else 1
        self.up_layers = nn.ModuleList()
        
        for i in range(downsample_levels):  # 0, 1, 2, 3
            in_ch = ch_mult[downsample_levels - i]  
            out_ch = ch_mult[downsample_levels - i - 1] // factor
            self.up_layers.append(Up(in_ch, out_ch, bilinear, norm_type))
            # i=0, in_ch=1024, out_ch=512//factor, if factor==1, out_ch=512 otherwise 256
            # i=1, in_ch=512, out_ch=256
            # i=2, in_ch=256, out_ch=128
            # i=3, in_ch=128, out_ch=64
        
        # Output layer
        self.outc = OutConv(ch_mult[0], out_channels)
        
        # Store skip connections for external access
        self.skip_connections = []
        
    def forward(self, x):
        # Clear previous skip connections
        self.skip_connections = []
        
        # Encoder path
        x1 = self.inc(x)
        self.skip_connections.append(x1)
        
        current = x1
        for down_layer in self.down_layers:
            current = down_layer(current)  # the output of the total layer will be stored in current, not the output of the block
            self.skip_connections.append(current)
        
        # Decoder path
        current = self.skip_connections.pop()  # Bottleneck
        
        for up_layer in self.up_layers:
            skip = self.skip_connections.pop()
            current = up_layer(current, skip)
        
        # Output
        logits = self.outc(current)
        return logits
    
    def get_features_at_level(self, x, level):
        """
        Get features at a specific encoder level
        level 0: after initial conv
        level 1: after first downsampling
        etc.
        """
        if level == 0:
            return self.inc(x)
        
        current = self.inc(x)
        for i, down_layer in enumerate(self.down_layers):
            current = down_layer(current)
            if i + 1 == level:
                return current
        
        return current  # Return bottleneck if level too high

