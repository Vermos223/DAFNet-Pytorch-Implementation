import torch
import torch.nn as nn
from .layers import spectral_norm_conv2d
from einops import rearrange


class BaseDiscriminator(nn.Module):
    def __init__(self,
                 resolution=128,
                 in_channels=1,
                 base_channels=64,
                 ch_mult=(1,2,4,8),
                 use_spectral_norm=True):
        super(BaseDiscriminator, self).__init__()
        
        self.in_channels = in_channels
        self.use_spectral_norm = use_spectral_norm
        self.base_channels = base_channels 
        
        conv_layer = spectral_norm_conv2d if use_spectral_norm else nn.Conv2d
        
        self.layers = nn.ModuleList()

        for i in range(len(ch_mult)):
            c_in = in_channels if i==0 else base_channels * ch_mult[i-1]
            c_out = base_channels * ch_mult[i]
            stride = 2 if i < len(ch_mult) - 1 else 1
            self.layers.append(
                nn.Sequential(conv_layer(c_in, c_out, kernel_size=4, stride=stride, padding=0),
                nn.LeakyReLU(0.2, inplace=True)   
                )
            )
        device = next(self.parameters()).device
        with torch.no_grad():
            h = torch.zeros(1, in_channels, resolution, resolution).to(device)
            for layer in self.layers:
                h = layer(h)
            in_features = h.shape[1] * h.shape[2] * h.shape[3]
        
        self.linear = nn.Linear(in_features, 1)
        
    def forward(self, x):
        h = x
        for layer in self.layers:
            h = layer(h)
        h = rearrange(h, 'b c h w -> b (c h w)')
        out = self.linear(h)
        return out
            

# class MaskDiscriminator(BaseDiscriminator):
#     def __init__(self, resolution, in_channels, base_channels=64, use_spectral_norm=True):
#         super(MaskDiscriminator, self).__init__(
#             resolution=resolution,
#             in_channels=in_channels,
#             base_channels=base_channels,
#             use_spectral_norm=use_spectral_norm
#         )


# class ImageDiscriminator(BaseDiscriminator):
#     def __init__(self, resolution, in_channels, base_channels=64, use_spectral_norm=True):
#         super(ImageDiscriminator, self).__init__(
#             resolution=resolution,
#             in_channels=in_channels,
#             base_channels=base_channels,
#             use_spectral_norm=use_spectral_norm
#         )


# class PatchGANDiscriminator(nn.Module):
#     """
#     PatchGAN discriminator (70x70 patches)
#     """
    
#     def __init__(self, in_channels, resolution, base_channels=64, num_layers=3, use_spectral_norm=True):
#         super(PatchGANDiscriminator, self).__init__()
        
#         conv_layer = spectral_norm_conv2d if use_spectral_norm else nn.Conv2d
        
#         # Build discriminator layers
#         layers = []
        
#         # First layer (no normalization)
#         layers.extend([
#             conv_layer(in_channels, base_channels, 4, stride=2, padding=1),
#             nn.LeakyReLU(0.2, inplace=True)
#         ])
        
#         # Intermediate layers
#         current_filters = base_channels
#         for i in range(1, num_layers):
#             next_filters = min(current_filters * 2, 512)
#             layers.extend([
#                 conv_layer(current_filters, next_filters, 4, stride=2, padding=1),
#                 nn.InstanceNorm2d(next_filters),
#                 nn.LeakyReLU(0.2, inplace=True)
#             ])
#             current_filters = next_filters
        
#         # Final layers
#         next_filters = min(current_filters * 2, 512)
#         layers.extend([
#             conv_layer(current_filters, next_filters, 4, stride=1, padding=1),
#             nn.InstanceNorm2d(next_filters),
#             nn.LeakyReLU(0.2, inplace=True),
#             conv_layer(next_filters, 1, 4, stride=1, padding=1)
#         ])
        
#         self.model = nn.Sequential(*layers)
        
#     def forward(self, x):
#         return self.model(x)


# class MultiScaleDiscriminator(nn.Module):
#     """
#     Multi-scale discriminator that operates at different image scales
#     """
    
#     def __init__(self, in_channels, resolution, num_discriminators=3, base_channels=64, 
#                  use_spectral_norm=True):
#         super(MultiScaleDiscriminator, self).__init__()
        
#         self.num_discriminators = num_discriminators
        
#         # Create discriminators for different scales
#         self.discriminators = nn.ModuleList()
#         for i in range(num_discriminators):
#             self.discriminators.append(
#                 PatchGANDiscriminator(
#                     in_channels=in_channels,
#                     resolution=resolution,
#                     base_channels=base_channels,
#                     use_spectral_norm=use_spectral_norm
#                 )
#             )
        
#         # Downsampling for multi-scale
#         self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)
        
#     def forward(self, x):
#         """
#         Forward pass through all discriminators at different scales
        
#         Returns:
#             List of discriminator outputs for each scale
#         """
#         outputs = []
#         current_input = x
        
#         for discriminator in self.discriminators:
#             outputs.append(discriminator(current_input))
#             current_input = self.downsample(current_input)
        
#         return outputs


# Factory function for creating discriminators
def create_discriminator( **kwargs): 
    """
    Factory function to create different types of discriminators
    Args:
        discriminator_type: Type of discriminator ('base', 'mask', 'image', 'patchgan', 'multiscale')
        **kwargs: Additional arguments for discriminator
    Returns:
        Discriminator instance
    """
    # if discriminator_type == 'base':
    #     return BaseDiscriminator(**kwargs)
    # elif discriminator_type == 'mask':
    #     return MaskDiscriminator(**kwargs)
    # elif discriminator_type == 'image':
    #     return ImageDiscriminator(**kwargs)
    # elif discriminator_type == 'patchgan':
    #     return PatchGANDiscriminator(in_channels, resolution, **kwargs)
    # elif discriminator_type == 'multiscale':
    #     return MultiScaleDiscriminator(in_channels, resolution, **kwargs)
    # else:
    #     raise ValueError(f"Unknown discriminator type: {discriminator_type}")
    return BaseDiscriminator(**kwargs)



if __name__ == "__main__":
    # testing
    img_arch_params = {
        'in_channels': 1,
        'resolution': 256,
        'base_channels': 16,
        'ch_mult': (1,2,4,8,16),
        'use_spectral_norm': True
    }
    mask_arch_params ={
        'in_channels': 4,
        'resolution': 128,
        'base_channels': 16,
        'ch_mult': (1,2,4,8),
        'use_spectral_norm': True
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_discriminator = BaseDiscriminator(**img_arch_params).to(device)
    mask_discriminator = BaseDiscriminator(**mask_arch_params).to(device)
    x = torch.randn(2, 1, 128, 128).to(device)
    x_mask = torch.randn(2, 4, 128, 128).to(device)
    print(image_discriminator(x).shape)
    print(mask_discriminator(x_mask).shape)
    