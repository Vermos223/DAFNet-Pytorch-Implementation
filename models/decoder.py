import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import SPADEBlock, FiLMLayer
from einops import rearrange


class FiLMDecoder(nn.Module):
    """
    FiLM-based decoder
    """
    def __init__(self, in_channels, base_channels, modality_dim=8, num_layers=4):
        super(FiLMDecoder, self).__init__()
        
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.modality_dim = modality_dim
        
        # Initial convolution on anatomy
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # FiLM layers
        self.film_layers = nn.ModuleList([
            FiLMLayer(base_channels, modality_dim) for _ in range(num_layers)
        ])
        
    def forward(self, anatomy, modality_vector):
        x = self.initial_conv(anatomy)
        
        for film_layer in self.film_layers:
            x = film_layer(x, modality_vector)
        
        return x


class SPADEDecoder(nn.Module):
    """
    SPADE-based decoder
    The input modality factor is a vector in the shape of [B, D], then linearly affine to B, C, h, w,
    followed by iteratively upsampling and SPADE blocks, finally output the image in the shape of [B, 1, H, W]
    """
    def __init__(self, 
                 resolution=128,  # the spatial resolution of the input anatomy facotr and the output reconstructed image
                 modality_dim=8,  # the dimension of the input modality factor
                 base_channels=128,  # base channels of the SPADE blocks
                 condition_channels=8,
                 ch_div=(1, 1, 1, 2, 4, 4)
                 ):
        super(SPADEDecoder, self).__init__()
        
        self.resolution = resolution
        self.modality_dim = modality_dim
        self.base_channels = base_channels
        
        self.ch_div = tuple(ch_div)
        
        self.in_ch_div = (1,) + self.ch_div
        
        self.initial_resolution = resolution // (2 **(len(ch_div) - 1))  # 128 // 32 = 4
         
        # Project modality vector to initial feature map
        self.modality_proj = nn.Sequential(
            nn.Linear(modality_dim, 
                      (self.initial_resolution ** 2) * base_channels // self.in_ch_div[0]),
            nn.ReLU(inplace=True)
        )
        
        self.spade_blocks = nn.ModuleList()
        curres = self.initial_resolution  
        for i in range(len(self.ch_div)):
            spade_layer = nn.Module()
            in_channels = base_channels // self.in_ch_div[i]
            out_channels = base_channels // self.ch_div[i]
            spade_layer.spade = SPADEBlock(in_channels, out_channels, condition_channels)
            if curres != resolution:
                spade_layer.upsample = nn.Upsample(scale_factor=2, mode='nearest')
                curres *= 2
            self.spade_blocks.append(spade_layer)
        
    def forward(self, anatomy, modality_vector):
        # Project modality vector to initial feature map
        x = self.modality_proj(modality_vector)
        x = rearrange(x, 'b (h w c) -> b c h w', h=self.initial_resolution, w=self.initial_resolution)
        
        for layers in self.spade_blocks:
            x = layers.spade(x, anatomy)
            if hasattr(layers, 'upsample'):
                x = layers.upsample(x)   
        return x
        

class Decoder(nn.Module):
    """
    Main decoder that supports both FiLM and SPADE conditioning
    """
    def __init__(self,
                 decoder_type='film',
                 decoder_params={},
                 out_channels=None,
                ):
        super(Decoder, self).__init__()
        
        self.decoder_type = decoder_type
        
        if decoder_type == 'film':
            self.decoder = FiLMDecoder(**decoder_params)
            output_channels = out_channels  # FiLM decoder output channels
        elif decoder_type == 'spade':
            self.decoder = SPADEDecoder(**decoder_params)
            output_channels = out_channels  # SPADE decoder output channels
        else:
            raise ValueError(f"Unknown decoder_type: {decoder_type}")
        
        # Final output layer
        self.output_conv = nn.Conv2d(output_channels, 1, kernel_size=1, padding=0)
        
    def forward(self, anatomy, modality_vector):
        """
        Forward pass
        
        Args:
            anatomy: Anatomy features [batch, channels, height, width]
            modality_vector: Modality vector [batch, modality_dim]
            
        Returns:
            Generated image [batch, 1, height, width]
        """
        # Pass through decoder
        features = self.decoder(anatomy, modality_vector)
        
        # Final output with tanh activation
        output = self.output_conv(features)
        output = torch.tanh(output)
        
        return output