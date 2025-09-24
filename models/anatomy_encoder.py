import torch.nn as nn
import torch.nn.functional as F

from .unet import UNet, DoubleConv, Down, Up, OutConv
from .layers import RoundingLayer


class AnatomyEncoders(nn.Module):
    def __init__(self, 
                 modalities,
                 in_channels,
                 out_channels,
                 base_channels,
                 ch_mult,
                 norm_type,
                 use_rounding,
                 bilinear,
                 ):
        super(AnatomyEncoders, self).__init__()
        self.modalities = modalities
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.ch_mult = ch_mult
        self.norm_type = norm_type
        self.use_rounding = use_rounding
        
        self.encoders = nn.ModuleDict()
        
        for modality in modalities:
            self.encoders[f'encoder_{modality}'] = self._build_separated_encoder(in_channels, base_channels, ch_mult, norm_type)
        
        if self.use_rounding:
            self.rounding = RoundingLayer()
        
        self.shared_decoder = self._build_shared_decoder(out_channels, base_channels, ch_mult, bilinear, norm_type)
        
    def _build_separated_encoder(self, in_channels, base_channels, ch_mult, norm_type):
        encoder_layers = nn.ModuleList()
        in_conv = DoubleConv(in_channels, ch_mult[0]*base_channels, norm_type)
        encoder_layers.append(in_conv)
        # Downsampling layers
        for i in range(len(ch_mult)-1):  # 0, 1, 2, 3
            in_ch = ch_mult[i]*base_channels
            out_ch = ch_mult[i+1]*base_channels
            encoder_layers.append(Down(in_ch, out_ch, norm_type))
        return nn.Sequential(*encoder_layers)
    
    def _build_shared_decoder(self, out_channels, base_channels, ch_mult, bilinear, norm_type):  # ch_mult = (1, 2, 4, 8, 16)
        self.shared_decoder_layers = nn.ModuleList()
        for i in reversed(range(len(ch_mult)-1)):  # 3, 2, 1, 0
            in_ch = ch_mult[i+1]*base_channels
            out_ch = ch_mult[i]*base_channels
            self.shared_decoder_layers.append(Up(in_ch+out_ch, out_ch, bilinear, norm_type))
        self.outc = OutConv(ch_mult[0]*base_channels, out_channels)
    
    def _encode_with_skips(self, x, encoder):
        """
        Apply encoder and collect skip connections like in original UNet
        """
        skip_connections = []
        current = x
        
        for i, layer in enumerate(encoder):
            current = layer(current)

            if i < len(encoder) - 1:
                skip_connections.append(current)
                
        return current, skip_connections # Last one is bottleneck, others are skips
    
    # def _get_norm_layer(self, channels):
    #     if self.norm_type == 'batch':
    #         return nn.BatchNorm2d(channels)
    #     elif self.norm_type == 'instance':
    #         return nn.InstanceNorm2d(channels)
    #     else:
    #         return nn.Identity() 
        
    def _decode_with_shared_decoder(self, bottleneck_features, skip_connections):
        current = bottleneck_features
        for up_layer in self.shared_decoder_layers:
            skip = skip_connections.pop()
            current = up_layer(current, skip)
        # Output
        output = self.outc(current)
        return output
    
    def forward(self, inputs):
        outputs = {}
        
        for modality in self.modalities:
            if f'input_{modality}' in inputs:
                x = inputs[f'input_{modality}']
                
                # Step 1: Encode with modality-specific encoder (downsampling only)
                encoder_features, skip_connections = self._encode_with_skips(
                    x, self.encoders[f'encoder_{modality}']
                )
                # Step 2: Pass through shared decoder with skip connections
                decoded_features = self._decode_with_shared_decoder(
                    encoder_features, skip_connections
                )
                
                # Step 3: Apply shared final convolution and softmax
                anatomy = F.softmax(decoded_features, dim=1)
                
                # Step 4: Apply rounding if enabled
                if self.use_rounding:
                    anatomy = self.rounding(anatomy)
                
                outputs[f'anatomy_{modality}'] = anatomy
        
        return outputs
    
    # def get_encoder_for_modality(self, modality):

    #     class SingleModalityEncoder(nn.Module):
    #         def __init__(self, parent, modality):
    #             super(SingleModalityEncoder, self).__init__()
    #             self.parent = parent
    #             self.modality = modality
            
    #         def forward(self, x):
    #             inputs = {f'input_{self.modality}': x}
    #             outputs = self.parent(inputs)
    #             return outputs[f'anatomy_{self.modality}']
        
    #     return SingleModalityEncoder(self, modality)