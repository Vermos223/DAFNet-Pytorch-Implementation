import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class SpectralNorm(nn.Module):
    """
    Spectral Normalization layer wrapper
    Can be applied to any layer with weight parameters
    """
    def __init__(self, module, name='weight', n_power_iterations=1, eps=1e-12):
        super(SpectralNorm, self).__init__()
        self.module = spectral_norm(module, name=name, 
                                   n_power_iterations=n_power_iterations, 
                                   eps=eps)
    
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def spectral_norm_conv2d(in_channels, out_channels, kernel_size, 
                        stride=1, padding=0, dilation=1, groups=1, bias=True):
    """
    Create a Conv2d layer with spectral normalization
    """
    conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                    stride=stride, padding=padding, dilation=dilation, 
                    groups=groups, bias=bias)
    return SpectralNorm(conv)


def spectral_norm_linear(in_features, out_features, bias=True):
    """
    Create a Linear layer with spectral normalization
    """
    linear = nn.Linear(in_features, out_features, bias=bias)
    return SpectralNorm(linear)
