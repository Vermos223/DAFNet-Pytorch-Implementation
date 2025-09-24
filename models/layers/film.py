import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) layer
    
    Used like: output = film_layer(x, gamma, beta)
    
    where:
        x: multi-channel feature map with shape (batch, channels, height, width)
        gamma: channel-wise scaling factors with shape (batch, channels)
        beta: channel-wise bias terms with shape (batch, channels)
    """
    
    def __init__(self):
        super(FiLM, self).__init__()
    
    def forward(self, x, gamma, beta):
        """
        Apply FiLM conditioning to input features
        
        Args:
            x: Input feature map [batch, channels, height, width]
            gamma: Scaling factors [batch, channels]
            beta: Bias terms [batch, channels]
            
        Returns:
            Conditioned feature map with same shape as x
        """
        # Reshape gamma and beta to match feature map dimensions
        # [batch, channels] -> [batch, channels, 1, 1]
        gamma = gamma.unsqueeze(2).unsqueeze(3)
        beta = beta.unsqueeze(2).unsqueeze(3)
        
        # Apply FiLM conditioning: x * gamma + beta
        return x * gamma + beta


class FiLMLayer(nn.Module):
    """
    Single FiLM conditioning layer
    """
    def __init__(self, channels, modality_dim):
        super(FiLMLayer, self).__init__()
        
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        
        # FiLM parameter generators
        self.gamma_fc = nn.Sequential(
            nn.Linear(modality_dim, channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.beta_fc = nn.Sequential(
            nn.Linear(modality_dim, channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.film = FiLM()
        
    def forward(self, x, modality_vector):
        identity = self.conv1(x)
        identity = F.leaky_relu(identity, 0.2)
        
        out = self.conv2(identity)
        
        gamma = self.gamma_fc(modality_vector)
        beta = self.beta_fc(modality_vector)
        
        # Apply FiLM conditioning
        out = self.film(out, gamma, beta)
        out = F.leaky_relu(out, 0.2)
        
        # Residual connection
        return out + identity


class FiLMGenerator(nn.Module):
    """
    Generates gamma and beta parameters for FiLM conditioning
    """
    def __init__(self, conditioning_dim, num_channels):
        super(FiLMGenerator, self).__init__()
        self.conditioning_dim = conditioning_dim
        self.num_channels = num_channels
        
        # Generate gamma and beta from conditioning vector
        self.gamma_fc = nn.Linear(conditioning_dim, num_channels)
        self.beta_fc = nn.Linear(conditioning_dim, num_channels)
        
    def forward(self, conditioning_vector):
        """
        Generate FiLM parameters from conditioning vector
        
        Args:
            conditioning_vector: Conditioning input [batch, conditioning_dim]
            
        Returns:
            gamma: Scaling factors [batch, num_channels]
            beta: Bias terms [batch, num_channels]
        """
        gamma = self.gamma_fc(conditioning_vector)
        beta = self.beta_fc(conditioning_vector)
        return gamma, beta
