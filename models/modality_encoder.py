import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('/data2/liaohx/Projects/DAFNet/')

class ModalityEncoder(nn.Module):
    """
    Variational encoder to extract modality/intensity information from images
    """
    def __init__(self, 
                 resolution,
                 in_channels,
                 base_channels,
                 anatomy_channels,
                 ch_mult =(1, 2, 4, 8), 
                 latent_dim=8
                 ):
        super(ModalityEncoder, self).__init__()
        
        self.resolution = resolution 
        self.anatomy_channels = anatomy_channels
        self.latent_dim = latent_dim
        
        # Calculate input channels (anatomy + image)
        self.input_channels = anatomy_channels + in_channels
        
        # Convolutional layers for feature extraction
        out_ch_mult = ch_mult[1:]

        kernel_size, stride, padding = 3, 2, 1
        
        self.conv_layers = nn.ModuleList()
        in_conv = nn.Conv2d(self.input_channels, base_channels, kernel_size, stride, padding)
        self.conv_layers.append(in_conv)
        
        for i in range(len(out_ch_mult)):
            self.conv_layers.append(nn.Conv2d(
                base_channels * ch_mult[i],
                base_channels * out_ch_mult[i],
                kernel_size,
                stride,
                padding
            ))
        
        out_channels = base_channels * ch_mult[-1]
        
        out_resolution = self.resolution // 2 ** len(ch_mult)
        self.flattened_size = out_channels * out_resolution * out_resolution
        
        self.fc_hidden = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_size, 32),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.fc_mean = nn.Linear(32, latent_dim)
        self.fc_log_var = nn.Linear(32, latent_dim) 
        
    def reparameterize(self, mean, log_var):
        """
        Reparameterization trick for VAE
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def kl_divergence(self, mean, log_var):
        """
        Compute KL divergence loss
        """
        kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=-1)
        return kl_loss.mean()
    
    def forward(self, anatomy, image):
        """
        Forward pass
        
        Args:
            anatomy: Anatomy features [batch, channels, height, width]
            image: Input image [batch, channels, height, width]
            
        Returns:
            z: Latent code [batch, latent_dim]
            kl_loss: KL divergence loss
        """
        # Concatenate anatomy and image
        x = torch.cat([anatomy, image], dim=1)
        
        # Pass through convolutional layers
        features = x
        for conv_layer in self.conv_layers:
            features = conv_layer(features)
        
        # Pass through hidden layers
        hidden = self.fc_hidden(features)
        
        # Get mean and log variance
        z_mean = self.fc_mean(hidden)
        z_log_var = self.fc_log_var(hidden)
        
        # Sample latent code
        z = self.reparameterize(z_mean, z_log_var)
        
        # Compute KL divergence
        kl_loss = self.kl_divergence(z_mean, z_log_var)
        
        return z, kl_loss
    
    def encode_mean(self, anatomy, image):
        """
        Encode to mean latent code (without sampling)
        """
        x = torch.cat([anatomy, image], dim=1)
        features = x
        for conv_layer in self.conv_layers:
            features = conv_layer(features)
        hidden = self.fc_hidden(features)
        z_mean = self.fc_mean(hidden)
        return z_mean