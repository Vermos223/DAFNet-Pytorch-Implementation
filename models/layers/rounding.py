import torch
import torch.nn as nn
import torch.nn.functional as F

class RoundingLayer(nn.Module):
    """
    Rounding layer that rounds input values to nearest integers
    Used for creating discrete anatomy factors
    """ 
    def __init__(self, method='straight_through'):
        super(RoundingLayer, self).__init__()
        self.method = method
    
    def forward(self, x):
        if self.method == 'straight_through':
            # Straight-through estimator: round in forward pass, 
            # but use identity in backward pass
            return x + (torch.round(x) - x).detach()
        elif self.method == 'standard':
            return torch.round(x)
        else:
            raise ValueError(f"Unknown rounding method: {self.method}")


class GumbelSoftmaxRounding(nn.Module):
    """
    Differentiable rounding using Gumbel-Softmax trick
    """
    
    def __init__(self, temperature=1.0, hard=True):
        super(GumbelSoftmaxRounding, self).__init__()
        self.temperature = temperature
        self.hard = hard
    
    def forward(self, x):
        # Convert continuous values to logits for discrete choices
        # This is a simplified version - might need adjustment based on use case
        logits = torch.stack([x, 1-x], dim=-1)
        samples = F.gumbel_softmax(logits, tau=self.temperature, hard=self.hard, dim=-1)
        return samples[..., 0]  # Take first component as rounded value
