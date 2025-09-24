import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange

def dice_coefficient(y_true, y_pred, epsilon=1e-12):
    # Flatten spatial dimensions
    y_true_flat = y_true.view(y_true.shape[0], y_true.shape[1], -1)  # B, C, ... -> B, C, N
    y_pred_flat = y_pred.view(y_pred.shape[0], y_pred.shape[1], -1)
    
    intersection = torch.sum(y_true_flat * y_pred_flat, dim=2)
    union = torch.sum(y_true_flat, dim=2) + torch.sum(y_pred_flat, dim=2)
    
    dice = (2 * intersection + epsilon) / (union + epsilon)
    return dice.mean()


def dice_loss(y_true, y_pred, epsilon=1e-12):
    return 1 - dice_coefficient(y_true, y_pred, epsilon)


def dice_loss_per_batch(y_true, y_pred, epsilon=1e-12):
    y_true_flat = y_true.view(y_true.shape[0], -1)
    y_pred_flat = y_pred.view(y_pred.shape[0], -1)
    
    intersection = torch.sum(y_true_flat * y_pred_flat, dim=1)
    union = torch.sum(y_true_flat, dim=1) + torch.sum(y_pred_flat, dim=1)
    
    dice = (2 * intersection + epsilon) / (union + epsilon)
    return 1 - dice


class DiceLoss(nn.Module):
    def __init__(self, restrict_channels=None, epsilon=1e-12):
        super(DiceLoss, self).__init__()
        self.restrict_channels = restrict_channels
        self.epsilon = epsilon
    
    def forward(self, y_pred, y_true):
        if self.restrict_channels is not None:
            y_pred = y_pred[:, :self.restrict_channels]
            y_true = y_true[:, :self.restrict_channels]
        
        return dice_loss(y_true, y_pred, self.epsilon)


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, epsilon=1e-12):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.epsilon = epsilon
    
    def forward(self, y_pred, y_true):
        num_classes = y_true.shape[1]
        
        # Calculate class weights
        n = torch.sum(y_true, dim=[0, 2, 3])  # Sum over batch and spatial dims
        n_total = torch.sum(n)
        weights = n_total / (n + self.epsilon)
        
        # Apply softmax to predictions
        y_pred_softmax = F.softmax(y_pred, dim=1)
        
        # Compute weighted cross entropy
        log_probs = torch.log(y_pred_softmax + self.epsilon)
        weighted_loss = -torch.sum(y_true * log_probs * weights.view(1, -1, 1, 1), dim=1)
        
        return torch.mean(weighted_loss)


class CombinedDiceBCELoss(nn.Module):
    """
    Combined Dice and Binary Cross Entropy Loss
    """
    def __init__(self, num_classes, lambda_bce=0.01, epsilon=1e-12):
        super(CombinedDiceBCELoss, self).__init__()
        self.num_classes = num_classes
        self.lambda_bce = lambda_bce
        self.dice_loss = DiceLoss(restrict_channels=num_classes, epsilon=epsilon)
        self.bce_loss = WeightedCrossEntropyLoss()
    
    def forward(self, y_pred, y_true):
        dice = self.dice_loss(y_pred, y_true)
        bce = self.bce_loss(y_pred, y_true)
        return dice + self.lambda_bce * bce


class KLDivergenceLoss(nn.Module):
    """
    KL Divergence Loss for VAE
    """
    def __init__(self):
        super(KLDivergenceLoss, self).__init__()
    
    def forward(self, mean, log_var):
        """
        Compute KL divergence between learned distribution and standard normal
        
        Args:
            mean: Mean of the learned distribution
            log_var: Log variance of the learned distribution
        """
        kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=-1)
        return torch.mean(kl_loss)


class MAELoss(nn.Module):
    """
    Mean Absolute Error Loss
    """
    def __init__(self):
        super(MAELoss, self).__init__()
    
    def forward(self, y_pred, y_true):
        return torch.mean(torch.abs(y_pred - y_true))


class SimilarityWeightedMAE(nn.Module):
    """
    Similarity-weighted MAE loss
    """
    def __init__(self):
        super(SimilarityWeightedMAE, self).__init__()
    
    def forward(self, y_pred, y_true, weights):
        """
        Compute weighted MAE loss
        
        Args:
            y_pred: Predictions
            y_true: Ground truth
            weights: Similarity weights [batch, 1, 1, 1] or [batch, 1]
        """
        mae = torch.abs(y_pred - y_true)
        
        # Expand weights to match mae dimensions if needed
        if weights.dim() == 2:  # [batch, 1]
            weights = weights.unsqueeze(2).unsqueeze(3)  # [batch, 1, 1, 1]
        
        # Tile weights to match all dimensions
        weights = weights.expand_as(mae)
        
        weighted_mae = mae * weights
        return torch.mean(weighted_mae)


class IdentityLoss(nn.Module):
    """
    Identity loss that returns the prediction as loss
    Used for regularization terms
    """
    def __init__(self):
        super(IdentityLoss, self).__init__()
    
    def forward(self, y_pred, y_true=None):
        return torch.mean(y_pred)


# Loss factory functions
def make_dice_loss(restrict_channels=1):
    """Factory function for dice loss"""
    return DiceLoss(restrict_channels=restrict_channels)


def make_combined_dice_bce(num_classes):
    """Factory function for combined dice + BCE loss"""
    return CombinedDiceBCELoss(num_classes=num_classes)


def ypred_loss(y_pred, y_true=None):
    """Return y_pred as loss (for identity loss)"""
    return torch.mean(y_pred)


def mae_single_input(inputs):
    """
    MAE between two inputs (used in automated pairing)
    """
    y1, y2 = inputs
    return torch.mean(torch.abs(y1 - y2), dim=[1, 2])  # Keep batch dimension





