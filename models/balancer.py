import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_coefficient(y_true, y_pred, epsilon=1e-12): 
    intersection = torch.sum(y_true * y_pred, dim=(1, 2, 3))  # [batch]
    union = torch.sum(y_true, dim=(1, 2, 3)) + torch.sum(y_pred, dim=(1, 2, 3))  # [batch]
    dice = (2 * intersection + epsilon) / (union + epsilon)
    return dice.unsqueeze(1)  # [batch, 1]


class Balancer(nn.Module):
    def __init__(self, n_pairs=3):
        super(Balancer, self).__init__()
        self.n_pairs = n_pairs
        self.dense1 = nn.Linear(n_pairs, 5)
        self.dense3 = nn.Linear(5, n_pairs)
    
    def forward(self, target_anatomy, candidate_anatomies):
        dice_values = []
        assert candidate_anatomies.shape[1] == self.n_pairs
        for i in range(self.n_pairs):
            candidate = candidate_anatomies[:, i, ...]
            dice = dice_coefficient(target_anatomy, candidate)
            dice_values.append(dice)
        dice_tensor = torch.cat(dice_values, dim=1)  # [batch, num_candidates]
        h1 = F.relu(self.dense1(dice_tensor))
        w = self.dense3(h1)
        w = F.softmax(w, dim=-1)
        return w


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    balancer = Balancer(n_pairs=3).to(device)
    target_anatomy = torch.randn(2, 8, 128, 128).to(device)
    candidate_anatomies = torch.randn(2, 3, 8, 128, 128).to(device)
    w = balancer(target_anatomy, candidate_anatomies)
    print(w.shape)