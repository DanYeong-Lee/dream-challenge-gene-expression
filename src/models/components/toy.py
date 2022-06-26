import torch
import torch.nn as nn
import torch.nn.functional as F


class Toy(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 64
    ):
        super().__init__()
        self.main = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4 * 110, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.main(x).squeeze()