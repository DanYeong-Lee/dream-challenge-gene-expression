import torch
import numpy as np
from torch import nn
from einops.layers.torch import Rearrange


class Aff(nn.Module):
    def __init__(
        self,
        dim
    ):
        super().__init__()

        self.alpha = nn.Parameter(torch.ones([1, 1, dim]))
        self.beta = nn.Parameter(torch.zeros([1, 1, dim]))

    def forward(self, x):
        x = x * self.alpha + self.beta
        return x
    

class CrossPatch(nn.Module):
    def __init__(
        self, 
        in_channels,
        length
        
    ):
        super().__init__()
        self.main = nn.Sequential(
            Aff(in_channels),
            Rearrange("N L C -> N C L"),
            nn.Linear(length, length),
            Rearrange("N C L -> N L C"),
            Aff(in_channels)
        )
        
    def forward(self, x):
        return x + self.main(x)
    
class CrossChannel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float
    ):
        super().__init__()
        self.main = nn.Sequential(
            Aff(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
            nn.Dropout(dropout)
        )
     
    def forward(self, x):
        return x + self.main(x)

class MLPBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        length,
        hidden_dim,
        dropout
    ):
        super().__init__()
        self.main = nn.Sequential(
            CrossPatch(in_channels, length),
            CrossChannel(in_channels, hidden_dim, dropout)
        )
    
    def forward(self, x):
        return self.main(x)
    

class ResMLP(nn.Module):
    def __init__(
        self,
        embed_dim,
        hidden_dim,
        dropout,
        n_blocks
    ):
        super().__init__()
        self.embed = nn.Linear(4, embed_dim)
        self.mlps = nn.ModuleList(
            [MLPBlock(embed_dim, 110, hidden_dim, dropout) for _ in range(n_blocks)]
        )
        self.out = nn.Linear(embed_dim, 1)
        
    def forward(self, x):
        x = self.embed(x)
        for mlp in self.mlps:
            x = mlp(x)
        x = x.mean(dim=1)
        x = self.out(x)
        
        return x.squeeze(-1)