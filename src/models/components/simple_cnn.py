import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable


class StrandConv(nn.Module):
    def __init__(
        self,
        conv_out_dim: int = 256,
        kernel_sizes: Iterable[int] = range(8, 37, 4),
        fc_out_dim: int = 256
    ):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(4, conv_out_dim, kernel_size),
                    nn.ReLU(),
                    nn.MaxPool1d(110 - kernel_size + 1)
                )
                for kernel_size in kernel_sizes
            ]
        )

        self.linear = nn.Sequential(
            nn.Linear(conv_out_dim * len(kernel_sizes), fc_out_dim),
            nn.BatchNorm1d(fc_out_dim),
            nn.ReLU()
        )
        
        
    def forward(self, x):
        # x: (N, C, L)
        h = []
        for conv in self.convs:
            h.append(conv(x).squeeze(-1))
        h = torch.cat(h, dim=1)
        
        h = self.linear(h)
        return h

class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256
    ):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        # x: (N, C)
        return self.main(x)

class SimpleCNN(nn.Module):
    def __init__(
        self,
        conv_out_dim: int = 256,
        kernel_sizes: Iterable[int] = range(8, 37, 4),
        strand_out_dim: int = 256,
        mlp_hidden_dim: int = 256
    ):
        super().__init__()
        self.fwd_conv = StrandConv(conv_out_dim, kernel_sizes, strand_out_dim)
        self.rev_conv = StrandConv(conv_out_dim, kernel_sizes, strand_out_dim)
        self.mlp = MLP(strand_out_dim * 2, mlp_hidden_dim)
        
    def forward(self, fwd_x, rev_x):
        # x: (N, L, C)
        fwd_x, rev_x = fwd_x.transpose(1, 2), rev_x.transpose(1, 2)
        fwd_x  = self.fwd_conv(fwd_x)
        rev_x = self.rev_conv(rev_x)
        h = torch.cat([fwd_x, rev_x], dim=1)
        out = self.mlp(h).squeeze(-1)
        return out