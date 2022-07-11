from typing import List
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(
        self,
        input_dim: int = 4,
        out_dim: int = 320,
        kernel_size: int = 15,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=out_dim, kernel_size=kernel_size, padding="same"),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # x: (N, C, L)
        
        return self.main(x)
    

class Encoder(nn.Module):
    def __init__(
        self,
        conv_out_dim: int = 320,
        conv_kernel_size: List = [9, 15],
        lstm_hidden_dim: int = 320,
        dropout1: float = 0.2
    ):
        super().__init__()
        conv_each_dim = int(conv_out_dim / len(conv_kernel_size))
        
        self.conv_blocks = nn.ModuleList([ConvBlock(4, conv_each_dim, k, dropout1) for k in conv_kernel_size])
        self.lstm = nn.LSTM(input_size=conv_out_dim, hidden_size=lstm_hidden_dim, bidirectional=True)
        
    def forward(self, x):
        # x: (N, L, C)
        x = x.transpose(1, 2)  # (N, C, L)
        conv_outs = []
        for conv in self.conv_blocks:
            conv_outs.append(conv(x))
        x = torch.cat(conv_outs, dim=1)  # (N, C, L)
        x = x.permute(2, 0, 1)  # (L, N, C)
        x, (h, c) = self.lstm(x)  # (L, N, C)
        x = x.transpose(0, 1)  # (N, L, C)
        
        return x


class Reconstructor(nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,
        conv_kernel_size: int = 9
    ):
        super().__init__()
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=embed_dim, batch_first=True)
        self.convt = nn.ConvTranspose1d(in_channels=embed_dim, out_channels=4, kernel_size=conv_kernel_size, padding=int((conv_kernel_size - 1) / 2))
        
    def forward(self, x, h):
        h, _ = self.lstm(h)  # (N, L, C)
        h = h.transpose(1, 2)  # (N, C, L)
        h = self.convt(h)  # (N, C, L)
        
        return h.transpose(1, 2)
    
    
class MLP(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 64,
        dropout: float = 0.5
    ):
        super().__init__()
        self.main = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.LazyLinear(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        x = self.main(x)
        
        return x.squeeze(-1)