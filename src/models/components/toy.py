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
        out_dim: int = 256,
        kernel_size: int = 10,
        pool_size: int = 3,
    ):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=out_dim, kernel_size=kernel_size),
            nn.ReLU(),
            nn.MaxPool1d(pool_size),
        )
    
    def forward(self, x):
        # x: (N, C, L)
        
        return self.main(x)


class Toy(nn.Module):
    def __init__(
        self,
        conv_out_dim: int = 256,
        conv_kernel_size: int = 10,
        pool_size: int = 3,
    ):
        super().__init__()
        conv_out_len = 110 - conv_kernel_size + 1
        pool_out_len = int(1 + ((conv_out_len - pool_size) / pool_size))
        fc_input_dim = conv_out_dim * pool_out_len
        
        self.conv_block = ConvBlock(4, conv_out_dim, conv_kernel_size, pool_size)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_input_dim, 1)
        )
        
    def forward(self, x):
        # x: (N, L, C)
        
        x = x.transpose(1, 2)  # (N, C, L)
        x = self.conv_block(x)
        x = x.transpose(1, 2)  # (N, L, C)
        x = self.fc(x).squeeze(-1)
        return x
