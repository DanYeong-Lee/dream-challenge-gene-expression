from typing import List
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class ConvBlock(nn.Module):
    def __init__(
        self,
        input_dim: int = 4,
        out_dim: int = 320,
        kernel_size: int = 15,
        pool_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=out_dim, kernel_size=kernel_size, padding="same"),
            nn.LeakyReLU(),
            nn.MaxPool1d(pool_size),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # x: (N, C, L)
        
        return self.main(x)


class DeepFamQ_CRC(nn.Module):
    def __init__(
        self,
        conv_out_dim: int = 320,
        conv_kernel_size: List = [9, 15],
        pool_size: int = 3,
        lstm_hidden_dim: int = 320,
        fc_hidden_dim: int = 64,
        dropout1: float = 0.2,
        dropout2: float = 0.5
    ):
        super().__init__()
        pool_out_len = int(1 + ((110 - pool_size) / pool_size))
        fc_input_dim = lstm_hidden_dim * 2 * pool_out_len // 2
        
        conv_each_dim = int(conv_out_dim / len(conv_kernel_size))
        self.conv_blocks1 = nn.ModuleList([ConvBlock(4, conv_each_dim, k, pool_size, dropout1) for k in conv_kernel_size])
        
        self.lstm = nn.LSTM(input_size=conv_out_dim, hidden_size=lstm_hidden_dim, bidirectional=True)
        
        conv_each_dim = int(lstm_hidden_dim / len(conv_kernel_size))
        self.conv_blocks2 = nn.ModuleList([ConvBlock(lstm_hidden_dim * 2, conv_each_dim, k, pool_size, dropout1) for k in conv_kernel_size])
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout2),
            nn.Linear(fc_input_dim, fc_hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(fc_hidden_dim, fc_hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(fc_hidden_dim, 1)
        )
        
    def forward(self, x):
        x = rearrange(x, "N L C -> N C L")
        
        conv_outs = []
        for conv in self.conv_blocks1:
            conv_outs.append(conv(x))
        x = torch.cat(conv_outs, dim=1)
        
        x = rearrange(x, "N C L -> L N C")
        x, (h, c) = self.lstm(x)
        
        x = rearrange(x, "L N C -> N C L")
        
        conv_outs = []
        for conv in self.conv_blocks2:
            conv_outs.append(conv(x))
        x = torch.cat(conv_outs, dim=1)
        
        x = self.fc(x)
        x = rearrange(x, "N 1 -> N")
        
        return x
    
