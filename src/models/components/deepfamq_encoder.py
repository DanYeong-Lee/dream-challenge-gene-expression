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
        pool_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=out_dim, kernel_size=kernel_size, padding="same"),
            nn.ReLU(),
            nn.MaxPool1d(pool_size),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # x: (N, C, L)
        
        return self.main(x)
    

class FC_block(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout: float
    ):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self.main(x)


class DeepFamQ_Encoder(nn.Module):
    def __init__(
        self,
        conv_out_dim: int = 320,
        conv_kernel_size: List = [9, 15],
        pool_size: int = 3,
        lstm_hidden_dim: int = 320,
        embed_dim: int = 320,
        dropout: float = 0.2
    ):
        super().__init__()
        pool_out_len = int(1 + ((110 - pool_size) / pool_size))
        fc_input_dim = lstm_hidden_dim * 2 * pool_out_len
        conv_each_dim = int(conv_out_dim / len(conv_kernel_size))
        
        self.conv_blocks = nn.ModuleList([ConvBlock(4, conv_each_dim, k, pool_size, dropout) for k in conv_kernel_size])
        self.lstm = nn.LSTM(input_size=conv_out_dim, hidden_size=lstm_hidden_dim, bidirectional=True)
        
        self.fc = FC_block(fc_input_dim, embed_dim, 0)
        
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
        x = self.fc(x)
        
        return x
    
    
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 320,
        hidden_dim: int = 128,
        n_layers: int = 3,
        dropout: float = 0.5
    ):
        super().__init__()
        self.fc1 = FC_block(input_dim, hidden_dim, dropout)
        self.fc_list = nn.ModuleList([FC_block(hidden_dim, hidden_dim, dropout) for k in range(n_layers - 2)])
        self.final = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        x = self.fc1(x)
        for fc in self.fc_list:
            x = fc(x)
        x = self.final(x)
        
        return x.squeeze(-1)