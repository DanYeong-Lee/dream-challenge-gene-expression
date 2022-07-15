import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

    
class ConvBlock(nn.Module):
    def __init__(
        self,
        input_dim: int = 4,
        out_dim: int = 320,
        kernel_size: int = 26,
        pool_size: int = 3,
        dropout: float = 0.2
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


class DanQ_CRC(nn.Module):
    def __init__(
        self,
        conv_out_dim: int = 320,
        conv_kernel_size: int = 26,
        pool_size: int = 3,
        lstm_hidden_dim: int = 320,
        fc_hidden_dim: int = 64,
        dropout1: float = 0.2,
        dropout2: float = 0.5
    ):
        super().__init__()
        conv_out_len = 110 - conv_kernel_size + 1
        pool_out_len = int(1 + ((conv_out_len - pool_size) / pool_size))
        fc_input_dim = lstm_hidden_dim * pool_out_len
        
        self.conv_block1 = ConvBlock(4, conv_out_dim, conv_kernel_size, pool_size, dropout1)
        
        self.lstm = nn.LSTM(input_size=conv_out_dim, hidden_size=lstm_hidden_dim, bidirectional=True)
        
        self.conv_block2 = ConvBlock(lstm_hidden_dim * 2, lstm_hidden_dim, conv_kernel_size, pool_size, dropout1)
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout2),
            nn.Linear(fc_input_dim, fc_hidden_dim),
            nn.ReLU(),
            nn.Linear(fc_hidden_dim, fc_hidden_dim),
            nn.ReLU(),
            nn.Linear(fc_hidden_dim, 1)
        )
        
    def forward(self, fwd_x):
        # x: (N, L, C)
        # Only use forward strand data
        
        x = fwd_x.transpose(1, 2)  # (N, C, L)
        x = self.conv_block1(x)
        x = x.permute(2, 0, 1)  # (L, N, C)
        x, (h, c) = self.lstm(x)
        x = x.permute(1, 2, 0)  # (N, C, L)
        x - self.conv_block2(x)
        x = self.fc(x).squeeze(-1)
        return x
