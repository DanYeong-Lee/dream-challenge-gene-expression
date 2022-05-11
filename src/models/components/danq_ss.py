import torch
import torch.nn as nn
import torch.nn.functional as F


class DanQ(nn.Module):
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
        fc_input_dim = lstm_hidden_dim * 2 * pool_out_len
        
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=conv_out_dim, kernel_size=conv_kernel_size),
            nn.ReLU(),
            nn.MaxPool1d(pool_size),
            nn.Dropout(dropout1)
        )
        self.lstm = nn.LSTM(input_size=conv_out_dim, hidden_size=lstm_hidden_dim, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout2),
            nn.Linear(fc_input_dim, fc_hidden_dim),
            nn.ReLU(),
            nn.Linear(fc_hidden_dim, fc_hidden_dim),
            nn.ReLU(),
            nn.Linear(fc_hidden_dim, 1)
        )
        
    def forward(self, x):
        # x: (N, L, C)
        
        x = x.transpose(1, 2)  # (N, C, L)
        x = self.conv_block(x)
        x = x.permute(2, 0, 1)  # (L, N, C)
        x, (h, c) = self.lstm(x)
        x = x.transpose(0, 1)  # (N, L, C)
        x = self.fc(x).squeeze(-1)
        return x