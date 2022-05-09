import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(
        self,
        conv_out_dim: int = 320,
        conv_kernel_size: int = 26,
        pool_size: int = 3,
        dropout: float = 0.2
    ):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=conv_out_dim, kernel_size=conv_kernel_size),
            nn.ReLU(),
            nn.MaxPool1d(pool_size),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # x: (N, C, L)
        
        return self.main(x)


class DanQ_DS(nn.Module):
    def __init__(
        self,
        conv_out_dim: int = 160,
        conv_kernel_size: int = 26,
        pool_size: int = 3,
        lstm_hidden_dim: int = 160,
        fc_hidden_dim: int = 64,
        dropout1: float = 0.2,
        dropout2: float = 0.5
    ):
        super().__init__()
        conv_out_len = 110 - conv_kernel_size + 1
        pool_out_len = int(1 + ((conv_out_len - pool_size) / pool_size))
        fc_input_dim = lstm_hidden_dim * 2 * 2 * pool_out_len
        
        self.fwd_conv = ConvBlock(conv_out_dim, conv_kernel_size, pool_size, dropout1)
        self.rev_conv = ConvBlock(conv_out_dim, conv_kernel_size, pool_size, dropout1)
        
        self.fwd_lstm = nn.LSTM(input_size=conv_out_dim, hidden_size=lstm_hidden_dim, bidirectional=True)
        self.rev_lstm = nn.LSTM(input_size=conv_out_dim, hidden_size=lstm_hidden_dim, bidirectional=True)
        
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Dropout(dropout2),
            nn.Linear(fc_input_dim, fc_hidden_dim),
            nn.ReLU(),
            nn.Linear(fc_hidden_dim, fc_hidden_dim),
            nn.ReLU(),
            nn.Linear(fc_hidden_dim, 1)
        )
        
    def forward(self, fwd_x, rev_x):
        # x: (N, L, C)
        
        fwd_x, rev_x = fwd_x.transpose(1, 2), rev_x.transpose(1, 2)  # (N, C, L)
        fwd_x, rev_x = self.fwd_conv(fwd_x), self.rev_conv(rev_x)
        fwd_x, rev_x = fwd_x.permute(2, 0, 1), rev_x.permute(2, 0, 1)  # (L, N, C)
        fwd_x, (h, c) = self.fwd_lstm(fwd_x)  
        rev_x, (h, c) = self.rev_lstm(rev_x)
        
        fwd_x, rev_x = fwd_x.transpose(0, 1), rev_x.transpose(0, 1)  # (N, L, C)
        fwd_x, rev_x = self.flatten(fwd_x), self.flatten(rev_x)  # (N, C)
        x = torch.cat([fwd_x, rev_x], dim=1)
        x = self.fc(x).squeeze(-1)
        
        return x