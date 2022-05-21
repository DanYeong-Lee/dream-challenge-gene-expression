from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SABlock(nn.Module):
    def __init__(
        self,
        seq_len: int = 110,
        input_dim: int = 320
    ):
        super().__init__()
        self.att = nn.Linear(seq_len, seq_len)
        
    def forward(self, x):
        # x: (N, C, L)
        att_score = self.att(x)
        att_score = F.softmax(att_score, dim=2)  # (N, C, L)
        att_score = att_score.mean(dim=1, keepdims=True)  # (N, 1, L)
        weighted = torch.mul(x, att_score)  # (N, C, L)
        out = weighted.sum(dim=2)  # (N, C)
        
        return out

    
class MHABlock(nn.Module):
    def __init__(
        self,
        input_dim: int = 320,
        qkv_dim: int = 320,
        num_heads: int = 4
    ):
        super().__init__()
        self.q = nn.Linear(input_dim, qkv_dim)
        self.k = nn.Linear(input_dim, qkv_dim)
        self.v = nn.Linear(input_dim, qkv_dim)
        self.mha = nn.MultiheadAttention(embed_dim=qkv_dim, num_heads=num_heads)
        
    def forward(self, x):
        # x: (L, N, C)
        query = self.q(x)  # (L, N, C)
        key = self.k(x)    # (L, N, C)
        value = self.v(x)  # (L, N, C)
        out, _ = self.mha(query, key, value, need_weights=False)  # (L, N, C)
        
        return out

    
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

    
class DeepGRN(nn.Module):
    def __init__(
        self,
        conv_out_dim: int = 320,
        conv_kernel_size: int = 15,
        pool_size: int = 3,
        lstm_hidden_dim: int = 160,
        num_heads: int = 4,
        fc_hidden_dim: int = 64,
        dropout1: float = 0.2,
        dropout2: float = 0.5
    ):
        super().__init__()
        pool_out_len = int(1 + ((110 - pool_size) / pool_size))
        fc_input_dim = lstm_hidden_dim * 2 * pool_out_len
        
        self.conv_block = ConvBlock(4, conv_out_dim, conv_kernel_size, pool_size, dropout1)
        self.lstm = nn.LSTM(input_size=conv_out_dim, hidden_size=lstm_hidden_dim, bidirectional=True)
        self.sa_layer = SABlock(pool_out_len, lstm_hidden_dim * 2)
        self.mha_layer = MHABlock(lstm_hidden_dim * 2, lstm_hidden_dim * 2, num_heads)
        
        self.sa_fc = nn.Sequential(
            nn.Dropout(dropout2),
            nn.Linear(lstm_hidden_dim * 2, fc_hidden_dim),
            nn.ReLU(),
            nn.Linear(fc_hidden_dim, fc_hidden_dim),
            nn.ReLU(),
            nn.Linear(fc_hidden_dim, 1)
        )
        
        self.mha_fc = nn.Sequential(
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
        x = self.conv_block(x)  # (N, C, L)
        x = x.permute(2, 0, 1)  # (L, N, C)
        x, (h, c) = self.lstm(x)  # (L, N, C)
        
        sa_x = x.permute(1, 2, 0)  # (N, C, L)
        sa_x = self.sa_layer(x)  # (N, C)
        sa_x = self.sa_fc(sa_x) 
        
        mha_x = self.mha_layer(sa_x)  # (L, N, C)
        mha_x = mha_x.transpose(0, 1)  # (N, L, C)
        mha_x = self.mha_fc(mha_x)  
        
        out = (sa_x + mha_x) / 2
        
        return out.squeeze()
    
    
class DeepFamGRN(nn.Module):
    def __init__(
        self,
        conv_out_dim: int = 320,
        conv_kernel_size: List = [10, 15],
        pool_size: int = 3,
        lstm_hidden_dim: int = 160,
        num_heads: int = 4,
        fc_hidden_dim: int = 64,
        dropout1: float = 0.2,
        dropout2: float = 0.5
    ):
        super().__init__()
        pool_out_len = int(1 + ((110 - pool_size) / pool_size))
        fc_input_dim = lstm_hidden_dim * 2 * pool_out_len
        conv_each_dim = int(conv_out_dim / len(conv_kernel_size))
        
        self.conv_blocks = nn.ModuleList([ConvBlock(4, conv_each_dim, k, pool_size, dropout1) for k in conv_kernel_size])
        self.lstm = nn.LSTM(input_size=conv_out_dim, hidden_size=lstm_hidden_dim, bidirectional=True)
        self.sa_layer = SABlock(pool_out_len, lstm_hidden_dim * 2)
        self.mha_layer = MHABlock(lstm_hidden_dim * 2, lstm_hidden_dim * 2, num_heads)
        
        self.sa_fc = nn.Sequential(
            nn.Dropout(dropout2),
            nn.Linear(lstm_hidden_dim * 2, fc_hidden_dim),
            nn.ReLU(),
            nn.Linear(fc_hidden_dim, fc_hidden_dim),
            nn.ReLU(),
            nn.Linear(fc_hidden_dim, 1)
        )
        
        self.mha_fc = nn.Sequential(
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
        conv_outs = []
        for conv in self.conv_blocks:
            conv_outs.append(conv(x))
        x = torch.cat(conv_outs, dim=1)  # (N, C, L)
        x = x.permute(2, 0, 1)  # (L, N, C)
        x, (h, c) = self.lstm(x)  # (L, N, C)
        
        sa_x = x.permute(1, 2, 0)  # (N, C, L)
        sa_x = self.sa_layer(sa_x)  # (N, C)
        sa_x = self.sa_fc(sa_x) 
        
        mha_x = self.mha_layer(x)  # (L, N, C)
        mha_x = mha_x.transpose(0, 1)  # (N, L, C)
        mha_x = self.mha_fc(mha_x)  
        
        out = (sa_x + mha_x) / 2
        
        return out.squeeze()