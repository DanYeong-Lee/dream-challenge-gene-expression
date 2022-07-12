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


class DeepFamQ(nn.Module):
    def __init__(
        self,
        conv_out_dim: int = 320,
        conv_kernel_size: List = [10, 15],
        pool_size: int = 3,
        lstm_hidden_dim: int = 320,
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
        conv_outs = []
        for conv in self.conv_blocks:
            conv_outs.append(conv(x))
        x = torch.cat(conv_outs, dim=1)  # (N, C, L)
        x = x.permute(2, 0, 1)  # (L, N, C)
        out, (h, c) = self.lstm(x)  # (L, N, C)
        x = out + torch.cat([x, x], dim=2)
        x = x.transpose(0, 1)  # (N, L, C)
        x = self.fc(x)
        x = x.squeeze()
        
        return x


class DeepFamQ_multilayerConv(nn.Module):
    def __init__(
        self,
        conv_out_dim: int = 320,
        conv_kernel_size: List = [10, 15],
        pool_size: int = 3,
        lstm_hidden_dim: int = 320,
        fc_hidden_dim: int = 64,
        dropout1: float = 0.2,
        dropout2: float = 0.5
    ):
        super().__init__()
        pool_out_len = int(1 + ((110 - pool_size) / pool_size))
        fc_input_dim = lstm_hidden_dim * 2 * pool_out_len
        conv_each_dim = int(conv_out_dim / len(conv_kernel_size))
        
        self.conv_blocks1 = nn.ModuleList(
            [ConvBlock(4, conv_each_dim, k, pool_size, dropout1) for k in conv_kernel_size]
        )
        self.conv_blocks2 = nn.ModuleList(
            [ConvBlock(conv_out_dim, conv_each_dim, k, pool_size, dropout1) for k in conv_kernel_size]
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
        conv_outs1 = []
        for conv in self.conv_blocks1:
            conv_outs1.append(conv(x))
        x = torch.cat(conv_outs1, dim=1)  # (N, C, L)
        conv_outs2 = []
        for conv in self.conv_blocks2:
            conv_outs2.append(conv(x))
        x = torch.cat(conv_outs2, dim=1)  # (N, C, L)
        
        x = x.permute(2, 0, 1)  # (L, N, C)
        x, (h, c) = self.lstm(x)  # (L, N, C)
        x = x.transpose(0, 1)  # (N, L, C)
        x = self.fc(x)
        x = x.squeeze()
        
        return x
    
    
class DeepFamQ_multilayerLSTM(nn.Module):
    def __init__(
        self,
        conv_out_dim: int = 320,
        conv_kernel_size: List = [10, 15],
        pool_size: int = 3,
        lstm_hidden_dim: int = 320,
        lstm_layers: int = 2,
        fc_hidden_dim: int = 64,
        dropout1: float = 0.2,
        dropout2: float = 0.5,
        dropout3: float = 0.5
    ):
        super().__init__()
        pool_out_len = int(1 + ((110 - pool_size) / pool_size))
        fc_input_dim = lstm_hidden_dim * 2 * pool_out_len
        conv_each_dim = int(conv_out_dim / len(conv_kernel_size))
        
        self.conv_blocks = nn.ModuleList([ConvBlock(4, conv_each_dim, k, pool_size, dropout1) for k in conv_kernel_size])
        self.lstm = nn.LSTM(input_size=conv_out_dim, 
                            hidden_size=lstm_hidden_dim, 
                            bidirectional=True, 
                            num_layers=lstm_layers, 
                            dropout=dropout2)
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout3),
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
        x = x.transpose(0, 1)  # (N, L, C)
        x = self.fc(x)
        x = x.squeeze()
        
        return x
    




class DeepFamQ_TRFM(nn.Module):
    def __init__(
        self,
        conv_out_dim: int = 320,
        conv_kernel_size: List = [10, 15],
        pool_size: int = 3,
        lstm_hidden_dim: int = 320,
        nhead: int = 4,
        dim_feedforward: int = 1024,
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
        self.trfm = nn.TransformerEncoderLayer(lstm_hidden_dim * 2, nhead, dim_feedforward, activation="gelu")
        
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
        conv_outs = []
        for conv in self.conv_blocks:
            conv_outs.append(conv(x))
        x = torch.cat(conv_outs, dim=1)  # (N, C, L)
        x = x.permute(2, 0, 1)  # (L, N, C)
        x, (h, c) = self.lstm(x)  # (L, N, C)
        x = self.trfm(x)
        
        x = x.transpose(0, 1)  # (N, L, C)
        x = self.fc(x)
        x = x.squeeze()
        
        return x    


class PositionalEncoding1D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding1D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 2) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.cached_penc = None

    def forward(self, tensor):
        """
        :param tensor: A 3d tensor of size (batch_size, x, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, ch)
        """
        if len(tensor.shape) != 3:
            raise RuntimeError("The input tensor has to be 3d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, orig_ch = tensor.shape
        pos_x = torch.arange(x).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1)
        emb = torch.zeros((x, self.channels)).type(tensor.type())
        emb[:, : self.channels] = emb_x

        self.cached_penc = emb[None, :, :orig_ch].repeat(batch_size, 1, 1)
        return self.cached_penc


class DeepFamMHA(nn.Module):
    def __init__(
        self,
        conv_out_dim: int = 320,
        conv_kernel_size: List = [10, 15],
        pool_size: int = 3,
        mha_embed_dim: int = 320,
        nhead: int = 4,
        fc_hidden_dim: int = 64,
        dropout1: float = 0.2,
        dropout2: float = 0.5
    ):
        super().__init__()
        pool_out_len = int(1 + ((110 - pool_size) / pool_size))
        fc_input_dim = mha_embed_dim * pool_out_len
        conv_each_dim = int(conv_out_dim / len(conv_kernel_size))
        
        self.conv_blocks = nn.ModuleList([ConvBlock(4, conv_each_dim, k, pool_size, dropout1) for k in conv_kernel_size])
        self.pos_enc = PositionalEncoding1D(conv_out_dim)
        self.q = nn.Linear(conv_out_dim, mha_embed_dim)
        self.k = nn.Linear(conv_out_dim, mha_embed_dim)
        self.v = nn.Linear(conv_out_dim, mha_embed_dim)
        self.mha = nn.MultiheadAttention(mha_embed_dim, nhead)
        
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
        conv_outs = []
        for conv in self.conv_blocks:
            conv_outs.append(conv(x))
        x = torch.cat(conv_outs, dim=1)  # (N, C, L)
        x = x.transpose(1, 2)  # (N, L, C)
        x = x + self.pos_enc(x)
        
        x = x.transpose(0, 1)  # (L, N, C)
        query = self.q(x)
        key = self.k(x)
        value = self.v(x)
        x, _ = self.mha(query, key, value)  # (L, N, C)
        
        x = x.transpose(0, 1)  # (N, L, C)
        x = self.fc(x)
        x = x.squeeze()
        
        return x 
    
class DeepFamTrfm(nn.Module):
    def __init__(
        self,
        conv_out_dim: int = 320,
        conv_kernel_size: List = [10, 15],
        pool_size: int = 3,
        trfm_d_model: int = 320,
        nhead: int = 4,
        dim_feedforward: int = 512,
        fc_hidden_dim: int = 64,
        dropout1: float = 0.2,
        dropout2: float = 0.5
    ):
        super().__init__()
        pool_out_len = int(1 + ((110 - pool_size) / pool_size))
        fc_input_dim = trfm_d_model * pool_out_len
        conv_each_dim = int(conv_out_dim / len(conv_kernel_size))
        
        self.conv_blocks = nn.ModuleList([ConvBlock(4, conv_each_dim, k, pool_size, dropout1) for k in conv_kernel_size])
        self.pos_enc = PositionalEncoding1D(conv_out_dim)
        self.trfm_encoder = nn.TransformerEncoderLayer(trfm_d_model, nhead, dim_feedforward)
        
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
        conv_outs = []
        for conv in self.conv_blocks:
            conv_outs.append(conv(x))
        x = torch.cat(conv_outs, dim=1)  # (N, C, L)
        x = x.transpose(1, 2)  # (N, L, C)
        x = x + self.pos_enc(x)
        
        x = x.transpose(0, 1)  # (L, N, C)
        x = self.trfm_encoder(x)  # (L, N, C)
        
        x = x.transpose(0, 1)  # (N, L, C)
        x = self.fc(x)
        x = x.squeeze()
        
        return x 