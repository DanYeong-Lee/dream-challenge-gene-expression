import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock_bn(nn.Module):
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
            nn.Conv1d(in_channels=input_dim, out_channels=out_dim, kernel_size=kernel_size),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(pool_size)
        )
    
    def forward(self, x):
        # x: (N, C, L)
        
        return self.main(x)

    
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
            nn.Conv1d(in_channels=input_dim, out_channels=out_dim, kernel_size=kernel_size),
            nn.ReLU(),
            nn.MaxPool1d(pool_size),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # x: (N, C, L)
        
        return self.main(x)


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
        
        self.conv_block = ConvBlock(4, conv_out_dim, conv_kernel_size, pool_size, dropout1)

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
        
    def forward(self, fwd_x, rev_x):
        # x: (N, L, C)
        # Only use forward strand data
        
        x = fwd_x.transpose(1, 2)  # (N, C, L)
        x = self.conv_block(x)
        x = x.permute(2, 0, 1)  # (L, N, C)
        x, (h, c) = self.lstm(x)
        x = x.transpose(0, 1)  # (N, L, C)
        x = self.fc(x).squeeze(-1)
        return x

class DanQ_SS(DanQ):
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
        super().__init__(
            conv_out_dim,
            conv_kernel_size,
            pool_size,
            lstm_hidden_dim,
            fc_hidden_dim,
            dropout1,
            dropout2
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
        
        self.fwd_conv = ConvBlock(4, conv_out_dim, conv_kernel_size, pool_size, dropout1)
        self.rev_conv = ConvBlock(4, conv_out_dim, conv_kernel_size, pool_size, dropout1)
        
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
    
    
class DanQ_Transformer(nn.Module):
    def __init__(
        self,
        conv_out_dim: int = 320,
        conv_kernel_size: int = 26,
        pool_size: int = 3,
        trfm_nhead: int = 8,
        trfm_fc_dim: int = 320,
        trfm_layers: int = 2,
        fc_hidden_dim: int = 64,
        dropout: float = 0.5
    ):
        super().__init__()
        conv_out_len = 110 - conv_kernel_size + 1
        pool_out_len = int(1 + ((conv_out_len - pool_size) / pool_size))
        fc_input_dim = conv_out_dim * pool_out_len
        
        self.conv_block = ConvBlock(4, conv_out_dim, conv_kernel_size, pool_size, dropout)
        
        self.pos_enc = PositionalEncoding1D(conv_out_dim)
        trfm_encoder = nn.TransformerEncoderLayer(
            d_model=conv_out_dim,
            nhead=trfm_nhead,
            dim_feedforward=trfm_fc_dim
        )
        self.transformer = nn.TransformerEncoder(trfm_encoder, num_layers=trfm_layers)
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
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
        x = x.transpose(1, 2)  # (N, L, C)
        x = x + self.pos_enc(x)
        x = x.transpose(0, 1)  # (L, N, C)
        x = self.transformer(x)
        x = x.transpose(0, 1)  # (N, L, C)
        x = self.fc(x).squeeze(-1)
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
    
    
class DanQ_EmbedBase(nn.Module):
    def __init__(
        self,
        embed_dim: int = 32,
        conv_out_dim: int = 320,
        conv_kernel_size: int = 15,
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
        
        self.embed = nn.Embedding(num_embeddings=5, embedding_dim=embed_dim)
        self.conv_block = ConvBlock_bn(embed_dim, conv_out_dim, conv_kernel_size, pool_size, dropout1)

        self.lstm = nn.LSTM(input_size=conv_out_dim, hidden_size=lstm_hidden_dim, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout2),
            nn.Linear(fc_input_dim, fc_hidden_dim),
            nn.BatchNorm1d(fc_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout2),
            nn.Linear(fc_hidden_dim, fc_hidden_dim),
            nn.BatchNorm1d(fc_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout2),
            nn.Linear(fc_hidden_dim, 1)
        )
    def forward(self, x):
        # x: (N, L, C)
        
        x = self.embed(x)  # (N, L, C)
        x = x.transpose(1, 2)  # (N, C, L)
        x = self.conv_block(x)
        x = x.permute(2, 0, 1)  # (L, N, C)
        x, (h, c) = self.lstm(x)
        x = x.transpose(0, 1)  # (N, L, C)
        x = self.fc(x).squeeze(-1)
        return x
