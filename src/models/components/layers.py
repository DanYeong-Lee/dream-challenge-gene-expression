from torch import nn


class SimpleDenseNet(nn.Module):
    def __init__(
        self,
        input_size: int = 784,
        lin1_size: int = 256,
        lin2_size: int = 256,
        lin3_size: int = 256,
        output_size: int = 10,
    ):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, lin1_size),
            nn.BatchNorm1d(lin1_size),
            nn.ReLU(),
            nn.Linear(lin1_size, lin2_size),
            nn.BatchNorm1d(lin2_size),
            nn.ReLU(),
            nn.Linear(lin2_size, lin3_size),
            nn.BatchNorm1d(lin3_size),
            nn.ReLU(),
            nn.Linear(lin3_size, output_size),
        )

    def forward(self, x):
        batch_size, channels, width, height = x.size()

        # (batch, 1, width, height) -> (batch, 1*width*height)
        x = x.view(batch_size, -1)

        return self.model(x)

import torch
import torch.nn as nn
import torch.nn.functional as F

###-------------------###
### Convolution Block ###
###-------------------###

class StrandSpecificConv(nn.Module):
    def __init__(
        self, 
        out_channels: int =256, 
        kernel_size: int = 30
    ):
        super().__init__()
        
        self.main = nn.Sequential(
            nn.Conv1d(4, out_channels, kernel_size=kernel_size, padding="same"),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        # Input size: (N, C, L)
        # Output size: (N, out_channels, L)
        x = self.main(x)
        
        return x

class ConcatedConv(nn.Module):
    def __init__(
        self,
        in_channels: int = 256, 
        out_channels: int = 64,
        kernel_size: int = 30
    ):
        super().__init__()
        pad_size = int(kernel_size / 2)
        
        self.conv2d = nn.Sequential(
            nn.ZeroPad2d((pad_size, pad_size-1, 0, 0)),
            nn.Conv2d(2, out_channels, kernel_size=(in_channels, kernel_size))
        )
        self.batch_norm = nn.Sequential(
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
        self.conv1d = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding="same"),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
    
    def forward(self, x):
        # Input size: (N, 2, C, L)
        # Output size: (N, out_channels, L)
        x = self.conv2d(x)
        x = x.squeeze(2)
        x = self.batch_norm(x)
        x = self.conv1d(x)
        
        return x
    
class ConvBlock(nn.Module):
    def __init__(
        self,
        fwd_conv: nn.Module,
        rev_conv: nn.Module,
        cat_conv: nn.Module
    ):
        super().__init__()
        self.fwd_conv = fwd_conv
        self.rev_conv = rev_conv
        self.cat_conv = cat_conv
        
    def forward(self, fwd_x, rev_x):
        # Input size: (N, C, L) x 2
        # Output size: (N, out_channels, L)
        
        fwd_h = self.fwd_conv(fwd_x).unsqueeze(1)
        rev_h = self.rev_conv(rev_x).unsqueeze(1)
        concated_h = torch.cat([fwd_h, rev_h], dim=1)
        h = self.cat_conv(concated_h)
        
        return h
    
###-------------------###
### Transformer Block ###
###-------------------###

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int = 64, 
        nhead: int = 8, 
        dim_feedforward: int = 8, 
        num_layers: int = 2
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x):
        # Input size: (L, N, d_model)
        # Output size: (L, N, d_model)
        x = self.encoder(x)   

        return x
    
    
###-------------------###
###    LSTM Block     ###
###-------------------###

class LSTMBlock(nn.Module):
    def __init__(
        self,
        input_dim: int = 64, 
        hidden_dim: int = 8, 
        dropoutL float = 0.05):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Input size: (L, N, C)
        # Output size: (L, N, hidden_dim * 2)
        x, _ = self.lstm(x)
        x = self.dropout(x)
        
        return x
    
###-------------------###
###     MLP Block     ###
###-------------------###

class MLPBlock(nn.Module):
    def __init__(
        self, 
        input_dim: int = 1760, 
        hidden_dim: int = 64, 
        dropout: float = 0.05
    ):
        super().__init__()
        self.main = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        # Input size: (N, L, C)
        # Output size: (N)
        x = self.main(x).squeeze()
        
        return x