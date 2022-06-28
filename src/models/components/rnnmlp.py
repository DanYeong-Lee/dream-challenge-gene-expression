from typing import List
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class GRUMLP(nn.Module):
    def __init__(
        self,
        rnn_hidden_dim: int = 320,
        fc_hidden_dim: int = 64,
        dropout: float = 0.5
    ):
        super().__init__()
        fc_input_dim = rnn_hidden_dim * 2 * 110
        self.rnn = nn.GRU(input_size=4, hidden_size=rnn_hidden_dim, num_layers=2, bidirectional=True, batch_first=True)
        
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
        x, _ = self.rnn(x)  # (N, L, C)
        x = self.fc(x)
        x = x.squeeze()
        
        return x
