"""2-layer LSTM that maps SEQUENCE_LENGTH hours of history to HORIZON hours ahead."""
from __future__ import annotations

import torch
from torch import nn


class TemperatureLSTM(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int, horizon: int, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, 1)
        _, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]  # (batch, hidden_size) — final layer's last time step
        return self.head(last_hidden)  # (batch, horizon)
