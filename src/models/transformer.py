"""Encoder-only Transformer forecaster with sinusoidal positional encoding. Owner: M5.
Same covariate/head design as the RNN models so the comparison is fair."""
import math

import torch
import torch.nn as nn

from .. import config


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerForecaster(nn.Module):
    def __init__(self, n_hist: int, n_fut: int, d_model: int = 64, nhead: int = 4, layers: int = 2,
                 ff: int = 128, dropout: float = 0.1, horizon: int = config.HORIZON,
                 n_targets: int = len(config.TARGETS), **_):
        super().__init__()
        self.proj = nn.Linear(n_hist, d_model)
        self.pos = PositionalEncoding(d_model)
        enc = nn.TransformerEncoderLayer(d_model, nhead, ff, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.head = nn.Sequential(nn.Linear(2 * d_model + horizon * n_fut, 256), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(256, horizon * n_targets))
        self.horizon, self.n_targets = horizon, n_targets

    def forward(self, x_hist: torch.Tensor, x_fut: torch.Tensor) -> torch.Tensor:
        h = self.encoder(self.pos(self.proj(x_hist)))
        z = torch.cat([h[:, -1], h.mean(dim=1), x_fut.flatten(1)], dim=1)
        return self.head(z).view(-1, self.horizon, self.n_targets)
