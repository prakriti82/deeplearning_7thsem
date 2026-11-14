"""Shared recurrent forecaster used by LSTM and GRU. Owners: M3, M4.
Encodes the history with a recurrent net, concatenates the known covariates of the
target day and predicts all 24 hours x targets at once (direct multi-output)."""
import torch
import torch.nn as nn

from .. import config


class RNNForecaster(nn.Module):
    def __init__(self, n_hist: int, n_fut: int, cell: str = "lstm", hidden: int = 64,
                 layers: int = 2, dropout: float = 0.1,
                 horizon: int = config.HORIZON, n_targets: int = len(config.TARGETS), **_):
        super().__init__()
        rnn = {"lstm": nn.LSTM, "gru": nn.GRU}[cell]
        self.rnn = rnn(n_hist, hidden, num_layers=layers, batch_first=True,
                       dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden + horizon * n_fut, 256), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(256, horizon * n_targets))
        self.horizon, self.n_targets = horizon, n_targets

    def forward(self, x_hist: torch.Tensor, x_fut: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x_hist)
        z = torch.cat([out[:, -1], x_fut.flatten(1)], dim=1)
        return self.head(z).view(-1, self.horizon, self.n_targets)
