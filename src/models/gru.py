"""GRU forecaster (same inputs/horizon/head as the LSTM for a fair comparison). Owner: M4."""
from .rnn_base import RNNForecaster


class GRUForecaster(RNNForecaster):
    def __init__(self, n_hist, n_fut, **kw):
        super().__init__(n_hist, n_fut, cell="gru", **kw)
