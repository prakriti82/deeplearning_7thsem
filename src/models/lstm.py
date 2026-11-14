"""LSTM forecaster. Owner: M3."""
from .rnn_base import RNNForecaster


class LSTMForecaster(RNNForecaster):
    def __init__(self, n_hist, n_fut, **kw):
        super().__init__(n_hist, n_fut, cell="lstm", **kw)
