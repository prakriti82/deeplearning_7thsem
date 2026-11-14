from .gru import GRUForecaster
from .lstm import LSTMForecaster
from .transformer import TransformerForecaster

REGISTRY = {"lstm": LSTMForecaster, "gru": GRUForecaster, "transformer": TransformerForecaster}


def build_model(name: str, n_hist: int, n_fut: int, **hparams):
    if name not in REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {list(REGISTRY)}")
    return REGISTRY[name](n_hist, n_fut, **hparams)
