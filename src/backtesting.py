"""Rolling day-ahead backtest. Owner: M6.
For every test day D the model sees only information up to 23:00 of D-1
(history) plus the known covariates of D, and predicts the 24 hours of D."""
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from . import config
from .dataset import Scaler, build_windows
from .models.baselines import naive_predict
from .training import predict_scaled


def to_long(model_name: str, days: List[pd.Timestamp], y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """(N,24,T) arrays in original units -> long DataFrame."""
    rows = []
    for k, d in enumerate(days):
        origin = d - pd.Timedelta(hours=1)
        for h in range(config.HORIZON):
            for j, t in enumerate(config.TARGETS):
                rows.append((model_name, origin, d + pd.Timedelta(hours=h), h + 1, t,
                             float(y_true[k, h, j]), float(y_pred[k, h, j])))
    return pd.DataFrame(rows, columns=["model", "origin", "timestamp", "horizon", "target", "y_true", "y_pred"])


def backtest_nn(model, name: str, df: pd.DataFrame, test_days: List[str], scaler: Scaler,
                hist_cols: List[str], fut_cols: List[str]) -> Tuple[pd.DataFrame, float]:
    xh, xf, y, days = build_windows(df, test_days, scaler, hist_cols, fut_cols)
    t0 = time.time()
    p = predict_scaled(model, xh, xf)
    per_day = (time.time() - t0) / max(len(days), 1)
    y_o = scaler.inverse(y, config.TARGETS)
    p_o = scaler.inverse(p, config.TARGETS)
    return to_long(name, days, y_o, p_o), per_day


def backtest_naive(name: str, lag_days: int, df: pd.DataFrame, days: List[pd.Timestamp]) -> pd.DataFrame:
    p = naive_predict(df, days, lag_days)
    y = np.stack([df.reindex(pd.date_range(d, periods=config.HORIZON, freq="h"))[config.TARGETS].values
                  for d in days])
    return to_long(name, days, y, p)
