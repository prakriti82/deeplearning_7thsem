"""Seasonal-naive baselines. Owner: M4.
seasonal_naive_24  : tomorrow = same hours as today (D-1).
seasonal_naive_168 : tomorrow = same hours one week earlier (D-7).
Both only use values known at the 23:00 forecast origin."""
from typing import List

import numpy as np
import pandas as pd

from .. import config


def naive_predict(df: pd.DataFrame, days: List[pd.Timestamp], lag_days: int) -> np.ndarray:
    """Return (N,24,T) in ORIGINAL units; NaN where history is missing."""
    out = np.full((len(days), config.HORIZON, len(config.TARGETS)), np.nan)
    for k, d in enumerate(days):
        src = pd.date_range(d - pd.Timedelta(days=lag_days), periods=config.HORIZON, freq="h")
        vals = df.reindex(src)[config.TARGETS].values
        out[k] = vals
    return out
