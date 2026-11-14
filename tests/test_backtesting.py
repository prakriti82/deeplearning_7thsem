import numpy as np
import pandas as pd

from src.backtesting import to_long
from src import config


def test_backtesting_timestamp_generation():
    day = pd.Timestamp("2025-01-01")

    y_true = np.zeros((1, config.HORIZON, len(config.TARGETS)))
    y_pred = np.ones((1, config.HORIZON, len(config.TARGETS)))

    result = to_long("test_model", [day], y_true, y_pred)

    assert result["timestamp"].min() == day
    assert result["timestamp"].max() == day + pd.Timedelta(hours=23)
    assert result["origin"].nunique() == 1
    assert result["horizon"].min() == 1
    assert result["horizon"].max() == 24
