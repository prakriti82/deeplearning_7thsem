"""Calendar features and the history/future column groups. Owner: M2.
Leakage rule: HISTORY columns only contain values up to the forecast origin;
FUTURE columns are only things known in advance for the target day
(calendar + weather forecast)."""
from typing import List, Tuple

import numpy as np
import pandas as pd

from . import config

CAL_COLS = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos",
            "is_weekend", "is_holiday"]


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    import holidays
    df = df.copy()
    idx = df.index
    df["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7)
    df["doy_sin"] = np.sin(2 * np.pi * idx.dayofyear / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * idx.dayofyear / 365.25)
    df["is_weekend"] = (idx.dayofweek >= 5).astype(float)
    hol = holidays.India(years=range(idx.year.min(), idx.year.max() + 1))
    df["is_holiday"] = pd.Series(idx.date, index=idx).isin(set(hol.keys())).astype(float).values
    return df


def build_features(grid: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Join hourly grid data with weather (left join on grid timestamps) + calendar."""
    df = grid.join(weather, how="left")
    return add_calendar(df)


def feature_columns(use_weather: bool = True) -> Tuple[List[str], List[str]]:
    wx = list(config.WEATHER_VARS) if use_weather else []
    hist = list(config.TARGETS) + wx + CAL_COLS
    fut = wx + CAL_COLS
    return hist, fut
