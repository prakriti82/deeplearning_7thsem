"""TEST FIXTURES ONLY. Synthetic data is used here to test the code and must
never be used for reported results."""
import numpy as np
import pandas as pd
import pytest

from src import config
from src.feature_engineering import add_calendar


@pytest.fixture(scope="session")
def fixture_df():
    idx = pd.date_range("2022-01-01", periods=24 * 120, freq="h")
    rng = np.random.default_rng(0)
    h = idx.hour.values
    df = pd.DataFrame(index=idx)
    df["solar_mw"] = np.clip(np.sin((h - 6) / 12 * np.pi), 0, None) * 100 + rng.normal(0, 2, len(idx))
    df["solar_mw"] = df["solar_mw"].clip(lower=0)
    df["demand_mw"] = 1000 + 100 * np.sin((h - 8) / 24 * 2 * np.pi) + rng.normal(0, 10, len(idx))
    for i, c in enumerate(config.WEATHER_VARS):
        df[c] = rng.normal(size=len(idx)) + i
    df = add_calendar(df)
    df.index.name = "timestamp"
    return df
