import numpy as np
import pandas as pd
import pytest
from src import config
from src.dataset import build_windows, fit_scaler, make_splits
from src.feature_engineering import feature_columns
from src.forecasting import forecast_day
from src.models import REGISTRY, build_model
from src.training import save_checkpoint, train_model

SMALL = {"epochs": 2, "hidden": 8, "d_model": 16, "nhead": 2, "ff": 16, "layers": 1, "batch_size": 16}


@pytest.mark.parametrize("name", list(REGISTRY))
def test_output_shape(name):
    import torch
    m = build_model(name, n_hist=17, n_fut=15, **SMALL)
    out = m(torch.randn(3, config.LOOKBACK, 17), torch.randn(3, config.HORIZON, 15))
    assert out.shape == (3, config.HORIZON, len(config.TARGETS))


def test_train_save_forecast_returns_24_correct_timestamps(fixture_df, tmp_path):
    hist, fut = feature_columns(True)
    s = make_splits(fixture_df, list(dict.fromkeys(hist + fut)))
    sc = fit_scaler(fixture_df, s, True)
    tr = build_windows(fixture_df, s["train"], sc, hist, fut)
    va = build_windows(fixture_df, s["val"], sc, hist, fut)
    model, info = train_model("gru", tr[:3], va[:3], SMALL, verbose=False)
    save_checkpoint(model, info, sc, tmp_path, "gru", "weather")
    day = s["test"][0]
    out = forecast_day("gru", "weather", fixture_df, day, models_dir=tmp_path)
    assert out.shape == (24, 2)
    assert list(out.index) == list(pd.date_range(day, periods=24, freq="h"))
    assert np.isfinite(out.values).all()
    naive = forecast_day("seasonal_naive_24", "baseline", fixture_df, day)
    assert naive.shape == (24, 2)
