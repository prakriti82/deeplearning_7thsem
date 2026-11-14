import numpy as np
import pandas as pd
from src import config
from src.dataset import build_windows, fit_scaler, make_splits
from src.feature_engineering import feature_columns


def _setup(df):
    hist, fut = feature_columns(True)
    splits = make_splits(df, list(dict.fromkeys(hist + fut)))
    return hist, fut, splits


def test_splits_are_chronological_and_disjoint(fixture_df):
    _, _, s = _setup(fixture_df)
    assert max(s["train"]) < min(s["val"]) and max(s["val"]) < min(s["test"])
    assert len(set(s["train"]) & set(s["test"])) == 0


def test_scaler_uses_train_period_only(fixture_df):
    _, _, s = _setup(fixture_df)
    sc = fit_scaler(fixture_df, s, True)
    train_rows = fixture_df[fixture_df.index < pd.Timestamp(s["val"][0])]
    assert abs(sc.mean[sc.cols.index("demand_mw")] - train_rows["demand_mw"].mean()) < 1e-6


def test_window_shapes_and_no_future_in_history(fixture_df):
    hist, fut, s = _setup(fixture_df)
    sc = fit_scaler(fixture_df, s, True)
    xh, xf, y, days = build_windows(fixture_df, s["test"], sc, hist, fut)
    assert xh.shape[1:] == (config.LOOKBACK, len(hist))
    assert xf.shape[1:] == (config.HORIZON, len(fut))
    assert y.shape[1:] == (config.HORIZON, len(config.TARGETS))
    # history of the first window ends one hour before the target day starts:
    d = days[0]
    i = fixture_df.index.get_loc(d)
    expected = sc.transform(fixture_df[hist].values[i - config.LOOKBACK:i], hist)
    assert np.allclose(xh[0], expected, atol=1e-5)
    expected_y = sc.transform(fixture_df[config.TARGETS].values[i:i + 24], config.TARGETS)
    assert np.allclose(y[0], expected_y, atol=1e-5)
    # target values must not appear in the history features
    assert not np.allclose(xh[0][-24:, :2], y[0])
