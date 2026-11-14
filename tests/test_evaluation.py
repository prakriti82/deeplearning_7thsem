import numpy as np
from src.evaluation import compute_metrics, mae, mape, rmse, smape


def test_basic_metrics():
    y = np.array([100.0, 200.0]); p = np.array([110.0, 180.0])
    assert mae(y, p) == 15.0
    assert abs(rmse(y, p) - np.sqrt((100 + 400) / 2)) < 1e-9
    assert abs(mape(y, p) - 10.0) < 1e-9
    assert smape(y, p) > 0


def test_solar_mape_ignores_night():
    y = np.array([0.0, 0.0, 100.0, 50.0]); p = np.array([5.0, 3.0, 90.0, 50.0])
    m = compute_metrics(y, p, "solar_mw")
    assert np.isfinite(m["MAPE"]) and abs(m["MAPE"] - 5.0) < 1e-9   # only the two daylight points
    assert m["n_points"] == 4                                        # MAE/RMSE use all points
