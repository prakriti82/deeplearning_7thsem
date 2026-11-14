"""Metrics. Owner: M6.
MAPE and sMAPE are only computed where the actual value is above a threshold
(threshold = frac * max(actual); for solar this excludes night hours where the
percentage error is undefined). MAE and RMSE use all points."""
from typing import Dict

import numpy as np
import pandas as pd

from . import config


def mae(y, p): return float(np.nanmean(np.abs(y - p))) """Calculate mean absolute error between actual and predicted values."""
def rmse(y, p): return float(np.sqrt(np.nanmean((y - p) ** 2))) """Calculate root mean squared error."""


def smape(y, p, frac: float = 0.0) -> float:
    thr = frac * np.nanmax(np.abs(y)) if frac > 0 else 0.0
    den = np.abs(y) + np.abs(p)
    m = (den > 1e-9) & (np.abs(y) > thr)
    return float(np.nanmean(2 * np.abs(y - p)[m] / den[m]) * 100)


def mape(y, p, frac: float = 0.0) -> float:
    thr = frac * np.nanmax(np.abs(y)) if frac > 0 else 1e-9
    m = np.abs(y) > thr
    if m.sum() == 0:
        return float("nan")
    return float(np.nanmean(np.abs((y - p)[m] / y[m])) * 100)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, target: str) -> Dict[str, float]:
    frac = config.SOLAR_MAPE_FRAC if target == "solar_mw" else config.DEMAND_MAPE_FRAC
    y, p = np.asarray(y_true, float).ravel(), np.asarray(y_pred, float).ravel()
    ok = ~np.isnan(y) & ~np.isnan(p)
    y, p = y[ok], p[ok]
    return {"MAE": mae(y, p), "RMSE": rmse(y, p), "MAPE": mape(y, p, frac), "sMAPE": smape(y, p, frac),
            "n_points": int(len(y))}


def metrics_table(frame: pd.DataFrame) -> pd.DataFrame:
    """frame: long format with columns model, target, y_true, y_pred."""
    rows = []
    for (m, t), g in frame.groupby(["model", "target"]):
        rows.append({"model": m, "target": t, **compute_metrics(g["y_true"], g["y_pred"], t)})
    return pd.DataFrame(rows)
