"""Chronological splits, train-only scaling and day-ahead windows. Owner: M2 (+M3, M6 review).

Forecast task: origin = 23:00 of day D-1. Inputs: last LOOKBACK hours of history
(x_hist) and the known covariates of the target day D (x_fut). Output: 24 hourly
values of every target for day D. One sample per target day => daily rolling origin."""
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from . import config


def complete_days(df: pd.DataFrame, cols: List[str]) -> List[pd.Timestamp]:
    """Days that have 24 hourly rows and no NaN in the given columns."""
    ok = df[cols].notna().all(axis=1)
    g = ok.groupby(df.index.normalize()).sum()
    return [d for d, n in g.items() if n == 24]


def make_splits(df: pd.DataFrame, cols: List[str]) -> Dict[str, List[str]]:
    """Chronological train/val/test split of target days (no shuffling)."""
    days = complete_days(df, cols)
    n = len(days)
    n_tr, n_va = int(n * config.TRAIN_FRAC), int(n * config.VAL_FRAC)
    return {"train": [str(d.date()) for d in days[:n_tr]],
            "val": [str(d.date()) for d in days[n_tr:n_tr + n_va]],
            "test": [str(d.date()) for d in days[n_tr + n_va:]]}


def save_splits(splits: Dict, path: Path) -> None:
    Path(path).write_text(json.dumps(splits, indent=2))


def load_splits(path: Path) -> Dict[str, List[str]]:
    return json.loads(Path(path).read_text())


class Scaler:
    """Per-column standardisation, fitted on TRAIN rows only."""
    def __init__(self, cols: List[str]):
        self.cols = list(cols)
        self.mean = None
        self.std = None

    def fit(self, df: pd.DataFrame) -> "Scaler":
        self.mean = df[self.cols].mean().values.astype(np.float64)
        self.std = df[self.cols].std().replace(0, 1.0).fillna(1.0).values.astype(np.float64)
        return self

    def transform(self, arr: np.ndarray, cols: List[str]) -> np.ndarray:
        idx = [self.cols.index(c) for c in cols]
        return (arr - self.mean[idx]) / self.std[idx]

    def inverse(self, arr: np.ndarray, cols: List[str]) -> np.ndarray:
        idx = [self.cols.index(c) for c in cols]
        return arr * self.std[idx] + self.mean[idx]

    def to_dict(self) -> Dict:
        return {"cols": self.cols, "mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d: Dict) -> "Scaler":
        s = cls(d["cols"])
        s.mean, s.std = np.array(d["mean"]), np.array(d["std"])
        return s


def fit_scaler(df: pd.DataFrame, splits: Dict, use_weather: bool) -> Scaler:
    """Fit on rows strictly before the first validation day (train period only)."""
    from .feature_engineering import feature_columns
    hist, fut = feature_columns(use_weather)
    cols = list(dict.fromkeys(hist + fut))
    cutoff = pd.Timestamp(splits["val"][0]) if splits["val"] else df.index.max() + pd.Timedelta(hours=1)
    return Scaler(cols).fit(df[df.index < cutoff])


def build_windows(df: pd.DataFrame, target_days: List[str], scaler: Scaler,
                  hist_cols: List[str], fut_cols: List[str],
                  lookback: int = config.LOOKBACK, horizon: int = config.HORIZON
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[pd.Timestamp]]:
    """Return x_hist (N,L,Fh), x_fut (N,H,Ff), y (N,H,T) [all scaled], and the days used.
    Days with missing rows/NaN anywhere in their window are skipped (never filled)."""
    xh_all = scaler.transform(df[hist_cols].values, hist_cols)
    xf_all = scaler.transform(df[fut_cols].values, fut_cols)
    y_all = scaler.transform(df[config.TARGETS].values, config.TARGETS)
    index = df.index
    X_h, X_f, Y, days = [], [], [], []
    for d in target_days:
        start = pd.Timestamp(d)
        if start not in index:
            continue
        i = index.get_loc(start)
        if i < lookback or i + horizon > len(df):
            continue
        # window must be contiguous hourly
        if index[i + horizon - 1] - index[i - lookback] != pd.Timedelta(hours=lookback + horizon - 1):
            continue
        xh, xf, y = xh_all[i - lookback:i], xf_all[i:i + horizon], y_all[i:i + horizon]
        if np.isnan(xh).any() or np.isnan(xf).any() or np.isnan(y).any():
            continue
        X_h.append(xh); X_f.append(xf); Y.append(y); days.append(start)
    n_h, n_f = len(hist_cols), len(fut_cols)
    return (np.array(X_h, dtype=np.float32).reshape(-1, lookback, n_h),
            np.array(X_f, dtype=np.float32).reshape(-1, horizon, n_f),
            np.array(Y, dtype=np.float32).reshape(-1, horizon, len(config.TARGETS)), days)
