"""Inference: forecast the target day. Owner: M9.
The dashboard uses this so its numbers match the model outputs exactly."""
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch

from . import config
from .dataset import Scaler
from .feature_engineering import feature_columns
from .models import build_model
from .models.baselines import naive_predict
from .training import checkpoint_path, predict_scaled


def load_checkpoint(name: str, variant: str, models_dir: Path = config.MODELS_DIR):
    p = checkpoint_path(models_dir, name, variant)
    if not p.exists():
        raise FileNotFoundError(f"No trained model at {p}. Run scripts/train_models.py first.")
    ck = torch.load(p, map_location="cpu", weights_only=False)
    info = ck["info"]
    model = build_model(name, info["n_hist"], info["n_fut"], **info["hparams"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, Scaler.from_dict(ck["scaler"]), info


def _get_frequency_string(df: pd.DataFrame) -> str:
    """Infer frequency string from dataframe index or fall back to config/hourly."""
    if df.index.freqstr:
        return df.index.freqstr
    if len(df) > 1:
        # Infer delta between first two rows
        delta_min = int((df.index[1] - df.index[0]).total_seconds() / 60)
        if delta_min == 15:
            return "15min"
        elif delta_min == 60:
            return "h"
    return getattr(config, "FREQ", "h")


def forecast_day(name: str, variant: str, df: pd.DataFrame, target_date,
                 models_dir: Path = config.MODELS_DIR) -> pd.DataFrame:
    """Return a DataFrame with timestamps of target_date and the target columns.
    df must contain the LOOKBACK periods before target_date and the covariates for target_date."""
    d = pd.Timestamp(target_date).normalize()
    
    # Infer frequency from the input dataframe index (e.g., '15min' or 'h')
    freq = _get_frequency_string(df)
    
    # Determine periods per day based on frequency
    steps_per_day = 96 if freq in ("15min", "15T") else 24
    horizon = getattr(config, "HORIZON", 24)
    if horizon == 24 and steps_per_day == 96:
        horizon = 96  # Full 24 hours at 15-min resolution

    stamps = pd.date_range(d, periods=horizon, freq=freq)
    
    if name.startswith("seasonal_naive_"):
        # Adjust lag multiplier if frequency changed
        lag = int(name.split("_")[-1])
        vals = naive_predict(df, [d], lag)[0]
        return pd.DataFrame(vals, index=stamps, columns=config.TARGETS)

    model, scaler, info = load_checkpoint(name, variant, models_dir)
    hist_cols, fut_cols = feature_columns(use_weather=(variant == "weather"))
    
    # Lookback steps calculation
    lookback = getattr(config, "LOOKBACK", 168)
    # If lookback was defined in hours but data is 15-min, scale it up if needed
    if freq in ("15min", "15T") and lookback <= 168:
        lookback = lookback * 4  # e.g., 168 hours -> 672 steps

    hist_idx = pd.date_range(end=d - pd.Timedelta(seconds=1), periods=lookback, freq=freq)
    
    hist = df.reindex(hist_idx)[hist_cols]
    fut = df.reindex(stamps)[fut_cols]
    
    if hist.isna().any().any() or fut.isna().any().any():
        missing_hist_count = hist.isna().any(axis=1).sum()
        missing_fut_count = fut.isna().any(axis=1).sum()
        raise ValueError(
            f"Missing inputs for {d.date()} (freq={freq}): "
            f"found {missing_hist_count} missing rows in history (needed {lookback}) "
            f"and {missing_fut_count} missing rows in future covariates (needed {horizon})."
        )
        
    xh = scaler.transform(hist.values, hist_cols).astype(np.float32)[None]
    xf = scaler.transform(fut.values, fut_cols).astype(np.float32)[None]
    p = predict_scaled(model, xh, xf)[0]
    return pd.DataFrame(scaler.inverse(p, config.TARGETS), index=stamps, columns=config.TARGETS)
