"""Clean raw series into a strict hourly table. Owner: M1."""
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from . import config

VALUE_COLS = ["demand_mw", "solar_mw", "wind_mw"]


def build_hourly(raw: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Return (hourly_df indexed by timestamp, report). Missing values stay NaN
    except gaps <= config.MAX_INTERPOLATE_HOURS which are interpolated and counted."""
    df = raw.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    cols = [c for c in VALUE_COLS if c in df.columns]
    report = {"rows_raw": int(len(df)),
              "duplicate_timestamps": int(df["timestamp"].duplicated().sum())}

    df = df.set_index("timestamp")[cols]
    hourly = df.resample("h").mean()            # also converts sub-hourly data to hourly
    report["hours_total"] = int(len(hourly))

    for c in cols:                              # invalid values
        bad = hourly[c] < 0
        report[f"{c}_negative_set_nan"] = int(bad.sum())
        hourly.loc[bad, c] = np.nan

    d = hourly["demand_mw"]                     # demand outliers (robust, 7-day window)
    med = d.rolling(24 * 7, center=True, min_periods=24).median()
    mad = (d - med).abs().rolling(24 * 7, center=True, min_periods=24).median()
    out = (d - med).abs() > config.OUTLIER_MADS * mad.replace(0, np.nan)
    report["demand_outliers_set_nan"] = int(out.sum())
    hourly.loc[out, "demand_mw"] = np.nan

    for c in cols:
        before = int(hourly[c].isna().sum())
        hourly[c] = hourly[c].interpolate(limit=config.MAX_INTERPOLATE_HOURS, limit_area="inside")
        report[f"{c}_missing_before"] = before
        report[f"{c}_missing_after"] = int(hourly[c].isna().sum())
    report["start"] = str(hourly.index.min())
    report["end"] = str(hourly.index.max())
    return hourly, report
