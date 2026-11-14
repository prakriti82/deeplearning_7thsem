# Interface contracts (agree on these in Week 1; change only via PR + coordinator)

## 1. Processed dataset (produced by M1 + M2, consumed by everyone)
- File: data/processed/dataset.csv, one row per hour, index column `timestamp`
  (IST, timezone-naive, strictly hourly, no duplicates).
- Targets: `demand_mw`, `solar_mw`. Optional: `wind_mw`.
- Weather columns (Open-Meteo): temperature_2m, relative_humidity_2m, cloud_cover,
  shortwave_radiation, direct_radiation, diffuse_radiation, wind_speed_10m.
- Calendar columns: hour, dow, month, is_weekend, is_holiday, cyclic sin/cos terms.
- Missing values stay NaN (never silently filled with fake values); short gaps
  (<= 3 h) may be interpolated and must be counted in the data report.
- Units are kept as in the source; every conversion is documented in data/README.md.
- Splits: data/processed/splits.json with train / val / test date boundaries
  (chronological). Scalers are fit on TRAIN only.

## 2. Forecast task (decision needed in Week 2, coordinator records it here)
- DEFAULT: forecast origin = 23:00 on day D-1 (uses data up to and including
  23:00); predicts hourly values for D 00:00 - 23:00 (24 values per target).
- Weather inputs for the target day must come from forecast-type data, or, if
  unavailable, actual weather treated as a perfect forecast (state this limitation).
- Only information available at the origin may be used as a feature.

## 3. Model interface (src/models/*.py)
Implemented as: every neural model is a torch Module with
    forward(x_hist[B,168,Fh], x_fut[B,24,Ff]) -> [B,24,2]   (scaled demand, solar)
registered in src/models/__init__.py (REGISTRY). Baselines live in src/models/baselines.py.
Training: src/training.py. Inference for one day: src/forecasting.forecast_day(), returning
24 rows indexed by the target-day timestamps with columns demand_mw, solar_mw.
All models use the same lookback, features, splits and horizon so the comparison is fair.

## 4. Output files
- outputs/predictions/<model>_<variant>.csv : model, origin, timestamp, horizon, target, y_true, y_pred, variant
- outputs/metrics/metrics_summary.csv : MAE, RMSE, MAPE, sMAPE per model/variant/target + train and inference time
- Solar MAPE is computed only where y_true > a stated threshold (daylight hours);
  also report MAE/RMSE and a normalised error. The rule is defined in src/evaluation.py.
