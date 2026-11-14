"""Forecast API. Owner: M9.

Run (from the project root, with the venv active):
    uvicorn api.main:app --reload
Then open http://127.0.0.1:8000/docs  or  call, for example:
    http://127.0.0.1:8000/forecast?date=2024-05-10&model=gru&variant=weather

Needs data/processed/dataset.csv and a trained model in models/ (see README).
"""
from functools import lru_cache
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from src import config
from src.forecasting import forecast_day, latest_forecastable_day

app = FastAPI(title="Electricity Load and Solar Forecast API", version="1.0")

VALID_MODELS = list(config.MODEL_NAMES) + list(config.BASELINES)
VALID_VARIANTS = ["weather", "noweather", "baseline"]


class HourForecast(BaseModel):
    timestamp: str
    demand_mw: float
    solar_mw: float


class ForecastResponse(BaseModel):
    model: str
    variant: str
    date: str
    horizon_hours: int
    forecast: List[HourForecast]


@lru_cache(maxsize=1)
def _load_dataset() -> pd.DataFrame:
    path = config.PROC_DIR / "dataset.csv"
    if not path.exists():
        raise FileNotFoundError(
            "data/processed/dataset.csv not found. Run scripts/prepare_dataset.py first.")
    return pd.read_csv(path, index_col=0, parse_dates=True)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/forecast", response_model=ForecastResponse)
def forecast(
    date: Optional[str] = Query(None, description="Target day YYYY-MM-DD (default: latest possible day)"),
    model: str = Query("gru", description="lstm | gru | transformer | seasonal_naive_24 | seasonal_naive_168"),
    variant: Optional[str] = Query(None, description="weather | noweather | baseline"),
):
    """Return the 24 hourly demand and solar forecasts for one day."""
    if model not in VALID_MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model '{model}'. Choose from {VALID_MODELS}.")
    if variant is None:
        variant = "baseline" if model.startswith("seasonal_naive") else "weather"
    if variant not in VALID_VARIANTS:
        raise HTTPException(status_code=400, detail=f"Unknown variant '{variant}'. Choose from {VALID_VARIANTS}.")

    try:
        df = _load_dataset()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    try:
        day = pd.Timestamp(date) if date else latest_forecastable_day(df, variant)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date: {e}")

    try:
        fc = forecast_day(model, variant, df, day)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    if fc.isna().any().any():
        raise HTTPException(status_code=422, detail=f"Not enough data to forecast {day.date()}.")

    rows = [HourForecast(timestamp=str(ts), demand_mw=float(r["demand_mw"]), solar_mw=float(r["solar_mw"]))
            for ts, r in fc.iterrows()]
    return ForecastResponse(model=model, variant=variant, date=str(day.date()),
                            horizon_hours=config.HORIZON, forecast=rows)
