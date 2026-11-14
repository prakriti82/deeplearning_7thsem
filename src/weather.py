"""Open-Meteo weather for the chosen region. Owner: M2.
mode='actual'   -> archive (observed/reanalysis). For day-ahead use it acts as a
                   'perfect weather forecast' assumption: state this in the report.
mode='forecast' -> historical forecast API (what was predicted at the time)."""
import time
from pathlib import Path
from typing import Dict

import pandas as pd
import requests

from . import config

URLS = {"actual": "https://archive-api.open-meteo.com/v1/archive",
        "forecast": "https://historical-forecast-api.open-meteo.com/v1/forecast"}


def _fetch_location(name: str, loc: Dict, start: str, end: str, mode: str, cache: Path) -> pd.DataFrame:
    f = cache / f"weather_{mode}_{name}_{start}_{end}.csv"
    if f.exists():
        return pd.read_csv(f, index_col=0, parse_dates=True)
    frames = []
    for year in range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1):
        s = max(pd.Timestamp(start), pd.Timestamp(f"{year}-01-01")).strftime("%Y-%m-%d")
        e = min(pd.Timestamp(end), pd.Timestamp(f"{year}-12-31")).strftime("%Y-%m-%d")
        params = {"latitude": loc["lat"], "longitude": loc["lon"], "start_date": s, "end_date": e,
                  "hourly": ",".join(config.WEATHER_VARS), "timezone": config.WEATHER_TZ}
        r = requests.get(URLS[mode], params=params, timeout=90)
        r.raise_for_status()
        h = pd.DataFrame(r.json()["hourly"])
        h["time"] = pd.to_datetime(h["time"])
        frames.append(h.set_index("time"))
        time.sleep(1)
    out = pd.concat(frames).sort_index()
    out.to_csv(f)
    return out


def fetch_weather(start: str, end: str, mode: str = "actual",
                  cache: Path = config.RAW_DIR) -> pd.DataFrame:
    """Weighted average of all configured locations, hourly, index 'timestamp'."""
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    total_w = sum(l["weight"] for l in config.WEATHER_LOCATIONS.values())
    acc = None
    for name, loc in config.WEATHER_LOCATIONS.items():
        w = _fetch_location(name, loc, start, end, mode, cache) * (loc["weight"] / total_w)
        acc = w if acc is None else acc.add(w, fill_value=0)
    acc.index.name = "timestamp"
    return acc[config.WEATHER_VARS]


def fetch_upcoming_forecast(days: int = 3) -> pd.DataFrame:
    """Live forecast for the next days (used by the dashboard for new data). Needs internet."""
    total_w = sum(l["weight"] for l in config.WEATHER_LOCATIONS.values())
    acc = None
    for name, loc in config.WEATHER_LOCATIONS.items():
        params = {"latitude": loc["lat"], "longitude": loc["lon"], "forecast_days": days,
                  "hourly": ",".join(config.WEATHER_VARS), "timezone": config.WEATHER_TZ}
        r = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=60)
        r.raise_for_status()
        h = pd.DataFrame(r.json()["hourly"])
        h["time"] = pd.to_datetime(h["time"])
        w = h.set_index("time") * (loc["weight"] / total_w)
        acc = w if acc is None else acc.add(w, fill_value=0)
    acc.index.name = "timestamp"
    return acc[config.WEATHER_VARS]
