"""Central configuration. Edit values here, not inside the code.
Owner: M10 (coordinator) with input from every member."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
PRED_DIR = ROOT / "outputs" / "predictions"
METRICS_DIR = ROOT / "outputs" / "metrics"
PLOTS_DIR = ROOT / "outputs" / "plots"

# ---------------------------------------------------------------- forecasting task
TARGETS = ["demand_mw", "solar_mw"]   # units are whatever the source file uses
LOOKBACK = 168        # hours of history given to the models (7 days)
HORIZON = 24          # forecast the next 24 hourly values
SEED = 42

# chronological split by whole days: train | validation | test (latest)
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15

# ---------------------------------------------------------------- raw-data parsing
# Fill these in AFTER running:  python scripts/prepare_dataset.py --inspect
# None = try to detect automatically from column names (fails loudly if ambiguous).
COLUMN_OVERRIDES = {
    "timestamp": None,   # e.g. "Date & Time"
    "demand_mw": None,   # e.g. "Northern Region Demand"
    "solar_mw": None,
    "wind_mw": None,
}
# If one sheet holds several regions as rows, set these two (else leave None):
REGION_COLUMN = None     # e.g. "Region"
REGION_VALUE = None      # e.g. "Northern"
# If the columns hold several regions, put a keyword here to pick one (else None):
REGION_KEYWORD = None    # e.g. "northern"
SHEET_NAME = None        # None = try every sheet

# ---------------------------------------------------------------- cleaning
MAX_INTERPOLATE_HOURS = 3     # only gaps up to this length are filled
OUTLIER_MADS = 6.0            # demand values this many MADs from a 7-day median are set to NaN

# ---------------------------------------------------------------- weather (Open-Meteo)
WEATHER_VARS = [
    "temperature_2m", "relative_humidity_2m", "cloud_cover",
    "shortwave_radiation", "direct_radiation", "diffuse_radiation", "wind_speed_10m",
]
# EDIT to match the grid region you chose. Weights are normalised automatically.
# Example below is the Northern region; one point is a simplification of a whole region.
WEATHER_LOCATIONS = {
    "Delhi":      {"lat": 28.61, "lon": 77.21, "weight": 1.0},
    "Jaipur":     {"lat": 26.91, "lon": 75.79, "weight": 1.0},
    "Jodhpur":    {"lat": 26.24, "lon": 73.02, "weight": 1.0},
    "Lucknow":    {"lat": 26.85, "lon": 80.95, "weight": 1.0},
    "Chandigarh": {"lat": 30.73, "lon": 76.78, "weight": 1.0},
}
WEATHER_TZ = "Asia/Kolkata"

# ---------------------------------------------------------------- evaluation
# MAPE on solar is only computed where actual > SOLAR_MAPE_FRAC * max(actual).
SOLAR_MAPE_FRAC = 0.05
DEMAND_MAPE_FRAC = 0.0

# ---------------------------------------------------------------- training
MODEL_NAMES = ["lstm", "gru", "transformer"]
BASELINES = ["seasonal_naive_24", "seasonal_naive_168"]
HPARAMS = {
    "epochs": 40, "batch_size": 64, "lr": 1e-3, "weight_decay": 1e-5,
    "patience": 6, "hidden": 64, "layers": 2, "dropout": 0.1,
    "d_model": 64, "nhead": 4, "ff": 128,
}
