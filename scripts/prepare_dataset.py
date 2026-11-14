"""Build data/processed/dataset.csv from the raw Excel files + Open-Meteo weather.
Owners: M1 + M2.
  python scripts/prepare_dataset.py --inspect        # look at the Excel structure first
  python scripts/prepare_dataset.py                  # build the dataset
  python scripts/prepare_dataset.py --weather-mode forecast"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.data_loader import inspect_excel_files, load_grid_data
from src.dataset import make_splits, save_splits
from src.feature_engineering import build_features, feature_columns
from src.preprocessing import build_hourly
from src.weather import fetch_weather


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true", help="print Excel sheets/columns and exit")
    ap.add_argument("--weather-mode", choices=["actual", "forecast"], default="actual")
    args = ap.parse_args()
    if args.inspect:
        inspect_excel_files()
        return

    raw = load_grid_data()
    hourly, report = build_hourly(raw)
    start, end = str(hourly.index.min().date()), str(hourly.index.max().date())
    print(f"Hourly data {start} -> {end}; weather mode = {args.weather_mode}")
    weather = fetch_weather(start, end, mode=args.weather_mode)
    df = build_features(hourly, weather)
    report["weather_mode"] = args.weather_mode
    report["weather_missing_rows"] = int(df[config.WEATHER_VARS].isna().any(axis=1).sum())

    hist, fut = feature_columns(use_weather=True)
    splits = make_splits(df, list(dict.fromkeys(hist + fut)))
    report["days"] = {k: len(v) for k, v in splits.items()}
    if min(report["days"].values()) == 0:
        raise SystemExit(f"A split is empty: {report['days']}. Too little complete data.")

    config.PROC_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.PROC_DIR / "dataset.csv", index_label="timestamp")
    save_splits(splits, config.PROC_DIR / "splits.json")
    (config.PROC_DIR / "data_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Saved dataset ({len(df)} rows) to {config.PROC_DIR}")


if __name__ == "__main__":
    main()
