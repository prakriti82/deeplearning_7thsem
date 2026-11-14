"""Train LSTM, GRU and Transformer on the chronological TRAIN split (early stopping on VAL).
Owners: M3, M4, M5.
  python scripts/train_models.py
  python scripts/train_models.py --models lstm gru --variants weather --epochs 20"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.dataset import build_windows, fit_scaler, load_splits
from src.feature_engineering import feature_columns
from src.training import save_checkpoint, train_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=config.MODEL_NAMES)
    ap.add_argument("--variants", nargs="+", default=["weather", "noweather"],
                    choices=["weather", "noweather"])
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    path = config.PROC_DIR / "dataset.csv"
    if not path.exists():
        raise SystemExit("data/processed/dataset.csv not found. Run scripts/prepare_dataset.py first.")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    splits = load_splits(config.PROC_DIR / "splits.json")
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    hp = {"epochs": args.epochs} if args.epochs else {}

    for variant in args.variants:
        use_wx = variant == "weather"
        hist, fut = feature_columns(use_wx)
        scaler = fit_scaler(df, splits, use_wx)          # fitted on train period only
        tr = build_windows(df, splits["train"], scaler, hist, fut)
        va = build_windows(df, splits["val"], scaler, hist, fut)
        print(f"[{variant}] train windows={len(tr[3])} val windows={len(va[3])}")
        for name in args.models:
            model, info = train_model(name, tr[:3], va[:3], hp)
            p = save_checkpoint(model, info, scaler, config.MODELS_DIR, name, variant)
            (config.METRICS_DIR / f"train_{name}_{variant}.json").write_text(json.dumps(info, indent=2))
            print(f"saved {p}  (train time {info['train_time_s']:.0f}s, best val loss {info['best_val_loss']:.4f})")


if __name__ == "__main__":
    main()
