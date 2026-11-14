#!/usr/bin/env bash
# Full pipeline once the raw Excel files are in data/raw/
set -e
python scripts/prepare_dataset.py        # clean + weather + features + splits
python scripts/train_models.py           # LSTM, GRU, Transformer (weather and no-weather)
python scripts/evaluate_models.py        # rolling day-ahead backtest, tables, plots
python -m pytest -q                      # tests (use synthetic fixtures only)
echo "Now run:  streamlit run app.py"
