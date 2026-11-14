# P6 - Electricity Load and Solar Generation Forecasting

Day-ahead (next 24 hours, hourly) forecasting of electricity demand and solar
generation for an Indian grid with weather covariates. LSTM, GRU and Transformer
(direct 24-hour multi-output) are compared against seasonal-naive baselines with a
rolling-origin daily backtest, MAE / RMSE / MAPE / sMAPE, a weather ablation and a
Streamlit dashboard.

Data: Grid-India / NERLDC hourly demand, solar and wind (Mendeley Data,
DOI 10.17632/y58jknpgs8) + Open-Meteo weather. The data describes regional/national
grids, NOT a single state. See `data/README.md`.

## Setup (Mac / Linux)
    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt

## Run (after putting the Excel files in data/raw/)
    python scripts/prepare_dataset.py --inspect     # 1. look at the files, edit src/config.py if needed
    python scripts/prepare_dataset.py               # 2. dataset + splits
    python scripts/train_models.py                  # 3. train all models (add --epochs 10 for a quick run)
    python scripts/evaluate_models.py               # 4. backtest, metrics table, plots
    streamlit run app.py                            # 5. dashboard
    python -m pytest -q                             # tests
or `./run_all.sh` for steps 2-4 and the tests.

## How it works let's see
- **Task:** forecast origin 23:00 on day D-1; predict the 24 hours of day D for demand and solar.
- **Inputs:** last 168 h of demand, solar, weather and calendar features + known covariates
  (weather, calendar) of the target day. Weather of the target day is taken as a known
  forecast (`--weather-mode actual` = "perfect forecast" assumption; `forecast` = historical forecasts).
- **Split:** chronological by whole days: 70 % train / 15 % validation / 15 % test.
  Scalers are fitted on the train period only. Early stopping uses validation only.
- **Backtest:** one forecast per test day (rolling daily origin), all models scored on the same days.
- **Metrics:** MAE, RMSE, MAPE, sMAPE. MAPE/sMAPE for solar only where actual > 5 % of max
  (night hours are excluded because percentage error is undefined there).
- **Models:** `src/models/` (LSTM, GRU, Transformer share the same inputs and head).
  Baselines: seasonal naive (yesterday, last week).

## Structure
    src/            config, data_loader, preprocessing, weather, feature_engineering, dataset,
                    models/, training, backtesting, evaluation, forecasting
    scripts/        prepare_dataset.py, train_models.py, evaluate_models.py
    tests/          pytest (synthetic fixtures for testing code only, never for results)
    app.py          Streamlit dashboard (reads real saved outputs only)
    docs/           TEAM_PLAN, CONTRACTS, GIT_WORKFLOW, weekly logs
    data/ models/ outputs/ reports/

## Limitations (keep updated)
- Regional/national grid data, not state-level; weather uses a few representative locations.
- About 3-4 years of data: small for deep models; results depend on the test period.
- Day-ahead weather is treated as known (perfect forecast) unless `--weather-mode forecast` is used.
- The loader auto-detects columns heuristically; verify with `--inspect` on the real files.
