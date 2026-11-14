# Team plan (10 members)

Split follows the course recommendation: 2 data, 3 modelling, 2 evaluation,
2 deployment, 1 coordination. Names go in docs/MEMBERS.md.

## Roles and tasks

### Data (M1, M2)
**M1 - Grid-India loading and cleaning** (src/data_loader.py, src/preprocessing.py, data/README.md)
- Download the dataset (check the latest version and licence), list sheets and columns.
- Parse all monthly Excel files, find demand / solar / wind columns, handle units.
- Merge chronologically; fix duplicates, gaps, bad values; strictly hourly index.
- Pick the region(s) to use with the team; write a data quality report with plots.
- Output: validated hourly table of demand_mw, solar_mw (and wind_mw).

**M2 - Weather and features** (src/weather.py, src/feature_engineering.py)
- Fetch Open-Meteo weather (cached); investigate forecast-type data vs actual.
- Choose representative locations for the region and document the limitation.
- Build calendar, holiday, lag (>= 24 h) and rolling features without leakage.
- Produce data/processed/dataset.csv and splits.json (chronological) with M1.
- Support EDA plots of weather vs demand and solar.

### Modelling (M3, M4, M5)
**M3 - Training framework + LSTM** (src/training.py, src/models/lstm.py)
- Windowing (lookback -> next 24 h), train-only scaling, early stopping, seeds, config.
- Direct 24-hour multi-output LSTM. Publish the shared framework early (Week 3).

**M4 - Baselines + GRU** (src/models/baselines.py, xgboost_baseline.py, gru.py)
- Seasonal naive (same hour yesterday / last week); optional extra baseline (e.g. LightGBM) as an extension.
- GRU with the same inputs and horizon as LSTM.

**M5 - Transformer** (src/models/transformer.py)
- Encoder-based forecaster with positional encoding, direct 24-hour output.
- Hyperparameter search on validation only; record training and inference time.

### Evaluation (M6, M7)
**M6 - Metrics, backtesting, tests** (src/evaluation.py, src/backtesting.py, tests/)
- MAE, RMSE, MAPE, sMAPE; documented solar-zero handling.
- Rolling-origin day-ahead backtest over the test period, same for all models.
- pytest tests: 24 outputs, correct timestamps, no leakage, metric correctness.

**M7 - Ablations and error analysis** (scripts/evaluate_models.py, outputs/plots/)
- Ablations: with/without weather, lookback length, feature groups, per-target models.
- Residual plots, errors by hour / season / day type, comparison tables and charts.
- Statistical comparison of models (e.g. paired error tests).

### Deployment (M8, M9)
**M8 - Dashboard** (app.py)
- Streamlit sections: Overview, Day-Ahead Forecast, Model Comparison,
  Forecast Evaluation, Weather Features. Reads only saved real outputs.

**M9 - Inference, API, packaging** (src/forecasting.py, api/)
- Load saved models, build features for the latest data, return 24 hourly values.
- Optional small FastAPI endpoint, requirements, run instructions, tidy scripts.

### Coordination (M10)
- Create the repo, protect `main`, set up issues/board, review PRs, enforce
  weekly commits, keep CONTRACTS.md current, write the report and slides,
  prepare the demo and viva questions, record decisions and limitations.

## Starter code already exists - what each member must still do
The repo ships a working, tested skeleton (tested only on synthetic fixtures; NOT yet run on the
real dataset). Everyone's job is to run it on real data, read and understand their part, fix what
breaks, improve it, add tests and document it. Each member must be able to explain their module
in the viva. Examples of real contributions: M1 verifies the loader against the real Excel files
and fixes column detection; M2 chooses weather locations and checks leakage; M3-M5 tune and
compare models; M6 adds tests and checks the metric rules; M7 runs ablations and writes the
analysis; M8-M9 polish the dashboard/inference; M10 writes the report and keeps records.

## Dependencies
M1 -> M2 -> (M3, M4, M5, M6) -> M7 -> M8, M9 -> M10 (report).
Until real data exists, M3-M9 build against the CONTRACTS.md interfaces and
small unit-test fixtures; test fixtures must be labelled and never used for results.

## Suggested timeline (adjust to your deadline)
| Week | Milestone |
|------|-----------|
| 1 | Repo set up, everyone commits first PR, dataset downloaded, contracts agreed |
| 2 | Loader + cleaning draft (M1), weather fetch (M2), forecast-origin decision, metrics skeleton (M6) |
| 3 | dataset.csv + splits.json, training framework (M3), baselines (M4), dashboard skeleton (M8) |
| 4 | LSTM, GRU, Transformer first runs, backtesting working (M6) |
| 5 | Tuned models, full backtest outputs, tests passing |
| 6 | Ablations and error analysis (M7), inference module (M9) |
| 7 | Dashboard wired to real outputs (M8), report draft (M10) |
| 8 | Final runs, README, demo rehearsal, viva prep |

## Review pairs (each PR is reviewed by the pair)
M1<->M2, M3<->M4, M5<->M6, M7<->M8, M9<->M10.
