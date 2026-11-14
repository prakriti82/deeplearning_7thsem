
# Project Coordination Log
## P6 - Electricity Load and Solar Generation Forecasting

**Coordinator:** Kajal Rana (Kajal0904)  
**Role:** M10 - Project Coordinator

## 1. Project Overview
The project aims to forecast day-ahead hourly electricity demand and solar generation for an Indian regional/national grid using historical electricity data and weather covariates.

The forecasting horizon is 24 hours, with LSTM, GRU and Transformer models compared against seasonal-naive baselines.

## 2. Team Responsibilities

| Member | Role | Responsibility |
|---|---|---|
| M1 | Data Engineer A | Grid data loading and cleaning |
| M2 | Data Engineer B | Weather and feature engineering |
| M3 | Modelling A | Training framework and LSTM |
| M4 | Modelling B | Baselines and GRU |
| M5 | Modelling C | Transformer |
| M6 | Evaluation A | Metrics, backtesting and tests |
| M7 | Evaluation B | Ablations and error analysis |
| M8 | Deployment A | Streamlit dashboard |
| M9 | Deployment B | Inference and API |
| M10 | Coordinator | Repository, report, demo and coordination |

## 3. Repository Management
- GitHub repository established for team collaboration.
- Member registration documented in MEMBERS.md.
- Individual branches and pull requests are used for contributions.
- Changes are merged into main through pull requests.
- Weekly progress is recorded in docs/logs/.
- Interface changes must follow CONTRACTS.md.

## 4. Coordination Activities
- Registered M10 coordinator details in MEMBERS.md.
- Created and merged the initial member-registration pull request.
- Added the M10 Week 1 progress log.
- Reviewed existing project documentation and repository structure.
- Identified pending project decisions and documentation requirements.

## 5. Technical Decisions and Pending Confirmations

| Topic | Current position |
|---|---|
| Forecast origin | 23:00 on previous day |
| Forecast horizon | 24 hourly values |
| Historical lookback | 168 hours |
| Models | LSTM, GRU, Transformer |
| Baselines | Yesterday and last-week seasonal naive |
| Weather mode | Team confirmation pending |
| Dataset region | Regional/national grid, not single-state |
| Evaluation | MAE, RMSE, MAPE, sMAPE |

The above follows the existing project documentation. Weather mode and other unresolved choices must be confirmed by the team.

## 6. Risks and Limitations
- Real dataset integration and validation are required.
- Dataset column formats and units must be verified.
- Historical forecast-type weather availability needs investigation.
- Actual weather as a perfect forecast introduces an evaluation limitation.
- Deep learning performance depends on data quality and test-period coverage.
- Model results must not be reported until real-data experiments are completed.

## 7. Upcoming Activities
- Coordinate weekly member progress.
- Track pending tasks and milestones.
- Record approved interface and forecasting decisions.
- Monitor pull requests and documentation updates.
- Coordinate real-data validation and model evaluation.
- Prepare final project report, presentation and demonstration.
- Maintain viva preparation material.

## 8. Submission and Reporting Policy
All reported metrics must come from actual evaluation outputs.
Synthetic test fixtures must not be presented as real experimental results.
Uncompleted tasks and unverified decisions must remain marked as pending.
