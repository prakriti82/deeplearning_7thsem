
# P6 - Electricity Load and Solar Generation Forecasting

## 1. Abstract
This project focuses on day-ahead forecasting of hourly electricity demand and solar generation for an Indian regional/national grid. It uses historical electricity data and weather covariates to predict the next 24 hours.

LSTM, GRU and Transformer architectures are compared with seasonal-naive forecasting baselines using rolling-origin evaluation.

## 2. Problem Statement
Accurate electricity demand and solar generation forecasts support grid scheduling, resource allocation and electricity planning.

The objective is to develop a forecasting system that predicts 24 hourly demand and solar values for the following day.

## 3. Objectives
- Develop a day-ahead forecasting pipeline.
- Integrate electricity and weather datasets.
- Perform preprocessing and feature engineering.
- Implement LSTM, GRU and Transformer models.
- Compare deep learning models with seasonal-naive baselines.
- Evaluate forecasting errors using standard metrics.
- Develop a Streamlit forecasting dashboard.

## 4. Dataset
Electricity data:
Grid-India / NERLDC hourly demand, solar and wind data.

Dataset reference:
Mendeley Data DOI: 10.17632/y58jknpgs8

Weather data:
Open-Meteo historical weather data.

The dataset represents regional/national grid information and should not be described as a single-state dataset.

## 5. Methodology
1. Data collection and inspection.
2. Data cleaning and hourly alignment.
3. Weather integration.
4. Feature engineering and chronological splitting.
5. Model training and validation.
6. Rolling-origin backtesting.
7. Performance comparison.
8. Streamlit dashboard deployment.

## 6. Model Architectures
- LSTM: captures sequential temporal dependencies.
- GRU: provides a recurrent forecasting architecture.
- Transformer: uses attention-based sequence modelling.
- Seasonal Naive: provides yesterday and last-week reference forecasts.

## 7. Evaluation Metrics
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)
- Symmetric Mean Absolute Percentage Error (sMAPE)

Solar percentage errors exclude night hours according to the documented threshold.

## 8. Experimental Results
PENDING: Populate this section after training and evaluating all models on the real dataset.

Include:
- Model comparison table.
- Actual MAE, RMSE, MAPE and sMAPE.
- Training and inference time.
- Weather ablation results.
- Residual and error analysis.

Do not insert estimated or fabricated results.

## 9. Dashboard
The planned Streamlit dashboard includes:
- Overview
- Day-Ahead Forecast
- Model Comparison
- Forecast Evaluation
- Weather Features

Screenshots will be added after successful execution using real outputs.

## 10. Limitations
- Regional/national grid data rather than state-specific data.
- Limited historical data for deep learning.
- Weather forecast availability and perfect-forecast assumptions.
- Data quality and missing-value handling.
- Results depend on the selected chronological test period.

## 11. Conclusion
The project establishes a framework for comparing multiple deep learning forecasting architectures against seasonal-naive baselines.

The final conclusions will be added after real-data training, evaluation and dashboard validation.

## 12. Future Scope
- Integrate additional regional datasets.
- Improve weather forecast integration.
- Explore hyperparameter optimization.
- Improve forecasting robustness.
- Extend dashboard capabilities.
