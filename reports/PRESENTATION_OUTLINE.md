
# Final Project Presentation Outline

## P6 - Electricity Load and Solar Generation Forecasting

### Slide 1: Title
- Project title
- Course: CS474 - Introduction to Deep Learning
- Team members and coordinator

### Slide 2: Problem Statement
- Day-ahead electricity demand forecasting.
- Solar generation forecasting.
- Importance of accurate grid scheduling.

### Slide 3: Project Objectives
- Predict the next 24 hourly values.
- Integrate electricity and weather data.
- Compare multiple neural architectures.

### Slide 4: Dataset
- Grid-India / NERLDC.
- Open-Meteo weather data.
- Dataset characteristics and limitations.

### Slide 5: System Architecture
- Data collection
- Preprocessing
- Feature engineering
- Model training
- Evaluation
- Dashboard

### Slide 6: Methodology
- 168-hour historical lookback.
- 24-hour forecasting horizon.
- Chronological train/validation/test split.
- Rolling-origin backtesting.

### Slide 7: Models
- LSTM
- GRU
- Transformer
- Seasonal-naive baselines

### Slide 8: Evaluation
- MAE
- RMSE
- MAPE
- sMAPE
- Weather ablation

### Slide 9: Results and Comparison
PENDING: Add verified real-data results, charts and comparison tables.

### Slide 10: Dashboard Demonstration
- Overview
- Day-Ahead Forecast
- Model Comparison
- Forecast Evaluation
- Weather Features

### Slide 11: Limitations and Future Scope
- Regional/national data limitation.
- Weather forecast assumptions.
- Data availability.
- Model improvement opportunities.

### Slide 12: Conclusion
- Summary of implementation.
- Verified experimental findings.
- Future improvements.

## Demo Checklist
- [ ] Dataset successfully loaded.
- [ ] Models trained on real data.
- [ ] Evaluation outputs generated.
- [ ] Dashboard successfully launched.
- [ ] Forecast visualizations verified.
- [ ] Team members prepared for viva.

## Viva Preparation Topics
- Why use LSTM, GRU and Transformer?
- What is a 168-hour lookback?
- What is multi-output forecasting?
- Why use chronological splitting?
- What is rolling-origin backtesting?
- What is data leakage?
- Why is solar MAPE treated differently?
