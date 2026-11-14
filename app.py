"""Electricity Load and Solar Generation Forecasting Dashboard. Owner: M8.
Shows ONLY real saved outputs (no dummy data). Run:  streamlit run app.py"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config
from src.dataset import load_splits
from src.forecasting import forecast_day

st.set_page_config(page_title="Electricity Forecasting", layout="wide")
st.title("Electricity Load and Solar Generation Forecasting")

DATA = config.PROC_DIR / "dataset.csv"
MET = config.METRICS_DIR / "metrics_summary.csv"
if not DATA.exists():
    st.error("data/processed/dataset.csv not found. Add the dataset and run:\n\n"
             "`python scripts/prepare_dataset.py`")
    st.stop()


@st.cache_data
def load_data():
    df = pd.read_csv(DATA, index_col=0, parse_dates=True)
    return df, load_splits(config.PROC_DIR / "splits.json")


@st.cache_data
def load_metrics():
    return pd.read_csv(MET) if MET.exists() else None


@st.cache_data
def load_preds():
    out = {}
    for f in sorted(config.PRED_DIR.glob("*.csv")):
        out[f.stem] = pd.read_csv(f, parse_dates=["origin", "timestamp"])
    return out


df, splits = load_data()
metrics, preds = load_metrics(), load_preds()
available = [p.stem.rsplit("_", 1) for p in config.MODELS_DIR.glob("*_*.pt")]
nn_choices = sorted({f"{m}|{v}" for m, v in available})
choices = nn_choices + [f"{b}|baseline" for b in config.BASELINES]

page = st.sidebar.radio("Section", ["Overview", "Day-Ahead Forecast", "Model Comparison",
                                    "Forecast Evaluation", "Weather Features"])
sel = st.sidebar.selectbox("Forecasting model", choices) if choices else None
test_days = [pd.Timestamp(d) for d in splits["test"]]


def line(fig, x, y, name, **kw):
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name, **kw))


if page == "Overview":
    c = st.columns(4)
    c[0].metric("Coverage", f"{df.index.min().date()} to {df.index.max().date()}")
    c[1].metric("Latest timestamp", str(df.index.max()))
    last = df.dropna(subset=config.TARGETS).iloc[-1]
    c[2].metric("Latest demand", f"{last['demand_mw']:.0f}")
    c[3].metric("Latest solar", f"{last['solar_mw']:.0f}")
    st.write(f"**Forecasting horizon:** next {config.HORIZON} hours (day-ahead, hourly). "
             f"**Selected model:** {sel or 'none trained yet'}.")
    st.write(f"Split (chronological, by day): train {len(splits['train'])}, "
             f"validation {len(splits['val'])}, test {len(splits['test'])} days.")
    rep = config.PROC_DIR / "data_report.json"
    if rep.exists():
        with st.expander("Data quality report"):
            st.json(json.loads(rep.read_text()))
    st.caption("Units are those of the source dataset. Data covers a regional/national grid, "
               "not a single state. Weather for the target day is treated as a known forecast.")

elif page == "Day-Ahead Forecast":
    if not sel:
        st.warning("No models found. Run `python scripts/train_models.py`.")
        st.stop()
    name, variant = sel.split("|")
    day = st.selectbox("Target day (test period)", test_days[::-1], index=0, format_func=lambda d: str(d.date()))
    try:
        fc = forecast_day(name, variant, df, day)
    except Exception as e:
        st.error(str(e)); st.stop()
    hist = df.loc[day - pd.Timedelta(days=7): day + pd.Timedelta(hours=23)]
    for tg, label in zip(config.TARGETS, ["Electricity demand", "Solar generation"]):
        fig = go.Figure()
        line(fig, hist.index, hist[tg], "actual")
        line(fig, fc.index, fc[tg], f"forecast ({name})", line=dict(dash="dash", width=3))
        fig.add_vrect(x0=day, x1=day + pd.Timedelta(hours=23), fillcolor="lightgrey", opacity=.25, line_width=0)
        fig.update_layout(title=f"{label}: previous 7 days + forecast for {day.date()}", height=350,
                          yaxis_title=tg)
        st.plotly_chart(fig)
    st.dataframe(fc.round(1))

elif page == "Model Comparison":
    if metrics is None:
        st.warning("Run `python scripts/evaluate_models.py` first."); st.stop()
    st.subheader("Test-set metrics (rolling day-ahead)")
    meta = config.METRICS_DIR / "eval_meta.json"
    if meta.exists():
        m = json.loads(meta.read_text())
        st.caption(f"{m['test_days']} test days ({m['first']} to {m['last']}). Solar MAPE/sMAPE rule: {m['solar_mape_rule']}.")
    for tg in config.TARGETS:
        st.markdown(f"**{tg}**")
        st.dataframe(metrics[metrics["target"] == tg].drop(columns="target").round(3), hide_index=True)
    metric = st.selectbox("Chart metric", ["MAE", "RMSE", "MAPE", "sMAPE"])
    cols = st.columns(2)
    for col, tg in zip(cols, config.TARGETS):
        s = metrics[(metrics["target"] == tg)]
        s = s.assign(label=s["model"] + " (" + s["variant"] + ")")
        fig = go.Figure(go.Bar(x=s["label"], y=s[metric]))
        fig.update_layout(title=f"{metric} - {tg}", height=350)
        col.plotly_chart(fig)
    st.subheader("Actual vs predicted")
    day = st.selectbox("Day", test_days[::-1], format_func=lambda d: str(d.date()), key="cmp_day")
    for tg in config.TARGETS:
        fig = go.Figure(); first = True
        for key, p in preds.items():
            g = p[(p["target"] == tg) & (p["origin"] == day - pd.Timedelta(hours=1))]
            if g.empty: continue
            if first:
                line(fig, g["timestamp"], g["y_true"], "actual", line=dict(color="black", width=3)); first = False
            line(fig, g["timestamp"], g["y_pred"], key)
        fig.update_layout(title=tg, height=350); st.plotly_chart(fig)

elif page == "Forecast Evaluation":
    if not preds:
        st.warning("Run `python scripts/evaluate_models.py` first."); st.stop()
    key = st.selectbox("Model", list(preds))
    p = preds[key]
    for tg in config.TARGETS:
        g = p[p["target"] == tg].assign(err=lambda d: d["y_pred"] - d["y_true"])
        st.subheader(tg)
        c = st.columns(3)
        daily = g.groupby(g["origin"].dt.date + pd.Timedelta(days=1))["err"].apply(lambda e: e.abs().mean())
        f1 = go.Figure(go.Scatter(x=daily.index, y=daily.values, mode="lines")); f1.update_layout(title="Daily MAE over test period", height=300)
        c[0].plotly_chart(f1)
        f2 = go.Figure(go.Histogram(x=g["err"], nbinsx=60)); f2.update_layout(title="Residuals (pred - actual)", height=300)
        c[1].plotly_chart(f2)
        hr = g.groupby(g["timestamp"].dt.hour)["err"].apply(lambda e: e.abs().mean())
        f3 = go.Figure(go.Bar(x=hr.index, y=hr.values)); f3.update_layout(title="MAE by hour of day", height=300)
        c[2].plotly_chart(f3)
    abl = config.PLOTS_DIR / "ablation_weather.png"
    if abl.exists():
        st.subheader("Ablation: with vs without weather features"); st.image(str(abl))

else:  # Weather Features
    wx = [c for c in config.WEATHER_VARS if c in df.columns]
    if not wx:
        st.warning("No weather columns in the dataset."); st.stop()
    var = st.selectbox("Weather variable", wx)
    s = df[[var] + config.TARGETS].dropna().resample("D").mean()
    fig = go.Figure(); line(fig, s.index, s[var], var); fig.update_layout(title=f"{var} (daily mean)", height=300)
    st.plotly_chart(fig)
    c = st.columns(2)
    for col, tg in zip(c, config.TARGETS):
        f = go.Figure(go.Scatter(x=s[var], y=s[tg], mode="markers", marker=dict(size=4, opacity=.5)))
        f.update_layout(title=f"{tg} vs {var} (r = {s[var].corr(s[tg]):.2f})", height=350)
        col.plotly_chart(f)
