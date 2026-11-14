"""Rolling day-ahead backtest on the TEST split + comparison tables, ablation and plots.
Owners: M6 (backtest/metrics), M7 (ablation/plots).
  python scripts/evaluate_models.py"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy 
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.backtesting import backtest_naive, backtest_nn
from src.dataset import build_windows, fit_scaler, load_splits
from src.evaluation import metrics_table
from src.feature_engineering import feature_columns
from src.forecasting import load_checkpoint
from src.training import checkpoint_path


def main():
    df = pd.read_csv(config.PROC_DIR / "dataset.csv", index_col=0, parse_dates=True)
    splits = load_splits(config.PROC_DIR / "splits.json")
    for d in (config.PRED_DIR, config.METRICS_DIR, config.PLOTS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    frames, extra = [], {}
    test_days = [pd.Timestamp(d) for d in splits["test"]]
    for lag, name in ((1, "seasonal_naive_24"), (7, "seasonal_naive_168")):
        f = backtest_naive(name, lag, df, test_days)
        f["variant"] = "baseline"
        frames.append(f)

    for variant in ("weather", "noweather"):
        hist, fut = feature_columns(variant == "weather")
        for name in config.MODEL_NAMES:
            if not checkpoint_path(config.MODELS_DIR, name, variant).exists():
                print(f"skip {name}/{variant}: no trained model")
                continue
            model, scaler, info = load_checkpoint(name, variant)
            f, per_day = backtest_nn(model, name, df, splits["test"], scaler, hist, fut)
            f["variant"] = variant
            frames.append(f)
            extra[(name, variant)] = {"train_time_s": info["train_time_s"], "inference_s_per_day": per_day}

    all_f = pd.concat(frames, ignore_index=True)
    # fair comparison: only origins where EVERY model produced a value
    valid = all_f.dropna(subset=["y_pred", "y_true"]).groupby(["model", "variant"])["origin"].apply(set)
    common = set.intersection(*valid.tolist())
    all_f = all_f[all_f["origin"].isin(common)].copy()
    all_f["key"] = all_f["model"] + "|" + all_f["variant"]
    print(f"Evaluating on {len(common)} common test days ({min(common).date()} to {max(common).date()})")

    for (m, v), g in all_f.groupby(["model", "variant"]):
        g.drop(columns="key").to_csv(config.PRED_DIR / f"{m}_{v}.csv", index=False)

    t = metrics_table(all_f.assign(model=all_f["key"]))
    t[["model", "variant"]] = t["model"].str.split("|", expand=True)
    t["train_time_s"] = [extra.get((m, v), {}).get("train_time_s", np.nan) for m, v in zip(t["model"], t["variant"])]
    t["inference_s_per_day"] = [extra.get((m, v), {}).get("inference_s_per_day", np.nan) for m, v in zip(t["model"], t["variant"])]
    t = t[["model", "variant", "target", "MAE", "RMSE", "MAPE", "sMAPE", "n_points", "train_time_s", "inference_s_per_day"]]
    t.round(4).to_csv(config.METRICS_DIR / "metrics_summary.csv", index=False)
    meta = {"test_days": len(common), "first": str(min(common).date()), "last": str(max(common).date()),
            "solar_mape_rule": f"actual > {config.SOLAR_MAPE_FRAC} * max(actual)"}
    (config.METRICS_DIR / "eval_meta.json").write_text(json.dumps(meta, indent=2))
    (config.METRICS_DIR / "RESULTS.md").write_text(
        "# Test-set results (rolling day-ahead)\n\n" + t.round(3).to_string(index=False) + "\n")
    print(t.round(3).to_string(index=False))
    make_plots(all_f, t)


def make_plots(all_f: pd.DataFrame, t: pd.DataFrame) -> None:
    main_v = all_f[all_f["variant"].isin(["baseline", "weather"])]
    # 1. model comparison (MAE per target)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, tg in zip(axes, config.TARGETS):
        s = t[(t["target"] == tg) & t["variant"].isin(["baseline", "weather"])]
        ax.bar(s["model"], s["MAE"]); ax.set_title(f"MAE - {tg}"); ax.tick_params(axis="x", rotation=30)
    fig.tight_layout(); fig.savefig(config.PLOTS_DIR / "model_comparison_mae.png", dpi=130); plt.close(fig)

    # 2. ablation: weather vs no weather
    s = t[t["variant"].isin(["weather", "noweather"])]
    if s["variant"].nunique() == 2:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, tg in zip(axes, config.TARGETS):
            p = s[s["target"] == tg].pivot(index="model", columns="variant", values="MAE")
            p.plot.bar(ax=ax, title=f"Ablation (MAE) - {tg}"); ax.tick_params(axis="x", rotation=0)
        fig.tight_layout(); fig.savefig(config.PLOTS_DIR / "ablation_weather.png", dpi=130); plt.close(fig)

    # 3. actual vs predicted over the last 7 test days
    last = sorted(main_v["origin"].unique())[-7:]
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for ax, tg in zip(axes, config.TARGETS):
        g = main_v[(main_v["target"] == tg) & main_v["origin"].isin(last)]
        act = g.drop_duplicates("timestamp").sort_values("timestamp")
        ax.plot(act["timestamp"], act["y_true"], "k", label="actual", lw=2)
        for (m, v), gg in g.groupby(["model", "variant"]):
            gg = gg.sort_values("timestamp"); ax.plot(gg["timestamp"], gg["y_pred"], label=m, alpha=.8)
        ax.set_ylabel(tg); ax.legend(ncol=5, fontsize=7)
    fig.tight_layout(); fig.savefig(config.PLOTS_DIR / "actual_vs_predicted_last_week.png", dpi=130); plt.close(fig)

    # 4. residuals + error by hour (weather models only)
    nn = main_v[~main_v["model"].str.startswith("seasonal")]
    if len(nn):
        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        for j, tg in enumerate(config.TARGETS):
            g = nn[nn["target"] == tg].assign(err=lambda d: d["y_pred"] - d["y_true"])
            for m, gg in g.groupby("model"):
                axes[0, j].hist(gg["err"], bins=60, alpha=.5, label=m)
                h = gg.assign(hour=gg["timestamp"].dt.hour).groupby("hour")["err"].apply(lambda e: e.abs().mean())
                axes[1, j].plot(h.index, h.values, label=m)
            axes[0, j].set_title(f"Residuals - {tg}"); axes[1, j].set_title(f"MAE by hour - {tg}")
            axes[0, j].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(config.PLOTS_DIR / "residuals_and_hourly_error.png", dpi=130); plt.close(fig)

    # 5. MAE by Day of Week (M7 Enhancement)
    if len(nn):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for j, tg in enumerate(config.TARGETS):
            g = nn[nn["target"] == tg].assign(err=lambda d: d["y_pred"] - d["y_true"])
            for m, gg in g.groupby("model"):
                dow = gg.assign(dow=gg["timestamp"].dt.dayofweek).groupby("dow")["err"].apply(lambda e: e.abs().mean())
                axes[j].plot(dow.index, dow.values, marker='o', label=m)
            axes[j].set_title(f"MAE by Day of Week - {tg}")
            axes[j].set_xticks(range(7))
            axes[j].set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
            axes[j].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(config.PLOTS_DIR / "mae_by_day_of_week.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
