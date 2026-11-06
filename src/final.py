"""Final evaluation. All model choices are fixed from validation first, and only then is the test window scored.

This is the only module that scores models on test rows. It writes the scoreboards with test columns, the
classical-vs-deep benchmark, predictions.parquet, importance tables, error analysis and success criteria.
Every joblib file loaded here was written by this pipeline into models/.
"""

import ast
import json
import time

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve

from src.clean import load_dataset
from src.dl_model import SequenceData, load_model, predict
from src.evaluate import (
    accident_metrics,
    accident_threshold,
    congestion_metrics,
    coverage,
    day_bootstrap,
    error_breakdown,
    regression_metrics,
    split_rows,
)
from src.features import FEATURE_GROUPS, _peak_flag, feature_columns, load_features
from src.ml_models import TASKS, choose_winner, predict_scores
from src.model_card import add_test_metrics, read_registry, write_registry
from src.serve import forecast_rows, forecast_volume, residual_path
from src.utils import ROOT, get_logger, load_config, project_path

logger = get_logger("final")

MEANINGFUL_GAIN_PCT = 2.0

def deep_positions(frame: pd.DataFrame, path: str, split: str, include_estimated: bool = False) -> tuple:
    """Load a saved network, returning it with its windows and the positions of a split's rows."""
    model, scaler, spec = load_model(path, 25)
    data = SequenceData(frame, scaler, spec["length"])
    if include_estimated:
        positions = np.flatnonzero((data.frame["split"] == split).to_numpy() & (data.history >= data.length))
    else:
        positions = data.positions(split)
    return model, data, positions


def validation_forecast(frame: pd.DataFrame, horizon: int, registry: dict) -> tuple:
    """Production volume forecast for measured validation rows.

    The forecast runs on the whole table and is filtered afterwards, because a sequence model needs unbroken history
    and removing rows would break its windows.
    """
    forecast, _, _ = forecast_volume(frame, horizon, registry)
    rows = (frame["split"] == "val") & frame["target_valid"]
    values = forecast[rows.to_numpy()]
    if np.isnan(values).any():
        raise ValueError(f"{int(np.isnan(values).sum())} validation rows have no volume forecast at h={horizon}")
    return frame[rows].reset_index(drop=True), values


def choose_production(registry: dict, scoreboard: pd.DataFrame, config: dict) -> dict:
    """Decide on validation scores alone which models serve the dashboard, then save their residuals."""
    volume_rows = scoreboard[(scoreboard["target"] == "volume") & (scoreboard["horizon"] == 1)]
    classical_rmse = float(volume_rows["val_rmse"].min())
    deep_rmse = registry["deep"]["horizons"]["1"]["val_rmse"]
    registry["production"] = {"volume": "deep" if deep_rmse < classical_rmse else "classical", "volume_val_rmse": {"classical": classical_rmse, "deep": deep_rmse}}

    # With the production volume decided, compare congestion candidates on validation.
    val, volume = validation_forecast(load_features(1, config), 1, registry)
    y = val["y_congestion"].to_numpy()
    half_capacity = val["road_capacity"].to_numpy() / 2
    published = config["congestion"]["bands"]
    cuts = ast_cuts(scoreboard)
    candidates = [{"model": "regress_then_band", "kind": "band", "cuts": published, "val_macro_f1": congestion_metrics(y, np.digitize(volume / half_capacity, published))["macro_f1"]}]
    candidates.append({"model": "regress_then_band_tuned", "kind": "band", "cuts": cuts, "val_macro_f1": congestion_metrics(y, np.digitize(volume / half_capacity, cuts))["macro_f1"]})
    direct = scoreboard[(scoreboard["target"] == "congestion") & (scoreboard["horizon"] == 1) & ~scoreboard["model"].str.startswith("regress")]
    best_direct = direct.loc[direct["val_macro_f1"].idxmax()]
    candidates.append({"model": best_direct["model"], "kind": "direct", "val_macro_f1": float(best_direct["val_macro_f1"])})
    best = max(candidates, key=lambda c: c["val_macro_f1"])
    if best["kind"] == "direct" and registry["targets"]["congestion"]["model"] != best["model"]:
        logger.warning("Best direct classifier has no per-horizon models; using the band rule instead")
        best = max((c for c in candidates if c["kind"] == "band"), key=lambda c: c["val_macro_f1"])
    registry["production"]["congestion"] = best
    registry["production"]["congestion_candidates"] = candidates
    logger.info("Production: volume=%s, congestion=%s (val macro-F1 %.3f)", registry["production"]["volume"], best["model"], best["val_macro_f1"])

    for horizon in config["horizons"]:
        val_h, forecast = validation_forecast(load_features(horizon, config), horizon, registry)
        pd.DataFrame({"residual": val_h["y_volume"].to_numpy() - forecast, "peak": _peak_flag(val_h["target_ts"], config).to_numpy()}).to_parquet(residual_path(horizon, config), index=False)
    write_registry(registry, config)
    return registry


def ast_cuts(scoreboard: pd.DataFrame) -> list:
    """Read the tuned V/C cut-offs back from the scoreboard row where the classical stage recorded them."""
    row = scoreboard[scoreboard["model"] == "regress_then_band_tuned"].iloc[0]
    return ast.literal_eval(row["params"])["cuts"]


def score_classical(scoreboard: pd.DataFrame, test: pd.DataFrame, columns: list, registry: dict) -> pd.DataFrame:
    """Add test metrics to every h=1 classical row, and record them on the model cards."""
    rows = []
    for row in scoreboard[scoreboard["horizon"] == 1].to_dict("records"):
        target, family = row["target"], row["model"]
        metrics = {}
        if family == "derived_from_speed":
            speed = predict_scores(joblib.load(ROOT / row["path"] / "model.joblib"), test, columns, "regression")
            metrics = regression_metrics(test["y_travel_time"], test["seg_length_km"].to_numpy() * 60 / np.clip(speed, 1.0, None))
        elif family.startswith("regress_then_band"):
            volume = predict_scores(joblib.load(ROOT / row["path"] / "model.joblib"), test, columns, "regression")
            cuts = ast.literal_eval(row["params"])["cuts"]
            metrics = congestion_metrics(test["y_congestion"], np.digitize(volume / (test["road_capacity"].to_numpy() / 2), cuts))
        else:
            task = TASKS[target]
            model = joblib.load(ROOT / row["path"] / "model.joblib")
            scores = predict_scores(model, test, columns, task["kind"])
            if task["kind"] == "regression":
                metrics = regression_metrics(test[task["column"]], scores)
            elif task["kind"] == "congestion":
                metrics = congestion_metrics(test[task["column"]], scores)
            else:
                card = json.loads((ROOT / row["path"] / "model_card.json").read_text(encoding="utf-8"))
                metrics = accident_metrics(test[task["column"]], scores, card["threshold"])
            add_test_metrics(ROOT / row["path"], metrics)
        rows.append({**row, **{f"test_{k}": v for k, v in metrics.items()}})
    return pd.DataFrame(rows)


def fill_horizon_cards(scoreboard: pd.DataFrame, registry: dict, config: dict) -> None:
    """Record test metrics on the model cards for the 60, 90 and 120-minute models (PRD 13.4). Nothing is chosen here."""
    for horizon in config["horizons"][1:]:
        frame = load_features(horizon, config)
        columns = feature_columns(frame)
        test = split_rows(frame, "test")
        for row in scoreboard[scoreboard["horizon"] == horizon].to_dict("records"):
            task = TASKS[row["target"]]
            scores = predict_scores(joblib.load(ROOT / row["path"] / "model.joblib"), test, columns, task["kind"])
            if task["kind"] == "regression":
                metrics = regression_metrics(test[task["column"]], scores)
            elif task["kind"] == "congestion":
                metrics = congestion_metrics(test[task["column"]], scores)
            else:
                card = json.loads((ROOT / row["path"] / "model_card.json").read_text(encoding="utf-8"))
                metrics = accident_metrics(test[task["column"]], scores, card["threshold"])
            add_test_metrics(ROOT / row["path"], metrics)
        path = registry["deep"]["horizons"][str(horizon)]["path"]
        model, data, positions = deep_positions(frame, path, "test")
        add_test_metrics(ROOT / path, regression_metrics(data.frame.iloc[positions]["y_volume"], predict(model, data, positions)[0]))


def baseline_rows(split_frames: dict) -> pd.DataFrame:
    """Naive forecasts on validation and test, in scoreboard form."""
    rows = []
    for name, column in {"Persistence (last window)": "vol_t", "Same window yesterday": "vol_day", "Same window last week": "vol_week", "Slot profile": "vol_profile"}.items():
        row = {"target": "volume", "model": name, "horizon": 1, "interpretability": "High: a stated rule, nothing fitted"}
        for split, frame in split_frames.items():
            row.update({f"{split}_{k}": v for k, v in regression_metrics(frame["y_volume"], frame[column]).items()})
        rows.append(row)
    return pd.DataFrame(rows)


def score_deep(frame: pd.DataFrame, registry: dict, config: dict) -> pd.DataFrame:
    """Validation and test metrics for every saved network, with congestion via bands and the multi-task head."""
    rows = []
    bands = config["congestion"]["bands"]
    for name, path in registry["deep"]["models"].items():
        row = {"target": "volume", "model": name, "horizon": 1, "path": path, "interpretability": "Low: hidden state, read through permutation only"}
        for split in ("val", "test"):
            model, data, positions = deep_positions(frame, path, split)
            started = time.time()
            forecast, logits = predict(model, data, positions)
            row["infer_ms_per_1k"] = round((time.time() - started) / len(positions) * 1e6, 2)
            rows_used = data.frame.iloc[positions]
            metrics = regression_metrics(rows_used["y_volume"], forecast)
            metrics["band_macro_f1"] = congestion_metrics(rows_used["y_congestion"], np.digitize(forecast / (rows_used["road_capacity"].to_numpy() / 2), bands))["macro_f1"]
            if logits is not None:
                metrics["head_macro_f1"] = congestion_metrics(rows_used["y_congestion"], logits.argmax(axis=1))["macro_f1"]
            row.update({f"{split}_{k}": v for k, v in metrics.items()})
            row[f"{split}_rows"] = len(positions)
        add_test_metrics(ROOT / path, {k.removeprefix("test_"): v for k, v in row.items() if k.startswith("test_")})
        rows.append(row)
    return pd.DataFrame(rows)


def write_benchmark(classical: pd.DataFrame, deep: pd.DataFrame, baselines: pd.DataFrame, frame: pd.DataFrame, test: pd.DataFrame, columns: list, registry: dict, config: dict) -> dict:
    """The single head-to-head table the PRD asks for, with a plain verdict computed from its numbers."""
    volume = classical[classical["target"] == "volume"]
    selected = volume.loc[volume["val_rmse"].idxmin()]
    best_on_test = volume.loc[volume["test_rmse"].idxmin()]
    deep_core = deep[deep["model"].isin(["lstm", "gru", "bilstm"]) | deep["model"].str.endswith("_multitask")]
    chosen = deep[deep["model"] == registry["deep"]["model"]].iloc[0]
    if (deep["test_rows"] != len(test)).any():
        raise ValueError(f"Head-to-head rows differ: classical {len(test)}, deep {deep['test_rows'].tolist()}")

    # How sure is the gap? Paired on the same rows and resampled by whole day.
    model, data, positions = deep_positions(frame, chosen["path"], "test")
    aligned = data.frame.iloc[positions]
    if not (aligned["road_id"].to_numpy() == test["road_id"].to_numpy()).all() or not (aligned["ts"].to_numpy() == test["ts"].to_numpy()).all():
        raise ValueError("Deep and classical test rows are not aligned")
    deep_forecast = predict(model, data, positions)[0]
    classical_forecast = predict_scores(joblib.load(ROOT / selected["path"] / "model.joblib"), test, columns, "regression")
    y = test["y_volume"].to_numpy(float)
    low, high = day_bootstrap(lambda idx: regression_metrics(y[idx], classical_forecast[idx])["rmse"] - regression_metrics(y[idx], deep_forecast[idx])["rmse"], test["target_ts"].dt.strftime("%Y-%m-%d").to_numpy(), seed=config["seed"])

    lines = []
    for _, r in pd.concat([baselines, volume, deep_core]).iterrows():
        lines.append(f"| {r['model']} | {r['val_rmse']:.2f} | {r['test_rmse']:.2f} | {r['test_mae']:.2f} | {r['test_mape_pct']:.2f}% | {r['test_r2']:.4f} |")
    gap = selected["test_rmse"] - chosen["test_rmse"]
    percent = gap / selected["test_rmse"] * 100
    beats_all = bool(chosen["test_rmse"] < volume["test_rmse"].min())
    measurable = low > 0
    earned = measurable and percent >= MEANINGFUL_GAIN_PCT
    cost = chosen["infer_ms_per_1k"] / max(selected["infer_ms_per_1k"], 1e-9)
    verdict = (
        f"**Verdict: the deep model {'earned' if earned else 'did not clearly earn'} its added complexity.** "
        f"The chosen {chosen['model'].upper()} scored a test RMSE of {chosen['test_rmse']:.2f}. That {'beats' if beats_all else 'does not beat'} every classical model, "
        f"so the PRD's benchmark is {'met' if beats_all else 'not met'}. Against {selected['model']}, the classical model chosen on validation ({selected['test_rmse']:.2f}), the gain is {gap:.2f} vehicles per window, or {percent:.1f}%. "
        f"The 95% interval for that gain, resampling whole test days, runs from {low:.2f} to {high:.2f}, so it is {'measurably above zero' if measurable else 'not distinguishable from zero'}. "
        f"The best classical model on test was {best_on_test['model']} ({best_on_test['test_rmse']:.2f}). "
        f"A gain under {MEANINGFUL_GAIN_PCT:.0f}% is counted here as too small to matter to operators, and the deep model costs {cost:.1f}x as much per prediction, needs its full window of history, and can't be read through a simple feature ranking. "
        f"The LSTM, GRU and BiLSTM are all within {deep_core['test_rmse'].max() - deep_core['test_rmse'].min():.2f} RMSE of each other, and accuracy barely falls from 30 to 120 minutes ahead, which suggests most of the remaining error is noise that no model can predict."
    )
    text = f"""# Classical vs deep benchmark

Generated by `python -m src.pipeline final`. Every row is scored on the same {len(test):,} test rows (9 to 31 May 2025, next-window volume, measured targets only). Models were chosen on validation before this table was computed.

| Model | Validation RMSE | Test RMSE | Test MAE | Test MAPE | Test R² |
|---|---:|---:|---:|---:|---:|
{chr(10).join(lines)}

{verdict}
"""
    (project_path("reports", config) / "benchmark.md").write_text(text, encoding="utf-8", newline="\n")
    return {
        "selected_classical": selected["model"],
        "selected_classical_test_rmse": float(selected["test_rmse"]),
        "best_classical": best_on_test["model"],
        "best_classical_test_rmse": float(best_on_test["test_rmse"]),
        "deep": chosen["model"],
        "deep_test_rmse": float(chosen["test_rmse"]),
        "gain_interval_95": [low, high],
        "deep_beats_classical": beats_all,
        "earned_complexity": bool(earned),
    }


def permutation_groups(model, frame: pd.DataFrame, columns: list, kind: str, y: np.ndarray, seed: int) -> pd.DataFrame:
    """Loss of skill when one group of correlated features is shuffled together. Collinearity makes single features misleading."""
    rng = np.random.default_rng(seed)

    def skill(data):
        scores = predict_scores(model, data, columns, kind)
        if kind == "regression":
            return regression_metrics(y, scores)["rmse"]
        return accident_metrics(y, scores)["roc_auc"]

    baseline = skill(frame)
    rows = []
    for group, names in FEATURE_GROUPS.items():
        shuffled = frame.copy()
        order = rng.permutation(len(frame))
        present = [c for c in names if c in columns]
        shuffled[present] = frame[present].to_numpy()[order]
        change = skill(shuffled) - baseline
        rows.append({"group": group, "skill_change": float(change if kind == "regression" else -change)})
    return pd.DataFrame(rows).sort_values("skill_change", ascending=False, ignore_index=True)


def importance_tables(classical: pd.DataFrame, test: pd.DataFrame, columns: list, registry: dict, config: dict) -> None:
    """Tree-ensemble importance for the dashboard, plus group permutation on the winners."""
    per_feature = []
    for target in ["volume", "congestion", "accident", "travel_time"]:
        for family in ["forest", "xgboost"]:
            row = classical[(classical["target"] == target) & (classical["model"] == family)]
            if row.empty:
                continue
            model = joblib.load(ROOT / row.iloc[0]["path"] / "model.joblib")
            per_feature.append(pd.DataFrame({"target": target, "model": family, "feature": columns, "importance": model.feature_importances_}))
    pd.concat(per_feature).round(5).to_csv(project_path("reports", config) / "feature_importance.csv", index=False, lineterminator="\n")

    groups = []
    for target, kind, column in [("volume", "regression", "y_volume"), ("accident", "accident", "y_accident")]:
        path = registry["targets"][target]["horizons"]["1"]["path"]
        table = permutation_groups(joblib.load(ROOT / path / "model.joblib"), test, columns, kind, test[column].to_numpy(), config["seed"])
        groups.append(table.assign(target=target, metric="RMSE increase" if kind == "regression" else "AUC loss"))
    pd.concat(groups).round(4).to_csv(project_path("reports", config) / "feature_importance_groups.csv", index=False, lineterminator="\n")


def diagnostics(predictions: pd.DataFrame, weather: pd.DataFrame, registry: dict, config: dict) -> dict:
    """Confusion matrices, ROC points, error breakdowns, interval coverage and the PRD 13.3 figures, on test rows."""
    reports, figures = project_path("reports", config), project_path("figures", config)
    test = predictions[(predictions["split"] == "test") & predictions["target_valid"]]
    h1 = test[test["horizon"] == 1].reset_index(drop=True)

    labels = config["congestion"]["labels"]
    matrix = confusion_matrix(h1["actual_congestion"].astype(int), h1["congestion_pred"], labels=range(4))
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(reports / "confusion_congestion.csv", lineterminator="\n")
    fpr, tpr, _ = roc_curve(h1["actual_accident"].astype(int), h1["accident_score"])
    step = max(1, len(fpr) // 500)
    pd.DataFrame({"fpr": fpr[::step], "tpr": tpr[::step]}).round(5).to_csv(reports / "roc_accident.csv", index=False, lineterminator="\n")

    wx = h1[["road_id", "target_ts"]].merge(weather, left_on=["road_id", "target_ts"], right_on=["road_id", "ts"], how="left")["weather_condition"]
    error_breakdown(h1, h1["actual_volume"], h1["volume_pred"], wx).to_csv(reports / "error_breakdown_volume.csv", index=False, lineterminator="\n")

    rows = []
    peak = _peak_flag(test["target_ts"], config).to_numpy()
    for horizon, part in test.groupby("horizon"):
        part_peak = peak[test["horizon"].to_numpy() == horizon]
        volume = regression_metrics(part["actual_volume"], part["volume_pred"])
        speed = regression_metrics(part["actual_speed"], part["speed_pred"])
        travel = regression_metrics(part["actual_travel_time"], part["travel_time_pred"])
        congestion = congestion_metrics(part["actual_congestion"], part["congestion_pred"])
        accident = accident_metrics(part["actual_accident"].astype(int), part["accident_score"].to_numpy())
        row = {"horizon": horizon, "minutes_ahead": horizon * 30, "volume_rmse": volume["rmse"], "volume_mape_pct": volume["mape_pct"], "speed_rmse": speed["rmse"], "speed_mape_pct": speed["mape_pct"], "travel_time_rmse": travel["rmse"], "travel_time_mape_pct": travel["mape_pct"], "congestion_macro_f1": congestion["macro_f1"], "accident_roc_auc": accident["roc_auc"], "accident_pr_auc": accident["pr_auc"]}
        for name in ("volume", "speed", "travel_time"):
            for level in (80, 95):
                row[f"{name}_cov{level}"] = coverage(part[f"actual_{name}"], part[f"{name}_lo{level}"], part[f"{name}_hi{level}"]) * 100
        for level in (80, 95):
            for tag, flag in (("peak", 1), ("offpeak", 0)):
                mask = part_peak == flag
                row[f"volume_cov{level}_{tag}"] = coverage(part["actual_volume"][mask], part[f"volume_lo{level}"][mask], part[f"volume_hi{level}"][mask]) * 100
        rows.append(row)
    by_horizon = pd.DataFrame(rows).round(3)
    by_horizon.to_csv(reports / "metrics_by_horizon.csv", index=False, lineterminator="\n")

    # Residual distribution, predicted against actual, and one week for a representative segment.
    residual = h1["actual_volume"] - h1["volume_pred"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].hexbin(h1["actual_volume"], h1["volume_pred"], gridsize=50, bins="log", cmap="Blues")
    limit = h1["actual_volume"].max()
    axes[0].plot([0, limit], [0, limit], color="#bc4749")
    axes[0].set_xlabel("Actual volume")
    axes[0].set_ylabel("Forecast volume")
    axes[0].set_title("Predicted against actual (test, next window)")
    axes[1].hist(residual, bins=100, color="#2a6f97")
    axes[1].set_xlabel("Actual minus forecast")
    axes[1].set_title(f"Residuals: mean {residual.mean():.1f}, std {residual.std():.1f}")
    axes[2].scatter(h1["volume_pred"], residual, s=1, alpha=0.15, color="#2a6f97")
    axes[2].axhline(0, color="#bc4749")
    axes[2].set_xlabel("Forecast volume")
    axes[2].set_ylabel("Residual")
    axes[2].set_title("Error spread grows with volume")
    fig.tight_layout()
    fig.savefig(figures / "final_volume_residuals.png", dpi=120)
    plt.close(fig)
    return {"by_horizon": by_horizon}


def write_selection(classical: pd.DataFrame, deep: pd.DataFrame, registry: dict, config: dict) -> None:
    """The PRD 11.4 selection write-up, stating the rule and what each winner was chosen over."""
    primary = {"volume": ("rmse", "RMSE"), "travel_time": ("rmse", "RMSE"), "speed": ("rmse", "RMSE"), "congestion": ("macro_f1", "macro-F1"), "accident": ("roc_auc", "ROC-AUC")}
    sections = []
    for target, (metric, label) in primary.items():
        rows = classical[(classical["target"] == target)].copy()
        winner = choose_winner(rows.to_dict("records"), metric)["model"] if len(rows) > 1 else rows.iloc[0]["model"]
        lines = [f"| {'**' + r['model'] + '**' if r['model'] == winner else r['model']} | {_fmt(r.get('cv_mean'))} ± {_fmt(r.get('cv_std'))} | {r[f'val_{metric}']:.4f} | {_fmt(r.get(f'test_{metric}'))} | {r['infer_ms_per_1k']:.2f} | {r['interpretability']} |" for _, r in rows.iterrows()]
        sections.append(f"## {target.replace('_', ' ').title()} ({label})\n\n| Model | CV mean ± std | Validation | Test | ms per 1k rows | Interpretability |\n|---|---:|---:|---:|---:|---|\n" + "\n".join(lines))
    production = registry["production"]
    candidates = "\n".join(f"- `{c['model']}`: validation macro-F1 {c['val_macro_f1']:.4f}" for c in production["congestion_candidates"])
    text = f"""# Model selection

Generated by `python -m src.pipeline final`. Bold marks the model the rule chose.

**The rule (PRD 11.4).** Take the best validation score on the metric that matters for the task. A faster model is preferred if it is statistically tied with the best, meaning both its validation gap and its mean CV gap are within the standard error of the paired per-fold CV differences between the two. Pairing folds removes the spread that comes from which days a fold holds, which affects every model alike. Test scores are shown for the record and played no part in the choice.

**Why each metric.** Volume and travel time use RMSE because large misses at peak are what hurt operations. Congestion uses macro-F1 because Heavy and Severe are rare, and accuracy would reward ignoring them. Accident risk uses ROC-AUC to rank segments for patrols, with precision and recall reported at an operating threshold chosen for recall of at least {config["accident"]["min_recall"]:.0%}.

**What serves the dashboard.** Volume comes from the {production["volume"]} model (validation RMSE: classical {production["volume_val_rmse"]["classical"]:.2f}, deep {production["volume_val_rmse"]["deep"]:.2f}). Congestion comes from `{production["congestion"]["model"]}`, compared using the production volume forecast:

{candidates}

{chr(10).join(sections)}
"""
    (project_path("reports", config) / "model_selection.md").write_text(text, encoding="utf-8", newline="\n")


def _fmt(value) -> str:
    return "-" if value is None or pd.isna(value) else f"{value:.4f}"


def run(config: dict | None = None) -> None:
    """Fix the production choice on validation, then score everything once on test and write all outputs."""
    cfg = config or load_config()
    reports, processed = project_path("reports", cfg), project_path("processed", cfg)
    registry = read_registry(cfg)
    scoreboard = pd.read_csv(reports / "scoreboard_validation.csv")

    registry = choose_production(registry, scoreboard, cfg)  # Validation only; test rows are first used below.

    frame = load_features(1, cfg)
    columns = feature_columns(frame)
    val, test = split_rows(frame, "val"), split_rows(frame, "test")
    classical = score_classical(scoreboard, test, columns, registry)
    deep = score_deep(frame, registry, cfg)
    fill_horizon_cards(scoreboard, registry, cfg)
    baselines = baseline_rows({"val": val, "test": test})
    benchmark = write_benchmark(classical, deep, baselines, frame, test, columns, registry, cfg)

    for target in TASKS:
        part = pd.concat([baselines if target == "volume" else None, classical[classical["target"] == target], deep if target == "volume" else None])
        part.to_csv(reports / f"scoreboard_{target}.csv", index=False, lineterminator="\n")
    write_selection(classical, deep, registry, cfg)

    # Forecasts for every segment, window and horizon in validation and test, including rows with estimated targets.
    parts, timings = [], {}
    for horizon in cfg["horizons"]:
        table = load_features(horizon, cfg)
        started = time.time()
        forecast = forecast_rows(table, horizon, registry, cfg)
        timings[str(horizon)] = round(time.time() - started, 2)
        parts.append(forecast[forecast["split"].isin(["val", "test"])])
    predictions = pd.concat(parts, ignore_index=True)
    if predictions["volume_pred"].isna().any():
        raise ValueError("Some forecast rows have no volume prediction")
    predictions.to_parquet(processed / "predictions.parquet", index=False)

    # The BiLSTM reads its window both ways, so it is kept as a context view for the Historical trends page (PRD 12.1).
    bilstm_path = registry["deep"]["models"]["bilstm"]
    overlays = []
    for split in ("val", "test"):
        model, data, positions = deep_positions(frame, bilstm_path, split, include_estimated=True)
        overlays.append(data.frame.iloc[positions][["road_id", "target_ts"]].assign(bilstm=predict(model, data, positions)[0]))
    pd.concat(overlays, ignore_index=True).to_parquet(processed / "bilstm_overlay.parquet", index=False)

    # Inference cost for one horizon across the whole corridor, on every row of a feature table (NFR).
    logger.info("Batch inference seconds per horizon (%d rows each): %s", len(table), timings)

    dataset = load_dataset(cfg)
    importance_tables(classical, test, columns, registry, cfg)
    outcome = diagnostics(predictions, dataset[["road_id", "ts", "weather_condition"]], registry, cfg)

    h1 = predictions[(predictions["split"] == "test") & predictions["target_valid"] & (predictions["horizon"] == 1)]
    val_h1 = predictions[(predictions["split"] == "val") & predictions["target_valid"] & (predictions["horizon"] == 1)]
    threshold = accident_threshold(val_h1["actual_accident"].astype(int), val_h1["accident_score"].to_numpy(), cfg["accident"]["min_recall"])
    volume = regression_metrics(h1["actual_volume"], h1["volume_pred"])
    congestion = congestion_metrics(h1["actual_congestion"], h1["congestion_pred"])
    accident = accident_metrics(h1["actual_accident"].astype(int), h1["accident_score"].to_numpy(), threshold)
    row = outcome["by_horizon"].iloc[0]
    days = h1["target_ts"].dt.strftime("%Y-%m-%d").to_numpy()
    actual_volume, forecast_volume_h1 = h1["actual_volume"].to_numpy(), h1["volume_pred"].to_numpy()
    actual_class, forecast_class = h1["actual_congestion"].to_numpy(), h1["congestion_pred"].to_numpy()
    actual_accident, accident_scores = h1["actual_accident"].astype(int).to_numpy(), h1["accident_score"].to_numpy()
    intervals = {
        "volume_mape_pct": day_bootstrap(lambda i: regression_metrics(actual_volume[i], forecast_volume_h1[i])["mape_pct"], days, seed=cfg["seed"]),
        "congestion_macro_f1": day_bootstrap(lambda i: congestion_metrics(actual_class[i], forecast_class[i])["macro_f1"], days, seed=cfg["seed"]),
        "accident_roc_auc": day_bootstrap(lambda i: accident_metrics(actual_accident[i], accident_scores[i])["roc_auc"], days, seed=cfg["seed"]),
    }
    metrics = {
        "intervals_95": intervals,
        "volume": volume,
        "congestion": congestion,
        "accident": accident,
        "speed": regression_metrics(h1["actual_speed"], h1["speed_pred"]),
        "travel_time": regression_metrics(h1["actual_travel_time"], h1["travel_time_pred"]),
        "benchmark": benchmark,
        "coverage": {"volume_80": float(row["volume_cov80"]), "volume_95": float(row["volume_cov95"])},
        "inference_seconds_per_horizon": timings,
        "rows_per_horizon": len(table),
        "test_rows": len(h1),
        "criteria": [
            {"criterion": "Volume MAPE <= 12%", "target": 12.0, "value": volume["mape_pct"], "met": bool(volume["mape_pct"] <= 12)},
            {"criterion": "Congestion macro-F1 >= 0.80", "target": 0.80, "value": congestion["macro_f1"], "met": bool(congestion["macro_f1"] >= 0.80)},
            {"criterion": "Accident ROC-AUC >= 0.75", "target": 0.75, "value": accident["roc_auc"], "met": bool(accident["roc_auc"] >= 0.75)},
            {"criterion": "Sequence model beats best classical on test RMSE", "target": benchmark["best_classical_test_rmse"], "value": benchmark["deep_test_rmse"], "met": benchmark["deep_beats_classical"]},
            {"criterion": "80% band covers 78-82% of test values", "target": 80.0, "value": float(row["volume_cov80"]), "met": bool(78 <= row["volume_cov80"] <= 82)},
            {"criterion": "Batch inference for one horizon <= 30 s", "target": 30.0, "value": max(timings.values()), "met": bool(max(timings.values()) <= 30)},
        ],
    }
    path = reports / "final_metrics.json"
    path.write_text(json.dumps(metrics, indent=2, default=float), encoding="utf-8", newline="\n")
    for criterion in metrics["criteria"]:
        logger.info("%s: %.3f (%s)", criterion["criterion"], criterion["value"], "met" if criterion["met"] else "NOT met")
