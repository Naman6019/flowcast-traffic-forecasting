"""Metrics, time-aware cross-validation folds and error breakdowns, shared by the classical and deep engines."""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import TimeSeriesSplit

from src import scratch

CONGESTION_CLASSES = 4


def split_rows(features: pd.DataFrame, split: str) -> pd.DataFrame:
    """Rows of one split whose target was actually measured. Estimated targets never count (plan D2)."""
    return features[(features["split"] == split) & features["target_valid"]].reset_index(drop=True)


def regression_metrics(y, p) -> dict:
    """RMSE, MAE, MAPE and R² for a regression target (PRD 13.2)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {"rmse": scratch.rmse(y, p), "mae": scratch.mae(y, p), "mape_pct": scratch.mape(y, p) * 100, "r2": scratch.r2(y, p)}


def wape(y, p) -> float:
    """Total absolute error as a share of total actual volume, which quiet hours can't inflate the way they do MAPE."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.abs(y - p).sum() / y.sum())


def congestion_metrics(y, p) -> dict:
    """Macro-F1, accuracy, macro precision and recall, and F1 per class for the four congestion levels (PRD 13.2)."""
    y, p = np.asarray(y, int), np.asarray(p, int)
    precision, recall, f1 = scratch.precision_recall_f1(y, p, CONGESTION_CLASSES)
    out = {"macro_f1": float(f1.mean()), "accuracy": float((y == p).mean()), "macro_precision": float(precision.mean()), "macro_recall": float(recall.mean())}
    out.update({f"f1_class{i}": float(v) for i, v in enumerate(f1)})
    return out


def accident_threshold(y, score, min_recall: float) -> float:
    """Highest-precision score threshold whose recall is at least `min_recall` (chosen on validation, plan D9)."""
    precision, recall, thresholds = precision_recall_curve(np.asarray(y, int), score)
    eligible = recall[:-1] >= min_recall
    return float(thresholds[eligible][np.argmax(precision[:-1][eligible])])


def top_share_lift(y, score, share: float = 0.1) -> float:
    """How many times more often accidents occur in the highest-scored share of windows than overall."""
    y = np.asarray(y, int)
    top = np.argsort(-np.asarray(score), kind="stable")[: max(1, int(len(y) * share))]
    return float(y[top].mean() / y.mean())


def accident_metrics(y, score, threshold: float | None = None) -> dict:
    """Ranking metrics for accident risk, plus precision, recall and F1 when a threshold is given (PRD 13.2)."""
    y = np.asarray(y, int)
    out = {"roc_auc": scratch.roc_auc(y, score), "pr_auc": float(average_precision_score(y, score)), "base_rate": float(y.mean()), "top_decile_lift": top_share_lift(y, score)}
    if threshold is not None:
        flagged = (score >= threshold).astype(int)
        precision, recall, f1 = scratch.precision_recall_f1(y, flagged, 2)
        out.update({"precision": float(precision[1]), "recall": float(recall[1]), "f1": float(f1[1]), "flag_rate": float(flagged.mean())})
    return out


def day_folds(dates: pd.Series, n_folds: int) -> list:
    """Expanding-window folds over whole days, as (train_positions, test_positions). Never a random split."""
    dates = pd.Series(np.asarray(dates))
    days = np.sort(dates.unique())
    folds = []
    for train_days, test_days in TimeSeriesSplit(n_splits=n_folds).split(days):
        folds.append((np.flatnonzero(dates.isin(days[train_days])), np.flatnonzero(dates.isin(days[test_days]))))
    return folds


def error_breakdown(frame: pd.DataFrame, y, p, weather: pd.Series | None = None) -> pd.DataFrame:
    """RMSE, MAE, bias and MAPE by hour of day, day of week and (when given) weather (PRD 13.3)."""
    rows = pd.DataFrame({"y": np.asarray(y, float), "p": np.asarray(p, float)}, index=frame.index)
    rows["hour"] = frame["target_ts"].dt.hour.astype("string")
    rows["day_of_week"] = frame["target_ts"].dt.strftime("%a")
    groups = ["hour", "day_of_week"]
    if weather is not None:
        rows["weather"] = weather.astype("string")
        groups.append("weather")
    parts = []
    for group in groups:
        for value, part in rows.groupby(group):
            metrics = regression_metrics(part["y"], part["p"])
            parts.append({"group": group, "value": value, "n": len(part), "rmse": metrics["rmse"], "mae": metrics["mae"], "mape_pct": metrics["mape_pct"], "bias": float((part["p"] - part["y"]).mean())})
    return pd.DataFrame(parts).round(3)


def day_bootstrap(metric, dates, n_resamples: int = 1000, seed: int = 42, level: float = 0.95) -> tuple:
    """Interval for a metric by resampling whole days, because windows within a day are correlated.

    `metric` receives an array of row positions. Returns the lower and upper bounds.
    """
    dates = np.asarray(dates)
    groups = [np.flatnonzero(dates == day) for day in np.unique(dates)]
    rng = np.random.default_rng(seed)
    values = [metric(np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])) for _ in range(n_resamples)]
    tail = (1 - level) / 2 * 100
    return float(np.percentile(values, tail)), float(np.percentile(values, 100 - tail))


def coverage(y, lower, upper) -> float:
    """Share of actual values that fall inside the interval."""
    y = np.asarray(y, float)
    return float(((y >= np.asarray(lower)) & (y <= np.asarray(upper))).mean())
