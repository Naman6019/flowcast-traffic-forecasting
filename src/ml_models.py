"""M5: train, tune and compare the classical models on cross-validation and validation only.

The test window is not read here. Held-out scores are computed once, after every choice is made, by src/final.py.
Every joblib file loaded here was written by this pipeline into models/; nothing loads a user-supplied model.
"""

import ast
import random
import time

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text

from src import checkpoint, scratch
from src.evaluate import (
    accident_metrics,
    accident_threshold,
    congestion_metrics,
    day_folds,
    regression_metrics,
    split_rows,
)
from src.features import feature_columns, load_features
from src.model_card import _plain, dataset_digest, model_dir, read_registry, write_card, write_registry
from src.uncertainty import conformal_groups, fit_conformal, segment_groups
from src.utils import get_logger, load_config, project_path

logger = get_logger("ml_models")

BOUNDED = ["occ_t", "wx_visibility", "share_hcv_t", "share_2w_t", "wx_temp_band", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
INTERPRETABILITY = {
    "linear_scratch": "High: one weight per feature, trained with NumPy gradient descent",
    "linear": "High: one weight per feature, though collinearity blurs individual weights",
    "logistic": "High: one weight per feature, though collinearity blurs individual weights",
    "tree": "High: readable if-then rules",
    "forest": "Medium: feature importance only",
    "xgboost": "Medium: feature importance only",
    "svm_rbf": "Low: support vectors, no direct feature reading",
    "svm_linear": "Medium: one weight per feature, no probabilities",
    "regress_then_band": "High: the published V/C rule applied to the volume forecast",
    "regress_then_band_tuned": "Medium: V/C rule with cut-offs tuned on training data",
    "derived_from_speed": "High: segment length x 60 / forecast speed",
}
SPACES = {
    "tree": {"max_depth": [6, 8, 10, 12, 16], "min_samples_leaf": [5, 20, 50, 100, 200]},
    "forest": {"n_estimators": [100], "max_depth": [10, 14, 18], "min_samples_leaf": [2, 5, 10], "max_features": [0.3, 0.5]},
    "xgboost": {
        "n_estimators": [300, 500, 800],
        "learning_rate": [0.02, 0.04, 0.08],
        "max_depth": [4, 6, 8],
        "min_child_weight": [1, 5, 20],
        "subsample": [0.7, 0.9],
        "colsample_bytree": [0.5, 0.7, 0.9],
        "reg_lambda": [1, 5],
    },
    "logistic": {"C": [0.01, 0.1, 1.0]},
    "svm_rbf": {"C": [0.1, 1.0, 10.0]},
    "svm_linear": {"C": [0.01, 0.1, 1.0]},
}
# Accident risk has a weak signal and overfits quickly, so its boosted trees and forests are searched shallow.
ACCIDENT_SPACES = {
    "xgboost": {
        "n_estimators": [100, 200, 400],
        "learning_rate": [0.01, 0.03, 0.05],
        "max_depth": [2, 3, 4],
        "min_child_weight": [20, 50, 100, 200],
        "subsample": [0.7, 0.9],
        "colsample_bytree": [0.3, 0.5, 0.7],
        "reg_lambda": [1, 5, 20],
    },
    "forest": {"n_estimators": [100], "max_depth": [4, 6, 8], "min_samples_leaf": [50, 100, 200], "max_features": [0.3, 0.5]},
    "tree": {"max_depth": [3, 4, 5, 6], "min_samples_leaf": [100, 200, 500]},
}
TASKS = {
    "volume": {"column": "y_volume", "kind": "regression", "primary": "rmse", "families": ["linear_scratch", "linear", "tree", "forest", "xgboost"]},
    "speed": {"column": "y_speed", "kind": "regression", "primary": "rmse", "families": []},
    "travel_time": {"column": "y_travel_time", "kind": "regression", "primary": "rmse", "families": ["linear", "tree", "forest", "xgboost"]},
    "congestion": {"column": "y_congestion", "kind": "congestion", "primary": "macro_f1", "families": ["logistic", "tree", "forest", "xgboost", "svm_rbf", "svm_linear"]},
    "accident": {"column": "y_accident", "kind": "accident", "primary": "roc_auc", "families": ["logistic", "tree", "forest", "xgboost", "svm_rbf", "svm_linear"]},
}


class ScratchLinearRegression(RegressorMixin, BaseEstimator):
    """Linear regression trained with the NumPy gradient descent in src/scratch.py (PRD build-it-once rule)."""

    def __init__(self, epochs: int = 1500):
        self.epochs = epochs

    def fit(self, X, y):
        """Train by batch gradient descent, keeping the weights and the loss at each epoch."""
        self.weights_, self.history_ = scratch.gradient_descent_lr(np.asarray(X, float), np.asarray(y, float), epochs=self.epochs)
        return self

    def predict(self, X):
        """Predict with the learned weights."""
        return scratch.predict_linear(np.asarray(X, float), self.weights_)


def make_preprocessor(columns: list) -> ColumnTransformer:
    """Scale for gradient and margin models: min-max for bounded fields, z-score otherwise, one-hot for the segment."""
    bounded = [c for c in BOUNDED if c in columns]
    numeric = [c for c in columns if c not in bounded and c != "road_code"]
    return ColumnTransformer(
        [
            ("zscore", StandardScaler(), numeric),
            ("minmax", MinMaxScaler(), bounded),
            ("segment", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["road_code"]),
        ]
    )


def make_model(family: str, kind: str, params: dict, columns: list, seed: int, scratch_epochs: int = 1500):
    """Build an unfitted estimator. Linear and margin models get the scaling pipeline, fit on the training rows only."""
    regression = kind == "regression"
    if family == "tree":
        cls = DecisionTreeRegressor if regression else DecisionTreeClassifier
        extra = {} if regression else {"class_weight": "balanced"}
        return cls(random_state=seed, **extra, **params)
    if family == "forest":
        cls = RandomForestRegressor if regression else RandomForestClassifier
        extra = {} if regression else {"class_weight": "balanced_subsample"}
        return cls(random_state=seed, n_jobs=-1, **extra, **params)
    if family == "xgboost":
        common = {"random_state": seed, "n_jobs": -1, "tree_method": "hist", **params}
        return xgb.XGBRegressor(**common) if regression else xgb.XGBClassifier(**common)
    estimators = {
        "linear_scratch": lambda: ScratchLinearRegression(scratch_epochs),
        "linear": lambda: LinearRegression(),
        "logistic": lambda: LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed, **params),
        "svm_rbf": lambda: SVC(kernel="rbf", gamma="scale", class_weight="balanced", random_state=seed, **params),
        "svm_linear": lambda: LinearSVC(class_weight="balanced", random_state=seed, max_iter=5000, **params),
    }
    return Pipeline([("prep", make_preprocessor(columns)), ("model", estimators[family]())])


def sample_weights(y: np.ndarray, kind: str) -> np.ndarray | None:
    """Balanced weights for boosted classifiers, which have no class_weight option."""
    if kind == "regression":
        return None
    counts = np.bincount(y.astype(int))
    return (len(y) / (len(counts) * counts))[y.astype(int)]


def subsample_by_day(frame: pd.DataFrame, rows: int, seed: int) -> pd.DataFrame:
    """Take the same number of rows from every day, so an SVM subsample still spans the whole period."""
    days = frame["ts"].dt.strftime("%Y-%m-%d")
    per_day = max(1, rows // days.nunique())
    return frame.groupby(days, group_keys=False).sample(n=per_day, random_state=seed).sort_values(["road_id", "ts"])


def fit_model(family: str, kind: str, params: dict, frame: pd.DataFrame, columns: list, target: str, config: dict):
    """Fit one model. RBF SVMs train on a by-day subsample because their cost grows with the square of the rows."""
    seed = config["seed"]
    if family == "svm_rbf":
        frame = subsample_by_day(frame, config["tuning"]["svm_subsample_rows"], seed)
    model = make_model(family, kind, params, columns, seed, config["tuning"]["scratch_epochs"])
    y = frame[target].to_numpy()
    if kind != "regression":
        y = y.astype(int)
    if family == "xgboost" and kind != "regression":
        return model.fit(frame[columns], y, sample_weight=sample_weights(y, kind))
    return model.fit(frame[columns], y)


def predict_scores(model, frame: pd.DataFrame, columns: list, kind: str) -> np.ndarray:
    """Regression value, congestion class, or a ranking score for the accident target."""
    X = frame[columns]
    if kind == "accident":
        return model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X)
    predicted = model.predict(X)
    return np.asarray(predicted, float) if kind == "regression" else np.asarray(predicted, int)


def primary_score(kind: str, y, p) -> dict:
    """The metric set for a task type."""
    if kind == "regression":
        return regression_metrics(y, p)
    if kind == "congestion":
        return congestion_metrics(y, p)
    return accident_metrics(y, p)


def sample_params(space: dict, trials: int, seed: int) -> list:
    """Draw distinct random configurations from a space, in a repeatable order."""
    rng = random.Random(seed)
    total = int(np.prod([len(v) for v in space.values()]))
    seen, drawn = set(), []
    while len(drawn) < min(trials, total):
        params = {name: rng.choice(values) for name, values in space.items()}
        key = tuple(sorted(params.items()))
        if key not in seen:
            seen.add(key)
            drawn.append(params)
    return drawn


def _fold_score(family, kind, params, frame, columns, target, config, train_idx, test_idx, primary) -> float:
    model = fit_model(family, kind, params, frame.iloc[train_idx], columns, target, config)
    held_out = frame.iloc[test_idx]
    return primary_score(kind, held_out[target].to_numpy(), predict_scores(model, held_out, columns, kind))[primary]


def cross_validate(family: str, kind: str, params: dict, frame: pd.DataFrame, columns: list, target: str, primary: str, config: dict) -> list:
    """Score one configuration on expanding whole-day folds of the training window."""
    is_svm = family in ("svm_rbf", "svm_linear")
    folds = day_folds(frame["ts"].dt.strftime("%Y-%m-%d"), config["tuning"]["svm_cv_folds"] if is_svm else config["splits"]["cv_folds"])
    if is_svm:
        # SVC is single-threaded, so its folds run side by side. Everything else already uses all cores per fit.
        return Parallel(n_jobs=len(folds))(delayed(_fold_score)(family, kind, params, frame, columns, target, config, a, b, primary) for a, b in folds)
    return [_fold_score(family, kind, params, frame, columns, target, config, a, b, primary) for a, b in folds]


def tune(family: str, task: dict, frame: pd.DataFrame, columns: list, config: dict, scope: str | None = None, checkpoint_stage: str = "classical") -> tuple:
    """Random search over time-series CV. Returns the best parameters and their fold scores."""
    kind, primary = task["kind"], task["primary"]
    higher_is_better = primary != "rmse"
    space = (ACCIDENT_SPACES if kind == "accident" else {}).get(family, SPACES.get(family))
    candidates = sample_params(space, config["tuning"]["trials"][family], config["seed"]) if space else [{}]
    best = None
    digest = scope or dataset_digest(config)
    for params in candidates:
        key = f"trial|{digest}|{task['column']}|{kind}|{family}|{sorted(params.items())}"
        saved = checkpoint.read(checkpoint_stage, key, config)
        if saved is None:
            scores = [float(v) for v in cross_validate(family, kind, params, frame, columns, task["column"], primary, config)]
            checkpoint.write(checkpoint_stage, key, {"scores": scores}, config)
        else:
            scores = saved["scores"]
        mean = float(np.mean(scores))
        if best is None or (mean > best[0] if higher_is_better else mean < best[0]):
            best = (mean, params, scores)
    return best[1], best[2]


def _fold_scores(row: dict) -> np.ndarray | None:
    """Per-fold CV scores of a scoreboard row, which may have been read back from a CSV as text."""
    scores = row.get("cv_scores")
    if isinstance(scores, str):
        scores = ast.literal_eval(scores)
    return np.asarray(scores, float) if isinstance(scores, list) and len(scores) > 1 else None


def choose_winner(rows: list, primary: str) -> dict:
    """Best validation score, except that a faster model statistically tied with the best is preferred (PRD 11.4).

    Two models tie when both their validation gap and their mean CV gap are within the standard error of their paired
    per-fold CV differences. Comparing fold by fold removes the spread that comes from which days a fold happens to
    hold, which affects all models alike. A row with no fold scores can only win outright.
    """
    sign = 1 if primary != "rmse" else -1
    best = max(rows, key=lambda r: sign * r[f"val_{primary}"])
    best_folds = _fold_scores(best)

    def ties_with_best(row: dict) -> bool:
        gap = sign * (best[f"val_{primary}"] - row[f"val_{primary}"])
        folds = _fold_scores(row)
        if row is best or gap <= 0:
            return True
        if folds is None or best_folds is None or len(folds) != len(best_folds):
            return False
        difference = folds - best_folds
        cv_gap = sign * -difference.mean()
        return max(gap, cv_gap) <= difference.std(ddof=1) / np.sqrt(len(folds))

    return min((r for r in rows if ties_with_best(r)), key=lambda r: r["infer_ms_per_1k"])


def train_candidate(family: str, target_name: str, task: dict, train: pd.DataFrame, val: pd.DataFrame, columns: list, version: str, config: dict, params: dict | None = None, horizon: int = 1) -> tuple:
    """Tune, fit, validate and persist one model. Returns its scoreboard row and the fitted model."""
    kind, primary, target = task["kind"], task["primary"], task["column"]
    key = f"candidate|{version}|{target_name}|{family}|h{horizon}|{params}"
    saved = checkpoint.read("classical", key, config)
    if saved is not None:
        logger.info("%s/%s h=%d: resumed from saved progress", target_name, family, horizon)
        return saved, joblib.load(project_path("models", config).parent / saved["path"] / "model.joblib")

    started = time.time()
    if params is None:
        params, cv_scores = tune(family, task, train, columns, config)
    else:
        cv_scores = []
    search_seconds = time.time() - started

    started = time.time()
    model = fit_model(family, kind, params, train, columns, target, config)
    fit_seconds = time.time() - started

    started = time.time()
    val_pred = predict_scores(model, val, columns, kind)
    infer_ms = (time.time() - started) / len(val) * 1e6

    metrics = primary_score(kind, val[target].to_numpy(), val_pred)
    extra = {}
    if kind == "accident":
        threshold = accident_threshold(val[target].to_numpy(), val_pred, config["accident"]["min_recall"])
        metrics = accident_metrics(val[target].to_numpy(), val_pred, threshold)
        extra["threshold"] = threshold

    directory = model_dir(target_name, family, f"h{horizon}-{version}", config)
    joblib.dump(model, directory / "model.joblib", compress=3)
    write_card(
        directory,
        model=family,
        target=target_name,
        horizon=horizon,
        features=columns,
        hyperparameters=params,
        cv_scores=cv_scores,
        validation_metrics=metrics,
        **extra,
    )
    row = {
        "target": target_name,
        "model": family,
        "horizon": horizon,
        "cv_mean": float(np.mean(cv_scores)) if cv_scores else None,
        "cv_std": float(np.std(cv_scores)) if cv_scores else None,
        "cv_scores": [round(float(v), 5) for v in cv_scores],
        **{f"val_{k}": v for k, v in metrics.items()},
        "search_s": round(search_seconds, 1),
        "fit_s": round(fit_seconds, 1),
        "infer_ms_per_1k": round(infer_ms, 2),
        "interpretability": INTERPRETABILITY[family],
        "params": str(params),
        "path": directory.relative_to(project_path("models", config).parent).as_posix(),
    }
    logger.info("%s/%s h=%d: cv %s, val %s=%.4f (search %.0fs)", target_name, family, horizon, f"{row['cv_mean']:.4f}" if cv_scores else "-", primary, metrics[primary], search_seconds)
    checkpoint.write("classical", key, _plain(row), config)
    return row, model


def tune_band_cuts(volume_pred, capacity, y, config: dict) -> list:
    """Coordinate search on the V/C cut-offs for macro-F1. Fit on out-of-fold training predictions, never on validation."""
    ratio = np.asarray(volume_pred, float) / (np.asarray(capacity, float) / 2)
    y = np.asarray(y, int)
    cuts = list(config["congestion"]["bands"])
    grids = [np.arange(c - 0.08, c + 0.0801, 0.01) for c in cuts]
    for _ in range(2):
        for i, grid in enumerate(grids):
            scores = []
            for value in grid:
                trial = cuts.copy()
                trial[i] = float(value)
                scores.append(scratch.macro_f1(y, np.digitize(ratio, trial), 4) if trial == sorted(trial) else -1.0)
            cuts[i] = float(grid[int(np.argmax(scores))])
    return [round(c, 3) for c in cuts]


def out_of_fold_tail(family: str, params: dict, train: pd.DataFrame, columns: list, config: dict) -> tuple:
    """Fit on all but the last CV fold of training days and predict that fold. Returns the frame and predictions."""
    fit_idx, tail_idx = day_folds(train["ts"].dt.strftime("%Y-%m-%d"), config["splits"]["cv_folds"])[-1]
    model = fit_model(family, "regression", params, train.iloc[fit_idx], columns, "y_volume", config)
    tail = train.iloc[tail_idx]
    return tail, predict_scores(model, tail, columns, "regression")


def band_row(name: str, cuts: list, val: pd.DataFrame, val_volume: np.ndarray, cv_scores: list, infer_ms: float, volume_row: dict) -> dict:
    """Scoreboard row for congestion read off the volume forecast through V/C bands."""
    predicted = np.digitize(val_volume / (val["road_capacity"].to_numpy() / 2), cuts)
    metrics = congestion_metrics(val["y_congestion"].to_numpy(), predicted)
    return {
        "target": "congestion",
        "model": name,
        "horizon": 1,
        "cv_mean": float(np.mean(cv_scores)) if cv_scores else None,
        "cv_std": float(np.std(cv_scores)) if cv_scores else None,
        **{f"val_{k}": v for k, v in metrics.items()},
        "search_s": 0.0,
        "fit_s": 0.0,
        "infer_ms_per_1k": infer_ms,
        "interpretability": INTERPRETABILITY[name],
        "params": str({"volume_model": volume_row["model"], "cuts": cuts}),
        "path": volume_row["path"],
    }


def write_tree_rules(train: pd.DataFrame, columns: list, config: dict) -> None:
    """Depth-3 trees whose rules can be read directly by the operations team (PRD 11.2)."""
    lines = []
    for title, kind, target in [("Next-window volume", "regression", "y_volume"), ("Congestion class", "congestion", "y_congestion")]:
        tree = make_model("tree", kind, {"max_depth": 3, "min_samples_leaf": 200}, columns, config["seed"])
        tree.fit(train[columns], train[target].astype(int) if kind != "regression" else train[target])
        lines += [f"## {title}", "```", export_text(tree, feature_names=columns, decimals=2), "```", ""]
    path = project_path("reports", config) / "decision_tree_rules.md"
    path.write_text("# Decision tree rules (depth 3)\n\n" + "\n".join(lines), encoding="utf-8", newline="\n")


def fit_calibrators(registry_entry: dict, val: pd.DataFrame, val_scores: np.ndarray, directory) -> None:
    """Isotonic calibration of accident scores, fit on validation (plan D9)."""
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(val_scores, val["y_accident"].to_numpy())
    joblib.dump(calibrator, directory / "calibrator.joblib")
    registry_entry["calibrator"] = (directory / "calibrator.joblib").relative_to(project_path("models").parent).as_posix()


def save_conformal(y, p, frame: pd.DataFrame, static: pd.DataFrame, directory, config: dict) -> str:
    """Fit conformal half-widths on validation residuals and store them next to the model."""
    table = fit_conformal(y, p, conformal_groups(frame, segment_groups(static), config), config["uncertainty"]["levels"])
    path = directory / "conformal.csv"
    table.to_csv(path, index=False, lineterminator="\n")
    return path.relative_to(project_path("models").parent).as_posix()


def run(config: dict | None = None) -> None:
    """Compare every model at h=1, choose winners, then train the winning family for h=2-4."""
    cfg = config or load_config()
    version = dataset_digest(cfg)[:8]
    static = pd.read_parquet(project_path("processed", cfg) / "segment_static.parquet")
    h1 = load_features(1, cfg)
    columns = feature_columns(h1)
    train, val = split_rows(h1, "train"), split_rows(h1, "val")
    write_tree_rules(train, columns, cfg)

    rows, models = [], {}
    for name in ["volume", "travel_time", "congestion", "accident"]:
        task = TASKS[name]
        for family in task["families"]:
            row, model = train_candidate(family, name, task, train, val, columns, version, cfg)
            rows.append(row)
            models[(name, family)] = model

    volume_rows = [r for r in rows if r["target"] == "volume"]
    volume_winner = choose_winner(volume_rows, "rmse")
    family = volume_winner["model"]
    params = ast.literal_eval(volume_winner["params"])
    logger.info("Volume winner: %s", family)

    # Speed uses the volume winner's family, with its own search (plan D7).
    speed_row, speed_model = train_candidate(family, "speed", TASKS["speed"], train, val, columns, version, cfg)
    rows.append(speed_row)
    models[("speed", family)] = speed_model

    # Travel time: derived from forecast speed, as the challenger to the direct regressors (plan D8).
    val_speed = predict_scores(speed_model, val, columns, "regression")
    derived = val["seg_length_km"].to_numpy() * 60 / np.clip(val_speed, 1.0, None)
    derived_metrics = regression_metrics(val["y_travel_time"].to_numpy(), derived)
    rows.append(
        {
            "target": "travel_time",
            "model": "derived_from_speed",
            "horizon": 1,
            "cv_mean": None,
            "cv_std": None,
            **{f"val_{k}": v for k, v in derived_metrics.items()},
            "search_s": 0.0,
            "fit_s": 0.0,
            "infer_ms_per_1k": speed_row["infer_ms_per_1k"],
            "interpretability": INTERPRETABILITY["derived_from_speed"],
            "params": str({"speed_model": family}),
            "path": speed_row["path"],
        }
    )

    # Congestion: regress-then-band, with the published cuts and with cuts tuned on out-of-fold training predictions.
    val_volume = predict_scores(models[("volume", family)], val, columns, "regression")
    tail, tail_volume = out_of_fold_tail(family, params, train, columns, cfg)
    cuts = tune_band_cuts(tail_volume, tail["road_capacity"], tail["y_congestion"], cfg)
    logger.info("Tuned V/C cut-offs: %s", cuts)
    published = cfg["congestion"]["bands"]
    tail_f1 = scratch.macro_f1(tail["y_congestion"].to_numpy().astype(int), np.digitize(tail_volume / (tail["road_capacity"].to_numpy() / 2), published), 4)
    rows.append(band_row("regress_then_band", published, val, val_volume, [tail_f1], volume_winner["infer_ms_per_1k"], volume_winner))
    rows.append(band_row("regress_then_band_tuned", cuts, val, val_volume, [], volume_winner["infer_ms_per_1k"], volume_winner))

    scoreboard = pd.DataFrame(rows)
    reports = project_path("reports", cfg)
    scoreboard.to_csv(reports / "scoreboard_validation.csv", index=False, lineterminator="\n")

    winners = {}
    for name, primary in [("volume", "rmse"), ("speed", "rmse"), ("travel_time", "rmse"), ("congestion", "macro_f1"), ("accident", "roc_auc")]:
        winners[name] = choose_winner([r for r in rows if r["target"] == name], primary)
        logger.info("Winner for %s: %s (val %s=%.4f)", name, winners[name]["model"], primary, winners[name][f"val_{primary}"])

    registry = {"version": version, "band_cuts": cuts if winners["congestion"]["model"] == "regress_then_band_tuned" else published, "targets": {}}
    for name in ["volume", "speed", "travel_time", "congestion", "accident"]:
        registry["targets"][name] = {"model": winners[name]["model"], "horizons": {}}

    # Train the winners for every horizon, and fit their confidence on validation.
    def register(name, family_name, horizon, model, val_rows, path):
        directory = project_path("models", cfg).parent / path
        entry = {"family": family_name, "path": path}
        kind = TASKS[name]["kind"]
        scores = predict_scores(model, val_rows, columns, kind)
        if kind == "regression":
            entry["conformal"] = save_conformal(val_rows[TASKS[name]["column"]], scores, val_rows, static, directory, cfg)
        if kind == "accident":
            entry["threshold"] = accident_threshold(val_rows["y_accident"].to_numpy(), scores, cfg["accident"]["min_recall"])
            fit_calibrators(entry, val_rows, scores, directory)
        registry["targets"][name]["horizons"][str(horizon)] = entry

    for name in ["volume", "speed", "accident"]:
        winner = winners[name]
        params_used = ast.literal_eval(winner["params"])
        register(name, winner["model"], 1, models[(name, winner["model"])], val, winner["path"])
        for horizon in cfg["horizons"][1:]:
            frame = load_features(horizon, cfg)
            train_h, val_h = split_rows(frame, "train"), split_rows(frame, "val")
            row, model = train_candidate(winner["model"], name, TASKS[name], train_h, val_h, columns, version, cfg, params=params_used, horizon=horizon)
            rows.append(row)
            register(name, winner["model"], horizon, model, val_h, row["path"])

    # Congestion by direct classifier needs its own horizon models; regress-then-band reuses the volume models.
    congestion_winner = winners["congestion"]
    if congestion_winner["model"].startswith("regress_then_band"):
        for horizon in cfg["horizons"]:
            frame = load_features(horizon, cfg)
            val_h = split_rows(frame, "val")
            volume_entry = registry["targets"]["volume"]["horizons"][str(horizon)]
            model = joblib.load(project_path("models", cfg).parent / volume_entry["path"] / "model.joblib")
            residuals = val_h["y_volume"].to_numpy() - predict_scores(model, val_h, columns, "regression")
            np.save(project_path("models", cfg).parent / volume_entry["path"] / "val_residuals.npy", residuals)
            registry["targets"]["congestion"]["horizons"][str(horizon)] = {"family": congestion_winner["model"], "path": volume_entry["path"]}
    else:
        params_used = ast.literal_eval(congestion_winner["params"])
        registry["targets"]["congestion"]["horizons"]["1"] = {"family": congestion_winner["model"], "path": congestion_winner["path"]}
        for horizon in cfg["horizons"][1:]:
            frame = load_features(horizon, cfg)
            row, _ = train_candidate(congestion_winner["model"], "congestion", TASKS["congestion"], split_rows(frame, "train"), split_rows(frame, "val"), columns, version, cfg, params=params_used, horizon=horizon)
            rows.append(row)
            registry["targets"]["congestion"]["horizons"][str(horizon)] = {"family": congestion_winner["model"], "path": row["path"]}

    # Travel time: the derived estimate reuses the speed models; a direct winner gets its own.
    travel_winner = winners["travel_time"]
    for horizon in cfg["horizons"]:
        frame = load_features(horizon, cfg)
        train_h, val_h = split_rows(frame, "train"), split_rows(frame, "val")
        if travel_winner["model"] == "derived_from_speed":
            speed_entry = registry["targets"]["speed"]["horizons"][str(horizon)]
            model = joblib.load(project_path("models", cfg).parent / speed_entry["path"] / "model.joblib")
            forecast = val_h["seg_length_km"].to_numpy() * 60 / np.clip(predict_scores(model, val_h, columns, "regression"), 1.0, None)
            directory = model_dir("travel_time", "derived_from_speed", f"h{horizon}-{version}", cfg)
            conformal = save_conformal(val_h["y_travel_time"], forecast, val_h, static, directory, cfg)
            write_card(directory, model="derived_from_speed", target="travel_time", horizon=horizon, source_model=speed_entry["path"], features=columns, hyperparameters={}, validation_metrics=regression_metrics(val_h["y_travel_time"], forecast))
            registry["targets"]["travel_time"]["horizons"][str(horizon)] = {"family": "derived_from_speed", "path": directory.relative_to(project_path("models", cfg).parent).as_posix(), "speed_path": speed_entry["path"], "conformal": conformal}
        else:
            params_used = ast.literal_eval(travel_winner["params"])
            if horizon == 1:
                row, model = travel_winner, models[("travel_time", travel_winner["model"])]
            else:
                row, model = train_candidate(travel_winner["model"], "travel_time", TASKS["travel_time"], train_h, val_h, columns, version, cfg, params=params_used, horizon=horizon)
                rows.append(row)
            register("travel_time", travel_winner["model"], horizon, model, val_h, row["path"])

    write_registry({**read_registry(cfg), **registry}, cfg)
    pd.DataFrame(rows).to_csv(reports / "scoreboard_validation.csv", index=False, lineterminator="\n")
    logger.info("Classical stage complete: %d models", len(rows))
