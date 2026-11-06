"""Inference without retraining (FR-12): turn feature rows into full forecasts using the registered models.

Everything loaded here was written by this pipeline into models/. An uploaded CSV is parsed as data and never unpickled.
"""

import time
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.clean import clean_tables, final_checks, load_dataset
from src.dl_model import SequenceData, load_model, predict
from src.features import FEATURE_GROUPS, _peak_flag, build_features, feature_columns, load_features
from src.ingest import load_all, traffic_keys
from src.ml_models import predict_scores
from src.model_card import read_registry
from src.uncertainty import apply_conformal, band_probabilities, conformal_groups, predictive_entropy, segment_groups
from src.utils import ROOT, get_logger, load_config, project_path

logger = get_logger("serve")
CLASS_COLUMNS = ["p_free_flow", "p_moderate", "p_heavy", "p_severe"]


@lru_cache(maxsize=64)
def _load_joblib(path: str):
    return joblib.load(ROOT / path)


@lru_cache(maxsize=16)
def _load_deep(path: str, n_segments: int) -> tuple:
    return load_model(path, n_segments)


@lru_cache(maxsize=32)
def _load_table(path: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / path)


def residual_path(horizon: int, config: dict):
    """Where the production volume model's validation errors are saved for one horizon."""
    directory = project_path("models", config) / "production"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"volume_residuals_h{horizon}.parquet"


def forecast_volume(frame: pd.DataFrame, horizon: int, registry: dict) -> tuple:
    """Volume forecast from the production model, as (values, model path, conformal table path)."""
    if registry["production"]["volume"] == "deep":
        entry = registry["deep"]["horizons"][str(horizon)]
        model, scaler, spec = _load_deep(entry["path"], 25)
        data = SequenceData(frame, scaler, spec["length"])
        forecast = np.full(len(frame), np.nan)
        positions = np.flatnonzero(data.history >= data.length)
        if len(positions):
            forecast[positions] = predict(model, data, positions)[0]
        return np.clip(forecast, 0, None), entry["path"], entry["conformal"]
    entry = registry["targets"]["volume"]["horizons"][str(horizon)]
    model = _load_joblib(f"{entry['path']}/model.joblib")
    return np.clip(predict_scores(model, frame, feature_columns(frame), "regression"), 0, None), entry["path"], entry["conformal"]


def forecast_congestion(frame: pd.DataFrame, volume: np.ndarray, horizon: int, registry: dict, config: dict) -> tuple:
    """Class and class probabilities. Bands read the volume forecast; probabilities come from its validation errors."""
    entry = registry["production"]["congestion"]
    if entry["kind"] == "band":
        residuals = pd.read_parquet(residual_path(horizon, config))
        peak = _peak_flag(frame["target_ts"], config).to_numpy()
        probabilities = np.zeros((len(frame), 4))
        for flag in (0, 1):
            rows = peak == flag
            if rows.any():
                probabilities[rows] = band_probabilities(volume[rows], frame["road_capacity"].to_numpy()[rows], residuals.loc[residuals["peak"] == flag, "residual"].to_numpy(), entry["cuts"])
        ratio = volume / (frame["road_capacity"].to_numpy() / 2)
        return np.digitize(ratio, entry["cuts"]), probabilities, f"band:{entry['cuts']}"
    path = registry["targets"]["congestion"]["horizons"][str(horizon)]["path"]
    model = _load_joblib(f"{path}/model.joblib")
    columns = feature_columns(frame)
    return model.predict(frame[columns]).astype(int), model.predict_proba(frame[columns]), path


def forecast_rows(frame: pd.DataFrame, horizon: int, registry: dict, config: dict | None = None) -> pd.DataFrame:
    """Forecast every row of a feature table: all four PRD outputs with confidence, version and input hash."""
    cfg = config or load_config()
    columns = feature_columns(frame)
    static = pd.read_parquet(project_path("processed", cfg) / "segment_static.parquet")
    cells = conformal_groups(frame, segment_groups(static), cfg)
    levels = cfg["uncertainty"]["levels"]
    out = frame[["road_id", "ts", "target_ts", "split", "target_valid"]].rename(columns={"ts": "issue_ts"}).reset_index(drop=True)
    out["horizon"] = horizon
    out["input_hash"] = pd.util.hash_pandas_object(frame[columns], index=False).map("{:016x}".format).to_numpy()

    def bands(name: str, values: np.ndarray, conformal: str) -> None:
        out[f"{name}_pred"] = values
        table = _load_table(conformal)
        for level in levels:
            lower, upper = apply_conformal(values, cells, table, level)
            out[f"{name}_lo{int(level * 100)}"], out[f"{name}_hi{int(level * 100)}"] = lower, upper

    volume, volume_path, volume_conformal = forecast_volume(frame, horizon, registry)
    bands("volume", volume, volume_conformal)
    out["volume_model"] = volume_path

    speed_entry = registry["targets"]["speed"]["horizons"][str(horizon)]
    speed = np.clip(predict_scores(_load_joblib(f"{speed_entry['path']}/model.joblib"), frame, columns, "regression"), 1.0, None)
    bands("speed", speed, speed_entry["conformal"])
    out["speed_model"] = speed_entry["path"]

    travel_entry = registry["targets"]["travel_time"]["horizons"][str(horizon)]
    if travel_entry["family"] == "derived_from_speed":
        travel = frame["seg_length_km"].to_numpy() * 60 / speed
    else:
        travel = np.clip(predict_scores(_load_joblib(f"{travel_entry['path']}/model.joblib"), frame, columns, "regression"), 0.1, None)
    bands("travel_time", travel, travel_entry["conformal"])
    out["travel_time_model"] = travel_entry["path"]

    congestion, probabilities, congestion_version = forecast_congestion(frame, volume, horizon, registry, cfg)
    out["congestion_pred"] = congestion
    out[CLASS_COLUMNS] = probabilities
    out["congestion_confidence"] = probabilities.max(axis=1)
    out["congestion_entropy"] = predictive_entropy(probabilities)
    out["congestion_model"] = congestion_version

    accident_entry = registry["targets"]["accident"]["horizons"][str(horizon)]
    scores = predict_scores(_load_joblib(f"{accident_entry['path']}/model.joblib"), frame, columns, "accident")
    out["accident_prob"] = _load_joblib(accident_entry["calibrator"]).predict(scores)
    out["accident_score"] = scores
    out["accident_flag"] = scores >= accident_entry["threshold"]
    out["accident_model"] = accident_entry["path"]

    for name, column in [("actual_volume", "y_volume"), ("actual_speed", "y_speed"), ("actual_travel_time", "y_travel_time"), ("actual_congestion", "y_congestion"), ("actual_accident", "y_accident")]:
        out[name] = frame[column].to_numpy() if column in frame else np.nan
    return out


@lru_cache(maxsize=4)
def _training_medians(horizon: int) -> pd.Series:
    """Typical value of each feature on training rows, the reference point for explanations."""
    frame = load_features(horizon)
    return frame.loc[frame["split"] == "train", feature_columns(frame)].median()


def explain_volume(frame: pd.DataFrame, horizon: int, registry: dict) -> pd.DataFrame | None:
    """How each kind of information moved one classical volume forecast.

    Each group of features is swapped for its typical training value in turn, and the change in the forecast is that
    group's contribution. Groups are used because the individual features are heavily collinear.
    """
    if frame.empty:
        return None
    model = _load_joblib(f"{registry['targets']['volume']['horizons'][str(horizon)]['path']}/model.joblib")
    columns = feature_columns(frame)
    row = frame[columns].iloc[[0]]
    medians = _training_medians(horizon)
    forecast = float(predict_scores(model, row, columns, "regression")[0])
    rows = []
    for group, names in FEATURE_GROUPS.items():
        present = [c for c in names if c in columns]
        replaced = row.copy()
        replaced[present] = medians[present].to_numpy()
        rows.append({"group": group, "contribution": forecast - float(predict_scores(model, replaced, columns, "regression")[0])})
    return pd.DataFrame(rows)


def uploaded_path(config: dict | None = None) -> Path:
    """Where the latest uploaded forecasts are kept, separately from the evaluated forecasts."""
    return project_path("processed", config) / "uploaded_forecasts.parquet"


def overlay_uploaded(stored: pd.DataFrame, uploaded: pd.DataFrame) -> pd.DataFrame:
    """Combine forecasts, with an uploaded row replacing a stored one for the same road, moment and look-ahead."""
    keys = ["road_id", "issue_ts", "horizon"]
    index = pd.MultiIndex.from_frame(uploaded[keys])
    kept = stored[~pd.MultiIndex.from_frame(stored[keys]).isin(index)]
    combined = pd.concat([kept.assign(source="stored"), uploaded.assign(source="uploaded")], ignore_index=True)
    return combined.sort_values(["horizon", "road_id", "issue_ts"]).reset_index(drop=True)


def save_uploaded(forecasts: pd.DataFrame, config: dict | None = None) -> None:
    """Keep an upload's forecasts so that the other pages can show them. They never change the evaluated forecasts."""
    forecasts.to_parquet(uploaded_path(config), index=False)


def clear_uploaded(config: dict | None = None) -> None:
    """Go back to showing the evaluated forecasts only."""
    uploaded_path(config).unlink(missing_ok=True)


class UploadError(ValueError):
    """A problem with an uploaded file, with a message written for the person who uploaded it."""


def run_upload(path, config: dict | None = None) -> dict:
    """Validate, clean and forecast an uploaded sensor log with the saved models, using stored rows as history."""
    cfg = config or load_config()
    started = time.time()
    processed = project_path("processed", cfg)
    try:
        tables = load_all(traffic_path=path, config=cfg)
    except ValueError as error:
        raise UploadError(str(error)) from error

    static = pd.read_parquet(processed / "segment_static.parquet")
    traffic = tables["traffic"]
    unknown = ~traffic["road_id"].isin(static["road_id"])
    tables["ledger"].add("ingest", "unknown_segment", "traffic", traffic_keys(traffic[unknown]), "row", traffic.loc[unknown, "road_id"])
    traffic = traffic[~unknown].reset_index(drop=True)
    if traffic.empty:
        raise UploadError("No usable rows: every row was rejected. Check the quarantine rules in the data dictionary.")
    tables["traffic"] = traffic

    first, last = traffic["ts"].min().strftime("%Y-%m-%d"), traffic["ts"].max().strftime("%Y-%m-%d")
    if first < cfg["grid"]["start"] or last > cfg["grid"]["end"]:
        raise UploadError(f"The file covers {first} to {last}, but weather and calendar data only exist for {cfg['grid']['start']} to {cfg['grid']['end']}.")

    roads = sorted(traffic["road_id"].unique())
    stats = {"static": static[static["road_id"].isin(roads)].reset_index(drop=True), "medians": pd.read_parquet(processed / "window_medians.parquet")}
    result = clean_tables(tables, cfg, stats=stats, grid=(first, last))
    cleaned = result["dataset"]
    final_checks(cleaned, None, cfg)

    # Stored rows before the upload supply the lags, the slot profile and the sequence history.
    stored = load_dataset(cfg)
    history = stored[stored["road_id"].isin(roads) & (stored["ts"] < cleaned["ts"].min())]
    combined = pd.concat([history, cleaned[history.columns]], ignore_index=True)
    if combined.groupby("road_id")["ts"].min().max() > pd.Timestamp(cfg["grid"]["start"], tz=cfg["timezone"]) + pd.Timedelta(days=cfg["splits"]["warmup_days"]):
        raise UploadError("A segment has too little history before this file to forecast it.")

    registry = read_registry(cfg)
    context_start = cleaned["ts"].min() - pd.Timedelta(days=1)
    parts = []
    for horizon in cfg["horizons"]:
        features = build_features(combined, horizon, cfg)
        forecast = forecast_rows(features[features["ts"] >= context_start].reset_index(drop=True), horizon, registry, cfg)
        parts.append(forecast[forecast["issue_ts"] >= cleaned["ts"].min()])
    if not parts or all(p.empty for p in parts):
        raise UploadError("The file is too short or starts too early to forecast: the first week of data is used as history.")
    predictions = pd.concat(parts, ignore_index=True)
    predictions["input_file"] = Path(path).name
    logger.info("Upload forecast: %d rows in %.1fs", len(predictions), time.time() - started)
    # The stored weather table is reprocessed too, but only what happened to this file is relevant to the person uploading it.
    ledger = result["ledger"].to_frame()
    return {"predictions": predictions, "quarantine": ledger[ledger["table"] == "traffic"].reset_index(drop=True), "rows_received": tables["raw_rows"]["traffic"], "seconds": time.time() - started}
