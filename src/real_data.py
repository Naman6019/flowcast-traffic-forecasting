"""Validate the approach on real traffic: hourly westbound I-94 volume, UCI Metro Interstate Traffic Volume (id 492).

It reuses FlowCast's cleaning helpers, slot profile, model families, tuning, sequence models, conformal bands and
metrics. The features are adapted, because this dataset has no speed, occupancy, capacity or accident columns, so
only volume can be validated. Models are chosen on 2017 and scored on 2018. Run `python -m src.real_data --validation-only`
to stop before the test window, and `python -m src.pipeline real` for everything.
"""

import argparse
import gzip
import hashlib
import json
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import scratch
from src.clean import fill_short_gaps
from src.dl_model import SequenceData, SequenceScaler, input_columns, predict, train_or_resume
from src.evaluate import coverage, day_bootstrap, error_breakdown, regression_metrics, split_rows, wape
from src.features import _peak_flag, feature_columns, slot_profile, split_of
from src.ingest import Ledger
from src.ml_models import choose_winner, fit_model, predict_scores, sample_params, tune
from src.uncertainty import apply_conformal, fit_conformal
from src.utils import ROOT, exit_cleanly, get_logger, load_config, project_path, seed_everything

logger = get_logger("real_data")

ROAD = "I94-W"
FAMILIES = ["linear", "forest", "xgboost"]
BASELINES = {"Persistence (last hour)": "vol_t", "Same hour yesterday": "vol_day", "Same hour last week": "vol_week", "Slot profile": "vol_profile"}


def real_config(config: dict | None = None) -> dict:
    """The main configuration with this dataset's grid, peak hours and splits swapped in, so shared code just works."""
    cfg = config or load_config()
    real = cfg["real_data"]
    return {
        **cfg,
        "grid": {"window_minutes": real["step_minutes"], "windows_per_day": real["windows_per_day"]},
        "peak_windows": real["peak_windows"],
        "splits": real["splits"],
        "horizons": real["horizons"],
        "tuning": {**cfg["tuning"], "deep_trials": real["deep_trials"]},
        "deep": {**cfg["deep"], "sequence_lengths": real["sequence_lengths"]},
    }


def fetch(config: dict | None = None) -> Path | None:
    """Return the verified CSV, downloading it if needed. None if it can't be fetched, so offline runs can skip this stage."""
    real = (config or load_config())["real_data"]
    folder = ROOT / real["folder"]
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / real["csv_name"]
    archive = folder / "i94.zip"
    if not path.exists():
        if not archive.exists():
            try:
                urllib.request.urlretrieve(real["url"], archive)
            except (urllib.error.URLError, OSError) as error:
                logger.warning("Couldn't download the I-94 dataset: %s", error)
                return None
        with zipfile.ZipFile(archive) as bundle:
            member = next(n for n in bundle.namelist() if n.endswith((".csv", ".csv.gz")))
            content = bundle.read(member)
        path.write_bytes(gzip.decompress(content) if member.endswith(".gz") else content)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != real["csv_sha256"]:
        raise ValueError(f"I-94 file checksum mismatch: expected {real['csv_sha256']}, found {digest}")
    return path


def _keys(frame: pd.DataFrame) -> pd.Series:
    return ROAD + "|" + frame["ts"].dt.strftime("%Y-%m-%d %H:%M")


def clean_i94(raw: pd.DataFrame, ledger: Ledger, config: dict) -> pd.DataFrame:
    """Repair the real defects and put the data on an hourly grid. Long gaps stay blank: there's nothing to estimate them from."""
    real = config["real_data"]
    limit = config["cleaning"]["interpolate_limit_windows"]
    df = raw.copy()
    df["ts"] = pd.to_datetime(df["date_time"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    bad = df["ts"].isna()
    ledger.add("ingest", "invalid_timestamp", "i94", df.loc[bad, "date_time"].astype("string"), "row", df.loc[bad, "date_time"])
    df = df[~bad].copy()

    # An hour with several weather conditions has one row per condition. Keep the most severe, since volumes are identical.
    severity = {label: rank for rank, label in enumerate(real["weather_severity"])}
    df["severity"] = df["weather_main"].map(severity).fillna(len(severity))
    df = df.sort_values(["ts", "severity"], kind="stable")
    duplicated = df.duplicated("ts", keep="first")
    ledger.add("dedup", "duplicate_hour", "i94", _keys(df[duplicated]), "row", df.loc[duplicated, "weather_main"])
    df = df[~duplicated].drop(columns="severity").reset_index(drop=True)

    # The holiday label is only set on a day's first hour, so it's applied to the whole date.
    named = df.loc[df["holiday"].notna() & (df["holiday"].astype("string") != "None"), "ts"].dt.strftime("%Y-%m-%d")
    holidays = set(named)

    frozen = df["temp"] <= 0
    ledger.add("outliers", "temperature_zero_kelvin", "i94", _keys(df[frozen]), "temp", df.loc[frozen, "temp"])
    df["temp"] = df["temp"].where(~frozen)
    flood = df["rain_1h"] > real["max_rain_mm"]
    ledger.add("outliers", "impossible_rainfall", "i94", _keys(df[flood]), "rain_1h", df.loc[flood, "rain_1h"])
    df["rain_1h"] = df["rain_1h"].where(~flood)
    silent = df["traffic_volume"] <= 0
    ledger.add("outliers", "zero_volume", "i94", _keys(df[silent]), "traffic_volume", df.loc[silent, "traffic_volume"])
    df["traffic_volume"] = df["traffic_volume"].astype("float64").where(~silent)
    df = df.rename(columns={"temp": "temperature", "rain_1h": "rainfall", "snow_1h": "snowfall", "clouds_all": "clouds"})

    grid = pd.DataFrame({"ts": pd.date_range(df["ts"].min(), df["ts"].max(), freq="h")})
    merged = grid.merge(df, on="ts", how="left", indicator=True)
    absent = merged["_merge"] == "left_only"
    run_id = (absent != absent.shift()).cumsum()
    run_length = absent.groupby(run_id).transform("sum")
    ledger.add("missing", "hour_absent", "i94", _keys(merged[absent]), "row", run_length[absent])
    measured = merged["traffic_volume"].notna()
    merged["traffic_volume"] = fill_short_gaps(merged["traffic_volume"], limit)
    merged["target_imputed"] = ~measured
    for col in ["temperature", "rainfall", "snowfall", "clouds"]:
        merged[col] = fill_short_gaps(merged[col], limit)
    merged["weather_main"] = merged["weather_main"].astype("object").ffill(limit=limit)

    merged["road_id"] = ROAD
    merged["public_holiday"] = merged["ts"].dt.strftime("%Y-%m-%d").isin(holidays).astype("int64")
    merged["event_flag"] = 0
    merged["roadwork_flag"] = 0
    return merged[["road_id", "ts", "traffic_volume", "target_imputed", "temperature", "rainfall", "snowfall", "clouds", "weather_main", "public_holiday", "event_flag", "roadwork_flag"]]


def build_real_features(df: pd.DataFrame, horizon: int, config: dict) -> pd.DataFrame:
    """Features for hour t+horizon, using only what's known at the end of hour t, with the same design as src/features.py."""
    per_day = config["grid"]["windows_per_day"]
    df = df.sort_values("ts").reset_index(drop=True)
    volume = df["traffic_volume"]
    target_ts = df["ts"].shift(-horizon)
    out = pd.DataFrame({"road_id": df["road_id"], "ts": df["ts"], "target_ts": target_ts, "road_code": 1})

    minutes = target_ts.dt.hour * 60
    weekday = target_ts.dt.day_of_week
    out["hour_sin"] = np.sin(2 * np.pi * minutes / 1440)
    out["hour_cos"] = np.cos(2 * np.pi * minutes / 1440)
    out["dow_sin"] = np.sin(2 * np.pi * weekday / 7)
    out["dow_cos"] = np.cos(2 * np.pi * weekday / 7)
    out["is_weekend"] = (weekday >= 5).astype("float64").where(target_ts.notna())
    peak = pd.Series(np.nan, index=df.index)
    known = target_ts.notna()
    peak[known] = _peak_flag(target_ts[known], config)
    out["is_peak"] = peak
    out["public_holiday"] = df["public_holiday"].shift(-horizon)
    out["holiday_x_peak"] = out["public_holiday"] * out["is_peak"]

    out["vol_t"] = volume
    out["vol_t1"] = volume.shift(1)
    out["vol_t2"] = volume.shift(2)
    out["vol_day"] = volume.shift(per_day - horizon)
    out["vol_week"] = volume.shift(per_day * 7 - horizon)
    for size in (4, 8):
        out[f"vol_mean{size}"] = volume.rolling(size, min_periods=size).mean()
        out[f"vol_std{size}"] = volume.rolling(size, min_periods=size).std()
    profile = slot_profile(df, "traffic_volume")
    out["vol_profile"] = profile.shift(-horizon)
    ratio = volume / profile
    out["vol_ratio_t"] = ratio
    out["vol_ratio_mean4"] = ratio.rolling(4, min_periods=4).mean()

    # An hour's weather is complete at the end of that hour, which is issue time. The target hour's weather isn't known.
    out["wx_rainfall"] = df["rainfall"]
    out["wx_snowfall"] = df["snowfall"]
    out["wx_temperature"] = df["temperature"]
    out["wx_clouds"] = df["clouds"]
    for name, labels in {"wx_rain": ["Rain", "Drizzle", "Thunderstorm", "Squall"], "wx_snow": ["Snow"], "wx_fog": ["Fog", "Mist", "Haze", "Smoke"]}.items():
        out[name] = df["weather_main"].isin(labels).astype("float64").where(df["weather_main"].notna())

    out["y_volume"] = volume.shift(-horizon)
    out["y_congestion"] = 0
    issue_split, target_split = split_of(out["ts"], config), split_of(target_ts, config)
    out["split"] = issue_split.where(issue_split == target_split, "none")
    out["target_valid"] = ~df["target_imputed"].shift(-horizon).astype("boolean").fillna(True).astype(bool)
    out["target_weather"] = df["weather_main"].shift(-horizon)

    # Rows inside or just after a long gap have no history to read, so they can't be forecast.
    return out[out[[*feature_columns(out.drop(columns="target_weather")), "y_volume"]].notna().all(axis=1)].reset_index(drop=True)


def real_features(frame: pd.DataFrame) -> list:
    """The model inputs in an I-94 feature table."""
    return [c for c in feature_columns(frame) if c != "target_weather"]


def score(y, p) -> dict:
    """The regression metrics plus WAPE, in percent."""
    return {**regression_metrics(y, p), "wape_pct": wape(y, p) * 100}


def _prefix(prefix: str, metrics: dict) -> dict:
    return {f"{prefix}_{k}": v for k, v in metrics.items()}


def run_horizon(table: pd.DataFrame, horizon: int, config: dict, plan: dict, score_test: bool) -> dict:
    """Train, tune (at h=1) and score every model for one horizon. All rows are the same for every model."""
    version = f"real-h{horizon}"
    columns = real_features(table)
    deep_columns = [c for c in input_columns(table) if c != "target_weather"]
    table = table.sort_values("ts").reset_index(drop=True)
    train = split_rows(table, "train")
    scaler = SequenceScaler().fit(train, deep_columns)
    longest = max(config["deep"]["sequence_lengths"])
    history = SequenceData(table, scaler, longest)
    positions = {name: np.flatnonzero((table["split"] == name).to_numpy() & table["target_valid"].to_numpy() & (history.history >= longest)) for name in ("val", "test")}
    frames = {name: table.iloc[index] for name, index in positions.items()}
    rows, forecasts = [], {}

    for name, column in BASELINES.items():
        row = {"model": name, "kind": "baseline", "horizon": horizon, **_prefix("val", score(frames["val"]["y_volume"], frames["val"][column]))}
        if score_test:
            row.update(_prefix("test", score(frames["test"]["y_volume"], frames["test"][column])))
        rows.append(row)

    task = {"column": "y_volume", "kind": "regression", "primary": "rmse"}
    for family in FAMILIES if horizon == 1 else [plan["classical"]]:
        started = time.time()
        if horizon == 1:
            params, cv_scores = tune(family, task, train, columns, config, scope="i94", checkpoint_stage="real")
            plan.setdefault("params", {})[family] = params
        else:
            params, cv_scores = plan["params"][family], []
        model = fit_model(family, "regression", params, train, columns, "y_volume", config)
        fit_seconds = time.time() - started
        started = time.time()
        val_pred = np.clip(predict_scores(model, frames["val"], columns, "regression"), 0, None)
        row = {
            "model": family,
            "kind": "classical",
            "horizon": horizon,
            "cv_mean": float(np.mean(cv_scores)) if cv_scores else None,
            "cv_scores": [float(v) for v in cv_scores],
            "params": str(params),
            "infer_ms_per_1k": (time.time() - started) / len(frames["val"]) * 1e6,
            "fit_s": round(fit_seconds, 1),
            **_prefix("val", score(frames["val"]["y_volume"], val_pred)),
        }
        forecasts[family] = {"val": val_pred}
        if score_test:
            forecasts[family]["test"] = np.clip(predict_scores(model, frames["test"], columns, "regression"), 0, None)
            row.update(_prefix("test", score(frames["test"]["y_volume"], forecasts[family]["test"])))
        rows.append(row)
        logger.info("h=%d %s: val rmse %.1f, wape %.1f%%", horizon, family, row["val_rmse"], row["val_wape_pct"])
    if horizon == 1:
        plan["classical"] = choose_winner([r for r in rows if r["kind"] == "classical"], "rmse")["model"]

    def evaluate_deep(spec: dict, label: str) -> tuple:
        data = SequenceData(table, scaler, spec["length"])
        model, history_frame = train_or_resume(spec, data, label, version, config, stage="real")
        started = time.time()
        val_pred = np.clip(predict(model, data, positions["val"])[0], 0, None)
        infer = (time.time() - started) / len(positions["val"]) * 1e6
        return model, data, val_pred, infer, history_frame

    if horizon == 1:
        deep = config["deep"]
        space = {"cell": ["lstm", "gru"], "length": deep["sequence_lengths"], "hidden": deep["hidden_sizes"], "layers": deep["num_layers"], "dropout": deep["dropouts"], "learning_rate": deep["learning_rates"], "batch_size": config["real_data"]["batch_sizes"]}
        best = None
        for i, spec in enumerate(sample_params(space, config["tuning"]["deep_trials"], config["seed"])):
            _, _, val_pred, _, history_frame = evaluate_deep(spec, f"trial{i + 1}")
            rmse = scratch.rmse(frames["val"]["y_volume"].to_numpy(float), val_pred)
            logger.info("h=1 deep trial %d (%s, length %d): val rmse %.1f, %d epochs", i + 1, spec["cell"], spec["length"], rmse, len(history_frame))
            if best is None or rmse < best[0]:
                best = (rmse, spec, f"trial{i + 1}")
        plan["deep_spec"], plan["deep_label"] = best[1], best[2]
    spec = plan["deep_spec"]
    model, data, val_pred, infer, history_frame = evaluate_deep(spec, plan["deep_label"] if horizon == 1 else "best")
    row = {"model": spec["cell"], "kind": "deep", "horizon": horizon, "params": str(spec), "epochs": len(history_frame), "infer_ms_per_1k": infer, **_prefix("val", score(frames["val"]["y_volume"], val_pred))}
    forecasts[spec["cell"]] = {"val": val_pred}
    if score_test:
        forecasts[spec["cell"]]["test"] = np.clip(predict(model, data, positions["test"])[0], 0, None)
        row.update(_prefix("test", score(frames["test"]["y_volume"], forecasts[spec["cell"]]["test"])))
    rows.append(row)
    logger.info("h=%d %s: val rmse %.1f, wape %.1f%%", horizon, spec["cell"], row["val_rmse"], row["val_wape_pct"])
    return {"rows": rows, "forecasts": forecasts, "frames": frames, "table": table}


def northline_comparison(config: dict) -> dict:
    """The same measures for the Northline test window, 30 minutes ahead."""
    from src.features import load_features

    table = load_features(1, config)
    test = split_rows(table, "test")
    predictions = pd.read_parquet(project_path("processed", config) / "predictions.parquet")
    rows = predictions[(predictions["split"] == "test") & (predictions["horizon"] == 1) & predictions["target_valid"]]
    y, p = rows["actual_volume"].to_numpy(float), rows["volume_pred"].to_numpy(float)
    model = score(y, p)
    return {
        "model": score(y, p),
        "persistence": score(test["y_volume"], test["vol_t"]),
        "yesterday": score(test["y_volume"], test["vol_day"]),
        "skill_vs_persistence": 1 - model["rmse"] / scratch.rmse(test["y_volume"].to_numpy(float), test["vol_t"].to_numpy(float)),
        "skill_vs_yesterday": 1 - model["rmse"] / scratch.rmse(test["y_volume"].to_numpy(float), test["vol_day"].to_numpy(float)),
    }


def write_report(results: dict, config: dict) -> None:
    """Write reports/real_data.md from the saved results."""
    best = results["headline"]
    near = results["northline"]
    names = {"linear": "Linear regression", "forest": "Random forest", "xgboost": "XGBoost", "gru": "GRU neural network", "lstm": "LSTM neural network"}
    classical = min((r for r in results["scoreboard"] if r["kind"] == "classical"), key=lambda r: r["val_rmse"])
    snow = next((r for r in results["by_weather"] if r["value"] == "Snow"), None)
    rows = "\n".join(
        f"| {names.get(r['model'], r['model'])} | {r['val_rmse']:.1f} | {r['test_rmse']:.1f} | {r['test_wape_pct']:.1f}% | {r['test_mape_pct']:.1f}% | {r['test_r2']:.3f} |" for r in results["scoreboard"]
    )
    horizons = "\n".join(f"| {r['horizon']} hour{'s' if r['horizon'] > 1 else ''} | {names.get(r['model'], r['model'])} | {r['test_rmse']:.1f} | {r['test_wape_pct']:.1f}% | {r['seasonal_test_wape_pct']:.1f}% |" for r in results["by_horizon"])
    cleaning = "\n".join(f"| {name} | {count:,} |" for name, count in results["cleaning"].items())
    weather = "\n".join(f"| {r['value']} | {r['n']:,} | {r['rmse']:.0f} | {r['mape_pct']:.1f}% | {r['bias']:+.0f} |" for r in results["by_weather"] if r["n"] >= 20)
    text = f"""# Real-data validation: I-94 traffic

Generated by `python -m src.pipeline real`. The data is hourly westbound traffic on I-94 in Minnesota, from the UCI Metro Interstate Traffic Volume dataset (id 492). It's real sensor data, unlike the Northline set. Models were chosen on 2017 and scored on {results["test_rows"]:,} hours in January to September 2018.

**What this tests:** volume only. This dataset has no speed, occupancy, capacity or accidents, so congestion, travel time and accident risk can't be validated here. It's also one road counted hourly rather than 25 roads every 30 minutes, so the two sets of numbers can't be compared directly.

## Result

The best model on validation, the {names.get(best["model"], best["model"])}, was off by {best["test_wape_pct"]:.1f}% of traffic on average one hour ahead (95% interval {results["wape_interval"][0]:.1f} to {results["wape_interval"][1]:.1f}), a test RMSE of {best["test_rmse"]:.0f} vehicles per hour. Its error is {results["skill_vs_persistence"]:.0%} lower than repeating the last hour, and {results["skill_vs_yesterday"]:.0%} lower than using the same hour yesterday. The best classical model on validation, {names.get(classical["model"], classical["model"])}, was close behind at {classical["test_rmse"]:.0f}, so as on Northline, the neural network's lead is small.

**This contradicts what the plan expected.** The plan assumed errors would be larger on real data, and they're about half Northline's in percentage terms. That doesn't mean real traffic is easier in general: this is one busy road counted hourly, averaging about 3,300 vehicles an hour, which smooths out much of the random variation that a 30-minute count on a smaller road shows.

**What this does and doesn't show.** The forecasting approach holds up on real, messy sensor data for volume. It says nothing about congestion, travel time or accident risk on real roads, because this dataset can't test them.

| Model | Validation RMSE | Test RMSE | Test WAPE | Test MAPE | Test R² |
|---|---:|---:|---:|---:|---:|
{rows}

WAPE is total error as a share of total traffic. MAPE is inflated here by quiet night hours, where a small miss is a large percentage.

## Compared with Northline

| Measure | Northline (synthetic, 30 min ahead) | I-94 (real, 1 hour ahead) |
|---|---:|---:|
| Average error (WAPE) | {near["model"]["wape_pct"]:.1f}% | {best["test_wape_pct"]:.1f}% |
| Error reduction against repeating the last reading | {near["skill_vs_persistence"]:.0%} | {results["skill_vs_persistence"]:.0%} |
| Error reduction against the same time yesterday | {near["skill_vs_yesterday"]:.0%} | {results["skill_vs_yesterday"]:.0%} |
| R² | {near["model"]["r2"]:.3f} | {best["test_r2"]:.3f} |

## By look-ahead

| Look-ahead | Model | Test RMSE | Test WAPE | Same hour yesterday, WAPE |
|---|---|---:|---:|---:|
{horizons}

## Confidence ranges

The 80% range contained the actual volume {results["coverage_80"]:.1f}% of the time on test hours, and the 95% range {results["coverage_95"]:.1f}%.

## By weather at the target hour

| Weather | Hours | RMSE | MAPE | Too high (+) or low (-) |
|---|---:|---:|---:|---:|
{weather}

Forecasts run too high in snow ({snow["bias"]:+.0f} vehicles), for the same reason they run too high in rain and fog on Northline: the model can't know the weather in the hour it's predicting. Conditions with fewer than 20 hours are left out of this table.

## Defects handled

| Defect | Entries |
|---|---:|
{cleaning}

Gaps of up to {config["cleaning"]["interpolate_limit_windows"]} hours are interpolated and never scored. Longer gaps stay blank, including one of about ten months in 2014 and 2015, and hours without enough history behind them aren't forecast.

## What was reused

Shared with the Northline pipeline: the quarantine ledger, short-gap filling, slot profile, chronological splits, time-series CV, tuning, model families, sequence models, conformal bands, metrics and selection rule. New for this dataset: the file adapter and a volume-only feature table, which follows the same design without the columns this dataset lacks.
"""
    (project_path("reports", config) / "real_data.md").write_text(text, encoding="utf-8", newline="\n")


def run(config: dict | None = None, score_test: bool = True) -> dict | None:
    """Clean, train and evaluate on I-94. Skips with a warning if the data can't be fetched."""
    base = config or load_config()
    path = fetch(base)
    if path is None:
        logger.warning("Skipping the real-data stage. Earlier results, if any, are left in place.")
        return None
    cfg = real_config(base)
    seed_everything(cfg["seed"])
    ledger = Ledger()
    raw = pd.read_csv(path)
    clean = clean_i94(raw, ledger, cfg)
    logger.info("Cleaned %d raw rows into %d hours, of which %d are measured", len(raw), len(clean), int((~clean["target_imputed"] & clean["traffic_volume"].notna()).sum()))

    plan, parts = {}, {}
    all_rows = []
    for horizon in cfg["horizons"]:
        parts[horizon] = run_horizon(build_real_features(clean, horizon, cfg), horizon, cfg, plan, score_test)
        all_rows += parts[horizon]["rows"]

    h1 = [r for r in all_rows if r["horizon"] == 1]
    classical = next(r for r in h1 if r["model"] == plan["classical"])
    deep = next(r for r in h1 if r["kind"] == "deep")
    winner = deep if deep["val_rmse"] < classical["val_rmse"] else classical
    logger.info("Validation scores at 1 hour: %s", {r["model"]: round(r["val_rmse"], 1) for r in h1})
    logger.info("Chosen on validation: classical %s, deep %s, overall %s", plan["classical"], plan["deep_spec"]["cell"], winner["model"])
    if not score_test:
        return {"rows": all_rows, "winner": winner["model"]}

    frame = parts[1]["frames"]["test"]
    y = frame["y_volume"].to_numpy(float)
    forecast = parts[1]["forecasts"][winner["model"]]["test"]
    val_frame = parts[1]["frames"]["val"]
    cells = _peak_flag(val_frame["target_ts"], cfg).map({1: "peak", 0: "off"}).astype("string")
    table = fit_conformal(val_frame["y_volume"].to_numpy(float), parts[1]["forecasts"][winner["model"]]["val"], cells, cfg["uncertainty"]["levels"])
    test_cells = _peak_flag(frame["target_ts"], cfg).map({1: "peak", 0: "off"}).astype("string")
    low80, high80 = apply_conformal(forecast, test_cells, table, 0.8)
    low95, high95 = apply_conformal(forecast, test_cells, table, 0.95)

    by_horizon = []
    for horizon in cfg["horizons"]:
        rows = [r for r in parts[horizon]["rows"] if r["kind"] != "baseline"]
        chosen = min(rows, key=lambda r: r["val_rmse"])
        seasonal = next(r for r in parts[horizon]["rows"] if r["model"] == "Same hour yesterday")
        by_horizon.append({"horizon": horizon, "model": chosen["model"], "test_rmse": chosen["test_rmse"], "test_wape_pct": chosen["test_wape_pct"], "seasonal_test_wape_pct": seasonal["test_wape_pct"]})

    breakdown = error_breakdown(frame, y, forecast, frame["target_weather"])
    persistence = next(r for r in h1 if r["model"] == "Persistence (last hour)")
    yesterday = next(r for r in h1 if r["model"] == "Same hour yesterday")
    counts = ledger.to_frame().groupby("rule").size()
    results = {
        "dataset": "UCI Metro Interstate Traffic Volume (id 492), westbound I-94",
        "test_rows": len(frame),
        "headline": winner,
        "scoreboard": h1,
        "all_rows": all_rows,
        "by_horizon": by_horizon,
        "wape_interval": day_bootstrap(lambda i: wape(y[i], forecast[i]) * 100, frame["target_ts"].dt.strftime("%Y-%m-%d").to_numpy(), seed=cfg["seed"]),
        "skill_vs_persistence": 1 - winner["test_rmse"] / persistence["test_rmse"],
        "skill_vs_yesterday": 1 - winner["test_rmse"] / yesterday["test_rmse"],
        "coverage_80": coverage(y, low80, high80) * 100,
        "coverage_95": coverage(y, low95, high95) * 100,
        "by_weather": breakdown[breakdown["group"] == "weather"].to_dict("records"),
        "by_hour": breakdown[breakdown["group"] == "hour"].to_dict("records"),
        "northline": northline_comparison(base),
        "cleaning": {
            "Duplicate hours (one row per weather condition)": int(counts.get("duplicate_hour", 0)),
            "Hours absent from the log": int(counts.get("hour_absent", 0)),
            "Temperatures of 0 Kelvin": int(counts.get("temperature_zero_kelvin", 0)),
            "Impossible rainfall readings": int(counts.get("impossible_rainfall", 0)),
            "Hours with a volume of zero": int(counts.get("zero_volume", 0)),
        },
    }
    (project_path("reports", base) / "real_data_results.json").write_text(json.dumps(results, indent=2, default=float), encoding="utf-8", newline="\n")
    pd.DataFrame(
        {"target_ts": frame["target_ts"].to_numpy(), "actual": y, "forecast": forecast, "low80": low80, "high80": high80, "low95": low95, "high95": high95, "weather": frame["target_weather"].to_numpy()}
    ).to_parquet(project_path("processed", base) / "real_predictions.parquet", index=False)
    write_report(results, base)

    week = frame.iloc[:24 * 7]
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(week["target_ts"], week["y_volume"], color="#222", label="What happened")
    ax.plot(week["target_ts"], forecast[: len(week)], color="#2a6f97", label="Forecast, 1 hour ahead")
    ax.set_ylabel("Vehicles per hour")
    ax.set_title("Real traffic on I-94, first week of the test period")
    ax.legend()
    fig.tight_layout()
    fig.savefig(project_path("figures", base) / "real_data_week.png", dpi=120)
    plt.close(fig)
    logger.info("Test at 1 hour: %s RMSE %.1f, WAPE %.1f%%, coverage %.1f%%", winner["model"], winner["test_rmse"], winner["test_wape_pct"], results["coverage_80"])
    return results


def main(argv: list | None = None) -> int:
    """Run the validation from the command line, optionally stopping before the test window."""
    parser = argparse.ArgumentParser(description="Validate FlowCast on real traffic data")
    parser.add_argument("--validation-only", action="store_true", help="choose models and stop before scoring the test window")
    run(score_test=not parser.parse_args(argv).validation_only)
    return 0


if __name__ == "__main__":
    exit_cleanly(main())
