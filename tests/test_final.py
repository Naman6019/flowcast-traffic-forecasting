"""Checks on the saved pipeline outputs. These are slow because they need a finished run."""

import json

import numpy as np
import pandas as pd
import pytest

from src.final import validation_forecast
from src.model_card import read_registry
from src.serve import residual_path
from src.utils import load_config, project_path

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def predictions():
    return pd.read_parquet(project_path("processed") / "predictions.parquet")


def test_stored_model_paths_work_on_every_operating_system(predictions):
    registry = (project_path("models") / "registry.json").read_text(encoding="utf-8")
    assert "\\\\" not in registry
    assert not predictions["volume_model"].str.contains("\\", regex=False).any()


def test_every_model_card_records_what_the_prd_requires():
    cards = [path for path in project_path("models").rglob("model_card.json") if "retrain" not in str(path)]
    assert len(cards) >= 41
    for path in cards:
        card = json.loads(path.read_text(encoding="utf-8"))
        for field in ["training_window", "features", "hyperparameters", "validation_metrics", "test_metrics", "dataset_sha256", "git_commit", "random_seed"]:
            assert card.get(field) is not None, f"{path}: {field}"


def test_every_forecast_row_is_complete(predictions):
    columns = ["volume_pred", "speed_pred", "travel_time_pred", "congestion_pred", "accident_prob", "input_hash", "volume_model"]
    assert predictions[columns].notna().all().all()


def test_every_segment_window_and_horizon_is_forecast(predictions):
    counts = predictions.groupby(["horizon", "split"]).size()
    # 25 segments x 48 windows x days, less the rows whose target falls past a split boundary.
    assert counts.loc[(1, "test")] == 25 * (48 * 23 - 1)
    assert counts.loc[(4, "test")] == 25 * (48 * 23 - 4)
    assert set(predictions["horizon"]) == {1, 2, 3, 4}


def test_intervals_are_ordered_and_nested(predictions):
    assert (predictions["volume_lo95"] <= predictions["volume_lo80"]).all()
    assert (predictions["volume_lo80"] <= predictions["volume_pred"]).all()
    assert (predictions["volume_pred"] <= predictions["volume_hi80"]).all()
    assert (predictions["volume_hi80"] <= predictions["volume_hi95"]).all()


def test_class_probabilities_are_valid(predictions):
    columns = ["p_free_flow", "p_moderate", "p_heavy", "p_severe"]
    assert np.allclose(predictions[columns].sum(axis=1), 1.0)
    assert (predictions[columns] >= 0).all().all()


def test_validation_residuals_have_no_gaps_and_are_centred():
    config = load_config()
    for horizon in config["horizons"]:
        residuals = pd.read_parquet(residual_path(horizon, config))
        assert residuals["residual"].notna().all()
        assert abs(residuals["residual"].mean()) < 15


def test_filtered_rows_would_break_a_sequence_model_so_forecasts_use_the_whole_table():
    from src.features import load_features

    registry = read_registry()
    frame = load_features(1)
    rows, values = validation_forecast(frame, 1, registry)
    assert len(rows) == len(values) == ((frame["split"] == "val") & frame["target_valid"]).sum()
    assert not np.isnan(values).any()


def test_head_to_head_uses_the_same_rows():
    metrics = json.loads((project_path("reports") / "final_metrics.json").read_text(encoding="utf-8"))
    board = pd.read_csv(project_path("reports") / "scoreboard_volume.csv")
    deep = board[board["model"].isin(["lstm", "gru", "bilstm"])]
    assert (deep["test_rows"] == metrics["test_rows"]).all()


def test_criteria_report_honestly_against_their_targets():
    metrics = json.loads((project_path("reports") / "final_metrics.json").read_text(encoding="utf-8"))
    by_name = {c["criterion"]: c for c in metrics["criteria"]}
    assert by_name["Volume MAPE <= 12%"]["met"] == (metrics["volume"]["mape_pct"] <= 12)
    assert by_name["Accident ROC-AUC >= 0.75"]["met"] == (metrics["accident"]["roc_auc"] >= 0.75)
