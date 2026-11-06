from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.insights import build_summary, to_html
from src.serve import UploadError, overlay_uploaded, run_upload

FIXTURE = Path(__file__).parent / "fixtures" / "sample_upload.csv"


def small_tables():
    dates = ["2025-05-20", "2025-05-21"]
    data = pd.DataFrame(
        {
            "date": np.repeat(dates, 2),
            "time": ["08:00", "12:00"] * 2,
            "road_id": ["NL-001", "NL-002"] * 2,
            "road_name": ["A", "B"] * 2,
            "target_imputed": [False, False, False, True],
            "traffic_volume": [800, 200, 900, 100],
            "avg_speed": [20.0, 50.0, 18.0, 55.0],
            "congestion_level": ["Severe", "Free-flow", "Severe", "Free-flow"],
            "accident_count": [1, 0, 0, 0],
            "weather_condition": ["Rain", "Clear", "Clear", "Clear"],
        }
    )
    predictions = pd.DataFrame(
        {
            "target_ts": pd.to_datetime(["2025-05-20 08:00", "2025-05-20 12:00", "2025-05-21 08:00"]),
            "target_valid": True,
            "horizon": 1,
            "actual_volume": [800.0, 200.0, 900.0],
            "volume_pred": [780.0, 210.0, 880.0],
            "actual_congestion": [3, 0, 3],
            "congestion_pred": [3, 0, 2],
            "actual_accident": [1.0, 0.0, 0.0],
            "accident_score": [0.9, 0.2, 0.1],
        }
    )
    return data, predictions


def test_summary_uses_real_numbers_and_leaves_out_estimated_rows():
    data, predictions = small_tables()
    summary = build_summary(data, predictions, "2025-05-20", "2025-05-21")
    assert "Windows measured:** 3" in summary
    assert "1 more were estimated" in summary
    assert "Reported accidents:** 1" in summary
    assert "Forecast accuracy for this range" in summary


def test_summary_says_when_no_forecasts_cover_the_range():
    data, predictions = small_tables()
    assert "No stored forecasts" in build_summary(data, predictions.iloc[0:0], "2025-05-20", "2025-05-21")


def test_summary_rejects_an_empty_range():
    data, predictions = small_tables()
    with pytest.raises(ValueError, match="No data"):
        build_summary(data, predictions, "2024-01-01", "2024-01-02")


def test_html_export_renders_tables():
    data, predictions = small_tables()
    html = to_html(build_summary(data, predictions, "2025-05-20", "2025-05-21"))
    assert html.startswith("<!doctype html>")
    assert "<table>" in html


def forecast_rows_for(road, hours, value):
    return pd.DataFrame({"road_id": road, "issue_ts": pd.date_range("2025-05-20 08:00", periods=hours, freq="30min"), "horizon": 1, "volume_pred": value})


def test_an_upload_replaces_only_the_rows_it_covers():
    stored = forecast_rows_for("NL-001", 4, 100.0)
    uploaded = forecast_rows_for("NL-001", 2, 999.0)
    combined = overlay_uploaded(stored, uploaded)
    assert len(combined) == 4
    assert combined["volume_pred"].tolist() == [999.0, 999.0, 100.0, 100.0]
    assert combined["source"].tolist() == ["uploaded", "uploaded", "stored", "stored"]


def test_an_upload_can_add_rows_that_were_not_stored():
    combined = overlay_uploaded(forecast_rows_for("NL-001", 1, 100.0), forecast_rows_for("NL-002", 1, 50.0))
    assert set(combined["road_id"]) == {"NL-001", "NL-002"}


def test_an_upload_does_not_replace_a_different_look_ahead():
    stored = forecast_rows_for("NL-001", 1, 100.0).assign(horizon=2)
    combined = overlay_uploaded(stored, forecast_rows_for("NL-001", 1, 999.0))
    assert len(combined) == 2


@pytest.mark.slow
def test_upload_is_cleaned_quarantined_and_forecast_within_the_time_limit():
    result = run_upload(FIXTURE)
    rules = set(result["quarantine"]["rule"])
    assert {"negative_volume", "speed_out_of_range", "invalid_timestamp", "unknown_segment"} <= rules
    assert set(result["quarantine"]["table"]) == {"traffic"}
    assert "weather_label_normalised" not in rules
    forecasts = result["predictions"]
    assert set(forecasts["road_id"]) == {"NL-003", "NL-011"}
    assert set(forecasts["horizon"]) == {1, 2, 3, 4}
    assert forecasts["volume_pred"].notna().all()
    assert (forecasts["volume_hi80"] >= forecasts["volume_lo80"]).all()
    assert forecasts["input_hash"].str.len().eq(16).all()
    assert result["seconds"] <= 30


@pytest.mark.slow
def test_upload_outside_the_weather_period_gets_a_clear_message(tmp_path):
    rows = pd.read_csv(FIXTURE, dtype=str)
    rows["date"] = rows["date"].str.replace("2025-05", "2026-05")
    path = tmp_path / "late.csv"
    rows.to_csv(path, index=False)
    with pytest.raises(UploadError, match="only exist"):
        run_upload(path)


@pytest.mark.slow
def test_upload_with_missing_columns_gets_a_clear_message(tmp_path):
    path = tmp_path / "broken.csv"
    pd.read_csv(FIXTURE, dtype=str).drop(columns="avg_speed").to_csv(path, index=False)
    with pytest.raises(UploadError, match="avg_speed"):
        run_upload(path)
