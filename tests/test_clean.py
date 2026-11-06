import numpy as np
import pandas as pd
import pytest

from src.clean import (
    MODELLING_COLUMNS,
    clean_tables,
    congestion_from_volume,
    dataset_hash,
    deduplicate,
    fill_short_gaps,
    final_checks,
    harmonise_weather,
)
from src.ingest import Ledger, load_all, validate_calendar, validate_traffic, validate_weather
from tests.conftest import make_calendar, make_traffic, make_weather


def run_clean(raw_traffic, config, raw_weather=None, days=2):
    ledger = Ledger()
    raw_weather = make_weather(days) if raw_weather is None else raw_weather
    tables = {
        "traffic": validate_traffic(raw_traffic, ledger, config),
        "weather": validate_weather(raw_weather, ledger, config),
        "calendar": validate_calendar(make_calendar(days), ledger),
        "ledger": ledger,
        "raw_rows": {"traffic": len(raw_traffic)},
    }
    end = (pd.Timestamp("2025-01-01") + pd.Timedelta(days=days - 1)).strftime("%Y-%m-%d")
    return clean_tables(tables, config, grid=("2025-01-01", end))


def row(df, road, date, time):
    return df[(df["road_id"] == road) & (df["date"] == date) & (df["time"] == time)].iloc[0]


def test_dedup_keeps_the_most_complete_row(config):
    raw = make_traffic(days=1, roads=("NL-001",))
    duplicate = raw.iloc[[3]].copy()
    raw.loc[3, "occupancy"] = None
    raw = pd.concat([raw, duplicate], ignore_index=True)
    ledger = Ledger()
    df = deduplicate(validate_traffic(raw, ledger, config), ledger)
    assert len(df) == 48
    assert df.loc[3, "occupancy"] == 30.0
    assert ledger.count("duplicate_key") == 1


def test_null_and_negative_volume_rebuilt_exactly_from_vehicle_count(config):
    raw = make_traffic()
    raw.loc[10, "traffic_volume"] = None
    raw.loc[11, "traffic_volume"] = "-40"
    expected = int(raw.loc[10, "vehicle_count"]), int(raw.loc[11, "vehicle_count"])
    df = run_clean(raw, config)["dataset"]
    got = row(df, "NL-001", raw.loc[10, "date"], raw.loc[10, "time"]), row(df, "NL-001", raw.loc[11, "date"], raw.loc[11, "time"])
    assert (got[0]["traffic_volume"], got[1]["traffic_volume"]) == expected
    assert got[0]["vol_source"] == "vehicle_count"
    assert not got[0]["target_imputed"]


def test_null_and_impossible_speed_rebuilt_from_travel_time(config):
    raw = make_traffic()
    expected = float(raw.loc[12, "avg_speed"]), float(raw.loc[13, "avg_speed"])
    raw.loc[12, "avg_speed"] = None
    raw.loc[13, "avg_speed"] = "319.6"
    df = run_clean(raw, config)["dataset"]
    got = row(df, "NL-001", raw.loc[12, "date"], raw.loc[12, "time"]), row(df, "NL-001", raw.loc[13, "date"], raw.loc[13, "time"])
    assert got[0]["avg_speed"] == pytest.approx(expected[0], abs=0.01)
    assert got[1]["avg_speed"] == pytest.approx(expected[1], abs=0.01)
    assert got[1]["speed_source"] == "travel_time"


def test_bad_occupancy_is_interpolated_and_flagged(config):
    raw = make_traffic()
    raw.loc[20, "occupancy"] = "259.9"
    df = run_clean(raw, config)["dataset"]
    got = row(df, "NL-001", raw.loc[20, "date"], raw.loc[20, "time"])
    assert got["occupancy"] == 30.0
    assert got["occ_imputed"]


def test_grid_is_complete_and_absent_windows_are_flagged(config):
    raw = make_traffic().drop(index=[30, 31]).reset_index(drop=True)
    df = run_clean(raw, config)["dataset"]
    assert len(df) == 2 * 2 * 48
    assert df["row_imputed"].sum() == 2
    assert df["target_imputed"].sum() == 2
    assert df[MODELLING_COLUMNS].notna().all().all()


def test_short_gap_interpolates_and_long_gap_stays_blank():
    values = pd.Series([1.0, np.nan, np.nan, 4.0, np.nan, np.nan, np.nan, 8.0])
    filled = fill_short_gaps(values, limit=2)
    assert filled.tolist()[:4] == [1.0, 2.0, 3.0, 4.0]
    assert filled.iloc[4:7].isna().all()
    assert filled.iat[7] == 8.0


def test_long_gap_uses_window_median_from_training_dates(config):
    raw = make_traffic(days=2)
    # Remove 5 consecutive windows on day 2, so that only day 1 can supply the medians.
    raw = raw.drop(index=range(48 + 10, 48 + 15)).reset_index(drop=True)
    df = run_clean(raw, config)["dataset"]
    got = row(df, "NL-001", "2025-01-02", "05:00")
    assert got["vol_source"] == "window_median"
    assert got["traffic_volume"] == 100 + 10 * 5


def test_congestion_bands_match_the_dictionary(config):
    capacity = pd.Series([2000] * 7)
    volume = pd.Series([0, 499, 500, 799, 800, 999, 1000])
    got = congestion_from_volume(volume, capacity, config).tolist()
    assert got == ["Free-flow", "Free-flow", "Moderate", "Moderate", "Heavy", "Heavy", "Severe"]


def test_blank_congestion_is_filled_from_cleaned_volume(config):
    raw = make_traffic()
    raw.loc[5, "congestion_level"] = None
    result = run_clean(raw, config)
    assert result["dataset"]["congestion_level"].notna().all()
    assert result["ledger"].count("congestion_filled") == 1
    assert result["ledger"].count("congestion_changed") == 0


def test_every_weather_label_maps_into_the_vocabulary(config):
    raw = make_weather()
    labels = ["Clear", "clear", "CLEAR", "Cloudy", "cloudy", "Overcast", "Rain", "rain ", "rainy", "RAIN", "Fog", "foggy", "FOG"]
    raw.loc[: len(labels) - 1, "weather_condition"] = labels
    ledger = Ledger()
    df = harmonise_weather(validate_weather(raw, ledger, config), ledger, config)
    assert set(df["weather_condition"]) <= set(config["weather"]["vocabulary"])
    assert df.loc[df["ts"].dt.hour == 7, "weather_condition"].iat[0] == "Rain"


def test_unknown_weather_label_raises(config):
    raw = make_weather()
    raw.loc[0, "weather_condition"] = "sleet"
    ledger = Ledger()
    with pytest.raises(ValueError, match="sleet"):
        harmonise_weather(validate_weather(raw, ledger, config), ledger, config)


def test_both_windows_of_an_hour_share_that_hours_weather(config):
    raw = make_weather()
    raw["temperature"] = [str(float(i)) for i in range(len(raw))]
    df = run_clean(make_traffic(), config, raw_weather=raw)["dataset"]
    a, b = row(df, "NL-001", "2025-01-01", "03:00"), row(df, "NL-001", "2025-01-01", "03:30")
    assert a["temperature"] == b["temperature"] == 3.0


def test_join_key_is_dropped_and_day_label_added(config):
    df = run_clean(make_traffic(), config)["dataset"]
    assert "weather_station_id" not in df.columns
    assert row(df, "NL-001", "2025-01-01", "00:00")["day_of_week"] == "Wed"


def test_final_checks_reject_nulls(config):
    df = run_clean(make_traffic(), config)["dataset"]
    final_checks(df, len(df), config)
    broken = df.copy()
    broken["temperature"] = broken["temperature"].astype(float)
    broken.loc[0, "temperature"] = np.nan
    with pytest.raises(ValueError, match="temperature"):
        final_checks(broken, len(df), config)
    with pytest.raises(ValueError, match="Expected"):
        final_checks(df, len(df) + 1, config)


def test_dataset_hash_is_repeatable_and_detects_changes(config):
    df = run_clean(make_traffic(), config)["dataset"]
    again = run_clean(make_traffic(), config)["dataset"]
    assert dataset_hash(df) == dataset_hash(again)
    again.loc[0, "traffic_volume"] += 1
    assert dataset_hash(df) != dataset_hash(again)


@pytest.mark.slow
def test_full_raw_data_matches_the_profile(config):
    tables = load_all(config=config)
    result = clean_tables(tables, config)
    df = result["dataset"]
    final_checks(df, 181_200, config)
    ledger = result["ledger"]
    assert ledger.count("duplicate_key") == 1_767
    assert ledger.count("row_added_to_grid") == 4_499
    assert ledger.count("negative_volume") == 241
    assert ledger.count("congestion_changed") == 0
    # Every interior gap is short. The median is only used where a series starts or ends on a gap.
    median_rows = df[df["vol_source"] == "window_median"]
    assert median_rows["ts"].isin([df["ts"].min(), df["ts"].max()]).all()
    assert (df["vol_source"].isin(["sensor", "vehicle_count", "interpolated", "window_median"])).all()
