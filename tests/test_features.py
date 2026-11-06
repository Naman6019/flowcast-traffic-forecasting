import numpy as np
import pandas as pd
import pytest

from src.clean import window_medians
from src.features import build_features, feature_columns, slot_profile


def synthetic(config, days=30, roads=("NL-001", "NL-002")):
    """A cleaned-looking dataset where volume equals the window index and temperature the hour index."""
    frames = []
    times = pd.date_range("2025-01-01", periods=48 * days, freq="30min", tz=config["timezone"])
    for road_number, road in enumerate(roads):
        n = len(times)
        index = np.arange(n, dtype=float)
        frames.append(
            pd.DataFrame(
                {
                    "road_id": road,
                    "ts": times,
                    "traffic_volume": index + 1_000_000 * road_number,
                    "avg_speed": index / 10,
                    "occupancy": 10.0,
                    "travel_time": 2.0,
                    "accident_count": (index % 100 == 0).astype(int),
                    "congestion_level": "Moderate",
                    "road_capacity": 2000,
                    "seg_length_km": 1.0,
                    "rainfall": 0.0,
                    "visibility": 5000.0,
                    "temperature": np.floor(index / 2),
                    "weather_condition": "Clear",
                    "public_holiday": 0,
                    "event_flag": 0,
                    "roadwork_flag": 0,
                    "share_hcv": 0.1,
                    "share_2w": 0.4,
                    "signal_timing": 60,
                    "target_imputed": False,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


@pytest.mark.parametrize("horizon", [1, 2, 3, 4])
def test_lags_and_rolling_never_read_past_issue_time(config, horizon):
    table = build_features(synthetic(config), horizon, config)
    road = table[table["road_id"] == "NL-001"]
    t = road["vol_t"]
    assert (road["vol_t1"] == t - 1).all()
    assert (road["vol_t2"] == t - 2).all()
    assert (road["vol_day"] == t + horizon - 48).all()
    assert (road["vol_week"] == t + horizon - 336).all()
    assert (road["vol_mean4"] == t - 1.5).all()
    assert (road["vol_mean8"] == t - 3.5).all()
    for col in ["vol_t1", "vol_t2", "vol_day", "vol_week", "vol_mean4", "vol_mean8"]:
        assert (road[col] <= t).all(), col


@pytest.mark.parametrize("horizon", [1, 4])
def test_target_is_exactly_h_windows_ahead(config, horizon):
    table = build_features(synthetic(config), horizon, config)
    road = table[table["road_id"] == "NL-001"]
    assert (road["y_volume"] == road["vol_t"] + horizon).all()
    assert ((road["target_ts"] - road["ts"]) == pd.Timedelta(minutes=30 * horizon)).all()


def test_lags_stay_within_each_segment(config):
    table = build_features(synthetic(config), 1, config)
    assert (table.loc[table["road_id"] == "NL-002", "vol_week"] >= 1_000_000).all()
    assert (table.loc[table["road_id"] == "NL-001", "vol_week"] < 1_000_000).all()


def test_weather_comes_from_the_last_completed_hour(config):
    df = synthetic(config)
    table = build_features(df, 1, config)
    road = table[table["road_id"] == "NL-001"]
    issue_time = road["ts"] + pd.Timedelta(minutes=30)
    # temperature equals the hour index, so the hour read by the feature can be recovered.
    start = df["ts"].min()
    hour_read = road["wx_temperature"]
    hour_end = start + pd.to_timedelta(hour_read + 1, unit="h")
    assert (hour_end <= issue_time).all()
    # And it is the latest such hour: the next hour would not have finished yet.
    assert (hour_end + pd.Timedelta(hours=1) > issue_time).all()


def test_no_feature_copies_a_target_window_value(config):
    df = synthetic(config)
    rng = np.random.default_rng(0)
    df["traffic_volume"] = rng.normal(500, 50, len(df))
    df["avg_speed"] = rng.normal(40, 5, len(df))
    df["accident_count"] = rng.integers(0, 2, len(df))
    table = build_features(df, 1, config)
    features = feature_columns(table)
    assert not {"vehicle_count", "travel_time", "congestion_level", "y_volume"} & set(features)
    for target in ["y_volume", "y_speed"]:
        correlations = table[features].select_dtypes("number").corrwith(table[target]).abs()
        assert correlations.max() < 0.2, correlations.idxmax()


def test_changing_the_future_does_not_change_past_features(config):
    df = synthetic(config)
    baseline = build_features(df, 1, config)
    cutoff = df["ts"].min() + pd.Timedelta(days=20)
    changed = df.copy()
    future = changed["ts"] > cutoff
    changed.loc[future, ["traffic_volume", "avg_speed", "temperature", "rainfall", "accident_count"]] = 9999
    after = build_features(changed, 1, config)
    columns = [c for c in feature_columns(baseline) if not c.startswith(("hour", "dow", "is_"))]
    past = baseline["ts"] <= cutoff
    pd.testing.assert_frame_equal(baseline.loc[past, columns], after.loc[past, columns])


def test_splits_never_straddle_a_boundary_and_warm_up_is_dropped(config):
    days = 151
    table = build_features(synthetic(config, days=days, roads=("NL-001",)), 4, config)
    for name in ("train", "val", "test"):
        start, end = config["splits"][name]
        part = table[table["split"] == name]
        assert part["ts"].dt.strftime("%Y-%m-%d").min() >= start
        assert part["target_ts"].dt.strftime("%Y-%m-%d").max() <= end
    train = table[table["split"] == "train"]
    assert train["ts"].min() == table["ts"].min()
    assert train["ts"].dt.strftime("%Y-%m-%d").min() == "2025-01-08"
    assert train["target_ts"].max() < table.loc[table["split"] == "val", "ts"].min()


def test_imputed_target_rows_are_marked_invalid(config):
    df = synthetic(config)
    position = df.index[(df["road_id"] == "NL-001") & (df["ts"] == df["ts"].min() + pd.Timedelta(days=10))][0]
    df.loc[position, "target_imputed"] = True
    table = build_features(df, 1, config)
    road = table[table["road_id"] == "NL-001"]
    invalid = road[~road["target_valid"]]
    assert len(invalid) == 1
    assert invalid["target_ts"].iat[0] == df.loc[position, "ts"]


def test_window_medians_ignore_dates_after_training(config):
    df = synthetic(config, days=151, roads=("NL-001",))
    df["time"] = df["ts"].dt.strftime("%H:%M")
    for col in ["vehicle_count", "share_car", "share_lcv"]:
        df[col] = 1.0
    baseline = window_medians(df, config)
    changed = df.copy()
    changed.loc[changed["ts"].dt.strftime("%Y-%m-%d") > config["splits"]["train"][1], "traffic_volume"] = 99999.0
    pd.testing.assert_frame_equal(baseline, window_medians(changed, config))


def test_slot_profile_reads_only_earlier_dates_of_the_same_slot(config):
    df = synthetic(config, days=30, roads=("NL-001",))
    profile = slot_profile(df, "traffic_volume")
    # Volume equals the window index, so Monday 08:00's expected value is the mean of earlier weekdays' 08:00 indices.
    position = df.index[df["ts"] == pd.Timestamp("2025-01-06 08:00", tz=config["timezone"])][0]
    earlier = df[(df["ts"] < df.loc[position, "ts"]) & (df["ts"].dt.hour == 8) & (df["ts"].dt.minute == 0) & (df["ts"].dt.day_of_week < 5)]
    assert profile[position] == pytest.approx(earlier["traffic_volume"].mean())


def test_slot_profile_skips_unusual_days(config):
    df = synthetic(config, days=30, roads=("NL-001",))
    baseline = slot_profile(df, "traffic_volume")
    changed = df.copy()
    holiday = changed["ts"].dt.strftime("%Y-%m-%d") == "2025-01-06"
    changed.loc[holiday, "public_holiday"] = 1
    changed.loc[holiday, "traffic_volume"] = 99999.0
    after = slot_profile(changed, "traffic_volume")
    later = changed["ts"].dt.strftime("%Y-%m-%d") >= "2025-01-08"
    assert (after[later] < 9000).all()
    earlier = changed["ts"].dt.strftime("%Y-%m-%d") < "2025-01-06"
    assert baseline[earlier].equals(after[earlier])
