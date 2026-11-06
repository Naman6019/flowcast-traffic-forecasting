import numpy as np
import pandas as pd
import pytest

from src.features import feature_columns
from src.ingest import Ledger
from src.real_data import build_real_features, clean_i94, fetch, real_config, score


@pytest.fixture
def cfg(config):
    return real_config(config)


def raw_i94(hours: int = 24 * 30, start: str = "2016-01-01 00:00:00") -> pd.DataFrame:
    """A defect-free raw table in the delivered format, where volume equals the hour's position."""
    times = pd.date_range(start, periods=hours, freq="h")
    return pd.DataFrame(
        {
            "holiday": None,
            "temp": 280.0,
            "rain_1h": 0.0,
            "snow_1h": 0.0,
            "clouds_all": 40,
            "weather_main": "Clouds",
            "weather_description": "scattered clouds",
            "date_time": times.strftime("%Y-%m-%d %H:%M:%S"),
            "traffic_volume": np.arange(1, hours + 1),
        }
    )


def rules(ledger: Ledger) -> set:
    return set(ledger.to_frame()["rule"])


def row_at(df: pd.DataFrame, when: str) -> pd.Series:
    return df[df["ts"] == pd.Timestamp(when)].iloc[0]


def test_clean_data_passes_through_unchanged(cfg):
    ledger = Ledger()
    df = clean_i94(raw_i94(), ledger, cfg)
    assert len(df) == 24 * 30
    assert ledger.to_frame().empty
    assert not df["target_imputed"].any()


def test_an_hour_with_two_weather_rows_keeps_the_more_severe(cfg):
    raw = raw_i94()
    extra = raw.iloc[[5]].assign(weather_main="Rain", rain_1h=2.0)
    ledger = Ledger()
    df = clean_i94(pd.concat([raw, extra], ignore_index=True), ledger, cfg)
    assert len(df) == 24 * 30
    assert ledger.count("duplicate_hour") == 1
    kept = row_at(df, raw.loc[5, "date_time"])
    assert kept["weather_main"] == "Rain"
    assert kept["rainfall"] == 2.0


def test_impossible_readings_are_logged_and_repaired(cfg):
    raw = raw_i94()
    raw.loc[10, "temp"] = 0.0
    raw.loc[11, "rain_1h"] = 9831.3
    raw.loc[12, "traffic_volume"] = 0
    ledger = Ledger()
    df = clean_i94(raw, ledger, cfg)
    assert {"temperature_zero_kelvin", "impossible_rainfall", "zero_volume"} <= rules(ledger)
    assert row_at(df, raw.loc[10, "date_time"])["temperature"] == 280.0
    assert row_at(df, raw.loc[11, "date_time"])["rainfall"] == 0.0
    repaired = row_at(df, raw.loc[12, "date_time"])
    assert repaired["traffic_volume"] == 13.0
    assert repaired["target_imputed"]


def test_short_gaps_are_estimated_but_never_count_as_measured(cfg):
    raw = raw_i94().drop(index=[20, 21]).reset_index(drop=True)
    ledger = Ledger()
    df = clean_i94(raw, ledger, cfg)
    filled = row_at(df, "2016-01-01 20:00:00")
    assert filled["traffic_volume"] == 21.0
    assert filled["target_imputed"]
    assert ledger.count("hour_absent") == 2


def test_long_gaps_stay_blank(cfg):
    raw = raw_i94().drop(index=range(100, 110)).reset_index(drop=True)
    df = clean_i94(raw, Ledger(), cfg)
    gap = df[df["ts"].between("2016-01-05 04:00:00", "2016-01-05 13:00:00")]
    assert len(gap) == 10
    assert gap["traffic_volume"].isna().all()


def test_a_holiday_set_on_the_first_hour_covers_the_whole_day(cfg):
    raw = raw_i94()
    raw.loc[24, "holiday"] = "Some Day"
    df = clean_i94(raw, Ledger(), cfg)
    assert df.loc[df["ts"].dt.strftime("%Y-%m-%d") == "2016-01-02", "public_holiday"].eq(1).all()
    assert df.loc[df["ts"].dt.strftime("%Y-%m-%d") == "2016-01-03", "public_holiday"].eq(0).all()


@pytest.mark.parametrize("horizon", [1, 4])
def test_lags_never_read_past_issue_time_and_targets_are_h_hours_ahead(cfg, horizon):
    table = build_real_features(clean_i94(raw_i94(), Ledger(), cfg), horizon, cfg)
    t = table["vol_t"]
    assert (table["y_volume"] == t + horizon).all()
    assert (table["vol_t1"] == t - 1).all()
    assert (table["vol_day"] == t + horizon - 24).all()
    assert (table["vol_week"] == t + horizon - 168).all()
    assert (table["vol_mean4"] == t - 1.5).all()
    assert ((table["target_ts"] - table["ts"]) == pd.Timedelta(hours=horizon)).all()


def test_weather_is_from_the_issue_hour_not_the_target_hour(cfg):
    raw = raw_i94()
    raw["temp"] = 200.0 + np.arange(len(raw)) / 10
    table = build_real_features(clean_i94(raw, Ledger(), cfg), 2, cfg)
    position = (table["ts"] - pd.Timestamp("2016-01-01")) / pd.Timedelta(hours=1)
    assert np.allclose(table["wx_temperature"], 200.0 + position / 10)


def test_changing_the_future_does_not_change_past_features(cfg):
    df = clean_i94(raw_i94(), Ledger(), cfg)
    baseline = build_real_features(df, 1, cfg)
    changed = df.copy()
    cutoff = pd.Timestamp("2016-01-20")
    changed.loc[changed["ts"] > cutoff, ["traffic_volume", "temperature"]] = 99999.0
    after = build_real_features(changed, 1, cfg)
    columns = [c for c in feature_columns(baseline) if c != "target_weather"]
    # Rows at the cutoff forecast the first changed hour, so only their target differs.
    past = baseline["ts"] < cutoff
    pd.testing.assert_frame_equal(baseline.loc[past, columns], after.loc[after["ts"] < cutoff, columns])


def test_the_target_hours_weather_is_kept_out_of_the_inputs(cfg):
    table = build_real_features(clean_i94(raw_i94(), Ledger(), cfg), 1, cfg)
    assert "target_weather" in table
    from src.real_data import real_features

    assert "target_weather" not in real_features(table)
    assert not {"y_volume", "y_congestion", "split", "target_valid"} & set(real_features(table))


def test_estimated_targets_are_marked_invalid(cfg):
    raw = raw_i94().drop(index=[300]).reset_index(drop=True)
    table = build_real_features(clean_i94(raw, Ledger(), cfg), 1, cfg)
    invalid = table[~table["target_valid"]]
    assert len(invalid) == 1
    assert invalid["target_ts"].iat[0] == pd.Timestamp("2016-01-13 12:00:00")


def test_rows_with_no_history_after_a_long_gap_are_not_forecast(cfg):
    raw = raw_i94(hours=24 * 40).drop(index=range(400, 450)).reset_index(drop=True)
    table = build_real_features(clean_i94(raw, Ledger(), cfg), 1, cfg)
    assert table[feature_columns(table)].drop(columns="target_weather").notna().all().all()
    assert ((table["ts"] - pd.Timestamp("2016-01-01")) / pd.Timedelta(hours=1)).between(390, 457).sum() < 15


def test_splits_are_chronological_and_do_not_overlap(cfg):
    splits = cfg["splits"]
    assert splits["train"][1] < splits["val"][0] <= splits["val"][1] < splits["test"][0] <= splits["test"][1]


def test_score_includes_wape():
    metrics = score(np.array([100.0, 300.0]), np.array([110.0, 270.0]))
    assert metrics["wape_pct"] == pytest.approx(10.0)
    assert metrics["rmse"] > 0


def test_a_tampered_file_is_rejected(config, tmp_path):
    folder = tmp_path / "i94"
    folder.mkdir()
    (folder / "data.csv").write_text("not the real file")
    tampered = {**config, "real_data": {**config["real_data"], "folder": str(folder), "csv_name": "data.csv"}}
    with pytest.raises(ValueError, match="checksum"):
        fetch(tampered)
