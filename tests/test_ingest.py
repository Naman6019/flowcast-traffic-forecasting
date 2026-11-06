import pytest

from src.ingest import Ledger, _read, validate_calendar, validate_traffic, validate_weather
from tests.conftest import make_calendar, make_traffic, make_weather


def rules(ledger: Ledger) -> set:
    return set(ledger.to_frame()["rule"])


def test_clean_table_produces_no_ledger_entries(config):
    ledger = Ledger()
    df = validate_traffic(make_traffic(), ledger, config)
    assert len(df) == 96 * 2
    assert ledger.to_frame().empty


def test_one_of_each_defect_gets_the_right_code(config):
    raw = make_traffic()
    raw.loc[0, "traffic_volume"] = "-5"
    raw.loc[1, "avg_speed"] = "250"
    raw.loc[2, "occupancy"] = "120"
    raw.loc[3, "vehicle_type_dist"] = "{not json"
    raw.loc[4, "occupancy"] = "abc"
    raw.loc[5, "date"] = "not-a-date"
    raw.loc[6, "road_capacity"] = "0"
    raw.loc[7, "time"] = "07:15"
    raw.loc[8, "road_id"] = None
    ledger = Ledger()
    df = validate_traffic(raw, ledger, config)
    assert rules(ledger) == {
        "negative_volume",
        "speed_out_of_range",
        "occupancy_out_of_range",
        "invalid_json",
        "unparseable_number",
        "invalid_timestamp",
        "invalid_capacity",
        "off_grid_time",
        "missing_road_id",
    }
    assert len(df) == len(raw) - 4


def test_no_row_disappears_without_a_ledger_entry(config):
    raw = make_traffic()
    raw.loc[5, "date"] = "not-a-date"
    raw.loc[6, "road_capacity"] = "-1"
    ledger = Ledger()
    df = validate_traffic(raw, ledger, config)
    assert len(raw) - len(df) == (ledger.to_frame()["column"] == "row").sum()


def test_weather_date_parses_as_day_first(config):
    raw = make_weather()
    raw["date"] = "05/03/2025"
    df = validate_weather(raw.head(1), Ledger(), config)
    assert (df["ts"].dt.month.iat[0], df["ts"].dt.day.iat[0]) == (3, 5)


def test_timestamps_are_timezone_aware(config):
    df = validate_traffic(make_traffic(), Ledger(), config)
    assert str(df["ts"].dt.tz) == "Asia/Kolkata"


def test_bad_calendar_flag_is_logged_and_zeroed():
    raw = make_calendar()
    raw.loc[0, "event_flag"] = "yes"
    ledger = Ledger()
    df = validate_calendar(raw, ledger)
    assert rules(ledger) == {"invalid_flag"}
    assert df.loc[0, "event_flag"] == 0


def test_missing_column_raises(tmp_path):
    path = tmp_path / "t.csv"
    make_calendar().drop(columns="event_flag").to_csv(path, index=False)
    with pytest.raises(ValueError, match="event_flag"):
        _read(path, ["date", "event_flag"], "calendar")
