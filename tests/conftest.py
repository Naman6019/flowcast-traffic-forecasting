import json
from pathlib import Path

import pandas as pd
import pytest

from src.utils import load_config

DIST = json.dumps({"2W": 0.4, "Car": 0.4, "LCV": 0.1, "HCV": 0.1})


def make_traffic(days: int = 2, roads: tuple = ("NL-001", "NL-002")) -> pd.DataFrame:
    """A defect-free raw traffic table, as strings, covering whole days from 1 Jan 2025."""
    rows = []
    times = pd.date_range("2025-01-01", periods=48 * days, freq="30min")
    for road_number, road in enumerate(roads):
        length = 1.0 + road_number
        for i, ts in enumerate(times):
            volume = 100 + (i % 48) * 5
            speed = 40.0 + (i % 5)
            rows.append(
                {
                    "road_id": road,
                    "road_name": f"Road {road}",
                    "latitude": "28.6",
                    "longitude": "77.2",
                    "weather_station_id": "WS-NORTH",
                    "date": ts.strftime("%Y-%m-%d"),
                    "time": ts.strftime("%H:%M"),
                    "traffic_volume": str(float(volume)),
                    "vehicle_count": str(volume),
                    "vehicle_type_dist": DIST,
                    "avg_speed": str(speed),
                    "occupancy": "30.0",
                    "congestion_level": "Free-flow" if volume / 1000 < 0.5 else "Moderate",
                    "travel_time": str(round(length * 60 / speed, 6)),
                    "accident_count": "0",
                    "signal_timing": "60",
                    "road_capacity": "2000",
                }
            )
    return pd.DataFrame(rows)


def make_weather(days: int = 2) -> pd.DataFrame:
    times = pd.date_range("2025-01-01", periods=24 * days, freq="h")
    return pd.DataFrame(
        {
            "station_id": "WS-NORTH",
            "date": times.strftime("%d/%m/%Y"),
            "time": times.strftime("%H:%M"),
            "weather_condition": "Clear",
            "temperature": "15.0",
            "rainfall": "0.0",
            "visibility": "10000.0",
        }
    )


def make_calendar(days: int = 2) -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=days, freq="D")
    return pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "public_holiday": "0",
            "holiday_name": pd.NA,
            "event_flag": "0",
            "event_name": pd.NA,
            "roadwork_flag": "0",
        }
    )


@pytest.fixture
def config():
    return load_config()


RAW_TRAFFIC = Path(__file__).resolve().parents[1] / "data" / "raw" / "traffic_sensor_log.csv"


def pytest_collection_modifyitems(config, items):
    """Skip the slow tests when the brief's data files aren't present (they aren't redistributed)."""
    if RAW_TRAFFIC.exists():
        return
    skip = pytest.mark.skip(reason="needs the brief's data files in data/raw/, which aren't redistributed")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
