"""M1: load the three raw tables, enforce schema, types and ranges, and log every violation.

Rows that can't be placed on the segment-time grid are rejected here. Values that are merely out of
range stay in the table and are repaired by rule in M2, but each one is recorded in the ledger first.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.utils import ROOT, get_logger, load_config

logger = get_logger("ingest")

TRAFFIC_NUMERIC = [
    "latitude",
    "longitude",
    "traffic_volume",
    "vehicle_count",
    "avg_speed",
    "occupancy",
    "travel_time",
    "accident_count",
    "signal_timing",
    "road_capacity",
]
TRAFFIC_COLUMNS = ["road_id", "road_name", "weather_station_id", "date", "time", "vehicle_type_dist", "congestion_level"]
TRAFFIC_COLUMNS += TRAFFIC_NUMERIC
WEATHER_NUMERIC = ["temperature", "rainfall", "visibility"]
WEATHER_COLUMNS = ["station_id", "date", "time", "weather_condition"] + WEATHER_NUMERIC
CALENDAR_FLAGS = ["public_holiday", "event_flag", "roadwork_flag"]
CALENDAR_COLUMNS = ["date", "holiday_name", "event_name"] + CALENDAR_FLAGS
LEDGER_COLUMNS = ["stage", "rule", "table", "key", "column", "old_value", "new_value"]

TRAFFIC_DATE_FORMAT = "%Y-%m-%d %H:%M"
WEATHER_DATE_FORMAT = "%d/%m/%Y %H:%M"


@dataclass
class Ledger:
    """Append-only record of every rejected row and changed value (the quarantine file)."""

    parts: list = field(default_factory=list)

    def add(self, stage: str, rule: str, table: str, keys, column: str, old_values, new_values=None) -> None:
        """Record one rule's effect. `keys` and the values are aligned sequences."""
        keys = pd.Series(keys).astype("string").reset_index(drop=True)
        if keys.empty:
            return
        old = pd.Series(old_values).astype("string").reset_index(drop=True)
        new = pd.Series(new_values).astype("string").reset_index(drop=True) if new_values is not None else pd.NA
        self.parts.append(
            pd.DataFrame(
                {
                    "stage": stage,
                    "rule": rule,
                    "table": table,
                    "key": keys,
                    "column": column,
                    "old_value": old,
                    "new_value": new,
                }
            )
        )

    def to_frame(self) -> pd.DataFrame:
        """Return the whole ledger as one string-typed table."""
        if not self.parts:
            return pd.DataFrame(columns=LEDGER_COLUMNS).astype("string")
        return pd.concat(self.parts, ignore_index=True).astype("string")

    def count(self, rule: str) -> int:
        """Number of entries recorded under one rule."""
        return sum(len(p) for p in self.parts if p["rule"].iat[0] == rule)


def traffic_keys(df: pd.DataFrame) -> pd.Series:
    """Ledger key for a traffic row: road, date and window, even when the timestamp didn't parse."""
    return df["road_id"].astype("string").fillna("?") + "|" + df["date"].astype("string").fillna("?") + " " + df["time"].astype("string").fillna("?")


def _read(path: Path, required: list, table: str) -> pd.DataFrame:
    """Read a CSV as strings so that unparseable values can be detected rather than guessed."""
    df = pd.read_csv(path, dtype=str)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{table}: missing required columns {missing}")
    return df[required]


def _coerce_numeric(df: pd.DataFrame, columns: list, table: str, keys: pd.Series, ledger: Ledger) -> pd.DataFrame:
    """Convert columns to float, logging any non-blank value that isn't a number and blanking it."""
    df = df.copy()
    for col in columns:
        parsed = pd.to_numeric(df[col], errors="coerce")
        bad = parsed.isna() & df[col].notna()
        ledger.add("ingest", "unparseable_number", table, keys[bad], col, df.loc[bad, col])
        df[col] = parsed.astype("float64")
    return df


def _timestamp(date: pd.Series, time: pd.Series, fmt: str, tz: str) -> pd.Series:
    """Parse date and time columns with an explicit format into tz-aware timestamps (NaT if invalid)."""
    return pd.to_datetime(date + " " + time, format=fmt, errors="coerce").dt.tz_localize(tz)


def _valid_json_dist(values: pd.Series) -> pd.Series:
    """True where a vehicle_type_dist string parses to a JSON object."""

    def parses(text: str) -> bool:
        try:
            return isinstance(json.loads(text), dict)
        except (TypeError, ValueError):
            return False

    return values.map({v: parses(v) for v in values.dropna().unique()}).fillna(False).astype(bool)


def validate_traffic(raw: pd.DataFrame, ledger: Ledger, config: dict | None = None) -> pd.DataFrame:
    """Apply types and range checks to raw traffic rows, returning the rows that stay in the pipeline."""
    cfg = config or load_config()
    limits = cfg["ranges"]
    keys = traffic_keys(raw)
    df = _coerce_numeric(raw, TRAFFIC_NUMERIC, "traffic", keys, ledger)
    df["ts"] = _timestamp(df["date"], df["time"], TRAFFIC_DATE_FORMAT, cfg["timezone"])

    step = cfg["grid"]["window_minutes"]
    checks = {
        "missing_road_id": df["road_id"].isna(),
        "invalid_timestamp": df["ts"].isna(),
        "off_grid_time": df["ts"].notna() & (df["ts"].dt.minute % step != 0),
        "invalid_capacity": ~(df["road_capacity"] > 0),
    }
    reject = pd.Series(False, index=df.index)
    for rule, mask in checks.items():
        mask = mask & ~reject
        ledger.add("ingest", rule, "traffic", keys[mask], "row", raw.loc[mask, "time"])
        reject |= mask

    flags = {
        "negative_volume": ("traffic_volume", df["traffic_volume"] < limits["volume_min"]),
        "speed_out_of_range": ("avg_speed", (df["avg_speed"] <= limits["speed_min_exclusive"]) | (df["avg_speed"] > limits["speed_max"])),
        "occupancy_out_of_range": ("occupancy", (df["occupancy"] < limits["occupancy_min"]) | (df["occupancy"] > limits["occupancy_max"])),
    }
    for rule, (col, mask) in flags.items():
        ledger.add("ingest", rule, "traffic", keys[mask], col, df.loc[mask, col])

    bad_json = df["vehicle_type_dist"].notna() & ~_valid_json_dist(df["vehicle_type_dist"])
    ledger.add("ingest", "invalid_json", "traffic", keys[bad_json], "vehicle_type_dist", df.loc[bad_json, "vehicle_type_dist"])
    df.loc[bad_json, "vehicle_type_dist"] = pd.NA

    return df.loc[~reject].reset_index(drop=True)


def validate_weather(raw: pd.DataFrame, ledger: Ledger, config: dict | None = None) -> pd.DataFrame:
    """Apply types to raw weather rows. Weather dates are DD/MM/YYYY and are never guessed."""
    cfg = config or load_config()
    keys = raw["station_id"].astype("string").fillna("?") + "|" + raw["date"].astype("string").fillna("?") + " " + raw["time"].astype("string").fillna("?")
    df = _coerce_numeric(raw, WEATHER_NUMERIC, "weather", keys, ledger)
    df["ts"] = _timestamp(df["date"], df["time"], WEATHER_DATE_FORMAT, cfg["timezone"])
    reject = df["ts"].isna() | df["station_id"].isna()
    ledger.add("ingest", "invalid_timestamp", "weather", keys[reject], "row", raw.loc[reject, "date"])
    return df.loc[~reject].reset_index(drop=True)


def validate_calendar(raw: pd.DataFrame, ledger: Ledger) -> pd.DataFrame:
    """Apply types to the calendar. Flags must be 0 or 1; anything else is logged and treated as 0."""
    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    reject = df["date"].isna()
    ledger.add("ingest", "invalid_date", "calendar", raw.loc[reject, "date"].fillna("?"), "row", raw.loc[reject, "date"])
    df = df.loc[~reject].reset_index(drop=True)
    for col in CALENDAR_FLAGS:
        bad = ~df[col].isin(["0", "1"])
        ledger.add("ingest", "invalid_flag", "calendar", df.loc[bad, "date"].astype("string"), col, df.loc[bad, col])
        df[col] = df[col].where(~bad, "0").astype("int64")
    return df


def load_all(traffic_path: str | Path | None = None, config: dict | None = None) -> dict:
    """Load and validate all three tables. `traffic_path` overrides the sensor log, for uploads."""
    cfg = config or load_config()
    raw_dir = ROOT / cfg["paths"]["raw"]
    files = cfg["raw_files"]
    ledger = Ledger()

    calendar_raw = _read(raw_dir / files["calendar"], CALENDAR_COLUMNS, "calendar")
    weather_raw = _read(raw_dir / files["weather"], WEATHER_COLUMNS, "weather")
    traffic_raw = _read(Path(traffic_path) if traffic_path else raw_dir / files["traffic"], TRAFFIC_COLUMNS, "traffic")

    tables = {
        "calendar": validate_calendar(calendar_raw, ledger),
        "weather": validate_weather(weather_raw, ledger, cfg),
        "traffic": validate_traffic(traffic_raw, ledger, cfg),
        "ledger": ledger,
        "raw_rows": {"calendar": len(calendar_raw), "weather": len(weather_raw), "traffic": len(traffic_raw)},
        "raw_nulls": traffic_raw[["traffic_volume", "avg_speed", "occupancy", "congestion_level"]].isna().mean().to_dict(),
    }
    logger.info("Loaded traffic=%d weather=%d calendar=%d", *(tables["raw_rows"][k] for k in ("traffic", "weather", "calendar")))
    return tables
