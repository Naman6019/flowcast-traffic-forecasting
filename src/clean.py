"""M2: clean the validated tables and merge them into one segment-time-window dataset.

Stage order follows PRD 9.1: deduplicate, repair and flag outliers, fill missing values, harmonise labels,
align and merge, then run the final checks. Every changed value goes into the ledger, and
reports/data_quality.md is generated from it.
"""

import hashlib
import json

import numpy as np
import pandas as pd

from src.ingest import Ledger, load_all, traffic_keys
from src.utils import get_logger, load_config, project_path

logger = get_logger("clean")

SHARE_COLUMNS = {"2W": "share_2w", "Car": "share_car", "LCV": "share_lcv", "HCV": "share_hcv"}
STATIC_COLUMNS = ["road_name", "latitude", "longitude", "weather_station_id", "road_capacity"]
FILLED_COLUMNS = ["traffic_volume", "vehicle_count", "avg_speed", "occupancy", "signal_timing"] + list(SHARE_COLUMNS.values())
MODELLING_COLUMNS = [
    "traffic_volume",
    "avg_speed",
    "occupancy",
    "travel_time",
    "accident_count",
    "signal_timing",
    "road_capacity",
    "congestion_level",
    "weather_condition",
    "temperature",
    "rainfall",
    "visibility",
    "public_holiday",
    "event_flag",
    "roadwork_flag",
] + list(SHARE_COLUMNS.values())


def deduplicate(df: pd.DataFrame, ledger: Ledger) -> pd.DataFrame:
    """Keep one row per road and timestamp, preferring the most complete, then the first seen."""
    completeness = df.notna().sum(axis=1)
    order = completeness.sort_values(ascending=False, kind="stable").index
    ranked = df.loc[order]
    duplicated = ranked.duplicated(["road_id", "ts"], keep="first")
    dropped = ranked[duplicated]
    ledger.add("dedup", "duplicate_key", "traffic", traffic_keys(dropped), "row", dropped["traffic_volume"])
    return ranked[~duplicated].sort_values(["road_id", "ts"]).reset_index(drop=True)


def parse_vehicle_shares(df: pd.DataFrame) -> pd.DataFrame:
    """Expand the vehicle_type_dist JSON string into one share column per vehicle class."""
    df = df.copy()
    parsed = {text: json.loads(text) for text in df["vehicle_type_dist"].dropna().unique()}
    for key, col in SHARE_COLUMNS.items():
        df[col] = df["vehicle_type_dist"].map({t: float(d.get(key, np.nan)) for t, d in parsed.items()}).astype("float64")
    return df.drop(columns="vehicle_type_dist")


def segment_static_table(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Per-segment attributes, including the length implied by travel_time x speed / 60 on clean rows."""
    limits = config["ranges"]
    clean = df[df["avg_speed"].between(limits["speed_min_exclusive"], limits["speed_max"], inclusive="right") & (df["travel_time"] > 0)]
    lengths = (clean["travel_time"] * clean["avg_speed"] / 60).groupby(clean["road_id"]).median().rename("seg_length_km")
    static = df.groupby("road_id")[STATIC_COLUMNS].agg(lambda s: s.mode().iat[0])
    return static.join(lengths).reset_index()


def repair_rules(df: pd.DataFrame, static: pd.DataFrame, ledger: Ledger, config: dict) -> pd.DataFrame:
    """Rule-based repair, exact reconstruction first (plan D2). Occupancy has no twin, so it is blanked."""
    limits = config["ranges"]
    df = df.merge(static[["road_id", "seg_length_km"]], on="road_id", how="left")
    keys = traffic_keys(df)

    df["vol_source"] = "sensor"
    bad_vol = df["traffic_volume"].isna() | (df["traffic_volume"] < limits["volume_min"])
    usable = bad_vol & df["vehicle_count"].notna() & (df["vehicle_count"] >= limits["volume_min"])
    ledger.add("outliers", "volume_from_vehicle_count", "traffic", keys[usable], "traffic_volume", df.loc[usable, "traffic_volume"], df.loc[usable, "vehicle_count"])
    df.loc[usable, "vol_source"] = "vehicle_count"
    df["traffic_volume"] = df["traffic_volume"].where(~usable, df["vehicle_count"])
    df["traffic_volume"] = df["traffic_volume"].where(df["traffic_volume"] >= limits["volume_min"])

    df["speed_source"] = "sensor"
    bad_speed = ~df["avg_speed"].between(limits["speed_min_exclusive"], limits["speed_max"], inclusive="right")
    usable = bad_speed & (df["travel_time"] > 0)
    rebuilt = df["seg_length_km"] * 60 / df["travel_time"]
    ledger.add("outliers", "speed_from_travel_time", "traffic", keys[usable], "avg_speed", df.loc[usable, "avg_speed"], rebuilt[usable].round(2))
    df.loc[usable, "speed_source"] = "travel_time"
    df["avg_speed"] = df["avg_speed"].where(~usable, rebuilt)
    df["avg_speed"] = df["avg_speed"].where(df["avg_speed"].between(limits["speed_min_exclusive"], limits["speed_max"], inclusive="right"))

    bad_occ = df["occupancy"].notna() & ~df["occupancy"].between(limits["occupancy_min"], limits["occupancy_max"])
    ledger.add("outliers", "occupancy_blanked", "traffic", keys[bad_occ], "occupancy", df.loc[bad_occ, "occupancy"])
    df["occupancy"] = df["occupancy"].where(~bad_occ)
    return df


def flag_statistical_outliers(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Flag (never delete) values beyond |z| > threshold per segment, or outside the IQR fence per segment x hour."""
    z_limit = config["cleaning"]["zscore_threshold"]
    k = config["cleaning"]["iqr_multiplier"]
    df = df.copy()
    flag = pd.Series(False, index=df.index)
    hour = df["ts"].dt.hour
    for col in ["traffic_volume", "avg_speed", "occupancy"]:
        by_road = df.groupby("road_id")[col]
        z = (df[col] - by_road.transform("mean")) / by_road.transform("std")
        by_hour = df.groupby(["road_id", hour])[col]
        q1, q3 = by_hour.transform(lambda s: s.quantile(0.25)), by_hour.transform(lambda s: s.quantile(0.75))
        iqr = q3 - q1
        flag |= (z.abs() > z_limit) | (df[col] < q1 - k * iqr) | (df[col] > q3 + k * iqr)
    df["stat_outlier"] = flag
    return df


def reindex_grid(df: pd.DataFrame, static: pd.DataFrame, start, end, ledger: Ledger, config: dict) -> pd.DataFrame:
    """Put every segment on the full 30-minute grid, adding a flagged row for each absent window."""
    tz, step = config["timezone"], config["grid"]["window_minutes"]
    end_ts = pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(minutes=step)
    times = pd.date_range(pd.Timestamp(start), end_ts, freq=f"{step}min", tz=tz)
    roads = static["road_id"]
    grid = pd.MultiIndex.from_product([roads, times], names=["road_id", "ts"]).to_frame(index=False)

    merged = grid.merge(df, on=["road_id", "ts"], how="left", indicator=True)
    merged["row_imputed"] = merged["_merge"] == "left_only"
    merged = merged.drop(columns="_merge")

    added = merged[merged["row_imputed"]]
    keys = added["road_id"].astype("string") + "|" + added["ts"].dt.strftime("%Y-%m-%d %H:%M").astype("string")
    ledger.add("missing", "row_added_to_grid", "traffic", keys, "row", pd.Series([pd.NA] * len(added), dtype="string"))

    merged = merged.drop(columns=[*STATIC_COLUMNS, "seg_length_km"]).merge(static, on="road_id", how="left")
    merged["date"] = merged["ts"].dt.strftime("%Y-%m-%d")
    merged["time"] = merged["ts"].dt.strftime("%H:%M")
    for col, default in [("vol_source", "missing"), ("speed_source", "missing"), ("stat_outlier", False)]:
        merged[col] = merged[col].where(merged[col].notna(), default)
    merged["stat_outlier"] = merged["stat_outlier"].astype(bool)
    return merged.sort_values(["road_id", "ts"]).reset_index(drop=True)


def fill_short_gaps(values: pd.Series, limit: int) -> pd.Series:
    """Linearly interpolate interior gaps of at most `limit` steps. Longer gaps stay fully blank."""
    missing = values.isna()
    run_id = (missing != missing.shift()).cumsum()
    run_length = missing.groupby(run_id).transform("sum")
    filled = values.interpolate(method="linear", limit_area="inside")
    return filled.where(~(missing & (run_length > limit)))


def window_medians(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Segment x window-of-day medians over training dates only, used for long gaps."""
    train_end = config["splits"]["train"][1]
    train = df[df["ts"].dt.strftime("%Y-%m-%d") <= train_end]
    window = train["ts"].dt.strftime("%H:%M").rename("time")
    return train.groupby(["road_id", window])[FILLED_COLUMNS].median().reset_index()


def fill_missing(df: pd.DataFrame, medians: pd.DataFrame, ledger: Ledger, config: dict) -> pd.DataFrame:
    """Time-interpolate short gaps, then use training medians for the rest, marking the source of each value."""
    limit = config["cleaning"]["interpolate_limit_windows"]
    df = df.copy()
    df["occ_imputed"] = df["occupancy"].isna()
    before = df[FILLED_COLUMNS].isna()
    keys = traffic_keys(df)

    for col in FILLED_COLUMNS:
        df[col] = df.groupby("road_id", group_keys=False)[col].apply(lambda s: fill_short_gaps(s, limit))
    interpolated = before & df[FILLED_COLUMNS].notna()

    with_medians = df[["road_id", "time"]].merge(medians, on=["road_id", "time"], how="left")
    with_medians.index = df.index
    for col in FILLED_COLUMNS:
        df[col] = df[col].fillna(with_medians[col])
    from_median = before & ~interpolated & df[FILLED_COLUMNS].notna()

    for col in FILLED_COLUMNS:
        for rule, mask in (("interpolated", interpolated[col]), ("window_median", from_median[col])):
            ledger.add("missing", rule, "traffic", keys[mask], col, pd.Series([pd.NA] * int(mask.sum()), dtype="string"), df.loc[mask, col].round(3))

    df.loc[interpolated["traffic_volume"], "vol_source"] = "interpolated"
    df.loc[from_median["traffic_volume"], "vol_source"] = "window_median"
    df.loc[interpolated["avg_speed"], "speed_source"] = "interpolated"
    df.loc[from_median["avg_speed"], "speed_source"] = "window_median"

    for col in ["traffic_volume", "vehicle_count", "signal_timing"]:
        df[col] = df[col].round()

    no_travel_time = ~(df["travel_time"] > 0)
    df.loc[no_travel_time, "travel_time"] = (df["seg_length_km"] * 60 / df["avg_speed"])[no_travel_time]
    ledger.add("missing", "travel_time_from_speed", "traffic", keys[no_travel_time], "travel_time", pd.Series([pd.NA] * int(no_travel_time.sum()), dtype="string"), df.loc[no_travel_time, "travel_time"].round(3))

    no_accident = df["accident_count"].isna()
    ledger.add("missing", "accident_count_zero", "traffic", keys[no_accident], "accident_count", pd.Series([pd.NA] * int(no_accident.sum()), dtype="string"), pd.Series(["0"] * int(no_accident.sum()), dtype="string"))
    df["accident_count"] = df["accident_count"].fillna(0)

    # A row whose volume or speed was estimated can be an input but never a target or a metric (plan D2).
    estimated = ("interpolated", "window_median", "missing")
    df["target_imputed"] = df["row_imputed"] | df["vol_source"].isin(estimated) | df["speed_source"].isin(estimated)
    return df


def congestion_from_volume(volume: pd.Series, capacity: pd.Series, config: dict) -> pd.Series:
    """Band V/C = volume / (capacity / 2) using the Data Dictionary rule."""
    bins = [-np.inf, *config["congestion"]["bands"], np.inf]
    ratio = volume / (capacity / 2)
    return pd.cut(ratio, bins=bins, labels=config["congestion"]["labels"], right=False).astype("string")


def recompute_congestion(df: pd.DataFrame, ledger: Ledger, config: dict) -> pd.DataFrame:
    """Recompute every label from cleaned volume, logging filled blanks and any changed label."""
    df = df.copy()
    new = congestion_from_volume(df["traffic_volume"], df["road_capacity"], config)
    old = df["congestion_level"].astype("string")
    keys = traffic_keys(df)
    blank = old.isna()
    changed = old.notna() & (old != new)
    ledger.add("harmonise", "congestion_filled", "traffic", keys[blank], "congestion_level", old[blank], new[blank])
    ledger.add("harmonise", "congestion_changed", "traffic", keys[changed], "congestion_level", old[changed], new[changed])
    df["congestion_level"] = new
    return df


def harmonise_weather(weather: pd.DataFrame, ledger: Ledger, config: dict) -> pd.DataFrame:
    """Map labels to the controlled vocabulary, raising on unknowns, and interpolate null readings per station."""
    label_map = config["weather"]["label_map"]
    df = weather.sort_values(["station_id", "ts"]).drop_duplicates(["station_id", "ts"]).reset_index(drop=True)
    normalised = df["weather_condition"].astype("string").str.strip().str.lower()
    unknown = sorted(set(normalised.dropna()) - set(label_map))
    if unknown:
        raise ValueError(f"Unknown weather labels: {unknown}. Add them to weather.label_map in config.yaml.")
    new = normalised.map(label_map).astype("string")
    new = new.groupby(df["station_id"]).ffill().groupby(df["station_id"]).bfill()
    changed = df["weather_condition"].astype("string") != new
    keys = df["station_id"].astype("string") + "|" + df["ts"].dt.strftime("%Y-%m-%d %H:%M").astype("string")
    ledger.add("harmonise", "weather_label_normalised", "weather", keys[changed], "weather_condition", df.loc[changed, "weather_condition"], new[changed])
    df["weather_condition"] = new

    df["wx_imputed"] = df[["temperature", "visibility", "rainfall"]].isna().any(axis=1)
    for col in ["temperature", "visibility", "rainfall"]:
        missing = df[col].isna()
        df[col] = df.groupby("station_id", group_keys=False)[col].apply(lambda s: s.interpolate(method="linear", limit_direction="both"))
        ledger.add("harmonise", "weather_interpolated", "weather", keys[missing], col, pd.Series([pd.NA] * int(missing.sum()), dtype="string"), df.loc[missing, col].round(3))
    return df


def merge_sources(traffic: pd.DataFrame, weather: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Broadcast each hour's weather to its two windows, join the calendar by date, then drop the join key."""
    df = traffic.copy()
    df["hour_ts"] = df["ts"].dt.floor("h")
    wx = weather[["station_id", "ts", "weather_condition", "temperature", "rainfall", "visibility", "wx_imputed"]].rename(columns={"station_id": "weather_station_id", "ts": "hour_ts"})
    df = df.merge(wx, on=["weather_station_id", "hour_ts"], how="left")
    cal = calendar.copy()
    cal["date"] = cal["date"].dt.strftime("%Y-%m-%d")
    df = df.merge(cal, on="date", how="left")
    df["day_of_week"] = df["ts"].dt.strftime("%a")
    return df.drop(columns=["hour_ts", "weather_station_id"])


def finalise_types(df: pd.DataFrame) -> pd.DataFrame:
    """Set the integer columns the merged schema specifies and order the rows."""
    df = df.copy()
    for col in ["traffic_volume", "vehicle_count", "accident_count", "signal_timing", "road_capacity"]:
        df[col] = df[col].round().astype("int64")
    return df.sort_values(["road_id", "ts"]).reset_index(drop=True)


def final_checks(df: pd.DataFrame, expected_rows: int | None, config: dict) -> None:
    """Assert the PRD 9.1 final checks, failing loudly rather than passing bad data downstream."""
    nulls = df[MODELLING_COLUMNS].isna().sum()
    if nulls.any():
        raise ValueError(f"Nulls remain in modelling columns: {nulls[nulls > 0].to_dict()}")
    if expected_rows is not None and len(df) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows after the merge, found {len(df)}")
    if df.duplicated(["road_id", "ts"]).any():
        raise ValueError("Duplicate road_id + timestamp after the merge")
    limits = config["ranges"]
    if (df["traffic_volume"] < limits["volume_min"]).any() or (df["avg_speed"] > limits["speed_max"]).any() or (df["occupancy"] > limits["occupancy_max"]).any():
        raise ValueError("Out-of-range values remain after cleaning")
    unknown = set(df["weather_condition"]) - set(config["weather"]["vocabulary"])
    if unknown:
        raise ValueError(f"Weather labels outside the vocabulary: {unknown}")


def dataset_hash(df: pd.DataFrame) -> str:
    """Content hash of the dataset, independent of how the Parquet writer lays out bytes."""
    hashed = pd.util.hash_pandas_object(df, index=False).to_numpy()
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def clean_tables(tables: dict, config: dict, stats: dict | None = None, grid: tuple | None = None) -> dict:
    """Run the cleaning stages on validated tables. `stats` reuses saved medians and segment facts for uploads."""
    ledger: Ledger = tables["ledger"]
    stages = [("Raw traffic rows", tables["raw_rows"]["traffic"])]
    traffic = tables["traffic"]
    stages.append(("After validation (M1 rejects)", len(traffic)))

    traffic = deduplicate(traffic, ledger)
    stages.append(("After deduplication", len(traffic)))

    traffic = parse_vehicle_shares(traffic)
    static = stats["static"] if stats else segment_static_table(traffic, config)
    traffic = repair_rules(traffic, static, ledger, config)
    traffic = flag_statistical_outliers(traffic, config)

    start, end = grid or (config["grid"]["start"], config["grid"]["end"])
    traffic = reindex_grid(traffic, static, start, end, ledger, config)
    stages.append(("After reindexing to the full grid", len(traffic)))

    medians = stats["medians"] if stats else window_medians(traffic, config)
    traffic = fill_missing(traffic, medians, ledger, config)
    traffic = recompute_congestion(traffic, ledger, config)

    weather = harmonise_weather(tables["weather"], ledger, config)
    merged = merge_sources(traffic, weather, tables["calendar"])
    merged = finalise_types(merged)
    stages.append(("After merging weather and calendar", len(merged)))
    return {"dataset": merged, "static": static, "medians": medians, "stages": stages, "ledger": ledger, "weather": weather}


def _rule_table(ledger_df: pd.DataFrame) -> str:
    counts = ledger_df.groupby(["stage", "rule", "table", "column"], sort=False).size().reset_index(name="entries")
    lines = ["| Stage | Rule | Table | Column | Entries |", "|---|---|---|---|---:|"]
    lines += [f"| {r.stage} | `{r.rule}` | {r.table} | {r.column} | {r.entries:,} |" for r in counts.itertuples()]
    return "\n".join(lines)


def write_report(result: dict, raw_nulls: dict, dataset_file: str, digest: str, config: dict) -> None:
    """Generate reports/data_quality.md from the ledger, so that every number in it is traceable."""
    df, ledger = result["dataset"], result["ledger"]
    ledger_df = ledger.to_frame()
    after = df[["traffic_volume", "avg_speed", "occupancy", "congestion_level"]].isna().mean().to_dict()
    stages = "\n".join(f"| {name} | {rows:,} |" for name, rows in result["stages"])
    nulls = "\n".join(f"| `{c}` | {raw_nulls[c]:.2%} | {after[c]:.2%} |" for c in raw_nulls)
    flags = {
        "`row_imputed` (window absent from the log)": df["row_imputed"].sum(),
        "`target_imputed` (excluded from targets and metrics)": df["target_imputed"].sum(),
        "`occ_imputed`": df["occ_imputed"].sum(),
        "`stat_outlier` (flagged, not changed)": df["stat_outlier"].sum(),
        "`wx_imputed`": df["wx_imputed"].sum(),
        "`vol_source = vehicle_count`": (df["vol_source"] == "vehicle_count").sum(),
        "`speed_source = travel_time`": (df["speed_source"] == "travel_time").sum(),
    }
    flag_rows = "\n".join(f"| {k} | {int(v):,} | {v / len(df):.2%} |" for k, v in flags.items())
    defects = [
        ("Duplicate log rows", "duplicate_key", "Deduplicated on `road_id + timestamp`, keeping the most complete row."),
        ("Absent windows", "row_added_to_grid", "Reindexed to the full grid. Short gaps time-interpolated, longer gaps use training-date medians. These rows never count as targets or in metrics."),
        ("Null or negative volume", "volume_from_vehicle_count", "Rebuilt from `vehicle_count`, which equals volume on every valid row."),
        ("Null, zero or above-200 speed", "speed_from_travel_time", "Rebuilt as `seg_length_km x 60 / travel_time`."),
        ("Occupancy above 100%", "occupancy_blanked", "Blanked, then time-interpolated within the segment."),
        ("Blank congestion label", "congestion_filled", "Every label recomputed from cleaned volume using the Data Dictionary bands."),
        ("Congestion label disagreeing with the rule", "congestion_changed", "Replaced by the recomputed label."),
        ("Inconsistent weather labels", "weather_label_normalised", "Lower-cased, stripped and mapped through `config.yaml`. An unknown label raises an error."),
        ("Null temperature, visibility or rainfall", "weather_interpolated", "Time-interpolated within the station."),
    ]
    defect_rows = "\n".join(f"| {name} | {ledger.count(rule):,} | {how} |" for name, rule, how in defects)
    text = f"""# FlowCast data-quality report

Generated by `python -m src.pipeline data`; don't edit by hand. Every change below is one row in
`data/interim/quarantine.parquet`, with its stage, rule, key, old value and new value.

- **Dataset:** `{dataset_file}`
- **SHA-256 of contents:** `{digest}`
- **Coverage:** {df["road_id"].nunique()} segments, {df["ts"].min():%d %b %Y} to {df["ts"].max():%d %b %Y}, {config["grid"]["window_minutes"]}-minute windows

## Row counts by stage

| Stage | Rows |
|---|---:|
{stages}

## Defects found and how they were resolved

| Defect | Entries | Resolution |
|---|---:|---|
{defect_rows}

Mixed date formats are parsed explicitly: weather as `DD/MM/YYYY`, traffic and calendar as `YYYY-MM-DD`, all
to tz-aware {config["timezone"]} timestamps. Each hour's weather is applied to both of its 30-minute windows.

Statistical outliers (|z| > {config["cleaning"]["zscore_threshold"]:g} per segment, or outside the {config["cleaning"]["iqr_multiplier"]:g} x IQR fence per segment x hour) are flagged
and never changed. Traffic has real peaks and incidents, and these are legitimate rows, so the field policy is to keep them.

## Null rates before and after

| Column | Raw | Final |
|---|---:|---:|
{nulls}

## Row flags in the final dataset

| Flag | Rows | Share |
|---|---:|---:|
{flag_rows}

## Every rule applied

{_rule_table(ledger_df)}
"""
    path = project_path("reports", config) / "data_quality.md"
    path.write_text(text, encoding="utf-8", newline="\n")


def load_dataset(config: dict | None = None) -> pd.DataFrame:
    """Load the current versioned dataset named by the manifest."""
    cfg = config or load_config()
    processed = project_path("processed", cfg)
    manifest = json.loads((processed / "dataset_manifest.json").read_text(encoding="utf-8"))
    return pd.read_parquet(processed / manifest["file"])


def run(config: dict | None = None) -> pd.DataFrame:
    """Build the versioned dataset, quarantine file and data-quality report from the raw files."""
    cfg = config or load_config()
    tables = load_all(config=cfg)
    result = clean_tables(tables, cfg)
    df = result["dataset"]

    days = (pd.Timestamp(cfg["grid"]["end"]) - pd.Timestamp(cfg["grid"]["start"])).days + 1
    final_checks(df, result["static"].shape[0] * days * cfg["grid"]["windows_per_day"], cfg)

    interim, processed = project_path("interim", cfg), project_path("processed", cfg)
    result["ledger"].to_frame().to_parquet(interim / "quarantine.parquet", index=False)
    result["static"].to_parquet(processed / "segment_static.parquet", index=False)
    result["medians"].to_parquet(processed / "window_medians.parquet", index=False)

    digest = dataset_hash(df)
    name = f"flowcast_v{digest[:12]}.parquet"
    for old in processed.glob("flowcast_v*.parquet"):
        if old.name != name:
            old.unlink()
    df.to_parquet(processed / name, index=False)
    (processed / "dataset_manifest.json").write_text(json.dumps({"file": name, "sha256": digest, "rows": len(df)}, indent=2), encoding="utf-8")

    write_report(result, tables["raw_nulls"], name, digest, cfg)
    logger.info("Wrote %s (%d rows); %d ledger entries", name, len(df), sum(len(p) for p in result["ledger"].parts))
    return df
