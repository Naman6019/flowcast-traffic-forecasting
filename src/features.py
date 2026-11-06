"""M3: leakage-safe features and targets. A forecast is issued at the end of window t for window t+h.

Each row describes an issue time t. Anything known in advance about the target window (clock, calendar) may
be used; anything measured (traffic, weather) must be complete by the end of window t (plan D4, D5).
"""

import numpy as np
import pandas as pd

from src.clean import load_dataset
from src.utils import get_logger, load_config, project_path

logger = get_logger("features")

ID_COLUMNS = ["road_id", "ts", "target_ts", "split", "target_valid"]
TARGET_COLUMNS = ["y_volume", "y_speed", "y_travel_time", "y_congestion", "y_accident"]
NOT_FEATURES = set(ID_COLUMNS + TARGET_COLUMNS)
FEATURE_GROUPS = {
    "Recent volume": ["vol_t", "vol_t1", "vol_t2", "vol_mean4", "vol_mean8", "vol_std4", "vol_std8", "vc_t", "headroom_t", "occ_t"],
    "Recent speed": ["speed_t", "speed_t1", "speed_t2", "speed_mean4", "speed_mean8", "speed_std4", "speed_std8"],
    "Usual level for the slot": ["vol_profile", "speed_profile", "vol_day", "vol_week", "speed_day", "speed_week"],
    "Today against usual": ["vol_ratio_t", "vol_ratio_mean4", "speed_ratio_t", "speed_ratio_mean4"],
    "Clock": ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend", "is_peak"],
    "Calendar": ["public_holiday", "event_flag", "roadwork_flag", "holiday_x_peak", "event_x_peak"],
    "Weather": ["wx_rainfall", "wx_visibility", "wx_temperature", "wx_rain", "wx_low_vis", "wx_temp_band", "wx_is_clear", "wx_is_cloudy", "wx_is_rain", "wx_is_fog", "rain_x_peak", "low_vis_x_speed"],
    "Segment": ["road_code", "road_capacity", "seg_length_km"],
    "Mix and operations": ["share_hcv_t", "share_2w_t", "signal_timing_t", "accidents_last4"],
}


def _peak_flag(ts: pd.Series, config: dict) -> pd.Series:
    """1 when a window starts inside a configured peak period."""
    clock = ts.dt.strftime("%H:%M")
    flag = pd.Series(False, index=ts.index)
    for start, end in config["peak_windows"]:
        flag |= (clock >= start) & (clock < end)
    return flag.astype("int64")


def split_of(ts: pd.Series, config: dict) -> pd.Series:
    """Name the chronological split each timestamp's date belongs to ('none' if outside all of them)."""
    dates = ts.dt.strftime("%Y-%m-%d")
    out = pd.Series("none", index=ts.index, dtype="string")
    for name in ["train", "val", "test"]:
        start, end = config["splits"][name]
        out = out.where(~((dates >= start) & (dates <= end)), name)
    return out


def slot_profile(df: pd.DataFrame, col: str) -> pd.Series:
    """Mean of strictly earlier values for the same segment, window of day and day type, on ordinary days only.

    Each row reads only earlier dates of its own slot, so this is causal for any horizon below one day.
    """
    ordinary = (df["public_holiday"] == 0) & (df["event_flag"] == 0) & (df["roadwork_flag"] == 0)
    day_type = df["ts"].dt.day_of_week >= 5
    clock = df["ts"].dt.hour * 60 + df["ts"].dt.minute
    values = df[col].astype("float64").where(ordinary)
    profile = values.groupby([df["road_id"], day_type, clock]).transform(lambda s: s.shift(1).expanding().mean())
    # A slot with no ordinary history yet carries forward through the day type's earlier readings of the segment.
    return profile.groupby(df["road_id"]).ffill()


def build_features(df: pd.DataFrame, horizon: int, config: dict | None = None) -> pd.DataFrame:
    """Return one row per segment and issue time, holding features for window t+horizon and its targets."""
    cfg = config or load_config()
    per_day = cfg["grid"]["windows_per_day"]
    df = df.sort_values(["road_id", "ts"]).reset_index(drop=True)
    roads = df.groupby("road_id", sort=False)

    def shifted(col: str, n: int) -> pd.Series:
        return roads[col].shift(n)

    def rolling(col: str, size: int, how: str) -> pd.Series:
        return roads[col].transform(lambda s: getattr(s.rolling(size, min_periods=size), how)())

    target_ts = shifted("ts", -horizon)
    out = pd.DataFrame({"road_id": df["road_id"], "ts": df["ts"], "target_ts": target_ts})
    out["road_code"] = df["road_id"].str[-3:].astype("int64")

    # Clock and calendar of the target window are known in advance.
    minutes = target_ts.dt.hour * 60 + target_ts.dt.minute
    weekday = target_ts.dt.day_of_week
    out["hour_sin"] = np.sin(2 * np.pi * minutes / 1440)
    out["hour_cos"] = np.cos(2 * np.pi * minutes / 1440)
    out["dow_sin"] = np.sin(2 * np.pi * weekday / 7)
    out["dow_cos"] = np.cos(2 * np.pi * weekday / 7)
    out["is_weekend"] = (weekday >= 5).astype("int64")
    out["is_peak"] = _peak_flag(target_ts, cfg)
    for col in ["public_holiday", "event_flag", "roadwork_flag"]:
        out[col] = shifted(col, -horizon)
    out["holiday_x_peak"] = out["public_holiday"] * out["is_peak"]
    out["event_x_peak"] = out["event_flag"] * out["is_peak"]

    # Measurements up to and including window t.
    for col, short in [("traffic_volume", "vol"), ("avg_speed", "speed")]:
        out[f"{short}_t"] = df[col]
        out[f"{short}_t1"] = shifted(col, 1)
        out[f"{short}_t2"] = shifted(col, 2)
        out[f"{short}_day"] = shifted(col, per_day - horizon)
        out[f"{short}_week"] = shifted(col, per_day * 7 - horizon)
        for size in (4, 8):
            out[f"{short}_mean{size}"] = rolling(col, size, "mean")
            out[f"{short}_std{size}"] = rolling(col, size, "std")
        # The usual level for this slot, and how today compares with it. Single-day lags are noisy (EDA finding 1).
        profile = slot_profile(df, col)
        out[f"{short}_profile"] = profile.groupby(df["road_id"]).shift(-horizon)
        ratio = df[col] / profile
        out[f"{short}_ratio_t"] = ratio
        out[f"{short}_ratio_mean4"] = ratio.groupby(df["road_id"]).transform(lambda s: s.rolling(4, min_periods=4).mean())
    out["occ_t"] = df["occupancy"]

    capacity = df["road_capacity"] / 2
    out["vc_t"] = df["traffic_volume"] / capacity
    out["headroom_t"] = capacity - df["traffic_volume"]
    out["road_capacity"] = df["road_capacity"]
    out["seg_length_km"] = df["seg_length_km"]

    # An hour's weather is complete only at the end of that hour. At issue time hh:30 (window hh:00) the last
    # complete hour is hh-1, which window t-1 carries. At issue time hh+1:00 (window hh:30) it is hh, carried by t.
    on_the_half_hour = df["ts"].dt.minute == cfg["grid"]["window_minutes"]
    for col in ["rainfall", "visibility", "temperature", "weather_condition"]:
        out[f"wx_{col}"] = df[col].where(on_the_half_hour, shifted(col, 1))
    out["wx_rain"] = (out["wx_rainfall"] > 0).astype("int64")
    out["wx_low_vis"] = (out["wx_visibility"] < cfg["weather"]["low_visibility_m"]).astype("int64")
    bands = [-np.inf, *cfg["weather"]["temperature_bands"], np.inf]
    out["wx_temp_band"] = pd.cut(out["wx_temperature"], bins=bands, labels=False, right=False).astype("float64")
    for label in cfg["weather"]["vocabulary"]:
        out[f"wx_is_{label.lower()}"] = (out["wx_weather_condition"] == label).fillna(False).astype("int64")
    # Weather rows that precede the first complete hour stay blank rather than reading as "dry".
    known = out["wx_weather_condition"].notna()
    for col in ["wx_rain", "wx_low_vis"] + [f"wx_is_{v.lower()}" for v in cfg["weather"]["vocabulary"]]:
        out[col] = out[col].astype("float64").where(known)
    out = out.drop(columns="wx_weather_condition")

    out["share_hcv_t"] = df["share_hcv"]
    out["share_2w_t"] = df["share_2w"]
    out["signal_timing_t"] = df["signal_timing"]
    out["accidents_last4"] = roads["accident_count"].transform(lambda s: s.rolling(4, min_periods=4).sum())
    out["rain_x_peak"] = out["wx_rain"] * out["is_peak"]
    out["low_vis_x_speed"] = out["wx_low_vis"] * out["speed_t"]

    # Targets.
    out["y_volume"] = shifted("traffic_volume", -horizon)
    out["y_speed"] = shifted("avg_speed", -horizon)
    out["y_travel_time"] = shifted("travel_time", -horizon)
    codes = {label: i for i, label in enumerate(cfg["congestion"]["labels"])}
    out["y_congestion"] = shifted("congestion_level", -horizon).map(codes).astype("float64")
    out["y_accident"] = (shifted("accident_count", -horizon) > 0).astype("float64").where(target_ts.notna())
    target_imputed = shifted("target_imputed", -horizon).astype("boolean").fillna(True).astype(bool)

    # A row belongs to a split only if issue and target fall in it. Rows straddling a boundary stay in the table as
    # sequence history for the deep model, marked 'none'. Warm-up rows lack the weekly lag, and the last rows lack a target.
    issue_split, target_split = split_of(out["ts"], cfg), split_of(target_ts.astype(df["ts"].dtype), cfg)
    warm_up = out["ts"] < out["ts"].min() + pd.Timedelta(days=cfg["splits"]["warmup_days"])
    out["split"] = issue_split.where(issue_split == target_split, "none")
    out["target_valid"] = ~target_imputed
    return out[~warm_up & target_ts.notna()].reset_index(drop=True)


def feature_columns(frame: pd.DataFrame) -> list:
    """The model inputs in a feature table: everything except identifiers and targets."""
    return [c for c in frame.columns if c not in NOT_FEATURES]


def load_features(horizon: int, config: dict | None = None) -> pd.DataFrame:
    """Load the saved feature table for one horizon."""
    return pd.read_parquet(project_path("processed", config) / f"features_h{horizon}.parquet")


def run(config: dict | None = None) -> None:
    """Build and save the feature table for every horizon."""
    cfg = config or load_config()
    df = load_dataset(cfg)
    for horizon in cfg["horizons"]:
        table = build_features(df, horizon, cfg)
        columns = feature_columns(table)
        nulls = table[columns].isna().sum()
        if nulls.any():
            raise ValueError(f"Nulls in features for h={horizon}: {nulls[nulls > 0].to_dict()}")
        table.to_parquet(project_path("processed", cfg) / f"features_h{horizon}.parquet", index=False)
        counts = table.groupby("split")["target_valid"].sum().to_dict()
        logger.info("h=%d: %d features, valid rows per split %s", horizon, len(columns), counts)
