"""M4: analysis tables behind the EDA notebook and the dashboard. Training dates only where a choice depends on them."""

import numpy as np
import pandas as pd

from src import scratch
from src.utils import load_config

NUMERIC_COLUMNS = ["traffic_volume", "avg_speed", "occupancy", "travel_time", "temperature", "rainfall", "visibility"]
VIF_COLUMNS = ["vol_t", "vol_t1", "vol_t2", "vol_day", "vol_week", "vol_mean4", "vol_mean8", "speed_t", "occ_t", "vc_t", "wx_rainfall", "wx_visibility", "wx_temperature"]


def observed(df: pd.DataFrame) -> pd.DataFrame:
    """Rows that were actually measured: absent windows were estimated, so analysis leaves them out."""
    return df[~df["target_imputed"]]


def train_rows(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """Observed rows on training dates. Every modelling decision is drawn from these only."""
    cfg = config or load_config()
    return observed(df)[df["date"] <= cfg["splits"]["train"][1]]


def distribution_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Mean, median, standard deviation, skew and tails for the main numeric fields."""
    rows = observed(df)[NUMERIC_COLUMNS]
    summary = rows.agg(["mean", "median", "std", "min", "max", "skew"]).T
    summary["p01"] = rows.quantile(0.01)
    summary["p99"] = rows.quantile(0.99)
    return summary.round(2)


def daily_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Mean volume and speed per window of day, for weekdays and weekends."""
    rows = observed(df).assign(period=lambda d: np.where(d["ts"].dt.day_of_week >= 5, "Weekend", "Weekday"))
    return rows.groupby(["period", "time"])[["traffic_volume", "avg_speed", "occupancy"]].mean().round(2).reset_index()


def correlation_table(df: pd.DataFrame, columns: list | None = None) -> pd.DataFrame:
    """Pearson correlations computed with the from-scratch covariance."""
    cols = columns or NUMERIC_COLUMNS
    return pd.DataFrame(scratch.correlation(observed(df)[cols].to_numpy(float)), index=cols, columns=cols).round(3)


def vif_table(features: pd.DataFrame) -> pd.DataFrame:
    """Variance inflation factors over training rows, from the inverse of the scratch correlation matrix."""
    rows = features[(features["split"] == "train") & features["target_valid"]][VIF_COLUMNS]
    vif = scratch.variance_inflation(rows.to_numpy(float))
    return pd.DataFrame({"feature": VIF_COLUMNS, "vif": vif.round(1)}).sort_values("vif", ascending=False, ignore_index=True)


def weather_effects(df: pd.DataFrame) -> pd.DataFrame:
    """Mean speed, volume and accident rate by weather, within peak and off-peak hours so clock effects don't confound."""
    rows = observed(df).assign(period=lambda d: np.where(d["time"].between("07:30", "09:59") | d["time"].between("17:00", "19:59"), "Peak", "Off-peak"))
    grouped = rows.groupby(["period", "weather_condition"])
    out = grouped.agg(windows=("traffic_volume", "size"), volume=("traffic_volume", "mean"), speed=("avg_speed", "mean"), visibility=("visibility", "mean"))
    out["accident_rate_pct"] = grouped["accident_count"].apply(lambda s: (s > 0).mean() * 100)
    return out.round(2).reset_index()


def calendar_effects(df: pd.DataFrame) -> pd.DataFrame:
    """Mean volume and speed on holidays, event days, roadwork days and ordinary days, weekdays only."""
    rows = observed(df)
    rows = rows[rows["ts"].dt.day_of_week < 5]
    labels = np.select([rows["public_holiday"] == 1, rows["event_flag"] == 1, rows["roadwork_flag"] == 1], ["Holiday", "Event", "Roadwork"], "Ordinary")
    grouped = rows.groupby(labels)
    out = grouped.agg(days=("date", "nunique"), volume=("traffic_volume", "mean"), speed=("avg_speed", "mean"))
    out["accident_rate_pct"] = grouped["accident_count"].apply(lambda s: (s > 0).mean() * 100)
    return out.round(2).rename_axis("day_type").reset_index()


def accident_profile(df: pd.DataFrame) -> dict:
    """Accident rate per window by hour, weather, V/C band and segment."""
    rows = observed(df).assign(hour=lambda d: d["ts"].dt.hour, any_accident=lambda d: d["accident_count"] > 0)
    rows["vc_band"] = pd.cut(rows["traffic_volume"] / (rows["road_capacity"] / 2), [-np.inf, 0.5, 0.8, 1.0, np.inf], right=False, labels=["<0.5", "0.5-0.8", "0.8-1.0", ">=1.0"])
    return {
        name: (rows.groupby(key, observed=True)["any_accident"].mean() * 100).round(3).rename("accident_rate_pct").reset_index()
        for name, key in {"hour": "hour", "weather": "weather_condition", "vc_band": "vc_band", "segment": "road_id"}.items()
    }


def accident_autocorrelation(df: pd.DataFrame) -> pd.DataFrame:
    """P(accident now | accident in the window k steps back), compared with the base rate."""
    rows = df.sort_values(["road_id", "ts"])
    any_accident = rows["accident_count"] > 0
    out = [{"lag_windows": 0, "accident_rate_pct": any_accident.mean() * 100}]
    for k in (1, 2, 4, 48):
        previous = any_accident.groupby(rows["road_id"]).shift(k).astype("boolean").fillna(False).astype(bool)
        out.append({"lag_windows": k, "accident_rate_pct": any_accident[previous].mean() * 100})
    return pd.DataFrame(out).round(3)


def naive_residuals(features: pd.DataFrame) -> dict:
    """Residuals of the seasonal-naive forecast on training rows: their spread, skew, tails and error by peak."""
    rows = features[(features["split"] == "train") & features["target_valid"]]
    residual = rows["y_volume"] - rows["vol_day"]
    z = (residual - residual.mean()) / residual.std()
    return {
        "mean": float(residual.mean()),
        "std": float(residual.std()),
        "skew": float(residual.skew()),
        "excess_kurtosis": float(residual.kurt()),
        "share_beyond_3_sigma_pct": float((z.abs() > 3).mean() * 100),
        "std_peak": float(residual[rows["is_peak"] == 1].std()),
        "std_off_peak": float(residual[rows["is_peak"] == 0].std()),
    }


def baseline_scores(features: pd.DataFrame, split: str = "val") -> pd.DataFrame:
    """Persistence, yesterday and last-week forecasts of next-window volume, as the numbers any model must beat."""
    rows = features[(features["split"] == split) & features["target_valid"]]
    y = rows["y_volume"].to_numpy(float)
    return pd.DataFrame(
        [
            {"baseline": name, "rmse": scratch.rmse(y, rows[col].to_numpy(float)), "mape_pct": scratch.mape(y, rows[col].to_numpy(float)) * 100}
            for name, col in {"Persistence (t)": "vol_t", "Same window yesterday": "vol_day", "Same window last week": "vol_week"}.items()
        ]
    ).round(2)


def segment_coupling(df: pd.DataFrame) -> float:
    """Median correlation between segments' volume residuals after removing each one's own window x weekday profile."""
    rows = observed(df).assign(weekday=lambda d: d["ts"].dt.day_of_week)
    profile = rows.groupby(["road_id", "weekday", "time"])["traffic_volume"].transform("mean")
    wide = rows.assign(residual=rows["traffic_volume"] - profile).pivot(index="ts", columns="road_id", values="residual")
    corr = wide.corr().to_numpy()
    return float(np.median(corr[np.triu_indices_from(corr, k=1)]))
