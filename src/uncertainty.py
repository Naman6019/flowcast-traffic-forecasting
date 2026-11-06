"""Confidence for every prediction: split conformal intervals and calibrated class probabilities (plan D12)."""

import numpy as np
import pandas as pd

from src.features import _peak_flag


def segment_groups(static: pd.DataFrame) -> pd.Series:
    """Assign each road to a capacity tercile, indexed by road_id."""
    ranks = static.set_index("road_id")["road_capacity"].rank(method="first")
    return pd.cut(ranks, 3, labels=["low", "mid", "high"]).astype("string")


def conformal_groups(frame: pd.DataFrame, groups: pd.Series, config: dict) -> pd.Series:
    """Calibration cell for each row: peak or off-peak at the target window, times the segment's capacity tercile."""
    peak = _peak_flag(frame["target_ts"], config).map({1: "peak", 0: "off"}).astype("string")
    return peak + "|" + frame["road_id"].map(groups).astype("string")


def conformal_quantile(residuals: np.ndarray, level: float) -> float:
    """Split-conformal half-width: the ceil((n+1) * level)-th smallest absolute residual."""
    ordered = np.sort(np.abs(residuals))
    rank = int(np.ceil((len(ordered) + 1) * level))
    return float(ordered[min(rank, len(ordered)) - 1])


def fit_conformal(y, p, cells: pd.Series, levels: list) -> pd.DataFrame:
    """Half-width per calibration cell and level, fit on validation residuals."""
    residual = np.asarray(y, float) - np.asarray(p, float)
    rows = []
    for cell, positions in cells.reset_index(drop=True).groupby(cells.reset_index(drop=True)).groups.items():
        rows.append({"cell": cell, "n": len(positions), **{f"q{int(level * 100)}": conformal_quantile(residual[positions], level) for level in levels}})
    return pd.DataFrame(rows)


def apply_conformal(p, cells: pd.Series, table: pd.DataFrame, level: float, floor: float = 0.0) -> tuple:
    """Return the lower and upper bounds. A cell not seen in calibration takes that level's widest band."""
    column = f"q{int(level * 100)}"
    half = cells.reset_index(drop=True).map(table.set_index("cell")[column]).astype("float64").fillna(table[column].max()).to_numpy()
    p = np.asarray(p, float)
    return np.maximum(p - half, floor), p + half


def band_probabilities(volume_pred, capacity, residuals, cuts) -> np.ndarray:
    """Congestion class probabilities: the share of validation residuals that land the volume in each V/C band."""
    ordered = np.sort(np.asarray(residuals, float))
    half_capacity = np.asarray(capacity, float) / 2
    prediction = np.asarray(volume_pred, float)
    cumulative = [np.searchsorted(ordered, cut * half_capacity - prediction, side="left") / len(ordered) for cut in cuts]
    edges = np.column_stack([np.zeros(len(prediction)), *cumulative, np.ones(len(prediction))])
    return np.diff(edges, axis=1)


def predictive_entropy(probabilities: np.ndarray) -> np.ndarray:
    """Shannon entropy per row in bits. Zero means the model is certain of one class."""
    p = np.clip(probabilities, 1e-12, 1)
    return -(p * np.log2(p)).sum(axis=1)
