"""Model cards and the registry of chosen models (PRD 13.4, plan D13)."""

import json
import subprocess
from pathlib import Path

import numpy as np

from src.utils import ROOT, load_config, project_path


def git_commit() -> str:
    """Current commit, or 'unknown' outside a repository."""
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def dataset_digest(config: dict | None = None) -> str:
    """SHA-256 of the processed dataset, from its manifest."""
    return json.loads((project_path("processed", config) / "dataset_manifest.json").read_text(encoding="utf-8"))["sha256"]


def _plain(value):
    """Make NumPy values JSON-serialisable."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def model_dir(target: str, family: str, version: str, config: dict | None = None) -> Path:
    """models/<target>/<model>/<version>/, created if needed."""
    path = project_path("models", config) / target / family / version
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_card(directory: Path, **fields) -> dict:
    """Write model_card.json, recording what's needed to reproduce and trace this model."""
    cfg = load_config()
    card = {"random_seed": cfg["seed"], "training_window": cfg["splits"]["train"], "dataset_sha256": dataset_digest(), "git_commit": git_commit(), "test_metrics": None}
    card.update(_plain(fields))
    (directory / "model_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8", newline="\n")
    return card


def add_test_metrics(directory: Path, metrics: dict) -> None:
    """Fill in the held-out metrics once, in the final evaluation stage."""
    path = directory / "model_card.json"
    card = json.loads(path.read_text(encoding="utf-8"))
    card["test_metrics"] = _plain(metrics)
    path.write_text(json.dumps(card, indent=2), encoding="utf-8", newline="\n")


def read_registry(config: dict | None = None) -> dict:
    """The record of which models are in service, or an empty record if there isn't one yet."""
    path = project_path("models", config) / "registry.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def write_registry(registry: dict, config: dict | None = None) -> None:
    """Save the record of which models are in service."""
    path = project_path("models", config) / "registry.json"
    path.write_text(json.dumps(_plain(registry), indent=2), encoding="utf-8", newline="\n")
