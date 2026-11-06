"""Saved progress for the long training stages, so an interrupted run can continue with `--resume`.

Entries live in data/interim/run_cache/<stage>/. A normal run clears its stage first, so results always come from a
fresh computation unless resuming was asked for. Every model is seeded per run, so resuming doesn't change results.
"""

import hashlib
import json
import os
import shutil
from pathlib import Path

from src.utils import project_path


def stage_dir(stage: str, config: dict | None = None) -> Path:
    """The folder holding one stage's saved progress, created if needed."""
    path = project_path("interim", config) / "run_cache" / stage
    path.mkdir(parents=True, exist_ok=True)
    return path


def entry_path(stage: str, key: str, suffix: str = ".json", config: dict | None = None) -> Path:
    """Stable file name for a key. The key's text can be long and hold characters that aren't valid in paths."""
    return stage_dir(stage, config) / (hashlib.sha256(key.encode("utf-8")).hexdigest()[:32] + suffix)


def read(stage: str, key: str, config: dict | None = None) -> dict | None:
    """The saved value for a key, or None if there isn't one."""
    path = entry_path(stage, key, config=config)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def write(stage: str, key: str, value: dict, config: dict | None = None) -> None:
    """Save a value atomically: an interruption mid-write leaves the entry absent rather than corrupt."""
    path = entry_path(stage, key, config=config)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value), encoding="utf-8")
    os.replace(temporary, path)


def clear(stage: str, config: dict | None = None) -> None:
    """Forget a stage's saved progress, so that it starts fresh."""
    shutil.rmtree(stage_dir(stage, config), ignore_errors=True)
