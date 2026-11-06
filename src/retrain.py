"""Retrain one classical winner as a new version, and compare it with the one in service.

Run with `python -m src.retrain --target volume`. It writes a separate versioned folder and never replaces the
production model, so a reviewer can compare the two model cards first.
"""

import argparse
import ast
import json
import sys
import time

import pandas as pd

from src.evaluate import split_rows
from src.features import feature_columns, load_features
from src.ml_models import TASKS, train_candidate
from src.model_card import read_registry
from src.utils import ROOT, get_logger, load_config, project_path, seed_everything

logger = get_logger("retrain")
RETRAINABLE = ["volume", "speed", "accident"]


def run(target: str, config: dict | None = None) -> dict:
    """Refit the target's winning family on the training window with its saved hyperparameters."""
    cfg = config or load_config()
    seed_everything(cfg["seed"])
    registry = read_registry(cfg)
    entry = registry["targets"][target]
    family = entry["model"]
    current_path = entry["horizons"]["1"]["path"]
    current = json.loads((ROOT / current_path / "model_card.json").read_text(encoding="utf-8"))

    frame = load_features(1, cfg)
    train, val = split_rows(frame, "train"), split_rows(frame, "val")
    logger.info("Retraining %s (%s) on %d rows", target, family, len(train))
    started = time.time()
    version = f"retrain-{time.strftime('%Y%m%d-%H%M%S')}"
    row, _ = train_candidate(family, target, TASKS[target], train, val, feature_columns(frame), version, cfg, params=ast.literal_eval(str(current["hyperparameters"])))
    new_card = json.loads((ROOT / row["path"] / "model_card.json").read_text(encoding="utf-8"))
    primary = TASKS[target]["primary"]
    result = {
        "target": target,
        "model": family,
        "seconds": round(time.time() - started, 1),
        "primary_metric": primary,
        "current_path": current_path,
        "new_path": row["path"],
        "current_validation": current["validation_metrics"],
        "new_validation": new_card["validation_metrics"],
    }
    pd.DataFrame([result]).to_json(project_path("reports", cfg) / "last_retrain.json", orient="records", indent=2)
    logger.info("Done in %.1fs. Validation %s: current %.4f, new %.4f", result["seconds"], primary, current["validation_metrics"][primary], new_card["validation_metrics"][primary])
    return result


def main(argv: list | None = None) -> int:
    """Retrain the chosen target from the command line."""
    parser = argparse.ArgumentParser(description="Retrain one FlowCast model as a new version")
    parser.add_argument("--target", choices=RETRAINABLE, default="volume")
    run(parser.parse_args(argv).target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
