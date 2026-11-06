"""One-command orchestrator: `python -m src.pipeline all` rebuilds everything from the raw files."""

import argparse
import subprocess
import sys
import time
from pathlib import Path

from src import checkpoint
from src.utils import exit_cleanly, get_logger, seed_everything

logger = get_logger("pipeline")


def stage_data() -> None:
    """Validate, clean and merge the raw files."""
    from src.clean import run

    run()


def stage_features() -> None:
    """Build the feature tables."""
    from src.features import run

    run()


def stage_classical() -> None:
    """Tune and compare the classical models on CV and validation."""
    from src.ml_models import run

    run()


def stage_deep() -> None:
    """Train the recurrent models on validation."""
    from src.dl_model import run

    run()


def stage_final() -> None:
    """Fix the production models, then score the test window."""
    from src.final import run

    run()


def stage_real() -> None:
    """Validate the approach on real I-94 traffic."""
    from src.real_data import run

    run()


def stage_notebooks() -> None:
    """Rebuild and run the notebooks."""
    root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, str(root / "scripts" / "build_notebooks.py")], check=True, cwd=root)
    for name in ["01_eda", "02_cleaning", "03_modelling"]:
        subprocess.run([sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute", "--inplace", "--ExecutePreprocessor.timeout=600", f"notebooks/{name}.ipynb"], check=True, cwd=root)


def stage_report() -> None:
    """Export the final report to PDF."""
    root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, str(root / "scripts" / "export_report.py")], check=True, cwd=root)


STAGES = {"data": stage_data, "features": stage_features, "classical": stage_classical, "deep": stage_deep, "final": stage_final, "real": stage_real, "notebooks": stage_notebooks, "report": stage_report}


def main(argv: list | None = None) -> int:
    """Run one stage, or all of them in order."""
    parser = argparse.ArgumentParser(description="FlowCast pipeline")
    parser.add_argument("stage", choices=["all", *STAGES], nargs="?", default="all")
    parser.add_argument("--resume", action="store_true", help="continue an interrupted training stage from its saved progress")
    args = parser.parse_args(argv)
    seed_everything()
    for name in STAGES if args.stage == "all" else [args.stage]:
        started = time.time()
        logger.info("Starting stage: %s%s", name, " (resuming)" if args.resume else "")
        if not args.resume:
            checkpoint.clear(name)
        STAGES[name]()
        logger.info("Finished stage: %s in %.1fs", name, time.time() - started)
    return 0


if __name__ == "__main__":
    exit_cleanly(main())
