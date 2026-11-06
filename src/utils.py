"""Shared helpers: configuration, paths, logging, seeding and hashing."""

import hashlib
import logging
import os
import random
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def load_config(path: str | None = None) -> dict:
    """Load config.yaml from the repo root, or from an explicit path."""
    with open(path or ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def project_path(key: str, config: dict | None = None) -> Path:
    """Return the absolute path for a `paths:` entry, creating the directory if needed."""
    cfg = config or load_config()
    path = ROOT / cfg["paths"][key]
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_logger(name: str) -> logging.Logger:
    """Return a logger that writes one timestamped line per message."""
    logger = logging.getLogger(f"flowcast.{name}")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def seed_everything(seed: int | None = None) -> int:
    """Seed Python, NumPy and (when installed) PyTorch. Returns the seed used."""
    seed = load_config()["seed"] if seed is None else seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass
    return seed


def exit_cleanly(code: int) -> None:
    """End the process once all work is saved, skipping interpreter shutdown.

    On Windows, a process that has trained models can crash in native library cleanup (0xC0000409) after everything
    has finished, which would make a successful run look like a failure. os._exit still runs that cleanup there, so
    Windows needs TerminateProcess.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    logging.shutdown()
    if sys.platform == "win32":
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32")
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.TerminateProcess.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        kernel32.TerminateProcess(kernel32.GetCurrentProcess(), code)
    os._exit(code)


def file_sha256(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of a file, used for dataset versioning and model lineage."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
