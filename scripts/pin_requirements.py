"""Rewrite requirements.txt with the exact versions installed in the active environment."""

import re
from importlib.metadata import version
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "requirements.txt"
names = [re.split(r"[=<>]", line)[0].strip() for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")]
header = "# Python 3.11. Setup: python -m venv .venv, activate it, then pip install -r requirements.txt\n"
path.write_text(header + "".join(f"{name}=={version(name)}\n" for name in names), newline="\n")
