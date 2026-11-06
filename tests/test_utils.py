import os
import random
import subprocess
import sys

import numpy as np

from src.utils import ROOT, file_sha256, load_config, seed_everything


def test_exit_cleanly_passes_on_the_exit_code_and_flushes_output():
    script = "import sys; from src.utils import exit_cleanly; print('saved', flush=False); exit_cleanly(int(sys.argv[1]))"
    for code in (0, 3):
        result = subprocess.run([sys.executable, "-c", script, str(code)], cwd=ROOT, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(ROOT)})
        assert result.returncode == code
        assert "saved" in result.stdout


def test_config_splits_are_chronological_and_contiguous():
    splits = load_config()["splits"]
    assert splits["train"][1] < splits["val"][0] <= splits["val"][1] < splits["test"][0] <= splits["test"][1]


def test_seed_everything_repeats_draws():
    seed_everything(7)
    first = (random.random(), np.random.rand())
    seed_everything(7)
    assert first == (random.random(), np.random.rand())


def test_file_sha256_changes_with_content(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("a")
    before = file_sha256(path)
    path.write_text("b")
    assert before != file_sha256(path)
