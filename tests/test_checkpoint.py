import numpy as np
import pandas as pd
import pytest
import torch

from src import checkpoint, dl_model, ml_models
from src.dl_model import SequenceData, SequenceScaler, input_columns, train_or_resume
from tests.test_dl_model import SPEC, feature_table, small_config


@pytest.fixture
def isolated(config, tmp_path, monkeypatch):
    """Point saved progress at a temporary folder, leaving any real run's progress alone."""
    monkeypatch.setattr(checkpoint, "stage_dir", lambda stage, cfg=None: _folder(tmp_path, stage))
    return config


def _folder(root, stage):
    path = root / stage
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_read_returns_what_was_written_and_none_otherwise(isolated):
    assert checkpoint.read("classical", "missing") is None
    checkpoint.write("classical", "key with | odd [characters]", {"scores": [1.0, 2.0]})
    assert checkpoint.read("classical", "key with | odd [characters]") == {"scores": [1.0, 2.0]}


def test_no_temporary_file_survives_a_write(isolated):
    checkpoint.write("classical", "a", {"x": 1})
    assert not list(checkpoint.stage_dir("classical").glob("*.tmp"))


def test_clear_forgets_a_stage(config, tmp_path, monkeypatch):
    monkeypatch.setattr(checkpoint, "project_path", lambda key, cfg=None: tmp_path)
    checkpoint.write("classical", "a", {"x": 1})
    checkpoint.clear("classical")
    assert checkpoint.read("classical", "a") is None


def test_finished_trials_are_not_repeated_after_an_interruption(isolated, monkeypatch):
    calls = []

    def fake_cross_validate(family, kind, params, *args):
        calls.append(params["C"])
        if len(calls) == 2:
            raise KeyboardInterrupt
        return [float(params["C"])] * 2

    monkeypatch.setattr(ml_models, "cross_validate", fake_cross_validate)
    monkeypatch.setattr(ml_models, "dataset_digest", lambda cfg=None: "digest")
    config = {**isolated, "tuning": {**isolated["tuning"], "trials": {"logistic": 3}}}
    task = {"kind": "congestion", "primary": "macro_f1", "column": "y_congestion"}
    with pytest.raises(KeyboardInterrupt):
        ml_models.tune("logistic", task, pd.DataFrame(), [], config)
    done_before_interruption = calls[0]

    calls.clear()
    monkeypatch.setattr(ml_models, "cross_validate", lambda family, kind, params, *args: calls.append(params["C"]) or [float(params["C"])] * 2)
    best, scores = ml_models.tune("logistic", task, pd.DataFrame(), [], config)
    assert done_before_interruption not in calls
    assert len(calls) == 2
    assert best == {"C": 1.0}
    assert scores == [1.0, 1.0]


def test_resuming_a_deep_run_gives_the_same_model_without_training(isolated, monkeypatch):
    config = small_config(isolated)
    frame = feature_table(config)
    columns = input_columns(frame)
    data = SequenceData(frame, SequenceScaler().fit(frame[frame["split"] == "train"], columns), 4)
    spec = {**SPEC, "length": 4}
    first, history = train_or_resume(spec, data, "trial1", "v1", config)

    def must_not_train(*args, **kwargs):
        raise AssertionError("A finished run was trained again")

    monkeypatch.setattr(dl_model, "train_model", must_not_train)
    second, resumed = train_or_resume(spec, data, "trial1", "v1", config)
    for a, b in zip(first.state_dict().values(), second.state_dict().values(), strict=True):
        assert torch.equal(a, b)
    assert np.allclose(history["val_loss"], resumed["val_loss"])
