import numpy as np
import pandas as pd
import pytest
import torch

from src.dl_model import (
    SeqForecaster,
    SequenceData,
    SequenceScaler,
    input_columns,
    predict,
    resolve_device,
    train_model,
)


def feature_table(config, days=12, roads=(1, 2)):
    """A small feature table shaped like features_h1. Volume follows a noiseless daily pattern."""
    frames = []
    times = pd.date_range("2025-04-10", periods=48 * days, freq="30min", tz=config["timezone"])
    for code in roads:
        n = len(times)
        phase = np.arange(n) % 48
        volume = 300 + 100 * np.sin(2 * np.pi * phase / 48) + 20 * code
        frames.append(
            pd.DataFrame(
                {
                    "road_id": f"NL-{code:03d}",
                    "ts": times,
                    "target_ts": times + pd.Timedelta(minutes=30),
                    "road_code": code,
                    "vol_t": volume,
                    "occ_t": 30.0 + phase,
                    "hour_sin": np.sin(2 * np.pi * phase / 48),
                    "vol_profile": 300.0,
                    "road_capacity": 2000,
                    "y_volume": np.roll(volume, -1),
                    "y_congestion": 0,
                    "split": ["train" if t.strftime("%Y-%m-%d") <= "2025-04-15" else "val" for t in times],
                    "target_valid": True,
                    "y_speed": 40.0,
                    "y_travel_time": 2.0,
                    "y_accident": 0.0,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def build(config, length=4, **kwargs):
    frame = feature_table(config, **kwargs)
    columns = input_columns(frame)
    train = frame[frame["split"] == "train"]
    return SequenceData(frame, SequenceScaler().fit(train, columns), length), columns


SPEC = {"cell": "gru", "hidden": 16, "layers": 1, "dropout": 0.0, "learning_rate": 0.01, "batch_size": 64}


def small_config(config, **overrides):
    return {**config, "deep": {**config["deep"], "max_epochs": 3, **overrides}}


def test_road_code_is_not_an_input(config):
    assert "road_code" not in input_columns(feature_table(config))


def test_scaler_uses_training_rows_only(config):
    frame = feature_table(config)
    columns = input_columns(frame)
    train = frame[frame["split"] == "train"]
    baseline = SequenceScaler().fit(train, columns)
    changed = frame.copy()
    changed.loc[changed["split"] == "val", "vol_t"] = 1e9
    after = SequenceScaler().fit(changed[changed["split"] == "train"], columns)
    assert np.array_equal(baseline.shift, after.shift)
    assert np.array_equal(baseline.scale, after.scale)


def test_bounded_columns_are_min_max_scaled_into_unit_range(config):
    frame = feature_table(config)
    columns = input_columns(frame)
    train = frame[frame["split"] == "train"]
    scaled = SequenceScaler().fit(train, columns).transform(train)
    occupancy = scaled[:, columns.index("occ_t")]
    assert occupancy.min() == pytest.approx(0.0)
    assert occupancy.max() == pytest.approx(1.0)
    volume = scaled[:, columns.index("vol_t")]
    assert volume.mean() == pytest.approx(0.0, abs=1e-5)
    assert volume.std() == pytest.approx(1.0, abs=1e-5)


def test_batch_shape_and_window_ends_at_the_issue_row(config):
    data, columns = build(config, length=4)
    positions = data.positions("val")[:5]
    x, segment = data.batch(positions)
    assert x.shape == (5, 4, len(columns))
    assert segment.shape == (5,)
    assert torch.equal(x[:, -1], torch.from_numpy(data.X[positions]))
    assert torch.equal(x[:, 0], torch.from_numpy(data.X[positions - 3]))


def test_every_input_is_before_the_target_and_in_one_segment(config):
    data, _ = build(config, length=8)
    frame = data.frame
    for split in ("train", "val"):
        for position in data.positions(split)[::50]:
            window = frame.iloc[position - 7 : position + 1]
            assert window["road_id"].nunique() == 1
            assert window["ts"].max() < frame.loc[position, "target_ts"]
            assert (window["ts"].diff().dropna() == pd.Timedelta(minutes=30)).all()


def test_positions_exclude_short_history_and_estimated_targets(config):
    frame = feature_table(config)
    frame.loc[(frame["road_code"] == 1) & (frame["ts"] == frame["ts"].min() + pd.Timedelta(days=8)), "target_valid"] = False
    columns = input_columns(frame)
    data = SequenceData(frame, SequenceScaler().fit(frame[frame["split"] == "train"], columns), 4)
    positions = data.positions("val")
    rows = data.frame.iloc[positions]
    assert rows["target_valid"].all()
    assert len(data.positions("train")) == 2 * (48 * 6 - 3)


def test_broken_runs_do_not_form_windows(config):
    frame = feature_table(config, roads=(1,))
    gap = frame["ts"] == frame["ts"].min() + pd.Timedelta(days=8)
    frame = frame[~gap].reset_index(drop=True)
    columns = input_columns(frame)
    data = SequenceData(frame, SequenceScaler().fit(frame[frame["split"] == "train"], columns), 4)
    ts = data.frame["ts"]
    for position in data.positions("val"):
        assert ts.iloc[position] - ts.iloc[position - 3] == pd.Timedelta(minutes=90)


@pytest.mark.parametrize(("cell", "bidirectional", "multitask", "layers"), [("lstm", False, False, 1), ("gru", False, True, 2), ("lstm", True, True, 2)])
def test_forward_shapes(cell, bidirectional, multitask, layers):
    model = SeqForecaster(5, 3, cell=cell, hidden=8, layers=layers, bidirectional=bidirectional, multitask=multitask)
    volume, logits = model(torch.zeros(7, 4, 5), torch.zeros(7, dtype=torch.long))
    assert volume.shape == (7,)
    assert (logits.shape == (7, 4)) if multitask else (logits is None)


def test_a_tiny_batch_can_be_memorised():
    torch.manual_seed(0)
    model = SeqForecaster(3, 2, cell="gru", hidden=32, dropout=0.0)
    x, segment, y = torch.randn(8, 4, 3), torch.zeros(8, dtype=torch.long), torch.randn(8)
    optimiser = torch.optim.Adam(model.parameters(), lr=0.02)
    for _ in range(300):
        optimiser.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x, segment)[0], y)
        loss.backward()
        optimiser.step()
    assert loss.item() < 1e-3


def test_same_seed_gives_the_same_weights(config):
    data, _ = build(config)
    cfg = small_config(config)
    first, _ = train_model(SPEC, data, cfg)
    second, _ = train_model(SPEC, data, cfg)
    for a, b in zip(first.state_dict().values(), second.state_dict().values(), strict=True):
        assert torch.equal(a, b)


def test_training_reduces_loss_and_restores_the_best_weights(config):
    data, _ = build(config)
    model, history = train_model(SPEC, data, small_config(config, max_epochs=12))
    assert history["train_loss"].iloc[-1] < history["train_loss"].iloc[0]
    positions = data.positions("val")
    forecast, _ = predict(model, data, positions)
    assert np.sqrt(np.mean((forecast - data.y[positions]) ** 2)) < np.sqrt(np.mean((data.base[positions] - data.y[positions]) ** 2))


def test_device_setting_is_respected():
    assert resolve_device({"deep": {"device": "cpu"}}) == torch.device("cpu")
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert resolve_device({"deep": {"device": "auto"}}).type == expected


def test_saved_weights_load_on_the_cpu_regardless_of_the_training_device(config, tmp_path):
    data, _ = build(config)
    model, _ = train_model(SPEC, data, small_config(config, max_epochs=1))
    path = tmp_path / "model.pt"
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, path)
    state = torch.load(path, weights_only=True, map_location="cpu")
    assert all(v.device.type == "cpu" for v in state.values())
    fresh = SeqForecaster(data.X.shape[1], data.n_segments, cell="gru", hidden=16, dropout=0.0)
    fresh.load_state_dict(state)
    positions = data.positions("val")[:5]
    assert np.allclose(predict(fresh, data, positions)[0], predict(model.cpu(), data, positions)[0])


def test_forecast_is_in_vehicles_and_uses_the_profile_base(config):
    data, _ = build(config)
    model = SeqForecaster(data.X.shape[1], data.n_segments)
    with torch.no_grad():
        for parameter in model.head.parameters():
            parameter.zero_()
    forecast, _ = predict(model, data, data.positions("val")[:10])
    assert np.allclose(forecast, data.base[data.positions("val")[:10]])
