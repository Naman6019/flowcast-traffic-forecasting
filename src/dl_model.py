"""M6: LSTM, GRU and BiLSTM forecasters built and trained from scratch, with no pretrained weights (FR-09).

One global model covers all segments. Each step of a sequence is one feature row, so every step only holds what
was known at its own issue time. The network predicts the scaled residual over the slot profile, and the test
window is not read here: held-out scores come from src/final.py. Only files this pipeline wrote are loaded.
"""

import copy
import json
import os
import time

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

from src import checkpoint, scratch
from src.evaluate import congestion_metrics, regression_metrics
from src.features import feature_columns, load_features
from src.ml_models import BOUNDED, sample_params, save_conformal
from src.model_card import dataset_digest, model_dir, read_registry, write_card, write_registry
from src.utils import get_logger, load_config, project_path, seed_everything

logger = get_logger("dl_model")

INPUT_EXCLUDED = {"road_code"}


def resolve_device(config: dict | None = None) -> torch.device:
    """The configured device. 'auto' uses CUDA when a CUDA build of PyTorch can see a GPU, and CPU otherwise."""
    wanted = (config or load_config())["deep"]["device"]
    if wanted == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(wanted)


def input_columns(frame: pd.DataFrame) -> list:
    """Per-step inputs. The segment reaches the network through its embedding instead."""
    return [c for c in feature_columns(frame) if c not in INPUT_EXCLUDED]


class SequenceScaler:
    """Z-score for unbounded features, min-max for bounded ones, plus per-segment target scale. Fit on training rows only."""

    def fit(self, train: pd.DataFrame, columns: list) -> "SequenceScaler":
        """Learn each column's scaling and each segment's target spread from training rows."""
        self.columns = columns
        values = train[columns].to_numpy(float)
        self.shift, self.scale = scratch.zscore_fit(values)
        bounded = [i for i, c in enumerate(columns) if c in BOUNDED]
        low, high = scratch.minmax_fit(values[:, bounded])
        self.shift[bounded], self.scale[bounded] = low, high - low
        self.segment_std = train.groupby("road_code")["y_volume"].std().to_dict()
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        """Scale feature rows with the parameters learned on training rows."""
        return ((frame[self.columns].to_numpy(float) - self.shift) / self.scale).astype(np.float32)

    def target_std(self, road_codes: np.ndarray) -> np.ndarray:
        """The training-window volume spread for each row's segment."""
        return np.array([self.segment_std[c] for c in road_codes], dtype=np.float32)


class SequenceData:
    """Feature rows of every segment in time order, from which sliding windows are cut by position."""

    def __init__(self, frame: pd.DataFrame, scaler: SequenceScaler, length: int, step_minutes: int | None = None):
        frame = frame.sort_values(["road_id", "ts"]).reset_index(drop=True)
        self.frame = frame
        self.length = length
        self.X = scaler.transform(frame)
        self.road_codes = frame["road_code"].to_numpy()
        self.segment_index = np.searchsorted(np.unique(self.road_codes), self.road_codes)
        self.n_segments = int(self.segment_index.max()) + 1
        self.base = frame["vol_profile"].to_numpy(np.float32)
        self.std = scaler.target_std(self.road_codes)
        self.y = frame["y_volume"].to_numpy(np.float32)
        self.y_scaled = (self.y - self.base) / self.std
        self.congestion = frame["y_congestion"].to_numpy(np.int64)
        step = pd.Timedelta(minutes=step_minutes) if step_minutes else frame["ts"].diff().mode().iat[0]
        same_road_and_step = (frame["road_id"] == frame["road_id"].shift()) & (frame["ts"].diff() == step)
        run_start = (~same_road_and_step).cumsum()
        self.history = frame.groupby(run_start).cumcount().to_numpy() + 1

    def positions(self, split: str) -> np.ndarray:
        """Rows usable as targets in a split: measured target, and a full unbroken window of history behind it."""
        rows = (self.frame["split"] == split) & self.frame["target_valid"] & (self.history >= self.length)
        return np.flatnonzero(rows)

    def batch(self, positions: np.ndarray) -> tuple:
        """Windows ending at each position: inputs of shape (batch, length, features) and segment indices."""
        offsets = np.arange(self.length - 1, -1, -1)
        windows = positions[:, None] - offsets
        return torch.from_numpy(self.X[windows]), torch.from_numpy(self.segment_index[positions])


class SeqForecaster(nn.Module):
    """Recurrent core, then dense heads on the final hidden state: volume, and optionally congestion class (PRD 12.1)."""

    def __init__(self, n_features: int, n_segments: int, cell: str = "lstm", hidden: int = 64, layers: int = 1, dropout: float = 0.1, embedding_dim: int = 4, bidirectional: bool = False, multitask: bool = False):
        super().__init__()
        self.embedding = nn.Embedding(n_segments, embedding_dim)
        recurrent = {"lstm": nn.LSTM, "gru": nn.GRU}[cell]
        self.rnn = recurrent(n_features + embedding_dim, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0.0, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)
        out = hidden * (2 if bidirectional else 1)
        self.head = nn.Linear(out, 1)
        self.congestion_head = nn.Linear(out, 4) if multitask else None

    def forward(self, x: torch.Tensor, segment: torch.Tensor) -> tuple:
        """Return scaled volume residuals, and congestion logits for the multi-task variant."""
        embedded = self.embedding(segment).unsqueeze(1).expand(-1, x.shape[1], -1)
        output, _ = self.rnn(torch.cat([x, embedded], dim=-1))
        final = self.dropout(output[:, -1])
        logits = self.congestion_head(final) if self.congestion_head is not None else None
        return self.head(final).squeeze(-1), logits


def predict(model: SeqForecaster, data: SequenceData, positions: np.ndarray, batch_size: int = 4096) -> tuple:
    """Forecast volume in vehicles, and congestion logits for the multi-task variant."""
    model.eval()
    device = next(model.parameters()).device
    volumes, all_logits = [], []
    with torch.no_grad():
        for start in range(0, len(positions), batch_size):
            part = positions[start : start + batch_size]
            x, segment = data.batch(part)
            volume, logits = model(x.to(device), segment.to(device))
            volumes.append(volume.cpu().numpy() * data.std[part] + data.base[part])
            if logits is not None:
                all_logits.append(logits.cpu().numpy())
    return np.concatenate(volumes), (np.concatenate(all_logits) if all_logits else None)


def _loss(model, data, part, alpha: float, device: torch.device) -> torch.Tensor:
    x, segment = data.batch(part)
    volume, logits = model(x.to(device), segment.to(device))
    loss = nn.functional.mse_loss(volume, torch.from_numpy(data.y_scaled[part]).to(device))
    if logits is not None:
        loss = loss + alpha * nn.functional.cross_entropy(logits, torch.from_numpy(data.congestion[part]).to(device))
    return loss


def build_model(spec: dict, data: SequenceData, config: dict) -> SeqForecaster:
    """An untrained network for a configuration, on the configured device."""
    return SeqForecaster(
        data.X.shape[1],
        data.n_segments,
        cell=spec["cell"],
        hidden=spec["hidden"],
        layers=spec["layers"],
        dropout=spec["dropout"],
        embedding_dim=config["deep"]["embedding_dim"],
        bidirectional=spec.get("bidirectional", False),
        multitask=spec.get("multitask", False),
    ).to(resolve_device(config))


def train_or_resume(spec: dict, data: SequenceData, label: str, version: str, config: dict, stage: str = "deep") -> tuple:
    """Train one configuration, or reload it if a previous run already finished it."""
    key = f"{version}|{label}|{sorted(spec.items())}"
    weights = checkpoint.entry_path(stage, key, ".pt", config)
    saved = checkpoint.read(stage, key, config)
    if saved is not None and weights.exists():
        model = build_model(spec, data, config)
        model.load_state_dict(torch.load(weights, weights_only=True, map_location=resolve_device(config)))
        logger.info("%s: resumed from saved progress", label)
        return model, pd.DataFrame(saved["history"])
    model, history = train_model(spec, data, config)
    temporary = weights.with_suffix(".tmp")
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, temporary)
    os.replace(temporary, weights)
    checkpoint.write(stage, key, {"history": history.to_dict("records")}, config)
    return model, history


def train_model(spec: dict, data: SequenceData, config: dict, seed: int | None = None) -> tuple:
    """Adam with early stopping on validation loss, restoring the best weights. LR falls on plateau; gradients are clipped."""
    deep = config["deep"]
    seed = seed_everything(config["seed"] if seed is None else seed)
    # CUDA recurrent layers are only repeatable with this workspace setting, which must be set before they first run.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    device = resolve_device(config)
    model = build_model(spec, data, config)
    optimiser = torch.optim.Adam(model.parameters(), lr=spec["learning_rate"], weight_decay=deep["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, factor=0.5, patience=3)
    train_positions, val_positions = data.positions("train"), data.positions("val")
    rng = np.random.default_rng(seed)
    alpha = deep["multitask_alpha"]

    best_loss, best_state, stale, history = float("inf"), None, 0, []
    for epoch in range(deep["max_epochs"]):
        started = time.time()
        model.train()
        order = rng.permutation(train_positions)
        total = 0.0
        for start in range(0, len(order), spec["batch_size"]):
            part = order[start : start + spec["batch_size"]]
            optimiser.zero_grad()
            loss = _loss(model, data, part, alpha, device)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), deep["grad_clip"])
            optimiser.step()
            total += loss.item() * len(part)

        model.eval()
        with torch.no_grad():
            val_loss = sum(float(_loss(model, data, val_positions[s : s + 4096], alpha, device)) * len(val_positions[s : s + 4096]) for s in range(0, len(val_positions), 4096)) / len(val_positions)
        scheduler.step(val_loss)
        history.append({"epoch": epoch + 1, "train_loss": total / len(order), "val_loss": val_loss, "lr": optimiser.param_groups[0]["lr"], "seconds": round(time.time() - started, 1)})
        if val_loss < best_loss - 1e-5:
            best_loss, best_state, stale = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= deep["patience"]:
                break
    model.load_state_dict(best_state)
    return model, pd.DataFrame(history)


def plot_loss_curves(histories: dict, path) -> None:
    """Train and validation loss per run, to confirm convergence rather than memorisation (PRD 12.3)."""
    columns = min(3, len(histories))
    rows = int(np.ceil(len(histories) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(5 * columns, 3 * rows), squeeze=False)
    for ax, (name, history) in zip(axes.flat, histories.items(), strict=False):
        ax.plot(history["epoch"], history["train_loss"], label="Train")
        ax.plot(history["epoch"], history["val_loss"], label="Validation")
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=8)
    for ax in list(axes.flat)[len(histories) :]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_model(model: SeqForecaster, scaler: SequenceScaler, spec: dict, history: pd.DataFrame, name: str, horizon: int, version: str, metrics: dict, columns: list, config: dict):
    """Persist weights, the scaler, the loss curve and a model card."""
    directory = model_dir("volume", name, f"h{horizon}-{version}", config)
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, directory / "model.pt")
    joblib.dump(scaler, directory / "scaler.joblib")
    history.to_csv(directory / "loss_curve.csv", index=False, lineterminator="\n")
    write_card(directory, model=name, target="volume", horizon=horizon, features=columns, hyperparameters=spec, epochs_run=len(history), validation_metrics=metrics, pretrained_weights=False, trained_on=str(next(model.parameters()).device))
    return directory


def load_model(path, n_segments: int) -> tuple:
    """Rebuild a saved network from its card for inference without retraining (FR-12)."""
    directory = project_path("models").parent / path
    card = json.loads((directory / "model_card.json").read_text(encoding="utf-8"))
    spec = card["hyperparameters"]
    scaler = joblib.load(directory / "scaler.joblib")
    model = SeqForecaster(len(scaler.columns), n_segments, cell=spec["cell"], hidden=spec["hidden"], layers=spec["layers"], dropout=spec["dropout"], embedding_dim=load_config()["deep"]["embedding_dim"], bidirectional=spec.get("bidirectional", False), multitask=spec.get("multitask", False))
    model.load_state_dict(torch.load(directory / "model.pt", weights_only=True, map_location="cpu"))
    model.to(resolve_device())
    model.eval()
    return model, scaler, spec


def run(config: dict | None = None) -> None:
    """Random search at h=1, compare LSTM, GRU and BiLSTM, add the multi-task head, then train h=2-4."""
    cfg = config or load_config()
    deep = cfg["deep"]
    version = dataset_digest(cfg)[:8]
    static = pd.read_parquet(project_path("processed", cfg) / "segment_static.parquet")
    frame = load_features(1, cfg)
    columns = input_columns(frame)
    train_rows = frame[(frame["split"] == "train") & frame["target_valid"]]
    scaler = SequenceScaler().fit(train_rows, columns)

    def evaluate(model, data):
        positions = data.positions("val")
        rows = data.frame.iloc[positions]
        forecast, logits = predict(model, data, positions)
        metrics = regression_metrics(rows["y_volume"], forecast)
        if logits is not None:
            metrics["head_macro_f1"] = congestion_metrics(rows["y_congestion"], logits.argmax(axis=1))["macro_f1"]
        ratio = forecast / (rows["road_capacity"].to_numpy() / 2)
        metrics["band_macro_f1"] = congestion_metrics(rows["y_congestion"], np.digitize(ratio, cfg["congestion"]["bands"]))["macro_f1"]
        return metrics, forecast, rows

    space = {
        "cell": ["lstm", "gru"],
        "length": deep["sequence_lengths"],
        "hidden": deep["hidden_sizes"],
        "layers": deep["num_layers"],
        "dropout": deep["dropouts"],
        "learning_rate": deep["learning_rates"],
        "batch_size": deep["batch_sizes"],
    }
    results, histories = [], {}

    def attempt(spec: dict, label: str, stage: str) -> dict:
        data = SequenceData(frame, scaler, spec["length"])
        model, history = train_or_resume(spec, data, label, version, cfg)
        metrics, _, _ = evaluate(model, data)
        row = {"stage": stage, "label": label, **spec, **{f"val_{k}": v for k, v in metrics.items()}, "epochs": len(history), "fit_s": round(float(history["seconds"].sum()), 1)}
        results.append(row)
        histories[label] = history
        logger.info("%s: val rmse %.2f, mape %.2f%%, %d epochs in %.0fs (%s)", label, metrics["rmse"], metrics["mape_pct"], len(history), row["fit_s"], spec)
        pd.DataFrame(results).to_csv(project_path("reports", cfg) / "deep_search.csv", index=False, lineterminator="\n")
        return {"row": row, "model": model, "data": data, "history": history, "metrics": metrics, "spec": spec}

    trials = [attempt(spec, f"trial{i + 1}-{spec['cell']}", "search") for i, spec in enumerate(sample_params(space, cfg["tuning"]["deep_trials"], cfg["seed"]))]
    best = min(trials, key=lambda t: t["metrics"]["rmse"])
    logger.info("Best trial: %s", best["spec"])

    # The same configuration for each architecture the PRD compares (plan D10).
    candidates = {best["spec"]["cell"]: best}
    other = "gru" if best["spec"]["cell"] == "lstm" else "lstm"
    candidates[other] = attempt({**best["spec"], "cell": other}, other, "compare")
    candidates["bilstm"] = attempt({**best["spec"], "cell": "lstm", "bidirectional": True}, "bilstm", "compare")
    chosen_name = min(("lstm", "gru"), key=lambda n: candidates[n]["metrics"]["rmse"])
    chosen = candidates[chosen_name]
    multitask = attempt({**chosen["spec"], "multitask": True}, f"{chosen_name}_multitask", "compare")
    candidates[f"{chosen_name}_multitask"] = multitask

    paths = {}
    for name, result in candidates.items():
        directory = save_model(result["model"], scaler, result["spec"], result["history"], name, 1, version, result["metrics"], columns, cfg)
        paths[name] = directory.relative_to(project_path("models", cfg).parent).as_posix()

    registry = {"model": chosen_name, "multitask": f"{chosen_name}_multitask", "models": paths, "horizons": {}}

    # The chosen architecture at every horizon, with conformal bands fit on validation residuals.
    for horizon in cfg["horizons"]:
        if horizon == 1:
            result, frame_h, scaler_h = chosen, frame, scaler
        else:
            frame_h = load_features(horizon, cfg)
            scaler_h = SequenceScaler().fit(frame_h[(frame_h["split"] == "train") & frame_h["target_valid"]], columns)
            data = SequenceData(frame_h, scaler_h, chosen["spec"]["length"])
            model, history = train_or_resume(chosen["spec"], data, f"{chosen_name}_h{horizon}", version, cfg)
            metrics, _, _ = evaluate(model, data)
            result = {"model": model, "data": data, "history": history, "metrics": metrics, "spec": chosen["spec"]}
            histories[f"{chosen_name} h={horizon}"] = history
            results.append({"stage": "horizon", "label": f"{chosen_name}_h{horizon}", **chosen["spec"], **{f"val_{k}": v for k, v in metrics.items()}, "epochs": len(history)})
            logger.info("%s h=%d: val rmse %.2f, mape %.2f%%", chosen_name, horizon, metrics["rmse"], metrics["mape_pct"])
        directory = save_model(result["model"], scaler_h, result["spec"], result["history"], chosen_name, horizon, version, result["metrics"], columns, cfg)
        _, forecast, rows = evaluate(result["model"], result["data"])
        conformal = save_conformal(rows["y_volume"].to_numpy(), forecast, rows, static, directory, cfg)
        np.save(directory / "val_residuals.npy", rows["y_volume"].to_numpy() - forecast)
        registry["horizons"][str(horizon)] = {"path": directory.relative_to(project_path("models", cfg).parent).as_posix(), "conformal": conformal, "val_rmse": result["metrics"]["rmse"], "val_mape_pct": result["metrics"]["mape_pct"]}

    pd.DataFrame(results).to_csv(project_path("reports", cfg) / "deep_search.csv", index=False, lineterminator="\n")
    plot_loss_curves(histories, project_path("figures", cfg) / "deep_loss_curves.png")
    write_registry({**read_registry(cfg), "deep": registry}, cfg)
