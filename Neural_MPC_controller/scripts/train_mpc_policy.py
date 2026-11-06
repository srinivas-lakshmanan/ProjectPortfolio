#!/usr/bin/env python3
"""Train a feed-forward policy to imitate the MPC from a data-dump CSV."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


STATE_NAMES = ("X", "Y", "psi", "vx", "vy", "r", "w_fl", "w_fr", "w_rl", "w_rr")
BODY_STATE_NAMES = ("vx", "vy", "r", "w_fl", "w_fr", "w_rl", "w_rr")
INPUT_NAMES = ("delta", "T_fl", "T_fr", "T_rl", "T_rr")
OUTPUT_NAMES = tuple(f"expert_{name}" for name in INPUT_NAMES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path, help="MPC data-dump CSV")
    parser.add_argument("--output", type=Path, default=Path("mpc_policy.pt"),
                        help="Output PyTorch checkpoint (default: mpc_policy.pt)")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-sizes", default="256,256,128",
                        help="Comma-separated hidden layer widths")
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        parser.error("--epochs, --batch-size, and --patience must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("--learning-rate must be finite and positive")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("--weight-decay must be finite and nonnegative")
    if (not math.isfinite(args.validation_fraction) or
            not math.isfinite(args.test_fraction) or
            args.validation_fraction <= 0 or args.test_fraction <= 0 or
            args.validation_fraction + args.test_fraction >= 1):
        parser.error("validation/test fractions must be positive and sum to less than 1")
    try:
        args.hidden_sizes = [int(size) for size in args.hidden_sizes.split(",")]
    except ValueError:
        parser.error("--hidden-sizes must be comma-separated positive integers")
    if not args.hidden_sizes or any(size < 1 for size in args.hidden_sizes):
        parser.error("--hidden-sizes must contain positive integers")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")
    return args


def read_dataset(path: Path) -> tuple[list[dict[str, Any]], int]:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset CSV does not exist: {path}")

    samples: list[dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        headers = reader.fieldnames
        if not headers or len(headers) != len(set(headers)):
            raise ValueError("CSV must have a nonempty header with unique column names")

        required = (
            ["episode_id", "maneuver", "time_s"] +
            [f"state_{name}" for name in STATE_NAMES] +
            [f"previous_{name}" for name in INPUT_NAMES] +
            [f"expert_{name}" for name in INPUT_NAMES]
        )
        missing = [name for name in required if name not in headers]
        if missing:
            raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")

        state_preview_steps = _preview_steps(headers, "reference_", STATE_NAMES)
        input_preview_steps = _preview_steps(headers, "reference_input_", INPUT_NAMES)
        if len(state_preview_steps) < 2 or state_preview_steps != input_preview_steps + [len(input_preview_steps)]:
            raise ValueError("CSV must contain contiguous horizon+1 state and horizon input previews")
        horizon = len(input_preview_steps)
        required.extend(
            f"reference_{step}_{name}"
            for step in state_preview_steps for name in STATE_NAMES
        )
        required.extend(
            f"reference_input_{step}_{name}"
            for step in input_preview_steps for name in INPUT_NAMES
        )
        missing = [name for name in required if name not in headers]
        if missing:
            raise ValueError(f"CSV is missing preview columns: {', '.join(missing)}")

        for line_number, row in enumerate(reader, start=2):
            if None in row or any(row.get(name) is None or row[name] == "" for name in required):
                raise ValueError(f"CSV row {line_number} has missing or extra fields")
            if not row["episode_id"] or not row["maneuver"]:
                raise ValueError(f"CSV row {line_number} has an empty episode or maneuver")
            numeric: dict[str, float] = {}
            for name in required:
                if name in ("episode_id", "maneuver"):
                    continue
                try:
                    value = float(row[name])
                except ValueError as error:
                    raise ValueError(
                        f"CSV row {line_number}, column {name} is not numeric"
                    ) from error
                if not math.isfinite(value):
                    raise ValueError(
                        f"CSV row {line_number}, column {name} is not finite"
                    )
                numeric[name] = value
            samples.append({
                "episode_id": row["episode_id"],
                "maneuver": row["maneuver"],
                "values": numeric,
            })

    if not samples:
        raise ValueError("Dataset CSV contains no data rows")
    return samples, horizon


def _preview_steps(headers: list[str], prefix: str, names: tuple[str, ...]) -> list[int]:
    marker = prefix
    step_ids: set[int] = set()
    for header in headers:
        if not header.startswith(marker):
            continue
        step_text, separator, name = header[len(marker):].partition("_")
        if separator and name in names and step_text.isdigit():
            step_ids.add(int(step_text))
    return sorted(step_ids)


def feature_names(horizon: int) -> list[str]:
    names = [f"state_{name}" for name in BODY_STATE_NAMES]
    names.extend(f"previous_{name}" for name in INPUT_NAMES)
    for step in range(horizon + 1):
        names.extend((
            f"reference_{step}_dx_body",
            f"reference_{step}_dy_body",
            f"reference_{step}_dpsi",
        ))
        names.extend(f"reference_{step}_{name}" for name in BODY_STATE_NAMES)
    for step in range(horizon):
        names.extend(f"reference_input_{step}_{name}" for name in INPUT_NAMES)
    return names


def make_features(sample: dict[str, Any], horizon: int) -> list[float]:
    values = sample["values"]
    current_pose = [values[f"state_{name}"] for name in ("X", "Y", "psi")]
    x, y, psi = current_pose
    cosine, sine = math.cos(psi), math.sin(psi)

    features = [values[f"state_{name}"] for name in BODY_STATE_NAMES]
    features.extend(values[f"previous_{name}"] for name in INPUT_NAMES)
    for step in range(horizon + 1):
        dx = values[f"reference_{step}_X"] - x
        dy = values[f"reference_{step}_Y"] - y
        dpsi = values[f"reference_{step}_psi"] - psi
        features.extend((
            cosine * dx + sine * dy,
            -sine * dx + cosine * dy,
            math.atan2(math.sin(dpsi), math.cos(dpsi)),
        ))
        features.extend(
            values[f"reference_{step}_{name}"] for name in BODY_STATE_NAMES
        )
    for step in range(horizon):
        features.extend(
            values[f"reference_input_{step}_{name}"] for name in INPUT_NAMES
        )
    return features


def split_by_episode(
    samples: list[dict[str, Any]], validation_fraction: float,
    test_fraction: float, seed: int,
) -> tuple[dict[str, list[int]], dict[str, list[str]]]:
    episode_indices: dict[str, list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        episode_indices[sample["episode_id"]].append(index)
    episode_ids = sorted(episode_indices)
    if len(episode_ids) < 3:
        raise ValueError(
            "At least 3 distinct episodes are required for train/validation/test splits"
        )

    random.Random(seed).shuffle(episode_ids)
    test_count = max(1, round(len(episode_ids) * test_fraction))
    validation_count = max(1, round(len(episode_ids) * validation_fraction))
    while test_count + validation_count >= len(episode_ids):
        if test_count >= validation_count and test_count > 1:
            test_count -= 1
        elif validation_count > 1:
            validation_count -= 1
        else:
            raise ValueError("Not enough episodes for nonempty train/validation/test splits")
    episode_groups = {
        "test": episode_ids[:test_count],
        "validation": episode_ids[test_count:test_count + validation_count],
        "train": episode_ids[test_count + validation_count:],
    }
    indices = {
        split: [index for episode in episodes for index in episode_indices[episode]]
        for split, episodes in episode_groups.items()
    }
    return indices, episode_groups


def build_model(input_size: int, hidden_sizes: list[int], output_size: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    width = input_size
    for hidden_size in hidden_sizes:
        layers.extend((nn.Linear(width, hidden_size), nn.ReLU()))
        width = hidden_size
    layers.append(nn.Linear(width, output_size))
    return nn.Sequential(*layers)


class MpcPolicy:
    """CPU inference wrapper for a checkpoint produced by this module."""

    def __init__(self, checkpoint_path: Path) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict) or checkpoint.get("format_version") != 1:
            raise ValueError("Unsupported or invalid MPC policy checkpoint")
        horizon = checkpoint.get("horizon")
        hidden_sizes = checkpoint.get("hidden_sizes")
        if not isinstance(horizon, int) or horizon < 1:
            raise ValueError("Checkpoint has an invalid horizon")
        if not isinstance(hidden_sizes, list) or not all(
            isinstance(size, int) and size > 0 for size in hidden_sizes
        ):
            raise ValueError("Checkpoint has invalid hidden layer sizes")
        expected_features = feature_names(horizon)
        if checkpoint.get("feature_names") != expected_features:
            raise ValueError("Checkpoint feature ordering is unsupported")
        if checkpoint.get("input_size") != len(expected_features):
            raise ValueError("Checkpoint input size does not match its feature ordering")
        if checkpoint.get("output_names") != list(INPUT_NAMES):
            raise ValueError("Checkpoint output ordering is unsupported")

        normalization = checkpoint.get("normalization")
        if not isinstance(normalization, dict):
            raise ValueError("Checkpoint is missing normalization values")
        expected_sizes = {
            "feature_mean": len(expected_features),
            "feature_scale": len(expected_features),
            "target_mean": len(INPUT_NAMES),
            "target_scale": len(INPUT_NAMES),
        }
        for name, size in expected_sizes.items():
            values = normalization.get(name)
            if (not isinstance(values, list) or len(values) != size or
                    any(not math.isfinite(value) for value in values)):
                raise ValueError(f"Checkpoint has invalid normalization values: {name}")
        if any(value <= 0.0 for value in normalization["feature_scale"] +
               normalization["target_scale"]):
            raise ValueError("Checkpoint normalization scales must be positive")

        self.horizon = horizon
        self.model = build_model(len(expected_features), hidden_sizes, len(INPUT_NAMES))
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.model.eval()
        self.feature_mean = torch.tensor(normalization["feature_mean"], dtype=torch.float32)
        self.feature_scale = torch.tensor(normalization["feature_scale"], dtype=torch.float32)
        self.target_mean = torch.tensor(normalization["target_mean"], dtype=torch.float32)
        self.target_scale = torch.tensor(normalization["target_scale"], dtype=torch.float32)

    def predict(self, sample: dict[str, Any]) -> dict[str, float]:
        """Return the five raw network commands for one data-dump-shaped sample."""
        features = make_features(sample, self.horizon)
        if len(features) != self.feature_mean.numel() or not all(
            math.isfinite(value) for value in features
        ):
            raise ValueError("Policy sample has invalid or nonfinite features")
        feature_tensor = torch.tensor(features, dtype=torch.float32)
        if not torch.isfinite(feature_tensor).all():
            raise ValueError("Policy sample exceeds the finite float32 range")
        with torch.no_grad():
            normalized = (feature_tensor - self.feature_mean) / self.feature_scale
            prediction = self.model(normalized) * self.target_scale + self.target_mean
        if not torch.isfinite(prediction).all():
            raise ValueError("Policy produced nonfinite commands")
        return {
            name: float(value)
            for name, value in zip(INPUT_NAMES, prediction.tolist())
        }


def regression_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> dict[str, Any]:
    error = predictions - targets
    mae = error.abs().mean(dim=0)
    rmse = error.square().mean(dim=0).sqrt()
    return {
        "mae": {name: float(value) for name, value in zip(INPUT_NAMES, mae)},
        "rmse": {name: float(value) for name, value in zip(INPUT_NAMES, rmse)},
    }


def fixed_episode_split(
    samples: list[dict[str, Any]], episode_groups: dict[str, list[str]],
) -> dict[str, list[int]]:
    if set(episode_groups) != {"train", "validation", "test"}:
        raise ValueError("Fixed split must contain train, validation, and test groups")
    assigned = [episode for group in episode_groups.values() for episode in group]
    actual = {sample["episode_id"] for sample in samples}
    if (any(not group for group in episode_groups.values()) or
            len(assigned) != len(set(assigned)) or set(assigned) != actual):
        raise ValueError("Fixed split must assign every episode exactly once to nonempty groups")
    return {
        split: [index for index, sample in enumerate(samples)
                if sample["episode_id"] in episodes]
        for split, episodes in episode_groups.items()
    }


def train(
    args: argparse.Namespace, split_episode_ids: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    samples, horizon = read_dataset(args.csv)
    if split_episode_ids is None:
        split_indices, split_episodes = split_by_episode(
            samples, args.validation_fraction, args.test_fraction, args.seed
        )
    else:
        split_episodes = split_episode_ids
        split_indices = fixed_episode_split(samples, split_episodes)
    features = torch.tensor(
        [make_features(sample, horizon) for sample in samples], dtype=torch.float32
    )
    targets = torch.tensor(
        [[sample["values"][name] for name in OUTPUT_NAMES] for sample in samples],
        dtype=torch.float32,
    )
    if not torch.isfinite(features).all() or not torch.isfinite(targets).all():
        raise ValueError("Feature or target values exceed the finite float32 range")

    train_indices = split_indices["train"]
    feature_mean = features[train_indices].mean(dim=0)
    feature_scale = features[train_indices].std(dim=0, unbiased=False).clamp_min(1e-8)
    target_mean = targets[train_indices].mean(dim=0)
    target_scale = targets[train_indices].std(dim=0, unbiased=False).clamp_min(1e-8)
    normalized_features = (features - feature_mean) / feature_scale
    normalized_targets = (targets - target_mean) / target_scale

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    model = build_model(features.shape[1], args.hidden_sizes, targets.shape[1]).to(device)
    train_dataset = TensorDataset(
        normalized_features[train_indices], normalized_targets[train_indices]
    )
    loader_generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        generator=loader_generator,
    )
    validation_indices = split_indices["validation"]
    validation_x = normalized_features[validation_indices].to(device)
    validation_y = normalized_targets[validation_indices].to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    loss_function = nn.MSELoss()
    best_loss = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_loss = float(loss_function(model(validation_x), validation_y))
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break

    if best_state is None:
        raise RuntimeError("Training completed without a finite validation checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    test_indices = split_indices["test"]
    with torch.no_grad():
        test_predictions = model(normalized_features[test_indices].to(device)).cpu()
    test_predictions = test_predictions * target_scale + target_mean
    test_metrics = regression_metrics(test_predictions, targets[test_indices])

    checkpoint = {
        "format_version": 1,
        "model_state_dict": best_state,
        "input_size": features.shape[1],
        "hidden_sizes": args.hidden_sizes,
        "output_names": list(INPUT_NAMES),
        "feature_names": feature_names(horizon),
        "horizon": horizon,
        "normalization": {
            "feature_mean": feature_mean.tolist(),
            "feature_scale": feature_scale.tolist(),
            "target_mean": target_mean.tolist(),
            "target_scale": target_scale.tolist(),
        },
        "training": {
            "seed": args.seed,
            "device": str(device),
            "epochs_requested": args.epochs,
            "best_epoch": best_epoch,
            "best_validation_loss_normalized": best_loss,
            "dataset": str(args.csv),
            "sample_count": len(samples),
            "episode_count": len(split_episodes["train"] + split_episodes["validation"] +
                                 split_episodes["test"]),
            "split_episode_ids": split_episodes,
            "split_sample_counts": {
                split: len(indices) for split, indices in split_indices.items()
            },
            "test_metrics": test_metrics,
            "torch_version": str(torch.__version__),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)

    metrics_path = args.output.with_suffix(".metrics.json")
    report = {
        "checkpoint": str(args.output),
        **checkpoint["training"],
    }
    metrics_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Samples: {len(samples)} in {checkpoint['training']['episode_count']} episodes")
    print("Split samples: " + json.dumps(checkpoint["training"]["split_sample_counts"]))
    print(f"Best epoch: {best_epoch}; normalized validation MSE: {best_loss:.6g}")
    print("Held-out test metrics (physical units):")
    print(json.dumps(test_metrics, indent=2))
    print(f"Checkpoint: {args.output}")
    print(f"Metrics: {metrics_path}")
    if len(split_episodes["test"]) < 5:
        print(
            "WARNING: test split has fewer than 5 episodes; treat its metrics as "
            "a smoke check, not a reliable generalization estimate.",
            file=sys.stderr,
        )
    return report


def main() -> int:
    args = parse_args()
    try:
        train(args)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
