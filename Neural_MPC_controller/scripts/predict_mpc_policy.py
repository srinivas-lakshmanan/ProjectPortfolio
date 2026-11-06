#!/usr/bin/env python3
"""Run a trained MPC imitation policy over data-dump CSV samples."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from train_mpc_policy import INPUT_NAMES, MpcPolicy, read_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--csv", required=True, type=Path, help="MPC data-dump CSV")
    parser.add_argument("--output", required=True, type=Path, help="Prediction CSV")
    args = parser.parse_args()

    try:
        samples, horizon = read_dataset(args.csv)
        policy = MpcPolicy(args.checkpoint)
        if horizon != policy.horizon:
            raise ValueError(
                f"CSV horizon ({horizon}) does not match checkpoint ({policy.horizon})"
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["episode_id", "maneuver", "time_s"]
        fieldnames += [f"expert_{name}" for name in INPUT_NAMES]
        fieldnames += [f"predicted_{name}" for name in INPUT_NAMES]
        fieldnames += [f"absolute_error_{name}" for name in INPUT_NAMES]
        absolute_errors = {name: 0.0 for name in INPUT_NAMES}
        with args.output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for sample in samples:
                predicted = policy.predict(sample)
                row = {
                    "episode_id": sample["episode_id"],
                    "maneuver": sample["maneuver"],
                    "time_s": sample["values"]["time_s"],
                }
                for name in INPUT_NAMES:
                    expert = sample["values"][f"expert_{name}"]
                    error = abs(predicted[name] - expert)
                    absolute_errors[name] += error
                    row[f"expert_{name}"] = expert
                    row[f"predicted_{name}"] = predicted[name]
                    row[f"absolute_error_{name}"] = error
                writer.writerow(row)

        metrics = {
            name: error_sum / len(samples)
            for name, error_sum in absolute_errors.items()
        }
        print(f"Predicted {len(samples)} samples; MAE by actuator: "
              f"{json.dumps(metrics, sort_keys=True)}")
        print(f"Predictions saved to {args.output}")
    except (OSError, ValueError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
