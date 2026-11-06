#!/usr/bin/env python3
"""Collect MPC labels on learner-visited states and retrain on accumulated data."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
from typing import Any

import torch

from rollout_mpc_policy import constrain_control, rk4_step
from train_mpc_policy import (
    INPUT_NAMES, STATE_NAMES, MpcPolicy, fixed_episode_split, read_dataset, train,
)


class MpcExpert:
    """One persistent C++ MPC process, with bounded waits and explicit failures."""

    def __init__(self, executable: Path, horizon: int, dt: float,
                 substeps: int, iterations: int, jit: bool, timeout: float) -> None:
        self.timeout = timeout
        environment = os.environ.copy()
        environment["OPENBLAS_NUM_THREADS"] = "1"
        environment["OMP_NUM_THREADS"] = "1"
        self.process = subprocess.Popen(
            [str(executable.resolve()), "--horizon", str(horizon),
             "--sample-time", str(dt), "--rk4-substeps", str(substeps),
             "--max-iterations", str(iterations), "--jit", str(jit).lower()],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
            bufsize=1, env=environment,
        )
        self.lines: queue.Queue[str | None] = queue.Queue()
        self.reader = threading.Thread(target=self._read_output, daemon=True)
        self.reader.start()
        try:
            if self._receive() != "READY":
                raise RuntimeError("MPC expert did not return READY")
        except (RuntimeError, OSError):
            self.close()
            raise
        self.horizon = horizon

    def _read_output(self) -> None:
        assert self.process.stdout is not None
        for line in self.process.stdout:
            self.lines.put(line.rstrip("\r\n"))
        self.lines.put(None)

    def _receive(self) -> str:
        try:
            line = self.lines.get(timeout=self.timeout)
        except queue.Empty as error:
            raise RuntimeError("Timed out waiting for the MPC expert") from error
        if line is None:
            raise RuntimeError("MPC expert exited before replying; see its stderr output")
        if line.startswith("ERROR"):
            raise RuntimeError(line)
        return line

    def _request(self, line: str) -> str:
        assert self.process.stdin is not None
        self.process.stdin.write(line + "\n")
        self.process.stdin.flush()
        return self._receive()

    def reset(self) -> None:
        if self._request("RESET") != "OK":
            raise RuntimeError("Invalid MPC expert reset response")

    def solve(self, sample: dict[str, Any]) -> dict[str, float]:
        values = sample["values"]
        fields = [values[f"state_{name}"] for name in STATE_NAMES]
        fields += [values[f"previous_{name}"] for name in INPUT_NAMES]
        fields += [values[f"reference_{step}_{name}"]
                   for step in range(self.horizon + 1) for name in STATE_NAMES]
        fields += [values[f"reference_input_{step}_{name}"]
                   for step in range(self.horizon) for name in INPUT_NAMES]
        if not all(math.isfinite(value) for value in fields):
            raise ValueError("Expert request contains nonfinite values")
        response = self._request("SOLVE " + " ".join(format(value, ".17g") for value in fields))
        parts = response.split()
        if len(parts) != 9 or parts[0] != "OK":
            raise RuntimeError(f"Invalid MPC expert response: {response}")
        result = [float(value) for value in parts[1:]]
        if not all(math.isfinite(value) for value in result):
            raise RuntimeError("MPC expert returned nonfinite values")
        return dict(zip(
            [f"expert_{name}" for name in INPUT_NAMES] +
            ["solver_iterations", "solve_seconds", "constraint_violation"],
            result,
        ))

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.reader.join(timeout=5)
        if self.process.stdout is not None:
            self.process.stdout.close()


def episode_interval(samples: list[dict[str, Any]]) -> float:
    if len(samples) < 2:
        raise ValueError("DAgger episodes require at least two timed samples")
    times = [sample["values"]["time_s"] for sample in samples]
    dt = times[1] - times[0]
    if dt <= 0 or not math.isfinite(dt) or any(
        not math.isclose(b - a, dt, rel_tol=1e-6, abs_tol=1e-9)
        for a, b in zip(times, times[1:])
    ):
        raise ValueError("DAgger requires uniformly increasing episode sample times")
    return dt


def collect_episode(
    samples: list[dict[str, Any]], policy: MpcPolicy, expert: MpcExpert,
    dt: float, substeps: int, iteration: int,
) -> list[dict[str, Any]]:
    state = [samples[0]["values"][f"state_{name}"] for name in STATE_NAMES]
    previous = [samples[0]["values"][f"previous_{name}"] for name in INPUT_NAMES]
    limits = argparse.Namespace(steering_limit=0.15, steering_rate_limit=0.30,
                               torque_limit=200.0, torque_rate_limit=1500.0)
    if abs(previous[0]) > 0.15 or any(abs(value) > 200.0 for value in previous[1:]):
        raise ValueError("Initial applied input is outside the MPC actuator bounds")
    expert.reset()
    collected = []
    for source in samples:
        sample = {**source, "values": source["values"].copy()}
        values = sample["values"]
        values.update({f"state_{name}": value for name, value in zip(STATE_NAMES, state)})
        values.update({f"previous_{name}": value for name, value in zip(INPUT_NAMES, previous)})
        prediction = policy.predict(sample)
        applied = constrain_control(
            [prediction[name] for name in INPUT_NAMES], previous, dt, limits
        )
        # The expert labels this state; its action is never applied to the plant.
        values.update(expert.solve(sample))
        values.update({f"applied_{name}": value for name, value in zip(INPUT_NAMES, applied)})
        values["dagger_iteration"] = iteration
        collected.append(sample)
        for _ in range(substeps):
            state = rk4_step(state, applied, dt / substeps)
        previous = applied
    return collected


def write_aggregate(path: Path, samples: list[dict[str, Any]]) -> None:
    numeric_fields = sorted({name for sample in samples for name in sample["values"]})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["episode_id", "maneuver"] + numeric_fields)
        writer.writeheader()
        for sample in samples:
            writer.writerow({
                "episode_id": sample["episode_id"], "maneuver": sample["maneuver"],
                **sample["values"],
            })


def run(args: argparse.Namespace) -> None:
    samples, horizon = read_dataset(args.csv)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    policy = MpcPolicy(args.checkpoint)
    if horizon != policy.horizon:
        raise ValueError("Dataset and policy horizons do not match")
    training = checkpoint.get("training")
    if not isinstance(training, dict):
        raise ValueError("Initial checkpoint is missing training metadata")
    split = training.get("split_episode_ids")
    if not isinstance(split, dict) or any(
        not isinstance(group, list) or
        any(not isinstance(episode, str) for episode in group)
        for group in split.values()
    ):
        raise ValueError("Initial checkpoint has invalid episode split metadata")
    indices = fixed_episode_split(samples, split)
    if (training.get("sample_count") != len(samples) or
            training.get("split_sample_counts") !=
            {name: len(group) for name, group in indices.items()}):
        raise ValueError("Source CSV sample counts do not match the initial training dataset")
    if not isinstance(training.get("seed"), int):
        raise ValueError("Initial checkpoint has an invalid training seed")
    episodes: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        episodes.setdefault(sample["episode_id"], []).append(sample)
        sample["values"]["dagger_iteration"] = 0
    intervals = [episode_interval(episode) for episode in episodes.values()]
    dt = intervals[0]
    if any(not math.isclose(interval, dt, rel_tol=1e-6, abs_tol=1e-9)
           for interval in intervals):
        raise ValueError("All episodes must use the same MPC sample interval")
    if not args.expert.is_file():
        raise FileNotFoundError(f"MPC expert executable does not exist: {args.expert}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("DAgger output directory must be new or empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    expert = MpcExpert(args.expert, horizon, dt, args.rk4_substeps,
                       args.max_iterations, args.jit, args.expert_timeout)
    rounds = []
    try:
        for iteration in range(1, args.rounds + 1):
            new_samples = []
            for episode_id in split["train"]:
                print(f"DAgger round {iteration}: collecting episode {episode_id}", flush=True)
                try:
                    new_samples.extend(collect_episode(
                        episodes[episode_id], policy, expert, dt, args.rk4_substeps, iteration
                    ))
                except (ValueError, RuntimeError, OSError) as error:
                    raise RuntimeError(
                        f"DAgger round {iteration}, episode {episode_id}: {error}"
                    ) from error
            samples.extend(new_samples)
            aggregate_path = args.output_dir / f"round_{iteration}_data.csv"
            model_path = args.output_dir / f"round_{iteration}_policy.pt"
            write_aggregate(aggregate_path, samples)
            training_args = argparse.Namespace(
                csv=aggregate_path, output=model_path, epochs=args.epochs,
                batch_size=args.batch_size, learning_rate=args.learning_rate,
                weight_decay=1e-5, hidden_sizes=checkpoint["hidden_sizes"],
                validation_fraction=0.15, test_fraction=0.15, patience=args.patience,
                seed=training["seed"], device="cpu",
            )
            report = train(training_args, split_episode_ids=split)
            policy = MpcPolicy(model_path)
            rounds.append({
                "iteration": iteration, "new_labels": len(new_samples),
                "aggregate_samples": len(samples), "dataset": str(aggregate_path),
                "checkpoint": str(model_path),
                "best_validation_loss_normalized":
                    report["best_validation_loss_normalized"],
            })
            summary = {
                "initial_checkpoint": str(args.checkpoint), "source_csv": str(args.csv),
                "split_episode_ids": split, "sample_time_s": dt,
                "rk4_substeps": args.rk4_substeps, "expert_max_iterations": args.max_iterations,
                "expert_jit": args.jit, "learner_drives_plant": True, "rounds": rounds,
            }
            (args.output_dir / "dagger_summary.json").write_text(
                json.dumps(summary, indent=2) + "\n", encoding="utf-8"
            )
    finally:
        expert.close()
    print(f"DAgger complete. Final policy: {rounds[-1]['checkpoint']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--csv", required=True, type=Path,
                        help="Original dataset used to train the initial checkpoint")
    parser.add_argument("--expert", required=True, type=Path, help="C++ mpc_expert executable")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--rk4-substeps", type=int, default=50)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--jit", choices=("true", "false"), default="true")
    parser.add_argument("--expert-timeout", type=float, default=300.0,
                        help="Seconds allowed per startup/request")
    args = parser.parse_args()
    for name in ("rounds", "epochs", "batch_size", "patience", "rk4_substeps", "max_iterations"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("learning_rate", "expert_timeout"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    args.jit = args.jit == "true"
    try:
        run(args)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
