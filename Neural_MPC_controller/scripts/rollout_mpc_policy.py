#!/usr/bin/env python3
"""Closed-loop rollout of a learned policy against a recorded reference episode."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

from train_mpc_policy import (
    INPUT_NAMES,
    MpcPolicy,
    STATE_NAMES,
    read_dataset,
)


MASS = 1500.0
YAW_INERTIA = 2500.0
WHEEL_INERTIA = 1.2
LF, LR, TF, TR = 1.2, 1.4, 1.6, 1.6
CG_HEIGHT = 0.55
WHEEL_RADIUS = 0.31
MU = 1.0
BX, CX, EX = 12.0, 1.65, -0.5
BY, CY, EY = 9.0, 1.30, -1.0
FRONT_ROLL_SHARE = 0.55
AIR_DENSITY, DRAG_COEFFICIENT, FRONTAL_AREA = 1.225, 0.32, 2.2
ROLLING_RESISTANCE, GRAVITY, SPEED_EPSILON = 0.012, 9.81, 0.5
WHEEL_X = (LF, LF, -LR, -LR)
WHEEL_Y = (TF / 2.0, -TF / 2.0, TR / 2.0, -TR / 2.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--csv", required=True, type=Path,
                        help="MPC data-dump CSV containing reference episodes")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output trajectory CSV")
    parser.add_argument("--summary", type=Path,
                        help="Summary JSON (default: output path with .summary.json suffix)")
    parser.add_argument("--episode", help="Roll out this episode ID (default: all episodes)")
    parser.add_argument("--rk4-substeps", type=int, default=50)
    parser.add_argument("--steering-limit", type=float, default=0.15)
    parser.add_argument("--steering-rate-limit", type=float, default=0.30)
    parser.add_argument("--torque-limit", type=float, default=200.0)
    parser.add_argument("--torque-rate-limit", type=float, default=1500.0)
    args = parser.parse_args()
    if args.rk4_substeps < 1:
        parser.error("--rk4-substeps must be positive")
    for name in (
        "steering_limit", "steering_rate_limit", "torque_limit", "torque_rate_limit"
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    return args


def smooth_abs(value: float, epsilon: float) -> float:
    return math.sqrt(value * value + epsilon * epsilon)


def smooth_div(numerator: float, denominator: float, epsilon: float) -> float:
    return numerator / smooth_abs(denominator, epsilon)


def dynamics(state: list[float], control: list[float]) -> list[float]:
    x, y, psi, vx, vy, yaw_rate = state[:6]
    steering = control[0]
    slips: list[float] = []
    slip_angles: list[float] = []
    wheel_cos: list[float] = []
    wheel_sin: list[float] = []
    for index in range(4):
        vxi = vx - yaw_rate * WHEEL_Y[index]
        vyi = vy + yaw_rate * WHEEL_X[index]
        angle = steering if index < 2 else 0.0
        cosine, sine = math.cos(angle), math.sin(angle)
        vwx = cosine * vxi + sine * vyi
        vwy = -sine * vxi + cosine * vyi
        slips.append(smooth_div(
            WHEEL_RADIUS * state[6 + index] - vwx, vwx, SPEED_EPSILON
        ))
        slip_angles.append(-math.atan(smooth_div(vwy, vwx, SPEED_EPSILON)))
        wheel_cos.append(cosine)
        wheel_sin.append(sine)

    wheelbase = LF + LR
    front_static = MASS * GRAVITY * LR / (2.0 * wheelbase)
    rear_static = MASS * GRAVITY * LF / (2.0 * wheelbase)
    loads = [front_static, front_static, rear_static, rear_static]
    longitudinal_forces = [0.0] * 4
    lateral_forces = [0.0] * 4
    sum_fx = sum_fy = sum_mz = 0.0
    for pass_index in range(2):
        sum_fx = sum_fy = sum_mz = 0.0
        for index in range(4):
            fz_positive = 0.5 * (loads[index] + smooth_abs(loads[index], 1.0))
            xk = BX * slips[index]
            xa = BY * slip_angles[index]
            fx0 = MU * fz_positive * math.sin(
                CX * math.atan(xk - EX * (xk - math.atan(xk)))
            )
            fy0 = MU * fz_positive * math.sin(
                CY * math.atan(xa - EY * (xa - math.atan(xa)))
            )
            combined_slip = math.sqrt(
                slips[index] ** 2 + slip_angles[index] ** 2 + 1e-8
            )
            longitudinal_forces[index] = (
                fx0 * smooth_abs(slips[index], 1e-4) / combined_slip
            )
            lateral_forces[index] = (
                fy0 * smooth_abs(slip_angles[index], 1e-4) / combined_slip
            )
            fx_body = (
                wheel_cos[index] * longitudinal_forces[index]
                - wheel_sin[index] * lateral_forces[index]
            )
            fy_body = (
                wheel_sin[index] * longitudinal_forces[index]
                + wheel_cos[index] * lateral_forces[index]
            )
            sum_fx += fx_body
            sum_fy += fy_body
            sum_mz += WHEEL_X[index] * fy_body - WHEEL_Y[index] * fx_body

        if pass_index == 0:
            resistance = (
                0.5 * AIR_DENSITY * DRAG_COEFFICIENT * FRONTAL_AREA
                * vx * smooth_abs(vx, SPEED_EPSILON)
                + ROLLING_RESISTANCE * MASS * GRAVITY
                * vx / smooth_abs(vx, SPEED_EPSILON)
            )
            acceleration_x = (sum_fx - resistance) / MASS
            acceleration_y = sum_fy / MASS
            longitudinal_transfer = (
                MASS * CG_HEIGHT / (2.0 * wheelbase) * acceleration_x
            )
            lateral_transfer_front = (
                FRONT_ROLL_SHARE * MASS * CG_HEIGHT / TF * acceleration_y
            )
            lateral_transfer_rear = (
                (1.0 - FRONT_ROLL_SHARE) * MASS * CG_HEIGHT / TR * acceleration_y
            )
            loads = [
                front_static - longitudinal_transfer - lateral_transfer_front,
                front_static - longitudinal_transfer + lateral_transfer_front,
                rear_static + longitudinal_transfer - lateral_transfer_rear,
                rear_static + longitudinal_transfer + lateral_transfer_rear,
            ]

    resistance = (
        0.5 * AIR_DENSITY * DRAG_COEFFICIENT * FRONTAL_AREA
        * vx * smooth_abs(vx, SPEED_EPSILON)
        + ROLLING_RESISTANCE * MASS * GRAVITY
        * vx / smooth_abs(vx, SPEED_EPSILON)
    )
    acceleration_x = (sum_fx - resistance) / MASS
    acceleration_y = sum_fy / MASS
    derivative = [
        vx * math.cos(psi) - vy * math.sin(psi),
        vx * math.sin(psi) + vy * math.cos(psi),
        yaw_rate,
        acceleration_x + yaw_rate * vy,
        acceleration_y - yaw_rate * vx,
        sum_mz / YAW_INERTIA,
    ]
    derivative.extend(
        (control[index + 1] - WHEEL_RADIUS * longitudinal_forces[index])
        / WHEEL_INERTIA
        for index in range(4)
    )
    return derivative


def rk4_step(state: list[float], control: list[float], dt: float) -> list[float]:
    k1 = dynamics(state, control)
    k2 = dynamics([x + 0.5 * dt * dx for x, dx in zip(state, k1)], control)
    k3 = dynamics([x + 0.5 * dt * dx for x, dx in zip(state, k2)], control)
    k4 = dynamics([x + dt * dx for x, dx in zip(state, k3)], control)
    result = [
        x + dt / 6.0 * (a + 2.0 * b + 2.0 * c + d)
        for x, a, b, c, d in zip(state, k1, k2, k3, k4)
    ]
    if not all(math.isfinite(value) for value in result):
        raise RuntimeError("Plant integration produced a nonfinite state")
    return result


def constrain_control(
    requested: list[float], previous: list[float], dt: float, args: argparse.Namespace
) -> list[float]:
    limits = [args.steering_limit] + [args.torque_limit] * 4
    rates = [args.steering_rate_limit] + [args.torque_rate_limit] * 4
    applied = []
    for value, prior, limit, rate in zip(requested, previous, limits, rates):
        lower = max(-limit, prior - rate * dt)
        upper = min(limit, prior + rate * dt)
        applied.append(min(max(value, lower), upper))
    return applied


def reference_state(sample: dict[str, Any], step: int = 0) -> list[float]:
    values = sample["values"]
    return [values[f"reference_{step}_{name}"] for name in STATE_NAMES]


def state_error(state: list[float], reference: list[float]) -> dict[str, float]:
    heading_error = state[2] - reference[2]
    return {
        "lateral_error_m": state[1] - reference[1],
        "heading_error_rad": math.atan2(math.sin(heading_error), math.cos(heading_error)),
        "speed_error_m_s": state[3] - reference[3],
    }


def rollout(
    samples: list[dict[str, Any]], episode_id: str, policy: MpcPolicy,
    args: argparse.Namespace, writer: csv.DictWriter,
) -> dict[str, Any]:
    if len(samples) < 2:
        raise ValueError(f"Episode {episode_id} needs at least two samples to infer its interval")
    dt = samples[1]["values"]["time_s"] - samples[0]["values"]["time_s"]
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"Episode {episode_id} has invalid sample times")
    for previous_sample, sample in zip(samples, samples[1:]):
        actual_dt = sample["values"]["time_s"] - previous_sample["values"]["time_s"]
        if not math.isclose(actual_dt, dt, rel_tol=1e-6, abs_tol=1e-9):
            raise ValueError(f"Episode {episode_id} does not have a uniform sample interval")

    first_values = samples[0]["values"]
    state = [first_values[f"state_{name}"] for name in STATE_NAMES]
    previous_control = [first_values[f"previous_{name}"] for name in INPUT_NAMES]
    limits = [args.steering_limit] + [args.torque_limit] * 4
    if any(abs(value) > limit for value, limit in zip(previous_control, limits)):
        raise ValueError(f"Episode {episode_id} initial input exceeds configured actuator limits")
    steps = args.rk4_substeps
    errors = []
    max_steering = max_torque = max_steering_rate = max_torque_rate = 0.0
    for index, sample in enumerate(samples):
        policy_sample = {"episode_id": episode_id, "maneuver": sample["maneuver"],
                         "values": sample["values"].copy()}
        for name, value in zip(STATE_NAMES, state):
            policy_sample["values"][f"state_{name}"] = value
        for name, value in zip(INPUT_NAMES, previous_control):
            policy_sample["values"][f"previous_{name}"] = value
        prediction = policy.predict(policy_sample)
        requested = [prediction[name] for name in INPUT_NAMES]
        control = constrain_control(requested, previous_control, dt, args)
        reference = reference_state(sample)
        measured_errors = state_error(state, reference)
        row: dict[str, Any] = {
            "episode_id": episode_id,
            "maneuver": sample["maneuver"],
            "time_s": sample["values"]["time_s"],
        }
        row.update({f"state_{name}": value for name, value in zip(STATE_NAMES, state)})
        row.update({f"reference_{name}": value for name, value in zip(STATE_NAMES, reference)})
        row.update({f"requested_{name}": value for name, value in zip(INPUT_NAMES, requested)})
        row.update({f"applied_{name}": value for name, value in zip(INPUT_NAMES, control)})
        row.update({f"previous_{name}": value for name, value in zip(INPUT_NAMES, previous_control)})
        row.update(measured_errors)
        writer.writerow(row)
        errors.append(measured_errors)

        max_steering = max(max_steering, abs(control[0]))
        max_torque = max(max_torque, *(abs(value) for value in control[1:]))
        steering_rate = abs(control[0] - previous_control[0]) / dt
        torque_rate = max(
            abs(control[input_index] - previous_control[input_index]) / dt
            for input_index in range(1, 5)
        )
        max_steering_rate = max(max_steering_rate, steering_rate)
        max_torque_rate = max(max_torque_rate, torque_rate)

        if index + 1 < len(samples):
            substep = dt / steps
            for _ in range(steps):
                state = rk4_step(state, control, substep)
        previous_control = control

    count = len(errors)
    return {
        "episode_id": episode_id,
        "maneuver": samples[0]["maneuver"],
        "samples": count,
        "sample_time_s": dt,
        "max_lateral_error_m": max(abs(row["lateral_error_m"]) for row in errors),
        "rms_lateral_error_m": math.sqrt(
            sum(row["lateral_error_m"] ** 2 for row in errors) / count
        ),
        "max_abs_heading_error_rad": max(abs(row["heading_error_rad"]) for row in errors),
        "max_abs_speed_error_m_s": max(abs(row["speed_error_m_s"]) for row in errors),
        "max_abs_steering_rad": max_steering,
        "max_abs_torque_nm": max_torque,
        "max_steering_rate_rad_s": max_steering_rate,
        "max_torque_rate_nm_s": max_torque_rate,
    }


def main() -> int:
    args = parse_args()
    if args.summary is None:
        args.summary = args.output.with_suffix(".summary.json")
    try:
        samples, horizon = read_dataset(args.csv)
        policy = MpcPolicy(args.checkpoint)
        if horizon != policy.horizon:
            raise ValueError(
                f"CSV horizon ({horizon}) does not match checkpoint ({policy.horizon})"
            )
        episodes: dict[str, list[dict[str, Any]]] = {}
        for sample in samples:
            episodes.setdefault(sample["episode_id"], []).append(sample)
        if args.episode is not None:
            if args.episode not in episodes:
                raise ValueError(f"Episode ID not found in CSV: {args.episode}")
            episodes = {args.episode: episodes[args.episode]}

        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["episode_id", "maneuver", "time_s"]
        fieldnames += [f"state_{name}" for name in STATE_NAMES]
        fieldnames += [f"reference_{name}" for name in STATE_NAMES]
        fieldnames += [f"requested_{name}" for name in INPUT_NAMES]
        fieldnames += [f"applied_{name}" for name in INPUT_NAMES]
        fieldnames += [f"previous_{name}" for name in INPUT_NAMES]
        fieldnames += ["lateral_error_m", "heading_error_rad", "speed_error_m_s"]
        summaries = []
        with args.output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for episode_id, episode_samples in episodes.items():
                summaries.append(rollout(episode_samples, episode_id, policy, args, writer))

        report = {
            "checkpoint": str(args.checkpoint),
            "reference_csv": str(args.csv),
            "rk4_substeps": args.rk4_substeps,
            "limits": {
                "steering_rad": args.steering_limit,
                "steering_rate_rad_s": args.steering_rate_limit,
                "torque_nm": args.torque_limit,
                "torque_rate_nm_s": args.torque_rate_limit,
            },
            "episodes": summaries,
        }
        args.summary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Rolled out {len(summaries)} episodes to {args.output}")
        print(f"Summary: {args.summary}")
    except (OSError, ValueError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
