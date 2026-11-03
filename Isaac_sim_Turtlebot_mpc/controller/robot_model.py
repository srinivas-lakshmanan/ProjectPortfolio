"""Unwrapped no-slip differential-drive kinematics for CasADi."""

import math

import casadi as ca


def dynamics(
    state: ca.MX | ca.SX | ca.DM, control: ca.MX | ca.SX | ca.DM
) -> ca.MX | ca.SX | ca.DM:
    return ca.vertcat(
        control[0] * ca.cos(state[2]),
        control[0] * ca.sin(state[2]),
        control[1],
    )


def step_euler(
    state: ca.MX | ca.SX | ca.DM, control: ca.MX | ca.SX | ca.DM, dt: float
) -> ca.MX | ca.SX | ca.DM:
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("Time step must be finite and positive.")
    return state + dt * dynamics(state, control)
