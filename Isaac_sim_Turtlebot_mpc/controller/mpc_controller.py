"""Multiple-shooting trajectory tracking through Python CasADi and IPOPT."""

from dataclasses import dataclass
import math

import casadi as ca
import numpy as np
from numpy.typing import ArrayLike, NDArray

from .robot_model import step_euler


_FEASIBILITY_TOLERANCE = 1e-6


@dataclass(frozen=True)
class MpcConfig:
    horizon: int = 20
    dt: float = 0.1
    stage_weights: tuple[float, float, float] = (10.0, 10.0, 1.0)
    terminal_weights: tuple[float, float, float] = (50.0, 50.0, 5.0)
    control_weights: tuple[float, float] = (0.1, 0.1)
    velocity_margin: float = 1e-4

    def __post_init__(self) -> None:
        if isinstance(self.horizon, bool) or not isinstance(self.horizon, int) or self.horizon < 1:
            raise ValueError("Horizon must be a positive integer.")
        if not math.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("Sampling interval must be finite and positive.")
        if not math.isfinite(self.velocity_margin) or not (
            _FEASIBILITY_TOLERANCE < self.velocity_margin < 0.25
        ):
            raise ValueError("Velocity margin must be finite and in (1e-6, 0.25).")
        for name, size, positive in (
            ("stage_weights", 3, False),
            ("terminal_weights", 3, False),
            ("control_weights", 2, True),
        ):
            weights = tuple(getattr(self, name))
            if len(weights) != size or any(
                not math.isfinite(w) or w < 0.0 or (positive and w == 0.0)
                for w in weights
            ):
                raise ValueError(
                    f"{name} must contain {size} finite "
                    f"{'positive' if positive else 'nonnegative'} weights."
                )
            object.__setattr__(self, name, weights)


@dataclass(frozen=True)
class MpcSolution:
    states: NDArray[np.float64]
    controls: NDArray[np.float64]
    objective: float


def _tracking_cost(
    state: ca.MX, reference: ca.MX, weights: tuple[float, float, float]
) -> ca.MX:
    error = state - reference
    return (
        weights[0] * error[0] ** 2
        + weights[1] * error[1] ** 2
        + 2.0 * weights[2] * (1.0 - ca.cos(error[2]))
    )


class MpcController:
    def __init__(self, config: MpcConfig = MpcConfig()) -> None:
        self._config = config
        n = config.horizon
        self._opti = ca.Opti()
        self._states = self._opti.variable(3, n + 1)
        self._controls = self._opti.variable(2, n)
        self._measured = self._opti.parameter(3)
        self._reference = self._opti.parameter(3, n + 1)
        self._opti.subject_to(self._states[:, 0] == self._measured)
        cost = ca.MX(0.0)
        for k in range(n):
            state = self._states[:, k]
            control = self._controls[:, k]
            self._opti.subject_to(
                self._states[:, k + 1] == step_euler(state, control, config.dt)
            )
            self._opti.subject_to(
                self._opti.bounded(
                    config.velocity_margin, control[0], 0.5 - config.velocity_margin
                )
            )
            self._opti.subject_to(self._opti.bounded(-1.5, control[1], 1.5))
            cost += _tracking_cost(state, self._reference[:, k], config.stage_weights)
            cost += (
                config.control_weights[0] * control[0] ** 2
                + config.control_weights[1] * control[1] ** 2
            )
        cost += _tracking_cost(
            self._states[:, n], self._reference[:, n], config.terminal_weights
        )
        self._opti.minimize(cost)
        self._opti.solver(
            "ipopt",
            {"print_time": False},
            {
                "print_level": 0,
                "sb": "yes",
                "max_iter": 500,
                "tol": 1e-8,
                "constr_viol_tol": 1e-8,
                "acceptable_tol": 1e-8,
                "bound_relax_factor": 0.0,
            },
        )

    @property
    def config(self) -> MpcConfig:
        return self._config

    def solve(self, measured_state: ArrayLike, reference: ArrayLike) -> MpcSolution:
        """Return (N+1, 3) poses and (N, 2) commands; errors propagate, never fallback."""
        measured = np.asarray(measured_state, dtype=float)
        poses = np.asarray(reference, dtype=float)
        n = self.config.horizon
        if measured.shape != (3,) or not np.isfinite(measured).all():
            raise ValueError("Measured state must be a finite vector with shape (3,).")
        if poses.shape != (n + 1, 3) or not np.isfinite(poses).all():
            raise ValueError(f"Reference must be finite with shape ({n + 1}, 3).")
        self._opti.set_value(self._measured, measured)
        self._opti.set_value(self._reference, poses.T)

        guess = np.empty((n + 1, 3))
        guess[0] = measured
        commands = np.zeros((n, 2))
        commands[:, 0] = 0.25
        for k in range(n):
            guess[k + 1] = np.asarray(
                step_euler(ca.DM(guess[k]), ca.DM(commands[k]), self.config.dt)
            ).reshape(3)
        self._opti.set_initial(self._states, guess.T)
        self._opti.set_initial(self._controls, commands.T)

        solution = self._opti.solve()
        result = MpcSolution(
            states=np.asarray(solution.value(self._states), dtype=float).reshape(3, n + 1).T,
            controls=np.asarray(solution.value(self._controls), dtype=float).reshape(2, n).T,
            objective=float(solution.value(self._opti.f)),
        )
        self._validate_solution(result, measured)
        return result

    def _validate_solution(self, result: MpcSolution, measured: NDArray[np.float64]) -> None:
        if (
            not math.isfinite(result.objective)
            or not np.isfinite(result.states).all()
            or not np.isfinite(result.controls).all()
        ):
            raise RuntimeError("MPC returned a nonfinite solution.")
        v, omega = result.controls.T
        margin = self.config.velocity_margin
        tol = _FEASIBILITY_TOLERANCE
        if (
            np.any(v <= 0.0)
            or np.any(v >= 0.5)
            or np.any(v < margin - tol)
            or np.any(v > 0.5 - margin + tol)
            or np.any(omega < -1.5)
            or np.any(omega > 1.5)
        ):
            raise RuntimeError("MPC returned a control outside the velocity bounds.")
        if np.any(np.abs(result.states[0] - measured) > tol):
            raise RuntimeError("MPC prediction violates the initial-state constraint.")
        for k in range(self.config.horizon):
            expected = np.asarray(
                step_euler(
                    ca.DM(result.states[k]), ca.DM(result.controls[k]), self.config.dt
                )
            ).reshape(3)
            if np.any(np.abs(result.states[k + 1] - expected) > tol):
                raise RuntimeError("MPC prediction violates the dynamics constraints.")
