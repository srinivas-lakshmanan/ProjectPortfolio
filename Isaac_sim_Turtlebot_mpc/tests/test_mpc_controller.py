from dataclasses import replace
import math
import unittest
from unittest.mock import patch

import casadi as ca
import numpy as np

from controller.mpc_controller import MpcConfig, MpcController, MpcSolution
from controller.robot_model import dynamics, step_euler


class ModelTests(unittest.TestCase):
    def test_numeric_and_symbolic_dynamics(self) -> None:
        for scalar in (ca.SX, ca.MX):
            state = scalar.sym("state", 3)
            control = scalar.sym("control", 2)
            transition = ca.Function("transition", [state, control], [step_euler(state, control, 0.1)])
            for heading in (0.0, math.pi / 2, math.pi, 4 * math.pi):
                for v, omega in ((0.2, 0.1), (-0.2, -0.5), (0.0, 0.0)):
                    pose = np.array([1.0, 2.0, heading])
                    rate = np.array([v * math.cos(heading), v * math.sin(heading), omega])
                    np.testing.assert_allclose(
                        np.asarray(transition(pose, [v, omega])).reshape(3),
                        pose + 0.1 * rate,
                        atol=1e-12,
                    )
                    np.testing.assert_allclose(
                        np.asarray(dynamics(ca.DM(pose), ca.DM([v, omega]))).reshape(3),
                        rate,
                        atol=1e-12,
                    )

    def test_invalid_time_step(self) -> None:
        for dt in (0.0, -0.1, math.nan, math.inf):
            with self.subTest(dt=dt), self.assertRaises(ValueError):
                step_euler(ca.DM.zeros(3), ca.DM.zeros(2), dt)


class ControllerTests(unittest.TestCase):
    def check_solution(
        self, result: MpcSolution, config: MpcConfig, measured: np.ndarray, reference: np.ndarray
    ) -> None:
        self.assertEqual(result.states.shape, (config.horizon + 1, 3))
        self.assertEqual(result.controls.shape, (config.horizon, 2))
        self.assertTrue(np.isfinite(result.states).all())
        self.assertTrue(np.isfinite(result.controls).all())
        np.testing.assert_allclose(result.states[0], measured, atol=1e-6, rtol=0.0)
        v, omega = result.controls.T
        self.assertTrue(((v > 0) & (v < 0.5)).all())
        self.assertTrue((v >= config.velocity_margin - 1e-6).all())
        self.assertTrue((v <= 0.5 - config.velocity_margin + 1e-6).all())
        self.assertTrue(((omega >= -1.5) & (omega <= 1.5)).all())
        cost = 0.0
        for k, (linear, angular) in enumerate(result.controls):
            x, y, theta = result.states[k]
            expected = np.array([
                x + config.dt * linear * math.cos(theta),
                y + config.dt * linear * math.sin(theta),
                theta + config.dt * angular,
            ])
            np.testing.assert_allclose(result.states[k + 1], expected, atol=1e-6, rtol=0.0)
            error = result.states[k] - reference[k]
            wx, wy, wh = config.stage_weights
            cost += wx * error[0] ** 2 + wy * error[1] ** 2 + 2 * wh * (1 - math.cos(error[2]))
            cost += config.control_weights[0] * linear**2 + config.control_weights[1] * angular**2
        error = result.states[-1] - reference[-1]
        wx, wy, wh = config.terminal_weights
        cost += wx * error[0] ** 2 + wy * error[1] ** 2 + 2 * wh * (1 - math.cos(error[2]))
        self.assertAlmostEqual(result.objective, cost, delta=1e-6 * (1 + abs(cost)))

    def test_tracking_and_periodic_heading(self) -> None:
        config = MpcConfig(horizon=12)
        controller = MpcController(config)
        measured = np.array([0.0, 0.15, 0.1])
        initial_error = abs(measured[1])
        for step in range(10):
            reference = np.zeros((config.horizon + 1, 3))
            reference[:, 0] = 0.2 * (step + np.arange(config.horizon + 1)) * config.dt
            result = controller.solve(measured, reference)
            self.check_solution(result, config, measured, reference)
            measured = result.states[1].copy()
        self.assertLess(abs(measured[1]), initial_error)
        reference = np.tile([0.5, 0.0, 0.0], (config.horizon + 1, 1))
        normal = controller.solve(measured, reference)
        reference[:, 2] += 2 * math.pi
        wrapped = controller.solve(measured, reference)
        self.check_solution(wrapped, config, measured, reference)
        np.testing.assert_allclose(normal.controls, wrapped.controls, atol=1e-5, rtol=0.0)

    def test_terminal_cost_and_active_bounds(self) -> None:
        config = MpcConfig(
            horizon=1, stage_weights=(0.0, 0.0, 0.0),
            terminal_weights=(100.0, 100.0, 100.0),
        )
        controller = MpcController(config)
        measured = np.zeros(3)
        for heading in (2.0, -2.0):
            reference = np.tile([10.0, 0.0, heading], (2, 1))
            result = controller.solve(measured, reference)
            self.check_solution(result, config, measured, reference)
            self.assertGreater(result.controls[0, 0], 0.49)
            self.assertGreater(math.copysign(1, heading) * result.controls[0, 1], 1.49)
        stationary = controller.solve(measured, np.zeros((2, 3)))
        self.check_solution(stationary, config, measured, np.zeros((2, 3)))
        self.assertLess(stationary.controls[0, 0], config.velocity_margin + 1e-4)
        no_terminal = replace(config, terminal_weights=(0.0, 0.0, 0.0))
        reference = np.tile([10.0, 0.0, 2.0], (2, 1))
        result = MpcController(no_terminal).solve(measured, reference)
        self.check_solution(result, no_terminal, measured, reference)
        self.assertLess(result.controls[0, 0], 0.001)
        self.assertLess(abs(result.controls[0, 1]), 1e-5)

    def test_invalid_config(self) -> None:
        cases = [
            {"horizon": value} for value in (0, -1, 1.5, True)
        ] + [
            {"dt": value} for value in (0.0, -0.1, math.nan, math.inf)
        ] + [
            {"velocity_margin": value} for value in (0.0, 1e-7, 0.25, math.nan)
        ] + [
            {"stage_weights": (-1.0, 1.0, 1.0)},
            {"stage_weights": (1.0, 1.0)},
            {"terminal_weights": (1.0, 1.0, math.inf)},
            {"control_weights": (0.1, 0.0)},
            {"control_weights": (0.1, math.nan)},
        ]
        for changes in cases:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(MpcConfig(), **changes)
        weights = [1.0, 2.0, 3.0]
        config = MpcConfig(stage_weights=weights)
        weights[0] = -1.0
        self.assertEqual(config.stage_weights, (1.0, 2.0, 3.0))

    def test_invalid_inputs_and_solver_failure(self) -> None:
        controller = MpcController(MpcConfig(horizon=2))
        for measured, reference in (
            (np.zeros(3), []),
            (np.zeros(3), np.zeros((4, 3))),
            (np.zeros(3), np.zeros((3, 2))),
            (np.zeros((3, 1)), np.zeros((3, 3))),
            ([math.nan, 0, 0], np.zeros((3, 3))),
            (np.zeros(3), np.full((3, 3), math.inf)),
        ):
            with self.subTest(measured=measured, reference=reference), self.assertRaises(ValueError):
                controller.solve(measured, reference)
        with patch.object(controller._opti, "solve", side_effect=RuntimeError("solver failed")):
            with self.assertRaisesRegex(RuntimeError, "solver failed"):
                controller.solve(np.zeros(3), np.zeros((3, 3)))

    def test_reject_invalid_solver_results(self) -> None:
        controller = MpcController(MpcConfig(horizon=1))
        measured = np.zeros(3)
        valid = controller.solve(measured, np.zeros((2, 3)))
        for command in ((0.0, 0.0), (0.5, 0.0), (0.1, -1.51), (0.1, 1.51), (math.nan, 0.0)):
            with self.subTest(command=command), self.assertRaises(RuntimeError):
                controller._validate_solution(
                    replace(valid, controls=np.array([command])), measured
                )
        for bad in (
            replace(valid, objective=math.nan),
            replace(valid, states=valid.states + [0.1, 0.0, 0.0]),
            replace(valid, states=np.array([[0, 0, 0], [1, 0, 0]], dtype=float)),
        ):
            with self.assertRaises(RuntimeError):
                controller._validate_solution(bad, measured)


if __name__ == "__main__":
    unittest.main()
