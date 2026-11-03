import math
import unittest

import numpy as np

from controller.mpc_controller import MpcConfig, MpcController
from controller.trajectory import generate_reference


class TrajectoryTests(unittest.TestCase):
    def test_shape_circle_and_tangent_heading(self) -> None:
        for t in (0.0, 3.0, math.pi / 0.2, -2.0):
            with self.subTest(t=t):
                ref = generate_reference(t, N=7, dt=0.15)
                self.assertEqual(ref.shape, (3, 8))
                times = t + np.arange(8) * 0.15
                phase = 0.2 * times
                np.testing.assert_allclose(ref[0], 2 * np.cos(phase))
                np.testing.assert_allclose(ref[1], 2 * np.sin(phase))
                np.testing.assert_allclose(np.hypot(ref[0], ref[1]), 2.0)
                np.testing.assert_allclose(np.cos(ref[2]), -np.sin(phase), atol=1e-14)
                np.testing.assert_allclose(np.sin(ref[2]), np.cos(phase), atol=1e-14)
        np.testing.assert_allclose(generate_reference(0.0)[:, 0], [2, 0, math.pi / 2])
        self.assertEqual(generate_reference(0.0).shape, (3, 21))

    def test_rolling_window(self) -> None:
        first = generate_reference(1.0)
        second = generate_reference(1.1)
        np.testing.assert_allclose(first[:, 1:], second[:, :-1], atol=1e-14)

    def test_invalid_inputs(self) -> None:
        for kwargs in (
            {"t": math.nan}, {"t": math.inf}, {"N": 0}, {"N": -1},
            {"N": 1.5}, {"N": True}, {"dt": 0}, {"dt": -0.1},
            {"dt": math.nan}, {"dt": math.inf},
        ):
            arguments = {"t": 0.0, **kwargs}
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                generate_reference(**arguments)

    def test_mpc_tracks_circle_across_heading_wrap(self) -> None:
        config = MpcConfig()
        controller = MpcController(config)
        for start in (0.0, (math.pi / 2 - 0.05) / 0.2):
            measured = generate_reference(start)[:, 0].copy()
            for step in range(10):
                reference = generate_reference(
                    start + step * config.dt, config.horizon, config.dt
                ).T
                result = controller.solve(measured, reference)
                self.assertEqual(result.controls.shape, (config.horizon, 2))
                self.assertTrue(((result.controls[:, 0] > 0) & (result.controls[:, 0] < 0.5)).all())
                self.assertTrue((np.abs(result.controls[:, 1]) <= 1.5).all())
                measured = result.states[1].copy()
                next_reference = generate_reference(start + (step + 1) * config.dt)[:, 0]
                self.assertLess(np.linalg.norm(measured[:2] - next_reference[:2]), 0.05)


if __name__ == "__main__":
    unittest.main()
