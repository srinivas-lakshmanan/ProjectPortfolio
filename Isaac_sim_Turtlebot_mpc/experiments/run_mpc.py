"""Run with python -m experiments.run_mpc from the project root."""

import casadi as ca
import numpy as np

from controller.mpc_controller import MpcConfig, MpcController
from controller.robot_model import step_euler
from controller.trajectory import generate_reference


def main() -> None:
    config = MpcConfig()
    controller = MpcController(config)
    state = generate_reference(0.0, config.horizon, config.dt)[:, 0].copy()
    for step in range(10):
        reference = generate_reference(step * config.dt, config.horizon, config.dt).T
        solution = controller.solve(state, reference)
        command = solution.controls[0]
        print(
            f"step={step} v={command[0]:.6f} omega={command[1]:.6f} "
            f"cost={solution.objective:.6f}"
        )
        state = np.asarray(step_euler(ca.DM(state), ca.DM(command), config.dt)).reshape(3)


if __name__ == "__main__":
    main()
