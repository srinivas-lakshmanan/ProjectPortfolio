"""Run with python -m ros.command_publisher in a sourced ROS 2 environment."""

import math
import time

from geometry_msgs.msg import Twist
import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.signals import SignalHandlerOptions

from controller.mpc_controller import MpcConfig, MpcController
from controller.trajectory import generate_reference
from .state_subscriber import StateSubscriber


class CommandPublisher(StateSubscriber):
    def __init__(self) -> None:
        super().__init__()
        self.declare_parameter("horizon", 20)
        self.declare_parameter("dt", 0.1)
        self.declare_parameter("odom_timeout", 0.5)
        config = MpcConfig(
            horizon=self.get_parameter("horizon").value,
            dt=self.get_parameter("dt").value,
        )
        self._odom_timeout = self.get_parameter("odom_timeout").value
        if not math.isfinite(self._odom_timeout) or self._odom_timeout <= config.dt:
            raise ValueError("odom_timeout must be finite and greater than dt.")
        self._controller = MpcController(config)
        self._publisher = self.create_publisher(
            Twist, "/cmd_vel",
            QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
            ),
        )
        self._origin_ns: int | None = None
        self._last_solved_stamp_ns: int | None = None
        self._solver_fault: str | None = None
        self._stop_reason: str | None = None
        self._last_solve_duration = 0.0
        # A steady timer still stops commands when simulation time is paused.
        self._timer = self.create_timer(
            config.dt, self._control_step, clock=Clock(clock_type=ClockType.STEADY_TIME)
        )
        self.get_logger().info("MPC ready: /odom -> circular reference -> /cmd_vel.")

    def stop(self) -> None:
        """Safety stop deliberately bypasses the MPC's strictly positive speed."""
        self._publisher.publish(Twist())

    def _stop_with_reason(self, reason: str) -> None:
        if self._stop_reason != reason:
            details = ""
            if self.latest_state is not None:
                now = time.monotonic()
                callback_age = (
                    now - self.last_odom_received_at
                    if self.last_odom_received_at is not None else math.inf
                )
                details = (
                    f" [advancing_pose_age={now - self.latest_state.received_at:.3f}s,"
                    f" odom_callback_age={callback_age:.3f}s,"
                    f" repeated_stamp_count={self.duplicate_stamp_count},"
                    f" odom_stamp_ns={self.latest_state.stamp_ns},"
                    f" last_solve={self._last_solve_duration:.3f}s,"
                    f" timeout={self._odom_timeout:.3f}s]"
                )
            self.get_logger().warning(f"Publishing stop: {reason}{details}")
            self._stop_reason = reason
        self.stop()

    def _on_time_reset(self) -> None:
        super()._on_time_reset()
        self._origin_ns = None
        self._last_solved_stamp_ns = None
        self._stop_with_reason("Simulation time reset; waiting for advancing /odom.")

    def _control_step(self) -> None:
        if self.state_fault is not None or self._solver_fault is not None:
            self._stop_with_reason(self.state_fault or self._solver_fault)
            return
        state = self.latest_state
        if state is None:
            self._stop_with_reason("Waiting for valid /odom.")
            return
        if time.monotonic() - state.received_at > self._odom_timeout:
            self._stop_with_reason("Odometry is stale or simulation is paused.")
            return
        if state.stamp_ns == self._last_solved_stamp_ns:
            return
        if self._origin_ns is None:
            self._origin_ns = state.stamp_ns
        elapsed = (state.stamp_ns - self._origin_ns) * 1e-9
        config = self._controller.config
        self._last_solved_stamp_ns = state.stamp_ns
        try:
            reference = generate_reference(elapsed, config.horizon, config.dt).T
            solve_started = time.monotonic()
            solution = self._controller.solve(state.pose, reference)
            self._last_solve_duration = time.monotonic() - solve_started
            v, omega = (float(value) for value in solution.controls[0])
            if not (math.isfinite(v) and math.isfinite(omega) and 0 < v < 0.5 and -1.5 <= omega <= 1.5):
                raise ValueError("MPC command violates velocity limits.")
        except (ValueError, RuntimeError) as error:
            self._solver_fault = f"MPC failed; restart controller: {error}"
            self.get_logger().error(self._solver_fault)
            self._stop_with_reason(self._solver_fault)
            return
        if time.monotonic() - state.received_at > self._odom_timeout:
            self._stop_with_reason("Odometry expired during the solve.")
            return
        command = Twist()
        command.linear.x = v
        command.angular.z = omega
        if self._stop_reason is not None:
            self.get_logger().info("Fresh odometry available; resuming MPC commands.")
            self._stop_reason = None
        self._publisher.publish(command)


def main() -> None:
    # Keep the ROS context alive until the final stop has been published on Ctrl+C.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node: CommandPublisher | None = None
    try:
        node = CommandPublisher()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node._timer.cancel()
            if rclpy.ok():
                node.stop()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
