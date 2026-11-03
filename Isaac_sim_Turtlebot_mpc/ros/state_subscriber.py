"""Receive planar pose from Isaac Sim's ROS 2 odometry publisher."""

from dataclasses import dataclass
import math
import time

from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data


@dataclass(frozen=True)
class RobotState:
    pose: tuple[float, float, float]
    stamp_ns: int
    received_at: float


class StateSubscriber(Node):
    def __init__(self) -> None:
        super().__init__("turtlebot_mpc")
        self.declare_parameter("odom_frame", "odom")
        self._odom_frame = self.get_parameter("odom_frame").value
        if not isinstance(self._odom_frame, str) or not self._odom_frame:
            raise ValueError("odom_frame must be a nonempty frame name.")
        self.latest_state: RobotState | None = None
        self.state_fault: str | None = None
        self._last_stamp_ns: int | None = None
        self.last_odom_received_at: float | None = None
        self.duplicate_stamp_count = 0
        self._subscription = self.create_subscription(
            Odometry, "/odom", self._on_odometry, qos_profile_sensor_data
        )

    def _on_odometry(self, message: Odometry) -> None:
        if self.state_fault is not None:
            return
        try:
            if message.header.frame_id != self._odom_frame:
                raise ValueError(
                    f"Expected odometry frame '{self._odom_frame}', "
                    f"got '{message.header.frame_id}'."
                )
            position = message.pose.pose.position
            q = message.pose.pose.orientation
            if not all(math.isfinite(v) for v in (position.x, position.y, q.x, q.y, q.z, q.w)):
                raise ValueError("Odometry pose contains nonfinite values.")
            norm = math.hypot(q.x, q.y, q.z, q.w)
            if norm < 1e-12:
                raise ValueError("Odometry quaternion has zero norm.")
            qx, qy, qz, qw = (v / norm for v in (q.x, q.y, q.z, q.w))
            yaw = math.atan2(
                2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)
            )
            stamp = message.header.stamp
            if stamp.sec < 0 or not 0 <= stamp.nanosec < 1_000_000_000:
                raise ValueError("Invalid odometry timestamp.")
            stamp_ns = stamp.sec * 1_000_000_000 + stamp.nanosec
            self.last_odom_received_at = time.monotonic()
            if self._last_stamp_ns is not None:
                if stamp_ns < self._last_stamp_ns:
                    self._last_stamp_ns = stamp_ns
                    self.latest_state = None
                    self._on_time_reset()
                    return
                if stamp_ns == self._last_stamp_ns:
                    self.duplicate_stamp_count += 1
                    return
        except ValueError as error:
            self.latest_state = None
            self.state_fault = str(error)
            self.get_logger().error(f"Odometry fault: {error}")
            return
        self._last_stamp_ns = stamp_ns
        self.duplicate_stamp_count = 0
        self.latest_state = RobotState((position.x, position.y, yaw), stamp_ns, time.monotonic())

    def _on_time_reset(self) -> None:
        self.get_logger().warning(
            "Odometry time moved backward. Restarting reference time; "
            "waiting for advancing odometry. Check for multiple /odom publishers "
            "if this repeats without a simulation reset."
        )
