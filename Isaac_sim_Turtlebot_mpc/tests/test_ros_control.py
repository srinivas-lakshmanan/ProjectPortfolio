"""ROS adapter tests with message/node doubles; no ROS installation required."""

import importlib
import math
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from controller.mpc_controller import MpcConfig, MpcController


class Twist:
    def __init__(self) -> None:
        self.linear = SimpleNamespace(x=0.0, y=0.0, z=0.0)
        self.angular = SimpleNamespace(x=0.0, y=0.0, z=0.0)


class Node:
    def __init__(self, name: str) -> None:
        self.parameters = {}
        self.logger = Mock()
        self.publisher = Mock()

    def declare_parameter(self, name, default):
        self.parameters[name] = default

    def get_parameter(self, name):
        return SimpleNamespace(value=self.parameters[name])

    def get_logger(self):
        return self.logger

    def create_subscription(self, *args):
        self.subscription_args = args
        return Mock()

    def create_publisher(self, *args):
        self.publisher_args = args
        return self.publisher

    def create_timer(self, *args, **kwargs):
        self.timer_args = args
        return Mock()

    def destroy_node(self):
        pass


def module(name, **members):
    result = ModuleType(name)
    result.__dict__.update(members)
    return result


def odometry(stamp_ns=0, x=2.0, y=0.0, yaw=math.pi / 2, frame="odom"):
    return SimpleNamespace(
        header=SimpleNamespace(
            frame_id=frame,
            stamp=SimpleNamespace(sec=stamp_ns // 10**9, nanosec=stamp_ns % 10**9),
        ),
        pose=SimpleNamespace(pose=SimpleNamespace(
            position=SimpleNamespace(x=x, y=y),
            orientation=SimpleNamespace(x=0.0, y=0.0, z=math.sin(yaw / 2), w=math.cos(yaw / 2)),
        )),
    )


class RosControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ros_modules = {
            "nav_msgs": module("nav_msgs"),
            "nav_msgs.msg": module("nav_msgs.msg", Odometry=object),
            "geometry_msgs": module("geometry_msgs"),
            "geometry_msgs.msg": module("geometry_msgs.msg", Twist=Twist),
            "rclpy": module("rclpy", init=Mock(), spin=Mock(), ok=Mock(return_value=True), shutdown=Mock()),
            "rclpy.node": module("rclpy.node", Node=Node),
            "rclpy.qos": module(
                "rclpy.qos", qos_profile_sensor_data=object(),
                QoSProfile=lambda **kwargs: SimpleNamespace(**kwargs),
                ReliabilityPolicy=SimpleNamespace(RELIABLE=1),
                DurabilityPolicy=SimpleNamespace(VOLATILE=1),
            ),
            "rclpy.clock": module(
                "rclpy.clock", Clock=lambda **kwargs: object(),
                ClockType=SimpleNamespace(STEADY_TIME=1),
            ),
            "rclpy.signals": module("rclpy.signals", SignalHandlerOptions=SimpleNamespace(NO=0)),
        }
        with patch.dict(sys.modules, cls.ros_modules):
            cls.commands = importlib.import_module("ros.command_publisher")
            cls.states = importlib.import_module("ros.state_subscriber")

    @classmethod
    def tearDownClass(cls) -> None:
        for name in ("ros.command_publisher", "ros.state_subscriber"):
            sys.modules.pop(name, None)

    def setUp(self) -> None:
        self.clock = patch("time.monotonic", return_value=10.0).start()
        self.addCleanup(patch.stopall)
        self.controller = Mock()
        self.controller.config = MpcConfig()
        self.controller.solve.return_value = SimpleNamespace(controls=np.array([[0.4, 0.2]]))
        patch.object(self.commands, "MpcController", return_value=self.controller).start()
        self.node = self.commands.CommandPublisher()

    def assert_stopped(self) -> None:
        command = self.node.publisher.publish.call_args.args[0]
        self.assertEqual(vars(command.linear), {"x": 0.0, "y": 0.0, "z": 0.0})
        self.assertEqual(vars(command.angular), {"x": 0.0, "y": 0.0, "z": 0.0})

    def test_topics_and_waiting_for_odometry(self) -> None:
        self.assertEqual(self.node.subscription_args[1], "/odom")
        self.assertEqual(self.node.publisher_args[1], "/cmd_vel")
        self.assertEqual(self.node.timer_args[0], 0.1)
        self.node._control_step()
        self.assert_stopped()
        self.controller.solve.assert_not_called()
        self.node.logger.warning.assert_called_once()

    def test_quaternion_and_command_mapping(self) -> None:
        message = odometry()
        message.pose.pose.orientation.z *= 2
        message.pose.pose.orientation.w *= 2
        self.node._on_odometry(message)
        np.testing.assert_allclose(self.node.latest_state.pose, [2, 0, math.pi / 2])
        self.node._control_step()
        measured, reference = self.controller.solve.call_args.args
        np.testing.assert_allclose(measured, [2, 0, math.pi / 2])
        self.assertEqual(reference.shape, (21, 3))
        np.testing.assert_allclose(reference[0], measured)
        command = self.node.publisher.publish.call_args.args[0]
        self.assertEqual(command.linear.x, 0.4)
        self.assertEqual(command.angular.z, 0.2)
        self.assertEqual(command.linear.y, 0.0)
        self.assertEqual(command.angular.x, 0.0)

    def test_reference_uses_odometry_time(self) -> None:
        self.node._on_odometry(odometry(5_000_000_000))
        self.node._control_step()
        self.node._on_odometry(odometry(5_100_000_000))
        self.node._control_step()
        reference = self.controller.solve.call_args.args[1]
        np.testing.assert_allclose(reference[0, :2], [2 * math.cos(0.02), 2 * math.sin(0.02)])

    def test_duplicate_timestamps_do_not_refresh_watchdog(self) -> None:
        self.node._on_odometry(odometry())
        self.clock.return_value = 10.6
        self.node._on_odometry(odometry())
        self.node._control_step()
        self.assert_stopped()
        self.controller.solve.assert_not_called()
        warning = self.node.logger.warning.call_args.args[0]
        self.assertIn("advancing_pose_age=0.600s", warning)
        self.assertIn("odom_callback_age=0.000s", warning)
        self.assertIn("repeated_stamp_count=1", warning)
        self.node._on_odometry(odometry(100_000_000))
        self.node._control_step()
        self.assertEqual(self.node.publisher.publish.call_args.args[0].linear.x, 0.4)

    def test_missing_callbacks_are_distinguished_from_frozen_timestamps(self) -> None:
        self.node._on_odometry(odometry())
        self.node._control_step()
        self.clock.return_value = 10.6
        self.node._control_step()
        self.assert_stopped()
        warning = self.node.logger.warning.call_args.args[0]
        self.assertIn("advancing_pose_age=0.600s", warning)
        self.assertIn("odom_callback_age=0.600s", warning)
        self.assertIn("repeated_stamp_count=0", warning)

    def test_bad_odometry_latches_stop(self) -> None:
        for bad in (odometry(frame="map"), odometry(x=math.nan), odometry(stamp_ns=-1)):
            with self.subTest(message=bad):
                node = self.commands.CommandPublisher()
                node._on_odometry(bad)
                self.assertIsNotNone(node.state_fault)
                node._on_odometry(odometry())
                self.assertIsNone(node.latest_state)
                node._control_step()
                self.assertEqual(node.publisher.publish.call_args.args[0].linear.x, 0)
                node.logger.error.assert_called()
        bad = odometry()
        bad.pose.pose.orientation.z = bad.pose.pose.orientation.w = 0.0
        self.node._on_odometry(bad)
        self.assertIn("zero norm", self.node.state_fault)

    def test_time_reset_stops_then_resumes_with_new_reference_origin(self) -> None:
        self.node._on_odometry(odometry(1_000_000_000))
        self.node._control_step()
        self.controller.solve.reset_mock()
        self.node._on_odometry(odometry(0))
        self.assert_stopped()
        self.node._control_step()
        self.assert_stopped()
        self.assertIsNone(self.node.state_fault)
        self.controller.solve.assert_not_called()
        self.node._on_odometry(odometry(0))
        self.node._control_step()
        self.controller.solve.assert_not_called()
        self.node._on_odometry(odometry(100_000_000))
        self.node._control_step()
        reference = self.controller.solve.call_args.args[1]
        np.testing.assert_allclose(reference[0], [2, 0, math.pi / 2])
        self.assertEqual(self.node.publisher.publish.call_args.args[0].linear.x, 0.4)
        self.node._on_odometry(odometry(200_000_000))
        self.node._control_step()
        reference = self.controller.solve.call_args.args[1]
        np.testing.assert_allclose(reference[0, :2], [2 * math.cos(0.02), 2 * math.sin(0.02)])

    def test_reset_before_first_solve_and_repeated_resets(self) -> None:
        for high, low in ((10_000_000_000, 0), (2_000_000_000, 50_000_000)):
            self.node._on_odometry(odometry(high))
            self.node._on_odometry(odometry(low))
            self.node._control_step()
            self.assert_stopped()
            self.assertIsNone(self.node.latest_state)
            self.node._on_odometry(odometry(low + 100_000_000))
            self.node._control_step()
            np.testing.assert_allclose(
                self.controller.solve.call_args.args[1][0], [2, 0, math.pi / 2]
            )

    def test_reset_does_not_clear_solver_fault(self) -> None:
        self.node._on_odometry(odometry(1_000_000_000))
        self.controller.solve.side_effect = RuntimeError("IPOPT failed")
        self.node._control_step()
        self.controller.solve.reset_mock()
        self.node._on_odometry(odometry(0))
        self.node._on_odometry(odometry(100_000_000))
        self.node._control_step()
        self.assert_stopped()
        self.controller.solve.assert_not_called()
        self.assertIsNotNone(self.node._solver_fault)

    def test_same_sample_is_not_solved_twice(self) -> None:
        self.node._on_odometry(odometry())
        self.node._control_step()
        self.node._control_step()
        self.controller.solve.assert_called_once()

    def test_solver_failure_latches_stop(self) -> None:
        self.node._on_odometry(odometry())
        self.controller.solve.side_effect = RuntimeError("IPOPT failed")
        self.node._control_step()
        self.assert_stopped()
        self.node.logger.error.assert_called()
        self.controller.solve.reset_mock()
        self.node._control_step()
        self.controller.solve.assert_not_called()
        self.assert_stopped()

    def test_out_of_bounds_command_is_not_published(self) -> None:
        for command in ((0.5, 0), (0, 0), (0.4, 1.6), (math.nan, 0)):
            with self.subTest(command=command):
                node = self.commands.CommandPublisher()
                node._on_odometry(odometry())
                self.controller.solve.return_value = SimpleNamespace(controls=np.array([command]))
                node._control_step()
                self.assertEqual(node.publisher.publish.call_args.args[0].linear.x, 0)
                self.assertIsNotNone(node._solver_fault)

    def test_slow_solve_discards_expired_command(self) -> None:
        self.node._on_odometry(odometry())
        def slow_solve(*args):
            self.clock.return_value = 11.0
            return SimpleNamespace(controls=np.array([[0.4, 0.2]]))
        self.controller.solve.side_effect = slow_solve
        self.node._control_step()
        self.assert_stopped()
        self.assertIsNone(self.node._solver_fault)
        self.assertIn("last_solve=1.000s", self.node.logger.warning.call_args.args[0])

    def test_real_mpc_command_from_odometry(self) -> None:
        self.node._controller = MpcController(MpcConfig())
        self.node._on_odometry(odometry())
        self.node._control_step()
        command = self.node.publisher.publish.call_args.args[0]
        self.assertGreater(command.linear.x, 0.0)
        self.assertLess(command.linear.x, 0.5)
        self.assertLessEqual(abs(command.angular.z), 1.5)

    def test_shutdown_publishes_stop(self) -> None:
        with patch.object(self.commands, "CommandPublisher", return_value=self.node):
            with patch.object(self.commands.rclpy, "spin", side_effect=KeyboardInterrupt):
                self.commands.main()
        self.node._timer.cancel.assert_called_once()
        self.assert_stopped()


if __name__ == "__main__":
    unittest.main()
