# Isaac Sim MPC project

Trajectory-tracking MPC and the differential-drive model use **Python CasADi
and IPOPT**. No C++ code or CMake build is required.

## Project structure

```text
Isaac_sim_Turtlebot_mpc/
|-- simulation/
|   `-- isaac_sim_scene/          # Existing USD/USDA scene assets
|-- controller/
|   |-- __init__.py
|   |-- mpc_controller.py
|   |-- trajectory.py
|   `-- robot_model.py
|-- ros/
|   |-- __init__.py
|   |-- state_subscriber.py
|   `-- command_publisher.py
|-- experiments/
|   |-- __init__.py
|   `-- run_mpc.py
|-- results/                     # Future experiment plots
|-- tests/
|   |-- test_mpc_controller.py
|   |-- test_trajectory.py
|   `-- test_ros_control.py
|-- requirements.txt
`-- README.md
```

ROS adapters implement the odometry-to-MPC-to-command loop. The numeric experiment does not launch
Isaac Sim or ROS. Scene assets are preserved under `simulation/isaac_sim_scene`.
`results` is tracked with `.gitkeep`; plots will come from actual experiment data.

## Python setup and execution

Use Python 3.10+ and run these commands from the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
.\.venv\Scripts\python.exe -m experiments.run_mpc
```

Dependencies install into the project's ignored `.venv`, not the source tree or
global environment. Select `.venv\Scripts\python.exe` as the VS Code interpreter.
The CasADi Python package supplies the native solver libraries and IPOPT plugin;
no separate C++ CasADi SDK, compiler ABI configuration, or `casadi_DIR` is needed.
This project does not depend on a Copilot session-local installation.

## Isaac Sim ROS 2 control loop

In the Isaac Sim ROS 2 bridge, enable:

- `nav_msgs/msg/Odometry` publishing on `/odom` with simulation timestamps.
- `geometry_msgs/msg/Twist` subscription on `/cmd_vel`, connected to the robot's
  differential-drive controller.

Run Isaac Sim in Play mode. Use the same `ROS_DOMAIN_ID` on both sides and
compatible DDS/QoS settings. The odometry subscriber uses sensor-data QoS
(best effort, volatile), compatible with best-effort or reliable publishers.
Commands use reliable, volatile QoS with depth 1.

Run the controller **outside Isaac Sim**, from the project root in a sourced
ROS 2 environment providing `rclpy`, `nav_msgs`, and `geometry_msgs`:

```powershell
python -m ros.command_publisher
```

ROS packages come from your ROS 2 installation, not `requirements.txt`. Use that
installation's compatible Python interpreter with CasADi/NumPy installed.
A virtual environment must expose ROS's Python packages (for example
`python -m venv --system-site-packages .venv-ros` after sourcing ROS), and its
Python version must match ROS's binary extensions. The numeric-only `.venv`
created above does not by itself provide ROS 2.

The node is named `turtlebot_mpc`. Parameters can be set at launch:

```powershell
python -m ros.command_publisher --ros-args -p horizon:=20 -p dt:=0.1 -p odom_timeout:=0.5 -p odom_frame:=odom
```

`state_subscriber.py` converts normalized odometry quaternions to yaw and stores
`[x, y, theta]`. The odometry frame must match `odom_frame` (default `odom`).
The circular reference is expressed directly in that frame; no TF transformation
or automatic translation/rotation is applied. Its initial pose is
`[2, 0, pi/2]`; initialize the simulated robot nearby for tracking tests.

`command_publisher.py` runs a steady-clock timer every `dt` seconds (default
10 Hz), generates a horizon, solves MPC, and publishes only the first control:
`Twist.linear.x = v`, `Twist.angular.z = omega`; other components are zero.
Reference time is elapsed `/odom` timestamp time since the first valid control
sample, not wall time. `/clock` and `use_sim_time` are not required.

The timer publishes zero velocity while waiting for odometry and when no
advancing odometry has arrived for `odom_timeout` wall-clock seconds.
Duplicate timestamps do not refresh this watchdog, and a sample is solved at most
once. Fresh odometry resumes
tracking. If a solve takes long enough for its input to expire, its command is
discarded and zero is published. On a backward timestamp jump, the node immediately
publishes zero, clears the previous pose/reference clock, and waits for a timestamp
newer than the reset sample. It then automatically restarts the reference at
`t=0`. This handles simulation Stop/Play or reset without restarting the node.
A timestamp alone cannot distinguish a reset from reordered messages or competing
publishers: repeated reset warnings during normal playback require checking
`ros2 topic info /odom --verbose` and ensuring only one intended odometry source.
Malformed odometry, frame mismatches, invalid commands, or solver failures still
latch a stop and log the reason; restart after correcting those faults.
Ctrl+C attempts a final zero command before shutting down.

Stop warnings include advancing-pose age, odometry-callback age, repeated
timestamp count, last odometry timestamp, last successful solve duration, and
the configured timeout. A recent callback with an old advancing pose indicates
repeated timestamps; both ages being large indicates no recently processed valid
odometry. Long solve durations indicate the synchronous solver may be delaying
subscription processing. Topic rate alone does not distinguish these cases.

Zero-speed safety stops intentionally override the optimizer's `v > 0` bound.
This synchronous Python loop is not hard real-time: it cannot publish during a
blocked solve. Configure an independent command timeout/watchdog in the simulator
or downstream robot controller; a final ROS stop message is not guaranteed to be
delivered on shutdown or process failure. Avoid running multiple `/cmd_vel`
publishers at the same time.

## Differential-drive kinematics

The pose is at the midpoint of the drive axle. State `[x, y, theta]` uses
world-frame metres and radians; control `[v, omega]` uses m/s and rad/s.
Heading is counterclockwise from world positive x. Positive linear speed is
forward and positive angular speed turns counterclockwise.

```text
x_dot     = v * cos(theta)
y_dot     = v * sin(theta)
theta_dot = omega

state_next = state + dt * dynamics(state, control)
```

The Python `robot_model` implements these equations for CasADi `SX`, `MX`, and
`DM` values using forward Euler, not exact circular-arc integration. The sampling interval
must be finite and positive; heading is intentionally unwrapped. Model functions
do not clamp controls or validate state finiteness; the MPC validates its inputs.
Wheel dimensions are unnecessary because inputs are body velocities.

## MPC formulation

`MpcController` builds a CasADi `Opti` multiple-shooting problem once and
updates measured/reference parameters at each solve. Variables are `N+1`
predicted states and `N` controls:

```text
e_x[k] = x[k] - x_ref[k]
e_y[k] = y[k] - y_ref[k]
e_heading[k] = theta[k] - theta_ref[k]

L[k] = q_x * e_x[k]^2 + q_y * e_y[k]^2
       + 2*q_heading * (1 - cos(e_heading[k]))
       + r_v * v[k]^2 + r_omega * omega[k]^2

J_f = qf_x * e_x[N]^2 + qf_y * e_y[N]^2
      + 2*qf_heading * (1 - cos(e_heading[N]))

minimize sum(L[k], k=0..N-1) + J_f
subject to:
    state[0] = measured_state
    state[k+1] = step_euler(state[k], control[k], dt)
    velocity_margin <= v[k] <= 0.5 - velocity_margin
    -1.5 <= omega[k] <= 1.5
```

The periodic heading cost is smooth across the heading wrap boundary.
Supply reference headings along the direction of travel, for example
`atan2(dy/dt, dx/dt)`; retain the last valid heading at stationary points.
Controls are penalized by magnitude, not command changes. There are no obstacle,
acceleration, or terminal equality constraints.

`MpcConfig` defaults:

| Setting | Default |
| --- | --- |
| Horizon `N` | 20 |
| Sampling interval `dt` | 0.1 seconds |
| Stage weights `[x, y, heading]` | `(10, 10, 1)` |
| Terminal weights `[x, y, heading]` | `(50, 50, 5)` |
| Control weights `[v, omega]` | `(0.1, 0.1)` |
| Velocity margin | `1e-4` m/s |

These are starting values for tuning, not stability or performance guarantees.
State weights must be finite and nonnegative; control weights must be finite
and positive. Configuration is immutable after construction.

Strict inequalities `0 < v < 0.5` use a closed interior interval, by default
`0.0001 <= v <= 0.4999`. The margin must be in `(1e-6, 0.25)`.
The positive minimum prevents stopping or pure in-place rotation, even at the
final reference pose.

### Controller API

```python
import numpy as np
from controller.mpc_controller import MpcConfig, MpcController
from controller.trajectory import generate_reference

config = MpcConfig()
controller = MpcController(config)
reference = generate_reference(0.0, config.horizon, config.dt)
measured = reference[:, 0].copy()
reference = reference.T
solution = controller.solve(measured, reference)
command = solution.controls[0]  # [v, omega]
```

Inputs must be finite with shapes `(3,)` and `(N+1, 3)`. Output
`MpcSolution` contains NumPy arrays `states` of shape `(N+1, 3)`,
`controls` of shape `(N, 2)`, and a scalar `objective`.
Apply only the first control, then update the measured pose and reference window.
The example runs ten numeric receding-horizon steps tracking a counterclockwise
circle, initialized at its first reference pose. `generate_reference(t, N, dt)`
returns `(3, N+1)` columns; transpose once for `solve`. The circle has radius
2 m, angular rate 0.2 rad/s, and tangent speed 0.4 m/s:
`x=2*cos(0.2*t)`, `y=2*sin(0.2*t)`. Heading is `atan2(dy/dt, dx/dt)`;
its wrap at +/-pi is handled by the periodic orientation cost.

Each solve starts from a feasible constant-velocity rollout. Warm starting and
real-time scheduling are not implemented. Do not share one controller instance
between concurrent calls.

Invalid inputs raise `ValueError`. IPOPT failures propagate as exceptions,
never as fallback commands. IPOPT uses `1e-8` constraint tolerance and disables
bound relaxation. Returned solutions are checked for finite values, velocity
bounds, and initial/dynamics residuals (tolerance `1e-6`). IPOPT is a local solver:
convergence and global optimality are not guaranteed. Handle exceptions and
implement a safety policy before commanding real hardware.

## Tests

Python tests exercise actual IPOPT solves, output shapes, equality constraints,
velocity limits, costs including terminal influence, heading periodicity,
closed-loop tracking, failure propagation, invalid solutions, and model equations.
ROS adapter tests use node/message doubles and also exercise the real MPC solver.
They do not verify DDS discovery, Isaac Sim wiring, or ROS runtime compatibility;
those require a running ROS 2/Isaac Sim environment.
