# Neural MPC Controller — Double-Track Vehicle Model

A full 4-wheel (double-track) vehicle dynamics model in modern C++ (header-only),
designed to be coupled with a CasADi-based MPC controller.

## Model overview

7 dynamic degrees of freedom + global pose kinematics:

| | |
|---|---|
| **States** (NX = 10) | `X, Y, psi` (global pose), `vx, vy, r` (body velocities, yaw rate), `w_fl, w_fr, w_rl, w_rr` (wheel speeds) |
| **Inputs** (NU = 5) | `delta` (front road-wheel steer), `T_fl, T_fr, T_rl, T_rr` (independent wheel torques) |

Because each wheel receives an independent drive/brake torque, **differential
vehicle dynamics** (differential drive, torque vectoring, asymmetric braking)
are supported natively — see the `torque_vectoring` demo scenario where the
vehicle yaws with zero steering input.

### Physics included

- Planar rigid-body dynamics (longitudinal, lateral, yaw)
- Per-wheel spin dynamics: `Iw * w_dot = T - Re * Fx`
- Pacejka magic-formula tires (pure slip) with a smooth combined-slip
  friction-ellipse scaling
- Per-wheel slip ratio / slip angle computed from contact-patch velocities
- Quasi-static longitudinal and lateral load transfer (roll-stiffness split)
- Aerodynamic drag and rolling resistance

### CasADi function wrapper

The dynamics `x_dot = f(x, u)` are templated on the scalar type and written as
smooth, branch-free expressions (all divisions/abs/sign are ε-regularized), so
the identical code path works with:

- `double` — forward simulation (as in the demo/tests), and
- `casadi::SX` / `casadi::MX` — symbolic graphs for MPC transcription.

The optional `vehicle_model_casadi` CMake target supplies a working wrapper.
It constructs SX functions that can be evaluated numerically with DM or called
from SX/MX optimization graphs. Link your controller executable to this target.

```cpp
#include "vehicle_model/casadi_vehicle_model.hpp"

using vehicle_model::DoubleTrackModel;
using casadi::MX;

DoubleTrackModel model;
const double sample_time = 0.02;  // controller prediction interval [s]
auto vehicle = vehicle_model::make_casadi_vehicle_functions(model, sample_time, 20);

MX x = MX::sym("x", DoubleTrackModel::NX);
MX u = MX::sym("u", DoubleTrackModel::NU);
MX x_next = MX::sym("x_next", DoubleTrackModel::NX);
MX predicted = vehicle.discrete(std::vector<MX>{x, u}).at(0);
MX shooting_residual = x_next - predicted;  // impose == 0 in your MPC
```

All functions accept named column-vector inputs `x` (10 x 1) and `u` (5 x 1).

| Function member | Outputs |
|---|---|
| `continuous` | `xdot` (10 x 1) |
| `discrete` | `x_next` (10 x 1) |
| `continuous_jacobian` | `A = d(xdot)/dx` (10 x 10), `B = d(xdot)/du` (10 x 5) |
| `discrete_jacobian` | `A = d(x_next)/dx` (10 x 10), `B = d(x_next)/du` (10 x 5) |
| `diagnostics` | `kappa`, `alpha`, `Fz`, `Fx_w`, `Fy_w` (each 4 x 1, FL/FR/RL/RR), `ax`, `ay` (scalars) |

Parameters and integration settings are constants captured at construction;
rebuild the functions after changing vehicle parameters. Inputs are held
constant over each prediction interval. The discrete function uses
`rk4_substeps` RK4 steps (default 20), with internal step duration
`sample_time / rk4_substeps`. Nonpositive/nonfinite sample times and nonpositive
substep counts throw `std::invalid_argument`.

Wheel-spin dynamics can be stiff. The example retains the simulation's 1 ms
internal step, rather than using one large 20 ms RK4 step. Choose substeps based
on numerical convergence for your parameters and operating range; no automatic
stability guarantee is provided.

This wrapper does not add costs, constraints, a solver, or driving scenarios.
The tracking controller and maneuver runner below supply those separately.
Existing tire and load-transfer approximations are unchanged.

## Layout

```
include/vehicle_model/
  vehicle_params.hpp       # physical parameters (mid-size car defaults)
  smooth_math.hpp          # eps-regularized abs / sign / division
  tire_model.hpp           # Pacejka magic formula + combined slip
  double_track_model.hpp   # the 7-DOF double-track ODE  x_dot = f(x, u)
  integrator.hpp           # templated RK4 step
  casadi_vehicle_model.hpp # continuous/discrete CasADi functions and Jacobians
  tracking_mpc.hpp         # IPOPT nonlinear multiple-shooting tracking controller
  maneuver_reference.hpp  # configurable step-steer and double-lane-change references
src/main.cpp               # demo: step steer + torque-vectoring scenarios (CSV output)
src/mpc_maneuvers.cpp       # closed-loop MPC scenarios, CLI, CSVs and acceptance metrics
test/test_double_track_model.cpp  # physics sanity tests
test/test_casadi_vehicle_model.cpp # wrapper parity, Jacobians, MX composition
test/test_maneuver_reference.cpp   # reference profiles and controller contracts
```

## Build & run

```sh
cmake -S . -B build
cmake --build build
./build/test_double_track_model   # sanity tests
./build/vehicle_demo              # writes step_steer.csv, torque_vectoring.csv
```

(On Windows with MSYS2 g++, pass the generator/compiler explicitly, e.g.
`cmake -S . -B build -G "Unix Makefiles" -DCMAKE_MAKE_PROGRAM=C:/msys64/usr/bin/make.exe -DCMAKE_CXX_COMPILER=C:/msys64/ucrt64/bin/g++.exe`.)

### Build with CasADi

CasADi is optional and defaults to OFF, preserving the dependency-free build.
Provide a CasADi 3.7+ C++ SDK compiled with a compatible compiler/ABI and set
`casadi_DIR` to the folder containing `casadi-config.cmake`.

The following PowerShell commands reproduce the verified Windows build using
the CasADi wheel's C++ SDK and MSYS2 **mingw64** g++ (MSVCRT, not ucrt64).
Install the `mingw-w64-x86_64-gcc` package if that compiler is missing.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-casadi.txt
$env:PATH = "C:\msys64\mingw64\bin;$env:PATH"
cmake -S . -B build-mpc -G "Unix Makefiles" `
  "-DCMAKE_MAKE_PROGRAM=C:\msys64\usr\bin\make.exe" `
  "-DCMAKE_CXX_COMPILER=C:\msys64\mingw64\bin\g++.exe" `
  -DVEHICLE_MODEL_WITH_CASADI=ON `
  "-Dcasadi_DIR=$PWD\.venv\Lib\site-packages\casadi\cmake"
cmake --build build-mpc --parallel 4
ctest --test-dir build-mpc --output-on-failure
```

The Windows CasADi executable targets copy the CasADi DLL, bundled MinGW runtime
DLLs, and available IPOPT/JIT plugin dependencies beside the executables. Do
the same for your controller executable, or prepend the SDK's DLL directory
to its runtime PATH. MSVC builds require an MSVC-compatible CasADi SDK; do not
mix a MinGW wheel with MSVC.
The wheel used here is MSVCRT-based: mixing it with ucrt64 can compile and pass
basic wrapper tests yet crash during repeated IPOPT solves. Use a separate
build directory when changing compiler/ABI.

Wrapper tests verify numerical parity (including standstill and changed
parameters), output shapes/order, finite Jacobians and finite-difference
agreement, MX multiple-shooting composition, and function serialization.
The two additional closed-loop CTest cases run the complete default maneuvers.
Use `ctest --test-dir build-mpc -L closed-loop --output-on-failure` to run only
those, or `ctest --test-dir build-mpc -LE closed-loop --output-on-failure` for
the smaller model/wrapper/reference/controller-contract suite.

## Closed-loop MPC maneuvers

Both scenarios use the full ten-state/five-input vehicle model. Every control
interval, the MPC solves a nonlinear multiple-shooting problem with IPOPT,
applies only the first optimized input, and advances an independent `double`
RK4 plant. The warm start shifts the prior optimized controls and rolls out
the new measured state. All four wheel torques remain independent decisions.

The objective penalizes full-state reference error, input/feedforward error,
and input increments, with increased terminal state weight. Hard constraints
enforce the model's discrete dynamics, measured initial state, steering/torque
bounds, and steering/torque slew rates. Yaw error is wrapped. Solver failure,
nonfinite data, and excessive solution constraint violation produce explicit
errors and a nonzero exit; no fallback controller hides failures.

These are **illustrative, configurable tests**, not ISO lane-change or other
standards-certified maneuvers. Fish-hook is intentionally excluded.

### References and defaults

| Setting | Default |
|---|---|
| Initial/nominal speed | 10 m/s (36 km/h) |
| MPC interval / horizon | 0.05 s / 12 steps (0.6 s preview) |
| Internal RK4 step | 1 ms (50 substeps per interval) |
| Road-wheel steering bound / slew | +/-0.15 rad / 0.30 rad/s |
| Wheel torque bound / slew | +/-200 Nm / 1500 Nm/s per wheel |
| Step-steer | 0 to 0.025 rad (1.43 deg) at 1 s; 6 s simulation |
| Double lane change | 3 m left shift over 2.5 s, hold 1 s, return over 2.5 s |
| Lane-change start / final settling | 1 s / 2 s (9 s total) |
| Initial lateral perturbation | 0 m |

**Step-steer response tracking:** a nominal model receives the steering step
and a bounded speed-holding torque policy to generate a dynamically feasible
state/input reference. The closed-loop plant receives **MPC commands**, not
the imposed step: its steering can ramp and anticipate through preview. This
is distinct from the original open-loop `vehicle_demo` step-steer test.

**Double lane change:** the time-based path uses two C2 quintic lateral shifts,
`X_ref = speed * t`, tangent heading, and corresponding yaw-rate/speed
references. Kinematic steering and cruise torque are feedforward guesses,
not claims that the geometric reference exactly satisfies the dynamic model.
Negative `--lane-width` reverses the maneuver. Both references extend through
the entire final prediction horizon.

### Running and overriding settings

```powershell
$env:PATH = "C:\msys64\mingw64\bin;$env:PATH"
$env:OPENBLAS_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
New-Item -ItemType Directory -Force -Path .\build-mpc\mpc_results | Out-Null
.\build-mpc\mpc_maneuvers.exe --scenario all --output-dir .\build-mpc\mpc_results
.\build-mpc\mpc_maneuvers.exe --help

# Examples of overrides (adjust thresholds for your chosen operating point).
.\build-mpc\mpc_maneuvers.exe --scenario step-steer --speed 8 `
  --step-angle-deg 1 --start-time 0.5 --step-duration 4 `
  --initial-y-error 0.1 --output-dir .\build-mpc\mpc_results
.\build-mpc\mpc_maneuvers.exe --scenario double-lane-change --speed 8 `
  --lane-width -2 --lane-change-time 2.5 --lane-hold-time 0.5 `
  --output-dir .\build-mpc\mpc_results
```

The CLI also exposes sample time, horizon, substeps, actuator limits/rates,
iteration cap, and acceptance thresholds; see `--help`. Cost weights and
solver tolerances are configurable in `TrackingMpcConfig` for controller
clients. `TrackingMpc::solve` takes the measured state, previous applied input,
and a `TrackingReference` of `horizon+1` states and `horizon` inputs.

JIT is enabled by default to compile CasADi's dynamics/derivative evaluations
with GCC-compatible commands (`gcc` on PATH). Initial construction includes
compilation time and needs a writable current directory; generated temporary
files are cleaned up by CasADi on normal exit. Override the compiler with
`--jit-compiler COMMAND` or disable compilation with `--jit false` (much slower).
No real-time solve-time guarantee is claimed. Thread limits in the examples
and CTest reduce small-problem BLAS overhead without changing global settings.

### Outputs and acceptance

`--output-dir` must already exist; default is the current directory.
Each run overwrites only its own trajectory and summary:

- `step_steer_mpc.csv`, `double_lane_change_mpc.csv`: states, full references,
  applied wheel torques/steering, tire diagnostics, tracking errors, solver
  iterations, solve time, and constraint violation.
- Matching `_summary.csv` files: effective maneuver/integration settings,
  maximum/RMS tracking errors, actuator/rate peaks, minimum normal load,
  average solve time, thresholds, and pass/fail.

There are `ceil(duration/sample_time)+1` trajectory rows, including the initial
state. At `t > 0`, the recorded input was held during the preceding interval;
state, reference, diagnostics, and errors are evaluated at the endpoint.
Reported solve time excludes warm-start rollout and initial JIT compilation.

Default acceptance requires maximum lateral error <= 0.5 m, lateral RMS <=
0.25 m, maximum heading error <= 0.15 rad, maximum body-forward speed error <=
1 m/s, finite plant states, successful feasible solves, and actuator/slew
limits respected to solver tolerance. Errors are measured at control samples;
plant finiteness is checked at every internal RK4 substep. Overrides can make
the maneuver infeasible or poorly tracked; the runner reports failure rather
than silently relaxing thresholds.

These tests use the same nominal parameters for prediction and plant, with no
noise/delay/road variation. Passing does not establish robust control, physical
rollover safety, or validated tire behavior. The existing planar model has no
roll/pitch/suspension dynamics. Minimum normal load is reported for inspection,
not a hard MPC contact-load constraint.

### Verified results

Both complete defaults passed on the documented Windows toolchain:

| Scenario | Max lateral error | Lateral RMS | Max heading error |
|---|---:|---:|---:|
| Step-steer response | 0.00018 m | 0.000025 m | 0.00013 rad |
| Double lane change | 0.0161 m | 0.00771 m | 0.0125 rad |

Overrides using 8 m/s, 0.1 m initial lateral error, a 40 ms controller interval,
10-step horizon, 40 RK4 substeps, and reversed/narrower lane change also passed.
Their maximum lateral error was the imposed 0.1 m initial offset, with lateral
RMS below 0.03 m. All trajectory CSVs were checked for exact row/column counts,
finite data, and agreement with the saved summary metrics.

Default mean IPOPT solve times were approximately 0.56 s (step-steer) and
1.09 s (lane change), exceeding the 50 ms control interval. These are offline
closed-loop simulation tests, not a real-time implementation.

## Conventions

- ISO vehicle coordinates: x forward, y left, z up; positive yaw = counter-clockwise.
- Wheel order everywhere: `FL, FR, RL, RR`.
- Positive `delta` steers left; positive right-side torque surplus yaws left.
