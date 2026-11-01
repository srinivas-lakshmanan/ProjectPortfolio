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
Those belong to the generic MPC and the test cases you define. Existing tire
and load-transfer approximations are unchanged.

## Layout

```
include/vehicle_model/
  vehicle_params.hpp       # physical parameters (mid-size car defaults)
  smooth_math.hpp          # eps-regularized abs / sign / division
  tire_model.hpp           # Pacejka magic formula + combined slip
  double_track_model.hpp   # the 7-DOF double-track ODE  x_dot = f(x, u)
  integrator.hpp           # templated RK4 step
  casadi_vehicle_model.hpp # continuous/discrete CasADi functions and Jacobians
src/main.cpp               # demo: step steer + torque-vectoring scenarios (CSV output)
test/test_double_track_model.cpp  # physics sanity tests
test/test_casadi_vehicle_model.cpp # wrapper parity, Jacobians, MX composition
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
the CasADi wheel's C++ SDK and MSYS2 MinGW g++:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-casadi.txt
cmake -S . -B build-casadi -G "Unix Makefiles" `
  "-DCMAKE_MAKE_PROGRAM=C:\msys64\usr\bin\make.exe" `
  "-DCMAKE_CXX_COMPILER=C:\msys64\ucrt64\bin\g++.exe" `
  -DVEHICLE_MODEL_WITH_CASADI=ON `
  "-Dcasadi_DIR=$PWD\.venv\Lib\site-packages\casadi\cmake"
cmake --build build-casadi --parallel 4
ctest --test-dir build-casadi --output-on-failure
```

The Windows test target copies the CasADi DLL and its bundled MinGW runtime
DLLs beside the executable to avoid loading incompatible DLLs from PATH. Do
the same for your controller executable, or prepend the SDK's DLL directory
to its runtime PATH. MSVC builds require an MSVC-compatible CasADi SDK; do not
mix a MinGW wheel with MSVC.

Wrapper tests verify numerical parity (including standstill and changed
parameters), output shapes/order, finite Jacobians and finite-difference
agreement, MX multiple-shooting composition, and function serialization.
They are integration checks, not additional driving scenarios.

## Conventions

- ISO vehicle coordinates: x forward, y left, z up; positive yaw = counter-clockwise.
- Wheel order everywhere: `FL, FR, RL, RR`.
- Positive `delta` steers left; positive right-side torque surplus yaws left.
