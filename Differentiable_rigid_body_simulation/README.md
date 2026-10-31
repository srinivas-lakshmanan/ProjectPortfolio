# Differentiable Articulated Dynamics

A C++17 fixed-base revolute/prismatic rigid-link tree simulator with shared
double/dual-number dynamics and RK4, bilateral coordinate constraints, optional
smooth frictionless plane contact, and state/control/physical-parameter gradients.

## Units and Frames

| Quantity | Convention |
| --- | --- |
| Length, mass, time | meters, kilograms, seconds |
| Revolute position, velocity, effort | radians, radians/second, newton-meters |
| Prismatic position, velocity, effort | meters, meters/second, newtons |
| COM inertia | kilogram-meter squared |
| World frame | right-handed; default gravity `(0, 0, -9.81)` |
| Base pose | fixed at the world origin with identity orientation |
| Positive joint rotation | right-hand rule about the joint axis |
| Joint axis | joint rest-frame; normalized at construction |
| Joint offset | parent-local displacement from parent endpoint |
| Joint rest rotation | maps joint rest-frame vectors into parent frame |
| Link and COM offsets | expressed in the rotated link's local frame |
| COM inertia | symmetric positive-definite tensor expressed in the link frame |
| Matrix storage | row-major; rotations map local vectors into the parent/world frame |

The historical `RevoluteJoint` aggregate now has trailing `type`, `parent`,
`joint_offset`, and `joint_rotation` fields. Type is `Revolute` or `Prismatic`.
Parent `-2` automatically selects the preceding joint, `-1` attaches to world,
and nonnegative indices must name earlier joints. Branches and multiple roots
are supported; child attachment starts at the parent's link endpoint.
Orientations are reconstructed with rest and axis-angle rotations, not integrated
quaternions. No quaternion normalization is needed. Models must outlive their
`DynamicsSolver`, which must outlive its `RolloutSolver`. The kinematics wrapper
returns the last link endpoint and its ancestor-only Jacobian.

## Dynamics and Differentiation

Forward dynamics solves `M(q) * acceleration = effort - bias + contact_effort`.
The bias contains gravity and velocity-dependent inertial terms. Mass-matrix
columns are constructed with unit-acceleration inverse-dynamics evaluations.
A scale-aware, equilibrated pivoted dense solve produces accelerations.
Inverse dynamics excludes constraints and contacts.

`DynamicsOptions::constraints` supplies independent linear bilateral relations
`A*q = target`. Each `CoordinateConstraint` defines one coefficient row, target,
frequency `w` (inverse seconds), and damping ratio `zeta`. The saddle solve is:

```text
[ M  -A^T ] [ acceleration ] = [ effort - bias + contact_effort ]
[ A    0  ] [ lambda       ]   [ -2*zeta*w*A*v - w*w*(A*q-target) ]
```

Returned `constraint_forces` are multipliers; generalized reaction is `A^T*lambda`.
Use feasible initial positions/velocities for physical constrained motion.
Stabilization corrects drift but does not project initial states. Mixed angular
and linear rows need dimensionally appropriate coefficients. These constraints
model coordinate coupling/gearing, not arbitrary nonlinear Cartesian loop closure.

Both ordinary and dual-number rollouts use the same scalar-templated dynamics,
linear solve and RK4 in `include/arm/simulation_core.hpp`. Each control vector is
held constant throughout its timestep, including all four RK4 stages. The final
legacy loss is `0.5 * sum((final_joint_position - target_joint_position)^2)` using
raw, unwrapped coordinate differences. Constraint/contact options flow identically
through both scalar paths. Stiff contact and stabilization require smaller steps;
RK4 is not an implicit stiff solver.

`automatic_control_gradient` preserves the control-only API.
`RolloutObjective` additionally supports terminal joint or Cartesian point error,
after-step trajectory tracking, and effort penalties, all weighted half-squared
errors summed without timestep scaling. Cartesian trajectory targets are XYZ
triples; joint targets have model-DOF size. Joint terminal targets remain required
even with zero terminal weight.

```cpp
arm::RolloutObjective objective;
objective.target_positions = {1.2};
const auto gradient = solver.automatic_gradient(initial, controls, 0.01, objective);
const double loss = solver.evaluate_objective(initial, controls, 0.01, objective);
```

`RolloutGradient` contains loss and derivatives for initial positions/velocities,
all controls, each mass, and symmetric inertia entries `xx, yy, zz, xy, xz, yz`.
Off-diagonal entries are perturbed together. Keep perturbations in the valid
positive-mass/positive-definite-inertia domain. Physical inertia triangle
inequalities are not enforced. For constrained initial states use tangent
directions; arbitrary perturbations also affect stabilization corrections.

`automatic_directional_derivative` uses one dual derivative component with a
same-shaped `RolloutGradient` direction (`direction.loss` is ignored), avoiding
the dense full-gradient cost when only a directional derivative is needed.
Gravity, geometry, constraint/contact parameters and timestep remain fixed.
Pivot decisions use primal values; singular systems are not valid gradient points.

## Build and Run

The CMake project requires a C++17 compiler. In VS Code, use CMake Tools to select
a compiler/configure preset, build, and run CTest. The optional
`windows-vs2022` preset requires CMake 3.21+ and Visual Studio 2022 with the
Desktop development with C++ workload and Windows SDK. Its matching build/test
preset is `windows-debug`; it uses a separate directory under `build`.
The `windows-gcc-ninja` configure preset uses installed MSYS2 UCRT64 GCC and
VS-bundled Ninja; matching build/test preset is `windows-gcc-debug`. Its tool
paths are machine-specific. Select a configure preset explicitly in CMake Tools:
no active preset means configuration cannot run. Without presets, the project
retains its CMake 3.16 minimum.

Executables:

- `arm_demo`: three-link kinematics and a short controlled rollout.
- `pendulum_demo`: a smooth single-link pendulum trajectory, energy, and torque gradient.
- `arm_solver_tests`: analytical, integration, and gradient regression tests.
- `eight_stage_tests`: independent multi-link physics, constraints/contact and full gradients.

The pendulum has mass 2 kg, COM distance 0.5 m, and axial COM inertia
1/6 kg-m^2. Its zero angle is horizontal along world +X, its axis is +Y,
and positive rotation moves the COM toward -Z. Thus the unforced hanging
equilibrium is `q = pi/2`, not zero. Its equation is
`(2/3) * acceleration = torque + 9.81 * cos(q)`.

## Validation Ladder

The test suite includes:

- Rejection of nonfinite axes and link offsets, and valid-axis normalization.
- Kinematics/Jacobian checks, mass-matrix symmetry, forward/inverse round trips,
  and gravity compensation.
- Independent pendulum acceleration checks at multiple angles and torques.
- Exact zero-gravity rotor trajectories and analytical control gradients.
- Two-second unforced pendulum energy drift with `dt = 0.04, 0.02, 0.01`.
- Nonlinear pendulum and three-link gradients at 5, 20, and 60 steps with
  `dt = 0.01`, sweeping epsilon from `1e-3` through `1e-6`.
- Gradient norm-relative error, cosine agreement, and a normalized deterministic
  directional derivative per scene/horizon, in addition to coordinate differences.

The nonlinear checks require relative error below `2e-4` and cosine agreement
above `0.99999`. Energy drift must improve by more than a factor of eight per
timestep halving, with final maximum absolute drift below `1e-6` joules.
These tests validate the discrete integrator; timestep refinement remains necessary
when assessing approximation to continuous dynamics. Forward/inverse consistency
alone does not establish physical correctness.

## Extended Validation

`eight_stage_tests` adds independent two-link mass/bias/acceleration references
at nine configurations, Cholesky positive-definiteness checks, rotated prismatic
gravity and branched decoupling. It checks analytical constraint forces, residuals,
feasible/stabilized drift, redundant-row rejection, contact forces and energy.

Full initial-state, control, mass and six-entry inertia FD checks sweep
`1e-3`, `1e-4`, `1e-5` with magnitude-scaled parameter perturbations at five steps.
Scenes progress from pendulum to two-link chain, constrained sliders, smooth
contact sliders, and a spatial mixed tree. Forty-step horizons use directional
FD/JVP checks against full-gradient dot products. Objectives cover Cartesian
terminal error, joint tracking and isolated effort penalties. Invalid-input and
empty-rollout checks are included. Component sweeps stay short to bound test cost.

## Smooth Contact

Contacts are disabled by default. `SmoothPlaneContact` selects a link/local point,
unit world normal, plane offset, stiffness and positive smoothing length. For
signed gap `normal dot point - plane_offset`, outward force magnitude is:

```text
force = stiffness * (sqrt(gap*gap + smoothing*smoothing) - gap) / 2
```

This conservative frictionless model permits penetration and has a small force
tail when separated. It is smooth across the plane, not hard unilateral contact.
Tests separately validate closed-form force, monotonic penetration response,
separated force tail, contact energy and contact gradients.

## Eight-Stage Coverage and Limits

1. Smooth starting problem: pendulum demo, explicit SI units and frames.
2. Representation: validated revolute/prismatic trees and rest frames.
3. Forward dynamics: independent physics, SPD tests and bilateral constraints.
4. Differentiable timestep: shared RK4, dynamics and linear solves.
5. Objectives: joint/Cartesian/trajectory losses and state/control/mass/inertia gradients.
6. Gradient validation: epsilon sweeps, relative norms, cosine agreement and JVP/FD checks.
7. Ladder: single link, chains, constraints and longer directional rollouts.
8. Contact: optional smooth plane model with separate physics/gradient tests.

This covers all eight stages within the stated model scope, not every possible
rigid-body system. Floating bases, spherical joints, general nonlinear loop
closure, self-collision, hard impacts, restitution and friction remain unsupported.
Hard impacts, friction transitions and discrete mode changes are nonsmooth and
must not be judged by ordinary smooth finite-difference agreement. Reverse-mode
or adjoint scalability remains future work; dense forward-mode cost grows with
gradient size. Passing tests establish the exercised regimes, not arbitrary
accuracy or stability. Timestep refinement remains necessary.