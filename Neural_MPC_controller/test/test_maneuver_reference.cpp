#include "vehicle_model/maneuver_reference.hpp"

#include <cstdlib>
#include <iostream>

namespace {
int failures = 0;
void check(bool condition, const char* message) {
    std::cout << (condition ? "[PASS] " : "[FAIL] ") << message << '\n';
    if (!condition) ++failures;
}

template <typename F>
void rejects(F&& action, const char* message) {
    bool rejected = false;
    try {
        action();
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    check(rejected, message);
}
} // namespace

int main() {
    using namespace vehicle_model;
    TrackingMpcConfig mpc;
    ManeuverConfig settings;
    DoubleTrackModel model;
    const auto lane = make_maneuver_reference(model, Maneuver::DoubleLaneChange, settings, mpc);
    check(std::abs(lane.states.front()[DoubleTrackModel::IY]) < 1e-12,
          "Lane change begins in original lane");
    const int hold = static_cast<int>(
        (settings.start_time + settings.lane_change_time + 0.5 * settings.lane_hold_time) /
        mpc.sample_time);
    check(std::abs(lane.states[hold][DoubleTrackModel::IY] - settings.lane_width) < 1e-12,
          "Lane change reaches requested adjacent lane");
    const auto& end = lane.states[lane.simulation_steps];
    check(std::abs(end[DoubleTrackModel::IY]) < 1e-12 &&
          std::abs(end[DoubleTrackModel::IPSI]) < 1e-12 &&
          std::abs(end[DoubleTrackModel::IR]) < 1e-12,
          "Double lane change returns to original lane and heading");
    const auto window = lane.window(lane.simulation_steps - 1, mpc.horizon);
    check(window.states.size() == static_cast<std::size_t>(mpc.horizon + 1) &&
          window.inputs.size() == static_cast<std::size_t>(mpc.horizon),
          "Last prediction window includes a full horizon");
    const auto before = detail::smooth_shift(settings.start_time, settings.start_time,
                                             settings.lane_change_time);
    const auto after = detail::smooth_shift(settings.start_time + settings.lane_change_time,
                                            settings.start_time, settings.lane_change_time);
    check(before.velocity == 0.0 && before.acceleration == 0.0 &&
          after.velocity == 0.0 && after.acceleration == 0.0,
          "Quintic transition endpoints have zero velocity and acceleration");

    const auto step = make_maneuver_reference(model, Maneuver::StepSteer, settings, mpc);
    const int onset = static_cast<int>(settings.start_time / mpc.sample_time);
    check(step.inputs[onset - 1][0] == 0.0 &&
          step.inputs[onset][0] == settings.step_angle,
          "Step-steer reference input changes at configured onset");
    check(step.states[onset + 10][DoubleTrackModel::IR] > 0.0,
          "Model-generated step reference contains positive yaw response");
    settings.speed = 8.0;
    settings.lane_width = -2.0;
    settings.start_time = 0.5;
    const auto right = make_maneuver_reference(model, Maneuver::DoubleLaneChange, settings, mpc);
    bool rightward = false;
    for (const auto& state : right.states)
        rightward = rightward || state[DoubleTrackModel::IY] < -1.9;
    check(rightward && right.states.front()[DoubleTrackModel::IVX] == 8.0,
          "Speed, direction, amplitude, and timing overrides change the reference");

    auto invalid = mpc;
    invalid.sample_time = 0.0;
    rejects([&] { TrackingMpc controller(model, invalid); }, "Invalid MPC time rejected before construction");
    invalid = mpc;
    invalid.horizon = 1;
    rejects([&] { invalid.validate(); }, "Invalid MPC horizon rejected");
    invalid = mpc;
    invalid.input_weights[0] = -1.0;
    rejects([&] { invalid.validate(); }, "Invalid MPC weight rejected");
    settings.lane_change_time = 0.0;
    rejects([&] { settings.validate(); }, "Invalid maneuver timing rejected");

    TrackingMpcConfig small;
    small.horizon = 2;
    small.sample_time = 0.005;
    small.rk4_substeps = 5;
    small.jit = false;
    TrackingMpc controller(model, small);
    auto measured = model.straight_driving_state(10.0);
    const double torque = detail::cruise_torque(model.params(), 10.0);
    DoubleTrackModel::Input<double> previous{0.0, torque, torque, torque, torque};
    TrackingReference valid;
    auto predicted = measured;
    valid.states.push_back(predicted);
    for (int k = 0; k < small.horizon; ++k) {
        valid.inputs.push_back(previous);
        for (int i = 0; i < small.rk4_substeps; ++i)
            predicted = rk4_step<DoubleTrackModel, double>(
                model, predicted, previous, small.sample_time / small.rk4_substeps);
        valid.states.push_back(predicted);
    }
    rejects([&] { controller.solve(measured, previous, TrackingReference{}); },
            "MPC rejects incomplete prediction reference");
    auto nonfinite = measured;
    nonfinite[0] = std::numeric_limits<double>::quiet_NaN();
    rejects([&] { controller.solve(nonfinite, previous, valid); },
            "MPC rejects nonfinite measurements");
    auto out_of_bounds = previous;
    out_of_bounds[0] = 1.0;
    rejects([&] { controller.solve(measured, out_of_bounds, valid); },
            "MPC rejects previous input outside actuator limits");
    for (int iteration = 0; iteration < 2; ++iteration) {
        const auto result = controller.solve(measured, previous, valid);
        check(result.constraint_violation <= small.feasibility_tolerance &&
              std::abs(result.input[0]) <= small.steering_limit &&
              std::isfinite(result.solve_seconds),
              "Repeated MPC solves return finite, feasible bounded inputs");
    }
    small.max_iterations = 1;
    TrackingMpc failing(model, small);
    measured[DoubleTrackModel::IY] = 0.5;
    bool reported = false;
    try {
        failing.solve(measured, previous, valid);
    } catch (const std::runtime_error&) {
        reported = true;
    }
    check(reported, "MPC solver failure is reported, not replaced with a fallback input");
    std::cout << failures << " failure(s)\n";
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
