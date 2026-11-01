#pragma once

#include "tracking_mpc.hpp"

namespace vehicle_model {

enum class Maneuver { StepSteer, DoubleLaneChange };

struct ManeuverConfig {
    double speed = 10.0;          // initial/nominal forward speed [m/s]
    double start_time = 1.0;
    double step_angle = 0.025;    // road-wheel radians, about 1.43 degrees
    double step_duration = 6.0;
    double lane_width = 3.0;
    double lane_change_time = 2.5; // duration of each lateral transition
    double lane_hold_time = 1.0;
    double settle_time = 2.0;
    double initial_y_error = 0.0;

    void validate() const {
        for (double value : {speed, step_duration, lane_change_time, settle_time}) {
            if (!std::isfinite(value) || value <= 0.0)
                throw std::invalid_argument("Maneuver speed and durations must be finite and positive");
        }
        for (double value : {start_time, lane_hold_time}) {
            if (!std::isfinite(value) || value < 0.0)
                throw std::invalid_argument("Maneuver start/hold times must be finite and nonnegative");
        }
        for (double value : {step_angle, lane_width, initial_y_error}) {
            if (!std::isfinite(value))
                throw std::invalid_argument("Maneuver angles, width, and initial error must be finite");
        }
        if (step_angle == 0.0 || lane_width == 0.0 || step_duration <= start_time)
            throw std::invalid_argument("Maneuvers require nonzero amplitude and step duration > start time");
    }

    double duration(Maneuver maneuver) const {
        return maneuver == Maneuver::StepSteer ? step_duration :
            start_time + 2.0 * lane_change_time + lane_hold_time + settle_time;
    }
};

struct ManeuverReference {
    std::vector<DoubleTrackModel::State<double>> states;
    std::vector<DoubleTrackModel::Input<double>> inputs;
    int simulation_steps;

    TrackingReference window(int step, int horizon) const {
        if (step < 0 || horizon < 1 ||
            static_cast<std::size_t>(step + horizon) >= states.size())
            throw std::out_of_range("Maneuver reference window exceeds the generated trajectory");
        return {{states.begin() + step, states.begin() + step + horizon + 1},
                {inputs.begin() + step, inputs.begin() + step + horizon}};
    }
};

namespace detail {

inline double cruise_torque(const VehicleParams& p, double speed) {
    const double resistance = 0.5 * p.rho_air * p.Cd * p.A_front *
        speed * std::sqrt(speed * speed + p.v_eps * p.v_eps) +
        p.c_rr * p.m * p.g * speed / std::sqrt(speed * speed + p.v_eps * p.v_eps);
    return p.Re * resistance / 4.0;
}

struct SmoothShift {
    double position, velocity, acceleration;
};

inline SmoothShift smooth_shift(double time, double start, double duration) {
    const double s = (time - start) / duration;
    if (s <= 0.0) return {0.0, 0.0, 0.0};
    if (s >= 1.0) return {1.0, 0.0, 0.0};
    return {s * s * s * (10.0 + s * (-15.0 + 6.0 * s)),
            30.0 * s * s * (1.0 - s) * (1.0 - s) / duration,
            60.0 * s * (1.0 - s) * (1.0 - 2.0 * s) / (duration * duration)};
}

} // namespace detail

inline ManeuverReference make_maneuver_reference(
    const DoubleTrackModel& model, Maneuver maneuver,
    const ManeuverConfig& settings, const TrackingMpcConfig& mpc) {
    settings.validate();
    mpc.validate();
    const double intervals = std::ceil(settings.duration(maneuver) / mpc.sample_time);
    if (!std::isfinite(intervals) || intervals > 1000000.0 ||
        mpc.horizon > 1000000)
        throw std::invalid_argument("Maneuver reference exceeds one million simulation/prediction steps");
    const int steps = static_cast<int>(intervals);
    const int count = steps + mpc.horizon + 1;
    ManeuverReference result{{}, {}, steps};
    result.states.reserve(count);
    result.inputs.reserve(count);
    const auto& p = model.params();
    auto state = model.straight_driving_state(settings.speed);
    for (int k = 0; k < count; ++k) {
        const double time = k * mpc.sample_time;
        if (maneuver == Maneuver::StepSteer) {
            const double angle = time >= settings.start_time ? settings.step_angle : 0.0;
            // Nominal speed-holding feedforward plant; the closed-loop plant
            // receives only the optimized MPC input, never this steering step.
            const double torque = std::clamp(
                detail::cruise_torque(p, state[DoubleTrackModel::IVX]) +
                80.0 * (settings.speed - state[DoubleTrackModel::IVX]),
                -mpc.torque_limit, mpc.torque_limit);
            const DoubleTrackModel::Input<double> input{angle, torque, torque, torque, torque};
            result.states.push_back(state);
            result.inputs.push_back(input);
            for (int i = 0; i < mpc.rk4_substeps; ++i) {
                state = rk4_step<DoubleTrackModel, double>(
                    model, state, input, mpc.sample_time / mpc.rk4_substeps);
            }
        } else {
            const auto out = detail::smooth_shift(time, settings.start_time, settings.lane_change_time);
            const auto back = detail::smooth_shift(time,
                settings.start_time + settings.lane_change_time + settings.lane_hold_time,
                settings.lane_change_time);
            const double y = settings.lane_width * (out.position - back.position);
            const double yd = settings.lane_width * (out.velocity - back.velocity);
            const double ydd = settings.lane_width * (out.acceleration - back.acceleration);
            const double vx = std::hypot(settings.speed, yd);
            const double r = settings.speed * ydd / (vx * vx);
            state = {};
            state[DoubleTrackModel::IX] = settings.speed * time;
            state[DoubleTrackModel::IY] = y;
            state[DoubleTrackModel::IPSI] = std::atan2(yd, settings.speed);
            state[DoubleTrackModel::IVX] = vx;
            state[DoubleTrackModel::IR] = r;
            const double delta = std::atan(p.wheelbase() * r / vx);
            const double lateral[4]{p.tf / 2.0, -p.tf / 2.0, p.tr / 2.0, -p.tr / 2.0};
            const double longitudinal[4]{p.lf, p.lf, -p.lr, -p.lr};
            for (int i = 0; i < 4; ++i) {
                const double steer = i < 2 ? delta : 0.0;
                state[DoubleTrackModel::IW_FL + i] =
                    ((vx - r * lateral[i]) * std::cos(steer) +
                     r * longitudinal[i] * std::sin(steer)) / p.Re;
            }
            const double torque = detail::cruise_torque(p, vx);
            result.states.push_back(state);
            result.inputs.push_back({delta, torque, torque, torque, torque});
        }
        for (double value : result.states.back()) {
            if (!std::isfinite(value))
                throw std::runtime_error("Reference integration became nonfinite; reduce internal step size");
        }
    }
    return result;
}

} // namespace vehicle_model
