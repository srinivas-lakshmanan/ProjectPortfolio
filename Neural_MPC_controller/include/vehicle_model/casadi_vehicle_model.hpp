#pragma once

#include <casadi/casadi.hpp>

#include <cmath>
#include <stdexcept>
#include <vector>

#include "double_track_model.hpp"
#include "integrator.hpp"

namespace vehicle_model {

struct CasadiVehicleFunctions {
    casadi::Function continuous;           // (x, u) -> xdot
    casadi::Function discrete;             // (x, u) -> x_next
    casadi::Function continuous_jacobian;  // (x, u) -> A, B
    casadi::Function discrete_jacobian;    // (x, u) -> A, B
    casadi::Function diagnostics;          // (x, u) -> kappa, alpha, Fz, Fx_w, Fy_w, ax, ay
};

namespace detail {

template <std::size_t N>
casadi::SX sx_column(const std::array<casadi::SX, N>& values) {
    return casadi::SX::vertcat(std::vector<casadi::SX>(values.begin(), values.end()));
}

} // namespace detail

// Parameters and sample time are captured in the graph. Rebuild to change them.
// Inputs are held constant over all RK4 substeps of a prediction interval.
inline CasadiVehicleFunctions make_casadi_vehicle_functions(
    const DoubleTrackModel& model, double sample_time, int rk4_substeps = 20) {
    if (!std::isfinite(sample_time) || sample_time <= 0.0) {
        throw std::invalid_argument("CasADi vehicle sample time must be finite and positive");
    }
    if (rk4_substeps <= 0) {
        throw std::invalid_argument("CasADi vehicle RK4 substeps must be positive");
    }
    const double dt = sample_time / rk4_substeps;
    if (dt == 0.0) {
        throw std::invalid_argument("CasADi vehicle RK4 substep duration underflows to zero");
    }

    using casadi::SX;
    const SX x = SX::sym("x", DoubleTrackModel::NX);
    const SX u = SX::sym("u", DoubleTrackModel::NU);
    DoubleTrackModel::State<SX> state;
    DoubleTrackModel::Input<SX> input;
    for (int i = 0; i < DoubleTrackModel::NX; ++i) state[i] = x(i);
    for (int i = 0; i < DoubleTrackModel::NU; ++i) input[i] = u(i);

    DoubleTrackModel::Diagnostics<SX> diag;
    const SX xdot = detail::sx_column(model.dynamics(state, input, diag));
    const casadi::Function step(
        "vehicle_rk4_substep", {x, u},
        {detail::sx_column(rk4_step<DoubleTrackModel, SX>(model, state, input, dt))},
        {"x", "u"}, {"x_next"});

    // Retain function calls rather than expanding the RK4 graph per substep.
    SX x_next = x;
    for (int i = 0; i < rk4_substeps; ++i) {
        std::vector<SX> result;
        step.call(std::vector<SX>{x_next, u}, result, false, true);
        x_next = result.at(0);
    }

    return {
        casadi::Function("vehicle_continuous", {x, u}, {xdot},
                         {"x", "u"}, {"xdot"}),
        casadi::Function("vehicle_discrete", {x, u}, {x_next},
                         {"x", "u"}, {"x_next"}),
        casadi::Function("vehicle_continuous_jacobian", {x, u},
                         {SX::jacobian(xdot, x), SX::jacobian(xdot, u)},
                         {"x", "u"}, {"A", "B"}),
        casadi::Function("vehicle_discrete_jacobian", {x, u},
                         {SX::jacobian(x_next, x), SX::jacobian(x_next, u)},
                         {"x", "u"}, {"A", "B"}),
        casadi::Function("vehicle_diagnostics", {x, u},
                         {detail::sx_column(diag.kappa), detail::sx_column(diag.alpha),
                          detail::sx_column(diag.Fz), detail::sx_column(diag.Fx_w),
                          detail::sx_column(diag.Fy_w), diag.ax, diag.ay},
                         {"x", "u"}, {"kappa", "alpha", "Fz", "Fx_w", "Fy_w", "ax", "ay"})
    };
}

} // namespace vehicle_model
