#pragma once

#include "casadi_vehicle_model.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <limits>
#include <string>

namespace vehicle_model {

struct TrackingMpcConfig {
    double sample_time = 0.05;
    int horizon = 12;
    int rk4_substeps = 50;
    double steering_limit = 0.15;       // road-wheel radians
    double steering_rate_limit = 0.30;  // rad/s
    double torque_limit = 200.0;        // signed Nm per wheel
    double torque_rate_limit = 1500.0;  // Nm/s per wheel
    int max_iterations = 100;
    double tolerance = 1e-5;
    double feasibility_tolerance = 1e-4;
    bool jit = true;
    std::string jit_compiler = "gcc";
    std::array<double, DoubleTrackModel::NX> state_weights{
        0.1, 30.0, 100.0, 8.0, 0.5, 10.0, 0.001, 0.001, 0.001, 0.001};
    std::array<double, DoubleTrackModel::NU> input_weights{
        0.5, 1e-5, 1e-5, 1e-5, 1e-5};
    std::array<double, DoubleTrackModel::NU> rate_weights{
        5.0, 1e-5, 1e-5, 1e-5, 1e-5};

    void validate() const {
        for (double value : {sample_time, steering_limit, steering_rate_limit,
                             torque_limit, torque_rate_limit, tolerance,
                             feasibility_tolerance}) {
            if (!std::isfinite(value) || value <= 0.0) {
                throw std::invalid_argument("MPC time, limits, and tolerances must be finite and positive");
            }
        }
        if (horizon < 2 || rk4_substeps < 1 || max_iterations < 1) {
            throw std::invalid_argument("MPC needs horizon >= 2, substeps >= 1, iterations >= 1");
        }
        if (jit && jit_compiler.empty())
            throw std::invalid_argument("MPC JIT compiler command must not be empty");
        for (double weight : state_weights) {
            if (!std::isfinite(weight) || weight < 0.0)
                throw std::invalid_argument("MPC state weights must be finite and nonnegative");
        }
        for (const auto* weights : {&input_weights, &rate_weights}) {
            for (double weight : *weights) {
                if (!std::isfinite(weight) || weight <= 0.0)
                    throw std::invalid_argument("MPC input/rate weights must be finite and positive");
            }
        }
    }
};

struct TrackingReference {
    std::vector<DoubleTrackModel::State<double>> states; // horizon + 1
    std::vector<DoubleTrackModel::Input<double>> inputs; // horizon
};

struct TrackingMpcResult {
    DoubleTrackModel::Input<double> input;
    int iterations;
    double solve_seconds;
    double constraint_violation;
};

class TrackingMpc {
public:
    using State = DoubleTrackModel::State<double>;
    using Input = DoubleTrackModel::Input<double>;

    explicit TrackingMpc(const DoubleTrackModel& model, TrackingMpcConfig config = {})
        : config_(validated(config)),
          functions_(make_casadi_vehicle_functions(model, config_.sample_time,
                                                   config_.rk4_substeps)) {
        using casadi::MX;
        const int n = config_.horizon;
        const MX X = MX::sym("X", DoubleTrackModel::NX, n + 1);
        const MX U = MX::sym("U", DoubleTrackModel::NU, n);
        const MX initial = MX::sym("initial", DoubleTrackModel::NX);
        const MX previous = MX::sym("previous", DoubleTrackModel::NU);
        const MX R = MX::sym("R", DoubleTrackModel::NX, n + 1);
        const MX V = MX::sym("V", DoubleTrackModel::NU, n);
        std::vector<MX> constraints{X(casadi::Slice(), 0) - initial};
        lower_g_.assign(DoubleTrackModel::NX, 0.0);
        upper_g_ = lower_g_;
        MX cost = 0.0;
        for (int k = 0; k <= n; ++k) {
            for (int i = 0; i < DoubleTrackModel::NX; ++i) {
                MX error = X(i, k) - R(i, k);
                if (i == DoubleTrackModel::IPSI) {
                    error = atan2(sin(error), cos(error));
                }
                cost += (k == n ? 3.0 : 1.0) * config_.state_weights[i] * error * error;
            }
            if (k == n) break;
            const MX uk = U(casadi::Slice(), k);
            const MX predicted = functions_.discrete(
                std::vector<MX>{X(casadi::Slice(), k), uk}).at(0);
            constraints.push_back(X(casadi::Slice(), k + 1) - predicted);
            lower_g_.insert(lower_g_.end(), DoubleTrackModel::NX, 0.0);
            upper_g_.insert(upper_g_.end(), DoubleTrackModel::NX, 0.0);
            const MX last = k == 0 ? previous : U(casadi::Slice(), k - 1);
            constraints.push_back(uk - last);
            for (int i = 0; i < DoubleTrackModel::NU; ++i) {
                const double rate = i == 0 ? config_.steering_rate_limit
                                          : config_.torque_rate_limit;
                lower_g_.push_back(-rate * config_.sample_time);
                upper_g_.push_back(rate * config_.sample_time);
                const MX effort = U(i, k) - V(i, k);
                const MX change = U(i, k) - last(i);
                cost += config_.input_weights[i] * effort * effort +
                        config_.rate_weights[i] * change * change;
            }
        }
        const MX decision = MX::vertcat({MX::vec(X), MX::vec(U)});
        const MX parameters = MX::vertcat({initial, previous, MX::vec(R), MX::vec(V)});
        casadi::Dict options;
        options["print_time"] = false;
        options["ipopt.print_level"] = 0;
        options["ipopt.sb"] = "yes";
        options["ipopt.max_iter"] = config_.max_iterations;
        options["ipopt.tol"] = config_.tolerance;
        options["ipopt.constr_viol_tol"] = config_.feasibility_tolerance;
        options["jit"] = config_.jit;
        if (config_.jit) {
            options["compiler"] = "shell";
            options["jit_options"] = casadi::Dict{
                {"compiler", config_.jit_compiler}, {"linker", config_.jit_compiler},
                {"compiler_setup", "-c"}, {"compiler_output_flag", "-o "},
                {"linker_setup", "-shared"}, {"linker_output_flag", "-o "},
                {"flags", std::vector<std::string>{"-O2"}}};
        }
        solver_ = casadi::nlpsol("vehicle_tracking_mpc", "ipopt",
            casadi::MXDict{{"x", decision}, {"p", parameters}, {"f", cost},
                           {"g", MX::vertcat(constraints)}}, options);

        const int state_count = DoubleTrackModel::NX * (n + 1);
        lower_x_.assign(state_count, -std::numeric_limits<double>::infinity());
        upper_x_.assign(state_count, std::numeric_limits<double>::infinity());
        for (int k = 0; k < n; ++k) {
            for (int i = 0; i < DoubleTrackModel::NU; ++i) {
                const double limit = i == 0 ? config_.steering_limit : config_.torque_limit;
                lower_x_.push_back(-limit);
                upper_x_.push_back(limit);
            }
        }
    }

    const TrackingMpcConfig& config() const { return config_; }

    void reset_warm_start() { warm_inputs_.clear(); }

    TrackingMpcResult solve(const State& measured, const Input& previous,
                            const TrackingReference& reference) {
        if (reference.states.size() != static_cast<std::size_t>(config_.horizon + 1) ||
            reference.inputs.size() != static_cast<std::size_t>(config_.horizon)) {
            throw std::invalid_argument("MPC reference must contain horizon+1 states and horizon inputs");
        }
        std::vector<double> parameters;
        append_finite(parameters, measured);
        append_finite(parameters, previous);
        for (const auto& state : reference.states) append_finite(parameters, state);
        for (const auto& input : reference.inputs) append_finite(parameters, input);
        for (int i = 0; i < DoubleTrackModel::NU; ++i) {
            const double limit = i == 0 ? config_.steering_limit : config_.torque_limit;
            if (std::abs(previous[i]) > limit + config_.feasibility_tolerance)
                throw std::invalid_argument("Previous MPC input exceeds actuator bounds");
        }

        std::vector<Input> guess_inputs(config_.horizon, previous);
        if (!warm_inputs_.empty()) {
            for (int k = 0; k < config_.horizon; ++k) {
                guess_inputs[k] = warm_inputs_[std::min(k + 1, config_.horizon - 1)];
            }
        }
        std::vector<double> guess;
        State predicted = measured;
        append_finite(guess, predicted);
        for (const auto& input : guess_inputs) {
            const auto value = functions_.discrete(
                std::vector<casadi::DM>{column(predicted), column(input)}).at(0).nonzeros();
            if (value.size() != predicted.size())
                throw std::runtime_error("MPC prediction returned an unexpected state size");
            std::copy(value.begin(), value.end(), predicted.begin());
            append_finite(guess, predicted);
        }
        for (const auto& input : guess_inputs) append_finite(guess, input);
        const auto start = std::chrono::steady_clock::now();
        const auto result = solver_(casadi::DMDict{
            {"x0", casadi::DM(guess)}, {"p", casadi::DM(parameters)},
            {"lbx", casadi::DM(lower_x_)}, {"ubx", casadi::DM(upper_x_)},
            {"lbg", casadi::DM(lower_g_)}, {"ubg", casadi::DM(upper_g_)}});
        const double seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();
        const auto stats = solver_.stats();
        if (!stats.at("success").to_bool()) {
            throw std::runtime_error("MPC solver failed: " + stats.at("return_status").to_string());
        }
        const auto solution = result.at("x").nonzeros();
        const auto constraints = result.at("g").nonzeros();
        double violation = bound_violation(solution, lower_x_, upper_x_);
        violation = std::max(violation, bound_violation(constraints, lower_g_, upper_g_));
        if (violation > config_.feasibility_tolerance) {
            throw std::runtime_error("MPC solution violates constraints by " + std::to_string(violation));
        }
        const int offset = DoubleTrackModel::NX * (config_.horizon + 1);
        warm_inputs_.resize(config_.horizon);
        for (int k = 0; k < config_.horizon; ++k) {
            std::copy_n(solution.begin() + offset + k * DoubleTrackModel::NU,
                        DoubleTrackModel::NU, warm_inputs_[k].begin());
        }
        return {warm_inputs_.front(), static_cast<int>(stats.at("iter_count").to_int()),
                seconds, violation};
    }

private:
    static TrackingMpcConfig validated(TrackingMpcConfig config) {
        config.validate();
        return config;
    }

    template <std::size_t N>
    static casadi::DM column(const std::array<double, N>& values) {
        return casadi::DM(std::vector<double>(values.begin(), values.end()));
    }

    template <std::size_t N>
    static void append_finite(std::vector<double>& out, const std::array<double, N>& values) {
        for (double value : values) {
            if (!std::isfinite(value))
                throw std::invalid_argument("MPC states and inputs must be finite");
            out.push_back(value);
        }
    }

    static double bound_violation(const std::vector<double>& values,
                                 const std::vector<double>& lower,
                                 const std::vector<double>& upper) {
        if (values.size() != lower.size())
            throw std::runtime_error("MPC solver returned an unexpected vector size");
        double violation = 0.0;
        for (std::size_t i = 0; i < values.size(); ++i) {
            if (!std::isfinite(values[i]))
                throw std::runtime_error("MPC solver returned nonfinite values");
            violation = std::max({violation, lower[i] - values[i], values[i] - upper[i]});
        }
        return violation;
    }

    TrackingMpcConfig config_;
    CasadiVehicleFunctions functions_;
    casadi::Function solver_;
    std::vector<double> lower_x_, upper_x_, lower_g_, upper_g_;
    std::vector<Input> warm_inputs_;
};

} // namespace vehicle_model
