#include "vehicle_model/casadi_vehicle_model.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using vehicle_model::DoubleTrackModel;
using vehicle_model::make_casadi_vehicle_functions;
using State = DoubleTrackModel::State<double>;
using Input = DoubleTrackModel::Input<double>;
using casadi::DM;

namespace {

int failures = 0;

void check(bool condition, const std::string& message) {
    std::cout << (condition ? "[PASS] " : "[FAIL] ") << message << '\n';
    if (!condition) ++failures;
}

template <std::size_t N>
DM column(const std::array<double, N>& values) {
    return DM(std::vector<double>(values.begin(), values.end()));
}

bool close(const DM& actual, const DM& expected, double tolerance = 1e-9) {
    if (actual.size1() != expected.size1() || actual.size2() != expected.size2()) {
        return false;
    }
    for (casadi_int j = 0; j < actual.size2(); ++j) {
        for (casadi_int i = 0; i < actual.size1(); ++i) {
            const double a = static_cast<double>(actual(i, j));
            const double e = static_cast<double>(expected(i, j));
            if (!std::isfinite(a) || !std::isfinite(e) ||
                std::abs(a - e) > tolerance * (1.0 + std::abs(e))) return false;
        }
    }
    return true;
}

void check_jacobian(const casadi::Function& f, const casadi::Function& jac,
                    const State& x, const Input& u) {
    const auto exact = jac(std::vector<DM>{column(x), column(u)});
    check(exact.at(0).size1() == DoubleTrackModel::NX &&
          exact.at(0).size2() == DoubleTrackModel::NX &&
          exact.at(1).size1() == DoubleTrackModel::NX &&
          exact.at(1).size2() == DoubleTrackModel::NU, f.name() + ": Jacobian shapes");

    bool matches = true;
    for (int variable = 0; variable < DoubleTrackModel::NX + DoubleTrackModel::NU;
         ++variable) {
        State xp = x, xm = x;
        Input up = u, um = u;
        const bool is_state = variable < DoubleTrackModel::NX;
        const int j = is_state ? variable : variable - DoubleTrackModel::NX;
        const double value = is_state ? x[j] : u[j];
        const double h = 1e-6 * std::max(1.0, std::abs(value));
        if (is_state) {
            xp[j] += h;
            xm[j] -= h;
        } else {
            up[j] += h;
            um[j] -= h;
        }
        const DM plus = f(std::vector<DM>{column(xp), column(up)}).at(0);
        const DM minus = f(std::vector<DM>{column(xm), column(um)}).at(0);
        const DM finite_difference = (plus - minus) / (2.0 * h);
        matches = close(exact.at(is_state ? 0 : 1)(casadi::Slice(), j),
                        finite_difference, 2e-4) && matches;
    }
    check(matches, f.name() + ": A and B match central finite differences");
}

void check_equivalence(const DoubleTrackModel& model, const State& x,
                       const Input& u, double dt, int substeps) {
    const auto functions = make_casadi_vehicle_functions(model, dt, substeps);
    const std::vector<DM> args{column(x), column(u)};
    DoubleTrackModel::Diagnostics<double> d;
    check(close(functions.continuous(args).at(0), column(model.dynamics(x, u, d))),
          "SX continuous dynamics match double");
    State expected = x;
    for (int i = 0; i < substeps; ++i) {
        expected = vehicle_model::rk4_step<DoubleTrackModel, double>(
            model, expected, u, dt / substeps);
    }
    check(close(functions.discrete(args).at(0), column(expected)),
          "SX discrete dynamics match double RK4 substeps");
    const auto diag = functions.diagnostics(args);
    check(diag.size() == 7 &&
          close(diag.at(0), column(d.kappa)) &&
          close(diag.at(1), column(d.alpha)) &&
          close(diag.at(2), column(d.Fz)) &&
          close(diag.at(3), column(d.Fx_w)) &&
          close(diag.at(4), column(d.Fy_w)) &&
          close(diag.at(5), DM(d.ax)) && close(diag.at(6), DM(d.ay)),
          "SX diagnostics match double, including per-wheel ordering");
    for (const auto* jac : {&functions.continuous_jacobian,
                            &functions.discrete_jacobian}) {
        const auto values = (*jac)(args);
        check(close(values.at(0), values.at(0)) && close(values.at(1), values.at(1)),
              jac->name() + ": derivatives are finite");
    }
}

} // namespace

int main() {
    const DoubleTrackModel model;
    State x = model.straight_driving_state(12.0);
    x[DoubleTrackModel::IPSI] = 0.2;
    x[DoubleTrackModel::IVY] = 0.15;
    x[DoubleTrackModel::IR] = 0.04;
    for (int i = 0; i < 4; ++i) x[DoubleTrackModel::IW_FL + i] += 0.4 + 0.1 * i;
    const Input u{0.04, 80.0, 160.0, -20.0, 120.0};

    check_equivalence(model, x, u, 0.02, 20);
    check_equivalence(model, State{}, Input{}, 0.02, 20);
    check_equivalence(model, State{}, u, 0.002, 4);
    check_equivalence(model, model.straight_driving_state(-5.0), Input{}, 0.001, 2);
    auto params = model.params();
    params.m = 1800.0;
    params.mu = 0.7;
    check_equivalence(DoubleTrackModel(params), x, u, 0.001, 1);

    const auto functions = make_casadi_vehicle_functions(model, 0.02);
    check(functions.discrete.name_in(0) == "x" &&
          functions.discrete.name_in(1) == "u" &&
          functions.discrete.name_out(0) == "x_next" &&
          functions.continuous.name_out(0) == "xdot",
          "MPC function inputs and outputs are named");
    check_jacobian(functions.continuous, functions.continuous_jacobian, x, u);
    check_jacobian(functions.discrete, functions.discrete_jacobian, x, u);

    using casadi::MX;
    const MX X = MX::sym("X", DoubleTrackModel::NX, 2);
    const MX U = MX::sym("U", DoubleTrackModel::NU);
    const MX next = functions.discrete(std::vector<MX>{X(casadi::Slice(), 0), U}).at(0);
    const MX residual = X(casadi::Slice(), 1) - next;
    const casadi::Function shooting("shooting_residual", {X, U},
        {residual, MX::jacobian(residual, U)});
    const DM xn = functions.discrete(std::vector<DM>{column(x), column(u)}).at(0);
    const auto result = shooting(std::vector<DM>{DM::horzcat({column(x), xn}), column(u)});
    check(close(result.at(0), DM::zeros(DoubleTrackModel::NX, 1)),
          "MX multiple-shooting residual is zero for predicted state");
    const auto jac = functions.discrete_jacobian(std::vector<DM>{column(x), column(u)});
    check(close(result.at(1), -jac.at(1)),
          "MX composition preserves automatic differentiation");

    const auto restored = casadi::Function::deserialize(functions.discrete.serialize());
    check(close(restored(std::vector<DM>{column(x), column(u)}).at(0), xn),
          "Discrete function survives CasADi serialization");

    for (const double dt : {0.0, -0.02, std::numeric_limits<double>::infinity(),
                            std::numeric_limits<double>::quiet_NaN()}) {
        bool rejected = false;
        try {
            make_casadi_vehicle_functions(model, dt);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        check(rejected, "Invalid sample time raises invalid_argument");
    }
    for (const int substeps : {0, -1}) {
        bool rejected = false;
        try {
            make_casadi_vehicle_functions(model, 0.02, substeps);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        check(rejected, "Invalid substep count raises invalid_argument");
    }

    std::cout << failures << " failure(s)\n";
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
