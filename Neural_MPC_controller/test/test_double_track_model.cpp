// Sanity tests for the double-track vehicle model (no external framework).

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "vehicle_model/double_track_model.hpp"
#include "vehicle_model/integrator.hpp"

using vehicle_model::DoubleTrackModel;
using vehicle_model::rk4_step;

using State = DoubleTrackModel::State<double>;
using Input = DoubleTrackModel::Input<double>;

namespace {

int failures = 0;

void check(bool cond, const char* msg) {
    if (cond) {
        std::printf("[PASS] %s\n", msg);
    } else {
        std::printf("[FAIL] %s\n", msg);
        ++failures;
    }
}

State simulate(const DoubleTrackModel& m, State x, const Input& u,
               double t_end, double dt = 1e-3) {
    for (double t = 0.0; t < t_end; t += dt) {
        x = rk4_step<DoubleTrackModel, double>(m, x, u, dt);
    }
    return x;
}

bool all_finite(const State& x) {
    for (double v : x) {
        if (!std::isfinite(v)) return false;
    }
    return true;
}

} // namespace

int main() {
    DoubleTrackModel model;
    const auto& p = model.params();

    // 1) Symmetric drive torques keep the vehicle straight and accelerate it.
    {
        const State x0 = model.straight_driving_state(10.0);
        const Input u{0.0, 200.0, 200.0, 200.0, 200.0};
        const State x = simulate(model, x0, u, 3.0);
        check(all_finite(x), "straight driving: states finite");
        check(x[DoubleTrackModel::IVX] > 10.5,
              "straight driving: vehicle accelerates");
        check(std::fabs(x[DoubleTrackModel::IVY]) < 1e-6 &&
              std::fabs(x[DoubleTrackModel::IR]) < 1e-6,
              "straight driving: no lateral motion or yaw");
        check(std::fabs(x[DoubleTrackModel::IY]) < 1e-6,
              "straight driving: stays on the x-axis");
    }

    // 2) Coast-down: zero torque -> resistances decelerate the vehicle.
    {
        const State x0 = model.straight_driving_state(20.0);
        const Input u{0.0, 0.0, 0.0, 0.0, 0.0};
        const State x = simulate(model, x0, u, 3.0);
        check(x[DoubleTrackModel::IVX] < 20.0 &&
              x[DoubleTrackModel::IVX] > 0.0,
              "coast-down: drag and rolling resistance decelerate");
    }

    // 3) Differential drive: right wheels driven harder -> positive yaw rate
    //    (left turn), with zero steering input.
    {
        const State x0 = model.straight_driving_state(10.0);
        const Input u{0.0, 50.0, 250.0, 50.0, 250.0};
        const State x = simulate(model, x0, u, 2.0);
        check(x[DoubleTrackModel::IR] > 1e-3,
              "differential drive: torque split alone yaws the vehicle");
        check(x[DoubleTrackModel::IY] > 0.0,
              "differential drive: vehicle curves to the left");
    }

    // 4) Step steer left (delta > 0) -> positive yaw rate and lateral accel;
    //    lateral load transfer loads the outer (right) wheels.
    {
        const State x0 = model.straight_driving_state(15.0);
        const Input u{3.0 * M_PI / 180.0, 100.0, 100.0, 100.0, 100.0};
        const State x = simulate(model, x0, u, 2.0);
        check(x[DoubleTrackModel::IR] > 0.05, "step steer: yaw rate develops");

        DoubleTrackModel::Diagnostics<double> d;
        model.dynamics(x, u, d);
        check(d.ay > 0.5, "step steer: lateral acceleration develops");
        check(d.Fz[1] > d.Fz[0] && d.Fz[3] > d.Fz[2],
              "step steer: load transfers to outer wheels");
        const double Fz_sum = d.Fz[0] + d.Fz[1] + d.Fz[2] + d.Fz[3];
        check(std::fabs(Fz_sum - p.m * p.g) < 1.0,
              "load transfer: total vertical load is conserved");
    }

    // 5) Smoothness at standstill: dynamics must stay finite at v = 0.
    {
        State x{};  // everything zero
        const Input u{0.1, 50.0, 50.0, 50.0, 50.0};
        DoubleTrackModel::Diagnostics<double> d;
        const State xdot = model.dynamics(x, u, d);
        check(all_finite(xdot), "standstill: dynamics finite at v = 0");
        const State x1 = simulate(model, x, u, 1.0);
        check(all_finite(x1) && x1[DoubleTrackModel::IVX] > 0.0,
              "standstill: vehicle launches from rest");
    }

    std::printf("\n%s (%d failure%s)\n",
                failures == 0 ? "ALL TESTS PASSED" : "TESTS FAILED",
                failures, failures == 1 ? "" : "s");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
