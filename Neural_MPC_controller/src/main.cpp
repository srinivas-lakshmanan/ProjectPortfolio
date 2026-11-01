// Demo: simulates the double-track model in two scenarios and writes CSV
// trajectories that can be plotted with any tool.
//
//  1) step_steer.csv       : accelerate, then apply a step steering input
//  2) torque_vectoring.csv : no steering; yaw the vehicle purely with a
//                            left/right drive-torque differential

#include <fstream>
#include <iostream>
#include <string>

#include "vehicle_model/double_track_model.hpp"
#include "vehicle_model/integrator.hpp"

using vehicle_model::DoubleTrackModel;
using vehicle_model::rk4_step;

using State = DoubleTrackModel::State<double>;
using Input = DoubleTrackModel::Input<double>;

namespace {

void write_header(std::ofstream& os) {
    os << "t,X,Y,psi,vx,vy,r,w_fl,w_fr,w_rl,w_rr,"
          "delta,T_fl,T_fr,T_rl,T_rr,ax,ay,"
          "Fz_fl,Fz_fr,Fz_rl,Fz_rr\n";
}

void write_row(std::ofstream& os, double t, const State& x, const Input& u,
               const DoubleTrackModel::Diagnostics<double>& d) {
    os << t;
    for (double v : x) os << ',' << v;
    for (double v : u) os << ',' << v;
    os << ',' << d.ax << ',' << d.ay;
    for (double v : d.Fz) os << ',' << v;
    os << '\n';
}

void simulate(const DoubleTrackModel& model, State x, double t_end, double dt,
              Input (*policy)(double), const std::string& csv_path) {
    std::ofstream os(csv_path);
    write_header(os);

    DoubleTrackModel::Diagnostics<double> diag;
    for (double t = 0.0; t <= t_end + 1e-12; t += dt) {
        const Input u = policy(t);
        model.dynamics(x, u, diag);  // diagnostics at current state
        write_row(os, t, x, u, diag);
        x = rk4_step<DoubleTrackModel, double>(model, x, u, dt);
    }
    std::cout << "wrote " << csv_path << "\n";
}

// Scenario 1: constant drive torque, step steer of 3 deg at t = 2 s.
Input step_steer_policy(double t) {
    const double delta = (t >= 2.0) ? 3.0 * M_PI / 180.0 : 0.0;
    const double T = 150.0;  // Nm per wheel (AWD)
    return {delta, T, T, T, T};
}

// Scenario 2: zero steering, yaw purely via torque differential
// (right wheels driven harder -> positive yaw / left turn).
Input torque_vectoring_policy(double t) {
    const double T_base = 120.0;
    const double dT = (t >= 2.0) ? 80.0 : 0.0;
    return {0.0, T_base - dT, T_base + dT, T_base - dT, T_base + dT};
}

} // namespace

int main() {
    DoubleTrackModel model;  // default mid-size car parameters

    const double dt = 1e-3;
    const double t_end = 8.0;
    const State x0 = model.straight_driving_state(15.0);  // 15 m/s initial

    simulate(model, x0, t_end, dt, &step_steer_policy, "step_steer.csv");
    simulate(model, x0, t_end, dt, &torque_vectoring_policy,
             "torque_vectoring.csv");
    return 0;
}
