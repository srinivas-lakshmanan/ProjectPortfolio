#include "arm/rollout.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>

int main() {
    const arm::ArmModel model({
        {"pendulum", {0.0, 1.0, 0.0}, {1.0, 0.0, 0.0}, 2.0,
         {0.5, 0.0, 0.0}, {{0.01, 0.0, 0.0, 0.0, 1.0 / 6.0, 0.0,
                            0.0, 0.0, 1.0 / 6.0}}}
    });
    const arm::DynamicsSolver dynamics(model);
    const arm::RolloutSolver solver(dynamics);
    const arm::ArmState initial{{0.7}, {0.2}};
    arm::ArmState state = initial;
    const double timestep = 0.01;
    std::cout << std::fixed << std::setprecision(8)
              << "time_s,angle_rad,velocity_rad_s,energy_J\n";
    for (std::size_t step = 0; step <= 200; ++step) {
        if (step % 20 == 0) {
            const double velocity = state.joint_velocities[0];
            const double energy = velocity * velocity / 3.0 -
                9.81 * std::sin(state.joint_positions[0]);
            std::cout << static_cast<double>(step) * timestep << ','
                      << state.joint_positions[0] << ',' << velocity << ','
                      << energy << '\n';
        }
        if (step < 200) {
            state = solver.step(state, {0.0}, timestep);
        }
    }
    const arm::TorqueSequence controls(60, {0.0});
    const auto gradient = solver.automatic_control_gradient(initial, controls, timestep, {1.2});
    const auto final_state = solver.rollout(initial, controls, timestep);
    std::cout << "Terminal loss at 0.6 s: "
              << solver.terminal_position_loss(final_state, {1.2}) << '\n'
              << "First control derivative: " << gradient.front()[0] << '\n';
}