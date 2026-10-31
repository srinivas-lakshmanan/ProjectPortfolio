#include "arm/arm_model.hpp"
#include "arm/dynamics_solver.hpp"
#include "arm/kinematics_solver.hpp"
#include "arm/rollout.hpp"

#include <iomanip>
#include <iostream>
#include <vector>

int main() {
    const arm::ArmModel model({
        {"shoulder_yaw", {0.0, 0.0, 1.0}, {0.0, 0.0, 0.30}, 0.4,
         {0.0, 0.0, 0.15}, {{0.002, 0.0, 0.0, 0.0, 0.002, 0.0, 0.0, 0.0, 0.001}}},
        {"shoulder_pitch", {0.0, 1.0, 0.0}, {0.30, 0.0, 0.0}, 1.5,
         {0.15, 0.0, 0.0}, {{0.012, 0.0, 0.0, 0.0, 0.012, 0.0, 0.0, 0.0, 0.003}}},
        {"elbow_pitch", {0.0, 1.0, 0.0}, {0.25, 0.0, 0.0}, 0.9,
         {0.125, 0.0, 0.0}, {{0.005, 0.0, 0.0, 0.0, 0.005, 0.0, 0.0, 0.0, 0.0015}}}
    });
    const arm::KinematicsSolver solver(model);
    const std::vector<double> positions{0.2, -0.5, 0.8};
    const arm::KinematicsResult result = solver.solve(positions);
    const arm::DynamicsSolver dynamics_solver(model);
    const arm::ForwardDynamicsResult dynamics =
        dynamics_solver.forward_dynamics(positions, {0.0, 0.0, 0.0}, {0.5, 0.0, 0.0});

    std::cout << std::fixed << std::setprecision(4)
              << "End-effector position: ("
              << result.end_effector.position.x << ", "
              << result.end_effector.position.y << ", "
              << result.end_effector.position.z << ")\n"
              << "Position Jacobian columns:\n";
    for (const arm::Vec3& column : result.position_jacobian_columns) {
        std::cout << "  (" << column.x << ", " << column.y << ", " << column.z << ")\n";
    }
    std::cout << "Joint accelerations under applied torques: ";
    for (double acceleration : dynamics.joint_accelerations) {
        std::cout << acceleration << " ";
    }
    std::cout << '\n';

    const arm::RolloutSolver rollout_solver(dynamics_solver);
    const arm::ArmState initial_state{positions, {0.0, 0.0, 0.0}};
    const arm::TorqueSequence torques(20, {0.5, 0.0, 0.0});
    const arm::ArmState final_state = rollout_solver.rollout(initial_state, torques, 0.01);
    std::cout << "Joint positions after rollout: ";
    for (double position : final_state.joint_positions) {
        std::cout << position << " ";
    }
    std::cout << '\n';
}