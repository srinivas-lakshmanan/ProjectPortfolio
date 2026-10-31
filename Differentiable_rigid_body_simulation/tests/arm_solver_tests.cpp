#include "arm/arm_model.hpp"
#include "arm/dynamics_solver.hpp"
#include "arm/kinematics_solver.hpp"
#include "arm/rollout.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

arm::ArmModel make_test_arm() {
    return arm::ArmModel({
        {"shoulder_yaw", {0.0, 0.0, 1.0}, {0.0, 0.0, 0.30}, 0.4,
         {0.0, 0.0, 0.15}, {{0.002, 0.0, 0.0, 0.0, 0.002, 0.0, 0.0, 0.0, 0.001}}},
        {"shoulder_pitch", {0.0, 1.0, 0.0}, {0.30, 0.0, 0.0}, 1.5,
         {0.15, 0.0, 0.0}, {{0.012, 0.0, 0.0, 0.0, 0.012, 0.0, 0.0, 0.0, 0.003}}},
        {"elbow_pitch", {0.0, 1.0, 0.0}, {0.25, 0.0, 0.0}, 0.9,
         {0.125, 0.0, 0.0}, {{0.005, 0.0, 0.0, 0.0, 0.005, 0.0, 0.0, 0.0, 0.0015}}}
    });
}

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void test_model_geometry_validation() {
    const arm::RevoluteJoint valid = make_test_arm().joints()[0];
    for (double invalid : {std::numeric_limits<double>::quiet_NaN(),
                           std::numeric_limits<double>::infinity()}) {
        for (std::size_t component = 0; component < 6; ++component) {
            arm::RevoluteJoint joint = valid;
            double* components[] = {&joint.axis.x, &joint.axis.y, &joint.axis.z,
                                    &joint.link_offset.x, &joint.link_offset.y,
                                    &joint.link_offset.z};
            *components[component] = invalid;
            bool threw = false;
            try {
                static_cast<void>(arm::ArmModel({joint}));
            } catch (const std::invalid_argument&) {
                threw = true;
            }
            require(threw, "Model must reject nonfinite axes and link offsets");
        }
    }
    arm::RevoluteJoint scaled = valid;
    scaled.axis = {0.0, 0.0, 2.0};
    const arm::ArmModel model({scaled});
    require(std::abs(arm::norm(model.joints()[0].axis) - 1.0) < 1e-12,
            "Valid joint axes must still be normalized");
}

void test_zero_configuration() {
    const arm::ArmModel model = make_test_arm();
    const arm::KinematicsSolver solver(model);
    const arm::KinematicsResult result = solver.solve({0.0, 0.0, 0.0});

    require(std::abs(result.end_effector.position.x - 0.55) < 1e-12,
            "Zero pose should place the hand at the sum of link lengths");
    require(std::abs(result.end_effector.position.y) < 1e-12,
            "Zero pose should have zero y position");
    require(std::abs(result.end_effector.position.z - 0.30) < 1e-12,
            "Zero pose should preserve shoulder height");
}

void test_jacobian_against_finite_differences() {
    const arm::ArmModel model = make_test_arm();
    const arm::KinematicsSolver solver(model);
    const std::vector<double> positions{0.31, -0.42, 0.67};
    const arm::KinematicsResult result = solver.solve(positions);
    const double epsilon = 1e-6;

    for (std::size_t joint = 0; joint < positions.size(); ++joint) {
        std::vector<double> plus = positions;
        std::vector<double> minus = positions;
        plus[joint] += epsilon;
        minus[joint] -= epsilon;

        const arm::Vec3 plus_position = solver.solve(plus).end_effector.position;
        const arm::Vec3 minus_position = solver.solve(minus).end_effector.position;
        const arm::Vec3 finite_difference = (plus_position - minus_position) * (0.5 / epsilon);
        const arm::Vec3 analytic = result.position_jacobian_columns[joint];

        require(arm::norm(finite_difference - analytic) < 1e-7,
                "Analytic position Jacobian does not match finite differences");
    }
}

void test_joint_count_validation() {
    const arm::ArmModel model = make_test_arm();
    const arm::KinematicsSolver solver(model);
    bool threw = false;
    try {
        static_cast<void>(solver.solve({0.0, 0.0}));
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    require(threw, "Solver should reject a joint vector with the wrong size");
}

void test_forward_inverse_dynamics_round_trip() {
    const arm::ArmModel model = make_test_arm();
    const arm::DynamicsSolver solver(model);
    const std::vector<double> positions{0.31, -0.42, 0.67};
    const std::vector<double> velocities{0.2, -0.3, 0.15};
    const std::vector<double> expected_accelerations{0.4, -1.2, 0.7};
    const std::vector<double> torques =
        solver.inverse_dynamics(positions, velocities, expected_accelerations);
    const arm::ForwardDynamicsResult result =
        solver.forward_dynamics(positions, velocities, torques);

    require(result.mass_matrix.size() == positions.size(),
            "Forward dynamics should return its mass matrix");
    for (std::size_t row = 0; row < positions.size(); ++row) {
        require(std::abs(result.joint_accelerations[row] - expected_accelerations[row]) < 1e-9,
                "Forward dynamics should recover inverse-dynamics accelerations");
        for (std::size_t column = 0; column < positions.size(); ++column) {
            require(std::abs(result.mass_matrix[row][column] -
                             result.mass_matrix[column][row]) < 1e-10,
                    "Arm mass matrix should be symmetric");
        }
    }
}

void test_gravity_compensation() {
    const arm::ArmModel model = make_test_arm();
    const arm::DynamicsSolver solver(model);
    const std::vector<double> positions{0.2, -0.5, 0.8};
    const std::vector<double> velocities(positions.size(), 0.0);
    const std::vector<double> accelerations(positions.size(), 0.0);
    const std::vector<double> gravity_torques =
        solver.inverse_dynamics(positions, velocities, accelerations);
    const arm::ForwardDynamicsResult result =
        solver.forward_dynamics(positions, velocities, gravity_torques);

    for (double acceleration : result.joint_accelerations) {
        require(std::abs(acceleration) < 1e-9,
                "Inverse-dynamics gravity torques should hold the arm stationary");
    }
}

arm::ArmModel make_pendulum() {
    return arm::ArmModel({
        {"pendulum", {0.0, 1.0, 0.0}, {1.0, 0.0, 0.0}, 2.0,
         {0.5, 0.0, 0.0}, {{0.01, 0.0, 0.0, 0.0, 1.0 / 6.0, 0.0, 0.0, 0.0, 1.0 / 6.0}}}
    });
}

void test_single_link_pendulum_acceleration() {
    const arm::ArmModel model = make_pendulum();
    const arm::DynamicsSolver solver(model);
    for (double angle : {-1.2, 0.0, 0.7, 1.5707963267948966, 2.4}) {
        for (double torque : {-0.4, 0.0, 0.6}) {
            const auto result = solver.forward_dynamics({angle}, {0.3}, {torque});
            const double expected = (torque + 9.81 * std::cos(angle)) / (2.0 / 3.0);
            require(std::abs(result.joint_accelerations[0] - expected) < 1e-10,
                    "Pendulum acceleration must match its independent analytical equation");
        }
    }
}

double pendulum_energy(const arm::ArmState& state) {
    const double velocity = state.joint_velocities[0];
    return velocity * velocity / 3.0 - 9.81 * std::sin(state.joint_positions[0]);
}

void test_pendulum_energy_convergence() {
    const arm::ArmModel model = make_pendulum();
    const arm::DynamicsSolver dynamics(model);
    const arm::RolloutSolver solver(dynamics);
    const arm::ArmState initial{{0.7}, {0.2}};
    const double initial_energy = pendulum_energy(initial);
    double previous_drift = 0.0;
    for (double timestep : {0.04, 0.02, 0.01}) {
        arm::ArmState state = initial;
        double maximum_drift = 0.0;
        const std::size_t steps = static_cast<std::size_t>(std::lround(2.0 / timestep));
        for (std::size_t step = 0; step < steps; ++step) {
            state = solver.step(state, {0.0}, timestep);
            maximum_drift = std::max(maximum_drift,
                std::abs(pendulum_energy(state) - initial_energy));
        }
        require(std::isfinite(maximum_drift), "Pendulum energy must remain finite");
        if (previous_drift > 0.0) {
            require(maximum_drift < previous_drift / 8.0,
                    "RK4 energy drift must decrease under timestep refinement");
        }
        previous_drift = maximum_drift;
        std::cout << "Pendulum maximum energy drift (dt=" << timestep
                  << "): " << maximum_drift << '\n';
    }
    require(previous_drift < 1e-6, "Fine-step pendulum energy drift must be small");
}

void check_nonlinear_gradients(const arm::ArmModel& model,
                             const arm::ArmState& initial,
                             const std::vector<double>& target) {
    const arm::DynamicsSolver dynamics(model);
    const arm::RolloutSolver solver(dynamics);
    const std::size_t dof = model.degrees_of_freedom();
    const double timestep = 0.01;
    for (std::size_t steps : {std::size_t(5), std::size_t(20), std::size_t(60)}) {
        arm::TorqueSequence controls(steps, std::vector<double>(dof));
        arm::TorqueSequence direction = controls;
        double direction_squared = 0.0;
        for (std::size_t step = 0; step < steps; ++step) {
            for (std::size_t joint = 0; joint < dof; ++joint) {
                const double phase = static_cast<double>(step * dof + joint + 1);
                controls[step][joint] = 0.1 * std::sin(phase * 0.3);
                direction[step][joint] = std::cos(phase * 0.7);
                direction_squared += direction[step][joint] * direction[step][joint];
            }
        }
        for (auto& row : direction) {
            for (double& value : row) {
                value /= std::sqrt(direction_squared);
            }
        }
        const auto automatic = solver.automatic_control_gradient(initial, controls, timestep, target);
        double automatic_squared = 0.0;
        double automatic_directional = 0.0;
        for (std::size_t step = 0; step < steps; ++step) {
            for (std::size_t joint = 0; joint < dof; ++joint) {
                automatic_squared += automatic[step][joint] * automatic[step][joint];
                automatic_directional += automatic[step][joint] * direction[step][joint];
            }
        }
        require(automatic_squared > 1e-16, "Gradient fixture must have a nonzero gradient");
        for (double epsilon : {1e-3, 1e-4, 1e-5, 1e-6}) {
            const auto finite = solver.central_difference_control_gradient(
                initial, controls, timestep, target, epsilon);
            double error_squared = 0.0;
            double finite_squared = 0.0;
            double agreement = 0.0;
            auto plus = controls;
            auto minus = controls;
            for (std::size_t step = 0; step < steps; ++step) {
                for (std::size_t joint = 0; joint < dof; ++joint) {
                    const double error = automatic[step][joint] - finite[step][joint];
                    error_squared += error * error;
                    finite_squared += finite[step][joint] * finite[step][joint];
                    agreement += automatic[step][joint] * finite[step][joint];
                    plus[step][joint] += epsilon * direction[step][joint];
                    minus[step][joint] -= epsilon * direction[step][joint];
                }
            }
            const double relative_error = std::sqrt(error_squared) /
                std::max(1e-12, std::max(std::sqrt(automatic_squared), std::sqrt(finite_squared)));
            const double cosine = agreement / std::max(1e-24,
                std::sqrt(automatic_squared) * std::sqrt(finite_squared));
            const double finite_directional = (
                solver.terminal_position_loss(solver.rollout(initial, plus, timestep), target) -
                solver.terminal_position_loss(solver.rollout(initial, minus, timestep), target)) /
                (2.0 * epsilon);
            require(relative_error < 2e-4, "Nonlinear gradient relative error is too large");
            require(cosine > 0.99999, "Nonlinear gradient directions must agree");
            require(std::abs(finite_directional - automatic_directional) <
                        1e-8 + 2e-4 * std::abs(automatic_directional),
                    "Centered directional derivative must match AD");
            std::cout << "Gradient check (dof=" << dof << ", steps=" << steps
                      << ", epsilon=" << epsilon << "): relative=" << relative_error
                      << ", cosine=" << cosine << '\n';
        }
    }
}

void test_nonlinear_gradient_ladder() {
    check_nonlinear_gradients(make_pendulum(), {{0.7}, {0.2}}, {1.2});
    check_nonlinear_gradients(make_test_arm(),
                             {{0.18, -0.37, 0.52}, {0.08, -0.12, 0.05}},
                             {0.25, -0.2, 0.4});
}

void test_rollout_gradient_against_finite_differences() {
    const arm::ArmModel model({
        {"rotor", {0.0, 1.0, 0.0}, {0.5, 0.0, 0.0}, 1.0,
         {0.0, 0.0, 0.0}, {{0.1, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.3}}}
    });
    const arm::DynamicsSolver dynamics_solver(model, {0.0, 0.0, 0.0});
    const arm::RolloutSolver rollout_solver(dynamics_solver);
    const arm::ArmState initial_state{{0.1}, {0.05}};
    const arm::TorqueSequence torques{{0.4}, {-0.2}, {0.8}, {0.1}};
    const double timestep = 0.1;
    const std::vector<double> target_positions{0.25};
    const arm::ArmState final_state = rollout_solver.rollout(initial_state, torques, timestep);
    const arm::TorqueSequence gradient = rollout_solver.central_difference_control_gradient(
        initial_state, torques, timestep, target_positions, 1e-5);
    const arm::TorqueSequence automatic_gradient = rollout_solver.automatic_control_gradient(
        initial_state, torques, timestep, target_positions);

    double torque_sum = 0.0;
    double weighted_torque_sum = 0.0;
    for (std::size_t index = 0; index < torques.size(); ++index) {
        torque_sum += torques[index][0];
        weighted_torque_sum +=
            (static_cast<double>(torques.size() - index) - 0.5) * torques[index][0];
    }
    const double inertia = 0.25;
    const double expected_position = initial_state.joint_positions[0] +
        initial_state.joint_velocities[0] * timestep * static_cast<double>(torques.size()) +
        timestep * timestep * weighted_torque_sum / inertia;
    const double expected_velocity = initial_state.joint_velocities[0] +
        timestep * torque_sum / inertia;
    const double loss_position_error = expected_position - target_positions[0];

    require(std::abs(final_state.joint_positions[0] - expected_position) < 1e-12,
            "RK4 rollout should match constant-acceleration rotor trajectory");
    require(std::abs(final_state.joint_velocities[0] - expected_velocity) < 1e-12,
            "RK4 rollout should match constant-acceleration rotor velocity");

    for (std::size_t index = 0; index < torques.size(); ++index) {
        const double position_sensitivity = timestep * timestep / inertia *
            (static_cast<double>(torques.size() - index) - 0.5);
        const double expected_gradient = loss_position_error * position_sensitivity;
        require(std::abs(gradient[index][0] - expected_gradient) < 1e-9,
                "Rollout control gradient should match the analytical and FD result");
        require(std::abs(automatic_gradient[index][0] - expected_gradient) < 1e-12,
                "Forward-mode AD gradient should match the analytical rotor gradient");
    }
}

void test_articulated_automatic_gradient_against_finite_differences() {
    const arm::ArmModel model = make_test_arm();
    const arm::DynamicsSolver dynamics_solver(model);
    const arm::RolloutSolver rollout_solver(dynamics_solver);
    const arm::ArmState initial_state{{0.18, -0.37, 0.52}, {0.08, -0.12, 0.05}};
    const arm::TorqueSequence torques{
        {0.3, -0.2, 0.1},
        {-0.1, 0.25, -0.15},
        {0.2, 0.05, 0.3}
    };
    const std::vector<double> target_positions{0.25, -0.2, 0.4};
    const double timestep = 0.015;
    const arm::TorqueSequence automatic_gradient = rollout_solver.automatic_control_gradient(
        initial_state, torques, timestep, target_positions);

    for (double epsilon : {1e-3, 1e-4, 1e-5}) {
        const arm::TorqueSequence finite_difference =
            rollout_solver.central_difference_control_gradient(
                initial_state, torques, timestep, target_positions, epsilon);
        double maximum_absolute_error = 0.0;
        for (std::size_t step_index = 0; step_index < torques.size(); ++step_index) {
            for (std::size_t joint_index = 0; joint_index < torques[step_index].size(); ++joint_index) {
                const double automatic = automatic_gradient[step_index][joint_index];
                const double finite = finite_difference[step_index][joint_index];
                const double tolerance = 2e-6 * std::max(1.0, std::abs(automatic));
                maximum_absolute_error = std::max(
                    maximum_absolute_error, std::abs(automatic - finite));
                require(std::abs(automatic - finite) < tolerance,
                        "Articulated rollout AD gradient should agree with centered differences");
            }
        }
        std::cout << "AD/FD maximum absolute gradient error (epsilon=" << epsilon
                  << "): " << maximum_absolute_error << '\n';
    }
}

}  // namespace

int main() {
    try {
        test_model_geometry_validation();
        test_zero_configuration();
        test_jacobian_against_finite_differences();
        test_joint_count_validation();
        test_forward_inverse_dynamics_round_trip();
        test_gravity_compensation();
        test_single_link_pendulum_acceleration();
        test_pendulum_energy_convergence();
        test_rollout_gradient_against_finite_differences();
        test_articulated_automatic_gradient_against_finite_differences();
        test_nonlinear_gradient_ladder();
    } catch (const std::exception& error) {
        std::cerr << "Test failure: " << error.what() << '\n';
        return 1;
    }

    std::cout << "All arm solver tests passed.\n";
}