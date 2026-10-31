#pragma once

#include "arm/dynamics_solver.hpp"

#include <array>
#include <vector>

namespace arm {

struct ArmState {
    std::vector<double> joint_positions;
    std::vector<double> joint_velocities;
};

using TorqueSequence = std::vector<std::vector<double>>;

struct RolloutObjective {
    std::vector<double> target_positions;
    std::vector<std::vector<double>> trajectory_targets;
    double terminal_weight = 1.0;
    double trajectory_weight = 0.0;
    double control_weight = 0.0;
    bool cartesian = false;
    std::size_t link = 0;
    Vec3 local_point{};
    Vec3 target_point{};
};

struct RolloutGradient {
    double loss = 0.0;
    ArmState initial_state;
    TorqueSequence controls;
    std::vector<double> masses;
    std::vector<std::array<double, 6>> inertias;
};

class RolloutSolver {
public:
    explicit RolloutSolver(const DynamicsSolver& dynamics_solver);

    ArmState step(const ArmState& state, const std::vector<double>& torques, double timestep) const;
    ArmState rollout(
        const ArmState& initial_state,
        const TorqueSequence& torque_sequence,
        double timestep) const;

    double terminal_position_loss(
        const ArmState& final_state,
        const std::vector<double>& target_positions) const;

    TorqueSequence central_difference_control_gradient(
        const ArmState& initial_state,
        const TorqueSequence& torque_sequence,
        double timestep,
        const std::vector<double>& target_positions,
        double epsilon = 1e-5) const;

    TorqueSequence automatic_control_gradient(
        const ArmState& initial_state,
        const TorqueSequence& torque_sequence,
        double timestep,
        const std::vector<double>& target_positions) const;

    double evaluate_objective(
        const ArmState& initial_state,
        const TorqueSequence& controls,
        double dt,
        const RolloutObjective& objective) const;

    RolloutGradient automatic_gradient(
        const ArmState& initial_state,
        const TorqueSequence& controls,
        double dt,
        const RolloutObjective& objective) const;

    double automatic_directional_derivative(
        const ArmState& initial_state,
        const TorqueSequence& controls,
        double dt,
        const RolloutObjective& objective,
        const RolloutGradient& direction) const;

private:
    void validate_rollout_inputs(
        const ArmState& initial_state,
        const TorqueSequence& controls,
        double timestep) const;
    void validate_objective(const RolloutObjective& objective, std::size_t steps) const;

    const DynamicsSolver& dynamics_solver_;
};

}  // namespace arm