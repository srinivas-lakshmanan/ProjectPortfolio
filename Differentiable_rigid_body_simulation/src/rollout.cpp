#include "arm/rollout.hpp"
#include "arm/simulation_core.hpp"

#include <cmath>
#include <stdexcept>

namespace arm {
namespace {

void validate_state(const ArmState& state, std::size_t dof) {
    detail::validate_vector(state.joint_positions, dof);
    detail::validate_vector(state.joint_velocities, dof);
}

void validate_timestep(double timestep) {
    if (!std::isfinite(timestep) || timestep <= 0.0) {
        throw std::invalid_argument("Timestep must be finite and positive");
    }
}

}  // namespace

RolloutSolver::RolloutSolver(const DynamicsSolver& dynamics_solver)
    : dynamics_solver_(dynamics_solver) {}

void RolloutSolver::validate_rollout_inputs(
    const ArmState& initial_state,
    const TorqueSequence& controls,
    double timestep) const {
    validate_timestep(timestep);
    const auto& model = dynamics_solver_.model();
    validate_state(initial_state, model.degrees_of_freedom());
    detail::validate_options(model, dynamics_solver_.gravity(), dynamics_solver_.options());
    for (const auto& torques : controls) {
        detail::validate_vector(torques, model.degrees_of_freedom());
    }
}

ArmState RolloutSolver::step(
    const ArmState& state,
    const std::vector<double>& torques,
    double timestep) const {
    validate_timestep(timestep);
    const auto& model = dynamics_solver_.model();
    const auto result = detail::rk4(
        model, dynamics_solver_.gravity(), dynamics_solver_.options(),
        detail::parameters<double>(model),
        detail::State<double>{state.joint_positions, state.joint_velocities}, torques, timestep);
    return {result.positions, result.velocities};
}

ArmState RolloutSolver::rollout(
    const ArmState& initial_state,
    const TorqueSequence& torque_sequence,
    double timestep) const {
    validate_rollout_inputs(initial_state, torque_sequence, timestep);

    ArmState state = initial_state;
    for (const std::vector<double>& torques : torque_sequence) {
        state = step(state, torques, timestep);
    }
    return state;
}

double RolloutSolver::terminal_position_loss(
    const ArmState& final_state,
    const std::vector<double>& target_positions) const {
    const auto dof = dynamics_solver_.model().degrees_of_freedom();
    validate_state(final_state, dof);
    detail::validate_vector(target_positions, dof);
    double loss = 0.0;
    for (std::size_t index = 0; index < target_positions.size(); ++index) {
        const double error = final_state.joint_positions[index] - target_positions[index];
        loss += 0.5 * error * error;
    }
    return loss;
}

TorqueSequence RolloutSolver::central_difference_control_gradient(
    const ArmState& initial_state,
    const TorqueSequence& torque_sequence,
    double timestep,
    const std::vector<double>& target_positions,
    double epsilon) const {
    if (!std::isfinite(epsilon) || epsilon <= 0.0) {
        throw std::invalid_argument("Gradient epsilon must be finite and positive");
    }
    validate_rollout_inputs(initial_state, torque_sequence, timestep);
    detail::validate_vector(target_positions, dynamics_solver_.model().degrees_of_freedom());

    TorqueSequence gradient(torque_sequence.size());
    for (std::size_t step_index = 0; step_index < torque_sequence.size(); ++step_index) {
        gradient[step_index].resize(torque_sequence[step_index].size());
        for (std::size_t joint_index = 0;
             joint_index < torque_sequence[step_index].size(); ++joint_index) {
            TorqueSequence plus = torque_sequence;
            TorqueSequence minus = torque_sequence;
            plus[step_index][joint_index] += epsilon;
            minus[step_index][joint_index] -= epsilon;
            const double plus_loss = terminal_position_loss(
                rollout(initial_state, plus, timestep), target_positions);
            const double minus_loss = terminal_position_loss(
                rollout(initial_state, minus, timestep), target_positions);
            gradient[step_index][joint_index] = (plus_loss - minus_loss) / (2.0 * epsilon);
        }
    }
    return gradient;
}

}  // namespace arm