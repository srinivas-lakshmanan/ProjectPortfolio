#include "arm/rollout.hpp"
#include "arm/simulation_core.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace arm {
namespace {

constexpr std::array<std::array<std::size_t, 2>, 6> inertia_entries{{
    {{0, 0}}, {{4, 4}}, {{8, 8}}, {{1, 3}}, {{2, 6}}, {{5, 7}}
}};

struct VariableLayout {
    std::size_t dof;
    std::size_t control_count;
    std::size_t masses;
    std::size_t inertias;
    std::size_t dimensions;
};

VariableLayout variable_layout(const ArmModel& model, std::size_t steps) {
    const auto dof = model.degrees_of_freedom();
    const auto maximum = std::numeric_limits<std::size_t>::max();
    if (dof > maximum / 9 || (dof != 0 && steps > (maximum - 9 * dof) / dof)) {
        throw std::length_error("Rollout derivative dimensions overflow");
    }
    const auto control_count = steps * dof;
    return {dof, control_count, 2 * dof + control_count,
            3 * dof + control_count, 9 * dof + control_count};
}

template <typename Scalar>
struct ObjectiveInputs {
    detail::State<Scalar> state;
    std::vector<std::vector<Scalar>> controls;
    detail::Parameters<Scalar> physical;
};

template <typename Scalar, typename Seed>
ObjectiveInputs<Scalar> objective_inputs(
    const ArmModel& model, const ArmState& initial_state,
    const TorqueSequence& controls, const VariableLayout& layout, const Seed& seed) {
    ObjectiveInputs<Scalar> result;
    result.physical = detail::parameters<Scalar>(model);
    result.state.positions.reserve(layout.dof);
    result.state.velocities.reserve(layout.dof);
    for (std::size_t joint_index = 0; joint_index < layout.dof; ++joint_index) {
        result.state.positions.push_back(seed(initial_state.joint_positions[joint_index], joint_index));
        result.state.velocities.push_back(
            seed(initial_state.joint_velocities[joint_index], layout.dof + joint_index));
        result.physical.masses[joint_index] =
            seed(model.joints()[joint_index].mass, layout.masses + joint_index);
        for (std::size_t entry_index = 0; entry_index < inertia_entries.size(); ++entry_index) {
            for (const auto matrix_index : inertia_entries[entry_index]) {
                result.physical.inertias[joint_index][matrix_index] = seed(
                    model.joints()[joint_index].inertia_com.values[matrix_index],
                    layout.inertias + joint_index * inertia_entries.size() + entry_index);
            }
        }
    }
    result.controls.resize(controls.size());
    for (std::size_t step_index = 0; step_index < controls.size(); ++step_index) {
        result.controls[step_index].reserve(layout.dof);
        for (std::size_t joint_index = 0; joint_index < layout.dof; ++joint_index) {
            result.controls[step_index].push_back(seed(
                controls[step_index][joint_index],
                2 * layout.dof + step_index * layout.dof + joint_index));
        }
    }
    return result;
}

template <typename Scalar>
Scalar position_loss(
    const ArmModel& model, const detail::State<Scalar>& state,
    const RolloutObjective& objective, const std::vector<double>& target) {
    if (objective.cartesian) {
        const auto point = detail::point_position(
            model, state.positions, objective.link, objective.local_point);
        const detail::Vector3<Scalar> error{
            point.x - Scalar(target[0]), point.y - Scalar(target[1]), point.z - Scalar(target[2])};
        return Scalar(0.5) * detail::dot(error, error);
    }
    Scalar loss(0.0);
    for (std::size_t joint_index = 0; joint_index < state.positions.size(); ++joint_index) {
        const Scalar error = state.positions[joint_index] - Scalar(target[joint_index]);
        loss = loss + Scalar(0.5) * error * error;
    }
    return loss;
}

template <typename Scalar>
Scalar scalar_objective(
    const DynamicsSolver& dynamics_solver, ObjectiveInputs<Scalar> inputs,
    double timestep, const RolloutObjective& objective) {
    Scalar loss(0.0);
    const auto& model = dynamics_solver.model();
    for (std::size_t step_index = 0; step_index < inputs.controls.size(); ++step_index) {
        const auto& controls = inputs.controls[step_index];
        inputs.state = detail::rk4(
            model, dynamics_solver.gravity(), dynamics_solver.options(),
            inputs.physical, inputs.state, controls, timestep);
        if (objective.trajectory_weight != 0.0) {
            loss = loss + Scalar(objective.trajectory_weight) * position_loss(
                model, inputs.state, objective, objective.trajectory_targets[step_index]);
        }
        if (objective.control_weight != 0.0) {
            for (const auto& control : controls) {
                loss = loss + Scalar(0.5 * objective.control_weight) * control * control;
            }
        }
    }
    if (objective.terminal_weight != 0.0) {
        const auto target = objective.cartesian ?
            std::vector<double>{objective.target_point.x, objective.target_point.y, objective.target_point.z} :
            objective.target_positions;
        loss = loss + Scalar(objective.terminal_weight) *
            position_loss(model, inputs.state, objective, target);
    }
    if (!detail::finite(loss)) {
        throw std::runtime_error("Rollout objective and derivatives must be finite");
    }
    return loss;
}

TorqueSequence control_gradient(
    const Dual& loss, std::size_t steps, std::size_t dof, std::size_t offset) {
    TorqueSequence result(steps, std::vector<double>(dof, 0.0));
    for (std::size_t step_index = 0; step_index < steps; ++step_index) {
        for (std::size_t joint_index = 0; joint_index < dof; ++joint_index) {
            result[step_index][joint_index] =
                derivative_at(loss, offset + step_index * dof + joint_index);
        }
    }
    return result;
}

void validate_direction(
    const RolloutGradient& direction, std::size_t steps, std::size_t dof) {
    detail::validate_vector(direction.initial_state.joint_positions, dof);
    detail::validate_vector(direction.initial_state.joint_velocities, dof);
    detail::validate_vector(direction.masses, dof);
    if (direction.controls.size() != steps || direction.inertias.size() != dof) {
        throw std::invalid_argument("Direction shapes must match rollout variables");
    }
    for (const auto& controls : direction.controls) {
        detail::validate_vector(controls, dof);
    }
    for (const auto& inertia : direction.inertias) {
        for (const auto entry : inertia) {
            if (!std::isfinite(entry)) {
                throw std::invalid_argument("Direction inertia entries must be finite");
            }
        }
    }
}

}  // namespace

void RolloutSolver::validate_objective(const RolloutObjective& objective, std::size_t steps) const {
    for (const auto weight : {objective.terminal_weight, objective.trajectory_weight, objective.control_weight}) {
        if (!std::isfinite(weight) || weight < 0.0) {
            throw std::invalid_argument("Objective weights must be finite and nonnegative");
        }
    }
    const auto dof = dynamics_solver_.model().degrees_of_freedom();
    if (!objective.cartesian || !objective.target_positions.empty()) {
        detail::validate_vector(objective.target_positions, dof);
    }
    if (!detail::finite(objective.local_point) || !detail::finite(objective.target_point) ||
        (objective.cartesian && objective.link >= dof)) {
        throw std::invalid_argument("Objective requires finite points and a valid Cartesian link");
    }
    if ((!objective.trajectory_targets.empty() || objective.trajectory_weight != 0.0) &&
        objective.trajectory_targets.size() != steps) {
        throw std::invalid_argument("Trajectory targets must match the number of rollout steps");
    }
    for (const auto& target : objective.trajectory_targets) {
        detail::validate_vector(target, objective.cartesian ? 3 : dof);
    }
}

double RolloutSolver::evaluate_objective(
    const ArmState& initial_state, const TorqueSequence& controls,
    double dt, const RolloutObjective& objective) const {
    validate_rollout_inputs(initial_state, controls, dt);
    validate_objective(objective, controls.size());
    const auto& model = dynamics_solver_.model();
    const auto layout = variable_layout(model, controls.size());
    return scalar_objective(dynamics_solver_, objective_inputs<double>(
        model, initial_state, controls, layout,
        [](double value, std::size_t) { return value; }), dt, objective);
}

RolloutGradient RolloutSolver::automatic_gradient(
    const ArmState& initial_state, const TorqueSequence& controls,
    double dt, const RolloutObjective& objective) const {
    validate_rollout_inputs(initial_state, controls, dt);
    validate_objective(objective, controls.size());
    const auto& model = dynamics_solver_.model();
    const auto layout = variable_layout(model, controls.size());
    const auto loss = scalar_objective(dynamics_solver_, objective_inputs<Dual>(
        model, initial_state, controls, layout,
        [&layout](double value, std::size_t index) {
            return Dual::variable(value, layout.dimensions, index);
        }), dt, objective);
    RolloutGradient result{};
    result.loss = loss.value;
    result.initial_state.joint_positions.resize(layout.dof);
    result.initial_state.joint_velocities.resize(layout.dof);
    result.controls = control_gradient(loss, controls.size(), layout.dof, 2 * layout.dof);
    result.masses.resize(layout.dof);
    result.inertias.resize(layout.dof);
    for (std::size_t joint_index = 0; joint_index < layout.dof; ++joint_index) {
        result.initial_state.joint_positions[joint_index] = derivative_at(loss, joint_index);
        result.initial_state.joint_velocities[joint_index] = derivative_at(loss, layout.dof + joint_index);
        result.masses[joint_index] = derivative_at(loss, layout.masses + joint_index);
        for (std::size_t entry_index = 0; entry_index < inertia_entries.size(); ++entry_index) {
            result.inertias[joint_index][entry_index] = derivative_at(
                loss, layout.inertias + joint_index * inertia_entries.size() + entry_index);
        }
    }
    return result;
}

double RolloutSolver::automatic_directional_derivative(
    const ArmState& initial_state, const TorqueSequence& controls,
    double dt, const RolloutObjective& objective, const RolloutGradient& direction) const {
    validate_rollout_inputs(initial_state, controls, dt);
    validate_objective(objective, controls.size());
    const auto& model = dynamics_solver_.model();
    const auto layout = variable_layout(model, controls.size());
    validate_direction(direction, controls.size(), layout.dof);
    const auto loss = scalar_objective(dynamics_solver_, objective_inputs<Dual>(
        model, initial_state, controls, layout,
        [&layout, &direction](double value, std::size_t index) {
            double tangent = 0.0;
            if (index < layout.dof) {
                tangent = direction.initial_state.joint_positions[index];
            } else if (index < 2 * layout.dof) {
                tangent = direction.initial_state.joint_velocities[index - layout.dof];
            } else if (index < layout.masses) {
                const auto control_index = index - 2 * layout.dof;
                tangent = direction.controls[control_index / layout.dof][control_index % layout.dof];
            } else if (index < layout.inertias) {
                tangent = direction.masses[index - layout.masses];
            } else {
                const auto inertia_index = index - layout.inertias;
                tangent = direction.inertias[inertia_index / inertia_entries.size()]
                                           [inertia_index % inertia_entries.size()];
            }
            Dual result(value, 1);
            result.derivative[0] = tangent;
            return result;
        }), dt, objective);
    return derivative_at(loss, 0);
}

TorqueSequence RolloutSolver::automatic_control_gradient(
    const ArmState& initial_state, const TorqueSequence& torque_sequence,
    double timestep, const std::vector<double>& target_positions) const {
    validate_rollout_inputs(initial_state, torque_sequence, timestep);
    RolloutObjective objective;
    objective.target_positions = target_positions;
    validate_objective(objective, torque_sequence.size());
    const auto& model = dynamics_solver_.model();
    const auto layout = variable_layout(model, torque_sequence.size());
    const auto loss = scalar_objective(dynamics_solver_, objective_inputs<Dual>(
        model, initial_state, torque_sequence, layout,
        [&layout](double value, std::size_t index) {
            if (index >= 2 * layout.dof && index < layout.masses) {
                return Dual::variable(value, layout.control_count, index - 2 * layout.dof);
            }
            return Dual(value);
        }), timestep, objective);
    return control_gradient(loss, torque_sequence.size(), layout.dof, 0);
}

}  // namespace arm