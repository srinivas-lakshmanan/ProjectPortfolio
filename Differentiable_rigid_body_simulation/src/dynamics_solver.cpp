#include "arm/dynamics_solver.hpp"
#include "arm/simulation_core.hpp"

#include <utility>

namespace arm {

DynamicsSolver::DynamicsSolver(const ArmModel& model, Vec3 gravity, DynamicsOptions options)
    : model_(model), gravity_(gravity), options_(std::move(options)) {
    detail::validate_options(model_, gravity_, options_);
}

std::vector<double> DynamicsSolver::inverse_dynamics(
    const std::vector<double>& joint_positions,
    const std::vector<double>& joint_velocities,
    const std::vector<double>& joint_accelerations) const {
    return detail::inverse_dynamics(model_, gravity_, detail::parameters<double>(model_),
                                    joint_positions, joint_velocities, joint_accelerations);
}

ForwardDynamicsResult DynamicsSolver::forward_dynamics(
    const std::vector<double>& joint_positions,
    const std::vector<double>& joint_velocities,
    const std::vector<double>& joint_torques) const {
    auto result = detail::forward_dynamics(model_, gravity_, options_, detail::parameters<double>(model_),
                                           joint_positions, joint_velocities, joint_torques);
    return {std::move(result.accelerations), std::move(result.mass_matrix),
            std::move(result.bias), std::move(result.constraint_forces)};
}

const ArmModel& DynamicsSolver::model() const {
    return model_;
}

Vec3 DynamicsSolver::gravity() const {
    return gravity_;
}

const DynamicsOptions& DynamicsSolver::options() const {
    return options_;
}

}