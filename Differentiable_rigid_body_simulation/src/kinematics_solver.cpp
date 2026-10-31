#include "arm/kinematics_solver.hpp"
#include "arm/simulation_core.hpp"

#include <utility>

namespace arm {

KinematicsSolver::KinematicsSolver(const ArmModel& model)
    : model_(model) {}

KinematicsResult KinematicsSolver::solve(
    const std::vector<double>& joint_positions) const {
    const std::size_t size = model_.degrees_of_freedom();
    const std::vector<double> zero(size, 0.0);
    const auto links = detail::kinematics(model_, joint_positions, zero, zero);
    const auto& endpoint = links.back();
    std::vector<Vec3> columns(size);
    int ancestor = static_cast<int>(size - 1);
    while (ancestor >= 0) {
        const auto index = static_cast<std::size_t>(ancestor);
        const auto& link = links[index];
        const auto column = model_.joints()[index].type == Prismatic ? link.axis :
            detail::cross(link.axis, endpoint.endpoint - link.origin);
        columns[index] = {column.x, column.y, column.z};
        ancestor = model_.joints()[index].parent;
    }
    return {{{endpoint.endpoint.x, endpoint.endpoint.y, endpoint.endpoint.z},
             {endpoint.orientation.values}}, std::move(columns)};
}

}