#pragma once

#include "arm/arm_model.hpp"

#include <vector>

namespace arm {

struct Pose {
    Vec3 position;
    Mat3 orientation;
};

struct KinematicsResult {
    Pose end_effector;
    std::vector<Vec3> position_jacobian_columns;
};

class KinematicsSolver {
public:
    explicit KinematicsSolver(const ArmModel& model);

    KinematicsResult solve(const std::vector<double>& joint_positions) const;

private:
    const ArmModel& model_;
};

}  // namespace arm