#pragma once

#include "arm/arm_model.hpp"

#include <vector>

namespace arm {

struct CoordinateConstraint {
    std::vector<double> coefficients;
    double target = 0.0;
    double frequency = 10.0;
    double damping_ratio = 1.0;
};

struct SmoothPlaneContact {
    std::size_t link = 0;
    Vec3 local_point;
    Vec3 normal{0.0, 0.0, 1.0};
    double plane_offset = 0.0;
    double stiffness = 1000.0;
    double smoothing = 0.001;
};

struct DynamicsOptions {
    std::vector<CoordinateConstraint> constraints;
    std::vector<SmoothPlaneContact> contacts;
};

struct ForwardDynamicsResult {
    std::vector<double> joint_accelerations;
    std::vector<std::vector<double>> mass_matrix;
    std::vector<double> bias_forces;
    std::vector<double> constraint_forces;
};

class DynamicsSolver {
public:
    explicit DynamicsSolver(const ArmModel& model, Vec3 gravity = {0.0, 0.0, -9.81},
                            DynamicsOptions options = {});

    std::vector<double> inverse_dynamics(
        const std::vector<double>& joint_positions,
        const std::vector<double>& joint_velocities,
        const std::vector<double>& joint_accelerations) const;

    ForwardDynamicsResult forward_dynamics(
        const std::vector<double>& joint_positions,
        const std::vector<double>& joint_velocities,
        const std::vector<double>& joint_torques) const;

    const ArmModel& model() const;
    Vec3 gravity() const;
    const DynamicsOptions& options() const;

private:
    const ArmModel& model_;
    Vec3 gravity_;
    DynamicsOptions options_;
};

}  // namespace arm