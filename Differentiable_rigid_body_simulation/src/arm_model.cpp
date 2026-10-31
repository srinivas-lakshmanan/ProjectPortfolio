#include "arm/arm_model.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace arm {

ArmModel::ArmModel(std::vector<RevoluteJoint> joints)
    : joints_(std::move(joints)) {
    if (joints_.empty()) {
        throw std::invalid_argument("An arm must contain at least one joint");
    }

    for (std::size_t index = 0; index < joints_.size(); ++index) {
        RevoluteJoint& joint = joints_[index];
        if (joint.parent == -2) {
            joint.parent = index == 0 ? -1 : static_cast<int>(index - 1);
        }
        if (joint.parent < -1 || (joint.parent >= 0 &&
            static_cast<std::size_t>(joint.parent) >= index)) {
            throw std::invalid_argument("Joint parent must be root or an earlier joint");
        }
        if (joint.type != Revolute && joint.type != Prismatic) {
            throw std::invalid_argument("Invalid joint type");
        }
        if (!std::isfinite(joint.joint_offset.x) ||
            !std::isfinite(joint.joint_offset.y) || !std::isfinite(joint.joint_offset.z)) {
            throw std::invalid_argument("Joint offset must be finite");
        }
        for (double value : joint.joint_rotation.values) {
            if (!std::isfinite(value)) {
                throw std::invalid_argument("Joint rest rotation must be finite");
            }
        }
        const Mat3 orthogonality = joint.joint_rotation.transpose() * joint.joint_rotation;
        for (std::size_t entry = 0; entry < 9; ++entry) {
            if (std::abs(orthogonality.values[entry] - (entry % 4 == 0 ? 1.0 : 0.0)) > 1e-10) {
                throw std::invalid_argument("Joint rest rotation must be orthogonal");
            }
        }
        const auto& rotation = joint.joint_rotation.values;
        const double rotation_determinant =
            rotation[0] * (rotation[4] * rotation[8] - rotation[5] * rotation[7]) -
            rotation[1] * (rotation[3] * rotation[8] - rotation[5] * rotation[6]) +
            rotation[2] * (rotation[3] * rotation[7] - rotation[4] * rotation[6]);
        if (std::abs(rotation_determinant - 1.0) > 1e-10) {
            throw std::invalid_argument("Joint rest rotation must have determinant one");
        }
        if (!std::isfinite(joint.axis.x) || !std::isfinite(joint.axis.y) ||
            !std::isfinite(joint.axis.z) || !std::isfinite(norm(joint.axis))) {
            throw std::invalid_argument("Joint axis must have finite components and norm");
        }
        joint.axis = normalized(joint.axis);
        if (!std::isfinite(joint.link_offset.x) || !std::isfinite(joint.link_offset.y) ||
            !std::isfinite(joint.link_offset.z)) {
            throw std::invalid_argument("Link offset must be finite");
        }
        if (!std::isfinite(joint.mass) || joint.mass <= 0.0) {
            throw std::invalid_argument("Link mass must be finite and positive");
        }
        if (!std::isfinite(joint.com_offset.x) || !std::isfinite(joint.com_offset.y) ||
            !std::isfinite(joint.com_offset.z)) {
            throw std::invalid_argument("Link center-of-mass offset must be finite");
        }

        const auto& inertia = joint.inertia_com.values;
        for (double value : inertia) {
            if (!std::isfinite(value)) {
                throw std::invalid_argument("Link inertia must be finite");
            }
        }
        if (std::abs(inertia[1] - inertia[3]) > 1e-10 ||
            std::abs(inertia[2] - inertia[6]) > 1e-10 ||
            std::abs(inertia[5] - inertia[7]) > 1e-10) {
            throw std::invalid_argument("Link inertia tensor must be symmetric");
        }

        const double minor_2 = inertia[0] * inertia[4] - inertia[1] * inertia[3];
        const double determinant =
            inertia[0] * (inertia[4] * inertia[8] - inertia[5] * inertia[7]) -
            inertia[1] * (inertia[3] * inertia[8] - inertia[5] * inertia[6]) +
            inertia[2] * (inertia[3] * inertia[7] - inertia[4] * inertia[6]);
        if (inertia[0] <= 0.0 || minor_2 <= 0.0 || determinant <= 0.0) {
            throw std::invalid_argument("Link inertia tensor must be positive definite");
        }
    }
}

const std::vector<RevoluteJoint>& ArmModel::joints() const {
    return joints_;
}

std::size_t ArmModel::degrees_of_freedom() const {
    return joints_.size();
}

}  // namespace arm