#pragma once

#include "arm/math.hpp"

#include <string>
#include <vector>

namespace arm {

enum JointType { Revolute, Prismatic };

struct RevoluteJoint {
    std::string name;
    Vec3 axis;
    Vec3 link_offset;
    double mass = 0.0;
    Vec3 com_offset;
    Mat3 inertia_com = Mat3::zero();
    JointType type = Revolute;
    int parent = -2;
    Vec3 joint_offset{};
    Mat3 joint_rotation{};
};

class ArmModel {
public:
    explicit ArmModel(std::vector<RevoluteJoint> joints);

    const std::vector<RevoluteJoint>& joints() const;
    std::size_t degrees_of_freedom() const;

private:
    std::vector<RevoluteJoint> joints_;
};

}  // namespace arm