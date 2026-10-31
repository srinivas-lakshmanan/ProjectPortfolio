#pragma once

#include "arm/autodiff.hpp"
#include "arm/dynamics_solver.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace arm::detail {

inline double scalar_value(double value) { return value; }
inline double scalar_value(const Dual& value) { return value.value; }
inline double scalar_sin(double value) { return std::sin(value); }
inline Dual scalar_sin(const Dual& value) { return arm::sin(value); }
inline double scalar_cos(double value) { return std::cos(value); }
inline Dual scalar_cos(const Dual& value) { return arm::cos(value); }
inline double scalar_sqrt(double value) { return std::sqrt(value); }
inline Dual scalar_sqrt(const Dual& value) {
    Dual result(std::sqrt(value.value), value.derivative.size());
    for (std::size_t index = 0; index < result.derivative.size(); ++index) {
        result.derivative[index] = value.derivative[index] / (2.0 * result.value);
    }
    return result;
}

inline bool finite(double value) { return std::isfinite(value); }
inline bool finite(const Dual& value) {
    return std::isfinite(value.value) &&
        std::all_of(value.derivative.begin(), value.derivative.end(),
                    [](double entry) { return std::isfinite(entry); });
}
inline bool finite(const Vec3& value) {
    return finite(value.x) && finite(value.y) && finite(value.z);
}

template <typename Scalar>
struct Vector3 {
    Scalar x = Scalar(0.0);
    Scalar y = Scalar(0.0);
    Scalar z = Scalar(0.0);

    Vector3 operator+(const Vector3& other) const {
        return {x + other.x, y + other.y, z + other.z};
    }
    Vector3 operator-(const Vector3& other) const {
        return {x - other.x, y - other.y, z - other.z};
    }
    Vector3 operator*(const Scalar& scale) const {
        return {x * scale, y * scale, z * scale};
    }
};

template <typename Scalar>
Vector3<Scalar> vector3(const Vec3& value) {
    return {Scalar(value.x), Scalar(value.y), Scalar(value.z)};
}

template <typename Scalar>
Scalar dot(const Vector3<Scalar>& lhs, const Vector3<Scalar>& rhs) {
    return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
}

template <typename Scalar>
Vector3<Scalar> cross(const Vector3<Scalar>& lhs, const Vector3<Scalar>& rhs) {
    return {lhs.y * rhs.z - lhs.z * rhs.y,
            lhs.z * rhs.x - lhs.x * rhs.z,
            lhs.x * rhs.y - lhs.y * rhs.x};
}

template <typename Scalar>
struct Matrix3 {
    std::array<Scalar, 9> values{
        Scalar(1.0), Scalar(0.0), Scalar(0.0),
        Scalar(0.0), Scalar(1.0), Scalar(0.0),
        Scalar(0.0), Scalar(0.0), Scalar(1.0)};

    Vector3<Scalar> operator*(const Vector3<Scalar>& value) const {
        return {values[0] * value.x + values[1] * value.y + values[2] * value.z,
                values[3] * value.x + values[4] * value.y + values[5] * value.z,
                values[6] * value.x + values[7] * value.y + values[8] * value.z};
    }
    Matrix3 operator*(const Matrix3& other) const {
        Matrix3 result;
        for (std::size_t row = 0; row < 3; ++row) {
            for (std::size_t column = 0; column < 3; ++column) {
                Scalar entry(0.0);
                for (std::size_t inner = 0; inner < 3; ++inner) {
                    entry = entry + values[row * 3 + inner] * other.values[inner * 3 + column];
                }
                result.values[row * 3 + column] = entry;
            }
        }
        return result;
    }
    Matrix3 transpose() const {
        return {{values[0], values[3], values[6], values[1], values[4], values[7],
                 values[2], values[5], values[8]}};
    }
};

template <typename Scalar>
Matrix3<Scalar> matrix3(const Mat3& value) {
    Matrix3<Scalar> result;
    for (std::size_t index = 0; index < 9; ++index) {
        result.values[index] = Scalar(value.values[index]);
    }
    return result;
}

template <typename Scalar>
Matrix3<Scalar> rotation(const Vec3& axis, const Scalar& angle) {
    const Scalar cosine = scalar_cos(angle);
    const Scalar sine = scalar_sin(angle);
    const Scalar remainder = Scalar(1.0) - cosine;
    return {{cosine + Scalar(axis.x * axis.x) * remainder,
             Scalar(axis.x * axis.y) * remainder - Scalar(axis.z) * sine,
             Scalar(axis.x * axis.z) * remainder + Scalar(axis.y) * sine,
             Scalar(axis.y * axis.x) * remainder + Scalar(axis.z) * sine,
             cosine + Scalar(axis.y * axis.y) * remainder,
             Scalar(axis.y * axis.z) * remainder - Scalar(axis.x) * sine,
             Scalar(axis.z * axis.x) * remainder - Scalar(axis.y) * sine,
             Scalar(axis.z * axis.y) * remainder + Scalar(axis.x) * sine,
             cosine + Scalar(axis.z * axis.z) * remainder}};
}

template <typename Scalar>
struct State {
    std::vector<Scalar> positions;
    std::vector<Scalar> velocities;
};

template <typename Scalar>
struct Parameters {
    std::vector<Scalar> masses;
    std::vector<std::array<Scalar, 9>> inertias;
};

template <typename Scalar>
Parameters<Scalar> parameters(const ArmModel& model) {
    Parameters<Scalar> result;
    for (const auto& joint : model.joints()) {
        result.masses.push_back(Scalar(joint.mass));
        result.inertias.push_back(matrix3<Scalar>(joint.inertia_com).values);
    }
    return result;
}

template <typename Scalar>
void validate_vector(const std::vector<Scalar>& values, std::size_t size) {
    if (values.size() != size) {
        throw std::invalid_argument("Joint vector size must match arm degrees of freedom");
    }
    for (const auto& value : values) {
        if (!finite(value)) {
            throw std::invalid_argument("Joint vector values and derivatives must be finite");
        }
    }
}

inline void validate_options(const ArmModel& model, Vec3 gravity, const DynamicsOptions& options) {
    if (!finite(gravity)) {
        throw std::invalid_argument("Gravity must be finite");
    }
    for (const auto& constraint : options.constraints) {
        validate_vector(constraint.coefficients, model.degrees_of_freedom());
        if (!finite(constraint.target) || !finite(constraint.frequency) ||
            !finite(constraint.damping_ratio) || constraint.frequency < 0.0 ||
            constraint.damping_ratio < 0.0) {
            throw std::invalid_argument("Invalid coordinate constraint stabilization parameters");
        }
    }
    for (const auto& contact : options.contacts) {
        if (contact.link >= model.degrees_of_freedom() || !finite(contact.local_point) ||
            !finite(contact.normal) || !finite(contact.plane_offset) ||
            !finite(contact.stiffness) || contact.stiffness < 0.0 ||
            !finite(contact.smoothing) || contact.smoothing <= 0.0 ||
            std::abs(norm(contact.normal) - 1.0) > 1e-10) {
            throw std::invalid_argument("Contact requires a valid link, unit normal and finite parameters");
        }
    }
}

template <typename Scalar>
struct LinkState {
    Vector3<Scalar> origin;
    Vector3<Scalar> axis;
    Matrix3<Scalar> orientation;
    Vector3<Scalar> center;
    Vector3<Scalar> endpoint;
    Vector3<Scalar> angular_velocity;
    Vector3<Scalar> angular_acceleration;
    Vector3<Scalar> origin_velocity;
    Vector3<Scalar> origin_acceleration;
    Vector3<Scalar> center_velocity;
    Vector3<Scalar> center_acceleration;
    Vector3<Scalar> endpoint_velocity;
    Vector3<Scalar> endpoint_acceleration;
};

template <typename Scalar>
std::vector<LinkState<Scalar>> kinematics(
    const ArmModel& model, const std::vector<Scalar>& positions,
    const std::vector<Scalar>& velocities, const std::vector<Scalar>& accelerations) {
    const std::size_t size = model.degrees_of_freedom();
    validate_vector(positions, size);
    validate_vector(velocities, size);
    validate_vector(accelerations, size);
    std::vector<LinkState<Scalar>> links(size);
    for (std::size_t index = 0; index < size; ++index) {
        const auto& joint = model.joints()[index];
        LinkState<Scalar> parent;
        if (joint.parent >= 0) {
            parent = links[static_cast<std::size_t>(joint.parent)];
        }
        auto& link = links[index];
        const auto offset = parent.orientation * vector3<Scalar>(joint.joint_offset);
        const auto rest = parent.orientation * matrix3<Scalar>(joint.joint_rotation);
        link.axis = rest * vector3<Scalar>(joint.axis);
        link.origin = parent.endpoint + offset;
        link.origin_velocity = parent.endpoint_velocity + cross(parent.angular_velocity, offset);
        link.origin_acceleration = parent.endpoint_acceleration +
            cross(parent.angular_acceleration, offset) +
            cross(parent.angular_velocity, cross(parent.angular_velocity, offset));
        link.angular_velocity = parent.angular_velocity;
        link.angular_acceleration = parent.angular_acceleration;
        link.orientation = rest;
        if (joint.type == Revolute) {
            link.orientation = rest * rotation(joint.axis, positions[index]);
            link.angular_velocity = link.angular_velocity + link.axis * velocities[index];
            link.angular_acceleration = link.angular_acceleration + link.axis * accelerations[index] +
                cross(parent.angular_velocity, link.axis * velocities[index]);
        } else {
            const auto displacement = link.axis * positions[index];
            link.origin = link.origin + displacement;
            link.origin_velocity = link.origin_velocity + cross(parent.angular_velocity, displacement) +
                link.axis * velocities[index];
            link.origin_acceleration = link.origin_acceleration +
                cross(parent.angular_acceleration, displacement) +
                cross(parent.angular_velocity, cross(parent.angular_velocity, displacement)) +
                cross(parent.angular_velocity, link.axis * velocities[index]) * Scalar(2.0) +
                link.axis * accelerations[index];
        }
        const auto center_offset = link.orientation * vector3<Scalar>(joint.com_offset);
        const auto endpoint_offset = link.orientation * vector3<Scalar>(joint.link_offset);
        link.center = link.origin + center_offset;
        link.endpoint = link.origin + endpoint_offset;
        link.center_velocity = link.origin_velocity + cross(link.angular_velocity, center_offset);
        link.endpoint_velocity = link.origin_velocity + cross(link.angular_velocity, endpoint_offset);
        link.center_acceleration = link.origin_acceleration +
            cross(link.angular_acceleration, center_offset) +
            cross(link.angular_velocity, cross(link.angular_velocity, center_offset));
        link.endpoint_acceleration = link.origin_acceleration +
            cross(link.angular_acceleration, endpoint_offset) +
            cross(link.angular_velocity, cross(link.angular_velocity, endpoint_offset));
    }
    return links;
}

template <typename Scalar>
void project_wrench(const ArmModel& model, const std::vector<LinkState<Scalar>>& links,
                    std::size_t link_index, const Vector3<Scalar>& point,
                    const Vector3<Scalar>& force, const Vector3<Scalar>& moment,
                    std::vector<Scalar>& generalized) {
    int ancestor = static_cast<int>(link_index);
    while (ancestor >= 0) {
        const auto index = static_cast<std::size_t>(ancestor);
        const auto& joint = model.joints()[index];
        const auto& link = links[index];
        generalized[index] = generalized[index] +
            (joint.type == Prismatic ? dot(link.axis, force) :
             dot(link.axis, moment + cross(point - link.origin, force)));
        ancestor = joint.parent;
    }
}

template <typename Scalar>
std::vector<Scalar> inverse_dynamics(
    const ArmModel& model, Vec3 gravity, const Parameters<Scalar>& physical,
    const std::vector<Scalar>& positions, const std::vector<Scalar>& velocities,
    const std::vector<Scalar>& accelerations) {
    const std::size_t size = model.degrees_of_freedom();
    validate_vector(physical.masses, size);
    if (physical.inertias.size() != size || !finite(gravity)) {
        throw std::invalid_argument("Invalid inertial parameters or gravity");
    }
    const auto links = kinematics(model, positions, velocities, accelerations);
    std::vector<Scalar> result(size, Scalar(0.0));
    for (std::size_t index = 0; index < size; ++index) {
        if (scalar_value(physical.masses[index]) <= 0.0) {
            throw std::invalid_argument("Mass must be positive");
        }
        for (const auto& value : physical.inertias[index]) {
            if (!finite(value)) {
                throw std::invalid_argument("Inertia values and derivatives must be finite");
            }
        }
        const auto& link = links[index];
        const Matrix3<Scalar> inertia{physical.inertias[index]};
        const auto inertia_world = link.orientation * inertia * link.orientation.transpose();
        const auto force = (link.center_acceleration - vector3<Scalar>(gravity)) * physical.masses[index];
        const auto moment = inertia_world * link.angular_acceleration +
            cross(link.angular_velocity, inertia_world * link.angular_velocity);
        project_wrench(model, links, index, link.center, force, moment, result);
    }
    return result;
}

template <typename Scalar>
std::vector<Scalar> solve_linear_system(std::vector<std::vector<Scalar>> matrix,
                                        std::vector<Scalar> rhs) {
    const std::size_t size = rhs.size();
    validate_vector(rhs, size);
    if (matrix.size() != size) {
        throw std::invalid_argument("Linear system must be square");
    }
    std::vector<double> column_scales(size, 0.0);
    for (std::size_t row = 0; row < size; ++row) {
        validate_vector(matrix[row], size);
        double scale = 0.0;
        for (const auto& entry : matrix[row]) {
            scale = std::max(scale, std::abs(scalar_value(entry)));
        }
        if (scale == 0.0) {
            throw std::runtime_error("Dynamics system is rank deficient");
        }
        rhs[row] = rhs[row] / Scalar(scale);
        for (std::size_t column = 0; column < size; ++column) {
            matrix[row][column] = matrix[row][column] / Scalar(scale);
            column_scales[column] = std::max(column_scales[column],
                                             std::abs(scalar_value(matrix[row][column])));
        }
    }
    for (std::size_t column = 0; column < size; ++column) {
        if (column_scales[column] == 0.0) {
            throw std::runtime_error("Dynamics system is rank deficient");
        }
        for (std::size_t row = 0; row < size; ++row) {
            matrix[row][column] = matrix[row][column] / Scalar(column_scales[column]);
        }
    }
    const double threshold = std::numeric_limits<double>::epsilon() *
        64.0 * static_cast<double>(std::max(size, std::size_t(1)));
    for (std::size_t pivot = 0; pivot < size; ++pivot) {
        std::size_t best = pivot;
        for (std::size_t row = pivot + 1; row < size; ++row) {
            if (std::abs(scalar_value(matrix[row][pivot])) >
                std::abs(scalar_value(matrix[best][pivot]))) {
                best = row;
            }
        }
        if (std::abs(scalar_value(matrix[best][pivot])) <= threshold) {
            throw std::runtime_error("Dynamics system is rank deficient");
        }
        std::swap(matrix[pivot], matrix[best]);
        std::swap(rhs[pivot], rhs[best]);
        for (std::size_t row = pivot + 1; row < size; ++row) {
            const Scalar factor = matrix[row][pivot] / matrix[pivot][pivot];
            for (std::size_t column = pivot + 1; column < size; ++column) {
                matrix[row][column] = matrix[row][column] - factor * matrix[pivot][column];
            }
            rhs[row] = rhs[row] - factor * rhs[pivot];
        }
    }
    std::vector<Scalar> result(size, Scalar(0.0));
    for (std::size_t remaining = size; remaining > 0; --remaining) {
        const std::size_t row = remaining - 1;
        Scalar value = rhs[row];
        for (std::size_t column = row + 1; column < size; ++column) {
            value = value - matrix[row][column] * result[column];
        }
        result[row] = value / matrix[row][row];
    }
    for (std::size_t index = 0; index < size; ++index) {
        result[index] = result[index] / Scalar(column_scales[index]);
    }
    validate_vector(result, size);
    return result;
}

template <typename Scalar>
struct ForwardResult {
    std::vector<Scalar> accelerations;
    std::vector<std::vector<Scalar>> mass_matrix;
    std::vector<Scalar> bias;
    std::vector<Scalar> constraint_forces;
};

template <typename Scalar>
ForwardResult<Scalar> forward_dynamics(
    const ArmModel& model, Vec3 gravity, const DynamicsOptions& options,
    const Parameters<Scalar>& physical, const std::vector<Scalar>& positions,
    const std::vector<Scalar>& velocities, const std::vector<Scalar>& torques) {
    validate_options(model, gravity, options);
    const std::size_t size = model.degrees_of_freedom();
    validate_vector(torques, size);
    const std::vector<Scalar> zero(size, Scalar(0.0));
    ForwardResult<Scalar> result;
    result.bias = inverse_dynamics(model, gravity, physical, positions, velocities, zero);
    result.mass_matrix.assign(size, std::vector<Scalar>(size, Scalar(0.0)));
    const auto inertial_zero = inverse_dynamics(model, Vec3{}, physical, positions, zero, zero);
    for (std::size_t column = 0; column < size; ++column) {
        auto unit = zero;
        unit[column] = Scalar(1.0);
        const auto effort = inverse_dynamics(model, Vec3{}, physical, positions, zero, unit);
        for (std::size_t row = 0; row < size; ++row) {
            result.mass_matrix[row][column] = effort[row] - inertial_zero[row];
        }
    }
    auto contact_forces = zero;
    if (!options.contacts.empty()) {
        const auto links = kinematics(model, positions, velocities, zero);
        for (const auto& contact : options.contacts) {
            const auto& link = links[contact.link];
            const auto point = link.origin + link.orientation * vector3<Scalar>(contact.local_point);
            const auto normal = vector3<Scalar>(contact.normal);
            const Scalar gap = dot(normal, point) - Scalar(contact.plane_offset);
            const Scalar root = scalar_sqrt(gap * gap + Scalar(contact.smoothing) * Scalar(contact.smoothing));
            const Scalar penetration = scalar_value(gap) > 0.0 ?
                Scalar(contact.smoothing) * Scalar(contact.smoothing) / (root + gap) : root - gap;
            const auto force = normal * (Scalar(0.5 * contact.stiffness) * penetration);
            project_wrench(model, links, contact.link, point, force, Vector3<Scalar>{}, contact_forces);
        }
    }
    const std::size_t total = size + options.constraints.size();
    std::vector<std::vector<Scalar>> system(total, std::vector<Scalar>(total, Scalar(0.0)));
    std::vector<Scalar> rhs(total, Scalar(0.0));
    for (std::size_t row = 0; row < size; ++row) {
        rhs[row] = torques[row] - result.bias[row] + contact_forces[row];
        for (std::size_t column = 0; column < size; ++column) {
            system[row][column] = result.mass_matrix[row][column];
        }
    }
    for (std::size_t index = 0; index < options.constraints.size(); ++index) {
        const auto& constraint = options.constraints[index];
        Scalar error = Scalar(0.0) - Scalar(constraint.target);
        Scalar speed(0.0);
        for (std::size_t coordinate = 0; coordinate < size; ++coordinate) {
            const Scalar coefficient(constraint.coefficients[coordinate]);
            system[size + index][coordinate] = coefficient;
            system[coordinate][size + index] = Scalar(0.0) - coefficient;
            error = error + coefficient * positions[coordinate];
            speed = speed + coefficient * velocities[coordinate];
        }
        rhs[size + index] = Scalar(-2.0 * constraint.damping_ratio) * Scalar(constraint.frequency) * speed -
            Scalar(constraint.frequency) * Scalar(constraint.frequency) * error;
    }
    auto solution = solve_linear_system(std::move(system), std::move(rhs));
    result.accelerations.assign(solution.begin(), solution.begin() + size);
    result.constraint_forces.assign(solution.begin() + size, solution.end());
    return result;
}

template <typename Scalar>
Vector3<Scalar> point_position(const ArmModel& model, const std::vector<Scalar>& positions,
                               std::size_t link, Vec3 point) {
    if (link >= model.degrees_of_freedom() || !finite(point)) {
        throw std::invalid_argument("Invalid link or local point");
    }
    const std::vector<Scalar> zero(model.degrees_of_freedom(), Scalar(0.0));
    const auto links = kinematics(model, positions, zero, zero);
    return links[link].origin + links[link].orientation * vector3<Scalar>(point);
}

template <typename Scalar>
State<Scalar> rk4(const ArmModel& model, Vec3 gravity, const DynamicsOptions& options,
                   const Parameters<Scalar>& physical, const State<Scalar>& state,
                   const std::vector<Scalar>& controls, double dt) {
    if (!finite(dt) || dt <= 0.0) {
        throw std::invalid_argument("Integration timestep must be finite and positive");
    }
    const std::size_t size = model.degrees_of_freedom();
    validate_vector(state.positions, size);
    validate_vector(state.velocities, size);
    const auto slope = [&](const State<Scalar>& input) {
        return State<Scalar>{input.velocities,
            forward_dynamics(model, gravity, options, physical, input.positions,
                             input.velocities, controls).accelerations};
    };
    const auto shifted = [&](const State<Scalar>& derivative, double scale) {
        State<Scalar> result = state;
        for (std::size_t index = 0; index < size; ++index) {
            result.positions[index] = result.positions[index] + derivative.positions[index] * Scalar(scale);
            result.velocities[index] = result.velocities[index] + derivative.velocities[index] * Scalar(scale);
        }
        return result;
    };
    const auto first = slope(state);
    const auto second = slope(shifted(first, dt * 0.5));
    const auto third = slope(shifted(second, dt * 0.5));
    const auto fourth = slope(shifted(third, dt));
    State<Scalar> result = state;
    for (std::size_t index = 0; index < size; ++index) {
        result.positions[index] = result.positions[index] + Scalar(dt / 6.0) *
            (first.positions[index] + Scalar(2.0) * second.positions[index] +
             Scalar(2.0) * third.positions[index] + fourth.positions[index]);
        result.velocities[index] = result.velocities[index] + Scalar(dt / 6.0) *
            (first.velocities[index] + Scalar(2.0) * second.velocities[index] +
             Scalar(2.0) * third.velocities[index] + fourth.velocities[index]);
    }
    validate_vector(result.positions, size);
    validate_vector(result.velocities, size);
    return result;
}

}