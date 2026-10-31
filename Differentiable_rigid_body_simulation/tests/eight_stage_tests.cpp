#include "arm/arm_model.hpp"
#include "arm/dynamics_solver.hpp"
#include "arm/rollout.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using Vector = std::vector<double>;
using Matrix = std::vector<Vector>;
constexpr std::array<std::array<std::size_t, 2>, 6> inertia_indices{{
    {{0, 0}}, {{4, 4}}, {{8, 8}}, {{1, 3}}, {{2, 6}}, {{5, 7}}
}};

void require(bool condition, const std::string& label) {
    if (!condition) {
        throw std::runtime_error(label);
    }
}

void near(double actual, double expected, double tolerance, const std::string& label) {
    require(std::isfinite(actual) && std::isfinite(expected) &&
            std::abs(actual - expected) <= tolerance * std::max(1.0, std::abs(expected)),
            label + " actual=" + std::to_string(actual) + " expected=" + std::to_string(expected));
}

template <typename Exception, typename Function>
void throws(Function function, const std::string& label) {
    bool caught = false;
    try {
        function();
    } catch (const Exception&) {
        caught = true;
    }
    require(caught, label);
}

arm::RevoluteJoint joint(const std::string& name, arm::Vec3 axis, double mass,
                         int parent, arm::JointType type = arm::Revolute) {
    arm::RevoluteJoint result;
    result.name = name;
    result.axis = axis;
    result.mass = mass;
    result.parent = parent;
    result.type = type;
    result.link_offset = {0.7, 0.0, 0.0};
    result.com_offset = {0.3, 0.04, -0.02};
    result.inertia_com = {{0.09, 0.006, -0.004, 0.006, 0.12, 0.005, -0.004, 0.005, 0.15}};
    return result;
}

std::vector<arm::RevoluteJoint> planar_joints() {
    auto first = joint("first", {0.0, 0.0, 1.0}, 1.7, -1);
    auto second = joint("second", {0.0, 0.0, 1.0}, 1.1, 0);
    first.link_offset = {0.8, 0.0, 0.0};
    first.com_offset = {0.35, 0.0, 0.0};
    second.com_offset = {0.28, 0.0, 0.0};
    return {first, second};
}

std::vector<arm::RevoluteJoint> slider_joints(std::size_t count) {
    std::vector<arm::RevoluteJoint> result;
    for (std::size_t index = 0; index < count; ++index) {
        result.push_back(joint("slider" + std::to_string(index), {0.0, 0.0, 1.0},
                               1.4 + 0.6 * static_cast<double>(index), -1, arm::Prismatic));
    }
    return result;
}

std::vector<arm::RevoluteJoint> branched_joints() {
    auto root = joint("root", arm::normalized({0.3, 0.7, 0.4}), 1.8, -1);
    auto slider = joint("slider", arm::normalized({0.6, -0.2, 0.5}), 1.3, 0, arm::Prismatic);
    auto sibling = joint("sibling", arm::normalized({-0.4, 0.8, 0.3}), 0.9, 0);
    root.joint_offset = {0.12, -0.08, 0.15};
    root.joint_rotation = arm::rotation_matrix(arm::normalized({0.2, 0.3, 0.8}), 0.35);
    slider.joint_offset = {0.11, 0.23, -0.07};
    slider.joint_rotation = arm::rotation_matrix({0.0, 1.0, 0.0}, -0.42);
    sibling.joint_offset = {-0.16, -0.12, 0.19};
    sibling.joint_rotation = arm::rotation_matrix({1.0, 0.0, 0.0}, 0.53);
    return {root, slider, sibling};
}

void cholesky(const Matrix& matrix, const std::string& label) {
    const auto size = matrix.size();
    Matrix lower(size, Vector(size, 0.0));
    for (std::size_t row = 0; row < size; ++row) {
        require(matrix[row].size() == size, label + " square");
        for (std::size_t column = 0; column <= row; ++column) {
            near(matrix[row][column], matrix[column][row], 1e-11, label + " symmetry");
            double entry = matrix[row][column];
            for (std::size_t inner = 0; inner < column; ++inner) {
                entry -= lower[row][inner] * lower[column][inner];
            }
            if (row == column) {
                require(std::isfinite(entry) && entry > 1e-10, label + " positive pivot");
                lower[row][column] = std::sqrt(entry);
            } else {
                lower[row][column] = entry / lower[column][column];
            }
        }
    }
    for (std::size_t row = 0; row < size; ++row) {
        for (std::size_t column = 0; column < size; ++column) {
            double reconstructed = 0.0;
            for (std::size_t inner = 0; inner < size; ++inner) {
                reconstructed += lower[row][inner] * lower[column][inner];
            }
            near(reconstructed, matrix[row][column], 1e-11, label + " reconstruction");
        }
    }
}

void planar_reference() {
    const auto joints = planar_joints();
    const arm::ArmModel model(joints);
    const arm::DynamicsSolver dynamics(model, {0.0, -9.81, 0.0});
    for (double first_angle : {-1.1, 0.27, 1.35}) {
        for (double second_angle : {-0.83, 0.46, 1.17}) {
            const double first_speed = 0.37;
            const double second_speed = -0.61;
            const double length = 0.8;
            const double first_com = 0.35;
            const double second_com = 0.28;
            const double first_mass = 1.7;
            const double second_mass = 1.1;
            const double coupling = second_mass * length * second_com;
            const double distal = joints[1].inertia_com.values[8] + second_mass * second_com * second_com;
            const double diagonal = joints[0].inertia_com.values[8] + first_mass * first_com * first_com +
                distal + second_mass * length * length + 2.0 * coupling * std::cos(second_angle);
            const double off_diagonal = distal + coupling * std::cos(second_angle);
            const double first_bias = -coupling * std::sin(second_angle) *
                (2.0 * first_speed * second_speed + second_speed * second_speed) +
                9.81 * ((first_mass * first_com + second_mass * length) * std::cos(first_angle) +
                        second_mass * second_com * std::cos(first_angle + second_angle));
            const double second_bias = coupling * std::sin(second_angle) * first_speed * first_speed +
                9.81 * second_mass * second_com * std::cos(first_angle + second_angle);
            const Vector torque{0.43, -0.29};
            const auto result = dynamics.forward_dynamics({first_angle, second_angle},
                                                          {first_speed, second_speed}, torque);
            near(result.mass_matrix[0][0], diagonal, 1e-11, "planar M00");
            near(result.mass_matrix[0][1], off_diagonal, 1e-11, "planar M01");
            near(result.mass_matrix[1][0], off_diagonal, 1e-11, "planar M10");
            near(result.mass_matrix[1][1], distal, 1e-11, "planar M11");
            near(result.bias_forces[0], first_bias, 1e-11, "planar bias0");
            near(result.bias_forces[1], second_bias, 1e-11, "planar bias1");
            const double determinant = diagonal * distal - off_diagonal * off_diagonal;
            near(result.joint_accelerations[0],
                 (distal * (torque[0] - first_bias) - off_diagonal * (torque[1] - second_bias)) /
                     determinant, 1e-10, "planar acceleration0");
            near(result.joint_accelerations[1],
                 (diagonal * (torque[1] - second_bias) - off_diagonal * (torque[0] - first_bias)) /
                     determinant, 1e-10, "planar acceleration1");
            cholesky(result.mass_matrix, "planar SPD");
        }
    }
}

void tree_reference() {
    const arm::ArmModel model(branched_joints());
    const arm::DynamicsSolver dynamics(model);
    for (double angle : {-0.9, 0.21, 1.4}) {
        const Vector positions{angle, 0.13, -0.38};
        const Vector velocities{0.31, -0.22, 0.19};
        const auto base = dynamics.forward_dynamics(positions, velocities, {0.0, 0.0, 0.0});
        cholesky(base.mass_matrix, "tree SPD");
        near(base.mass_matrix[1][2], 0.0, 1e-12, "siblings M12");
        near(base.mass_matrix[2][1], 0.0, 1e-12, "siblings M21");
        auto changed_positions = positions;
        auto changed_velocities = velocities;
        changed_positions[2] += 0.71;
        changed_velocities[2] -= 0.54;
        const auto changed = dynamics.forward_dynamics(changed_positions, changed_velocities, {0.0, 0.0, 0.0});
        near(changed.bias_forces[1], base.bias_forces[1], 1e-11, "sibling bias independence");
        near(changed.mass_matrix[1][1], base.mass_matrix[1][1], 1e-11, "sibling inertia independence");
        const auto response = dynamics.inverse_dynamics(positions, velocities, {0.0, 0.0, 0.8});
        near(response[1], base.bias_forces[1], 1e-11, "sibling acceleration effort independence");
        const auto& root = model.joints()[0];
        const auto& slider = model.joints()[1];
        const auto root_rotation = root.joint_rotation * arm::rotation_matrix(root.axis, angle);
        const auto axis = root_rotation * slider.joint_rotation * slider.axis;
        const auto resting = dynamics.forward_dynamics(positions, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0});
        near(resting.bias_forces[1], -slider.mass * arm::dot(axis, dynamics.gravity()),
             1e-11, "rotated child gravity projection");
    }
    auto sliders = slider_joints(2);
    sliders[0].joint_rotation = arm::rotation_matrix({0.0, 1.0, 0.0}, 0.63);
    sliders[0].joint_offset = {0.3, -0.4, 0.7};
    sliders[1].joint_rotation = arm::rotation_matrix({1.0, 0.0, 0.0}, -0.41);
    const arm::ArmModel free_model(sliders);
    const arm::Vec3 gravity{0.8, -1.2, -9.81};
    const arm::DynamicsSolver free_dynamics(free_model, gravity);
    const Vector torques{0.52, -0.38};
    const auto result = free_dynamics.forward_dynamics({0.27, -0.19}, {0.44, -0.36}, torques);
    for (std::size_t index = 0; index < sliders.size(); ++index) {
        const auto axis = sliders[index].joint_rotation * sliders[index].axis;
        near(result.joint_accelerations[index], torques[index] / sliders[index].mass + arm::dot(axis, gravity),
             1e-11, "prismatic analytic acceleration");
    }
}

arm::DynamicsOptions constraint_options() {
    arm::DynamicsOptions options;
    options.constraints.push_back({{1.0, -2.0}, 0.12, 7.0, 0.9});
    return options;
}

void constraint_reference() {
    const arm::ArmModel model(slider_joints(2));
    const auto options = constraint_options();
    const arm::DynamicsSolver dynamics(model, {}, options);
    const arm::RolloutSolver rollout(dynamics);
    const Vector torques{0.7, -0.4};
    const arm::ArmState initial{{0.31, -0.08}, {0.27, -0.16}};
    const auto& constraint = options.constraints[0];
    const double error = initial.joint_positions[0] - 2.0 * initial.joint_positions[1] - constraint.target;
    const double speed = initial.joint_velocities[0] - 2.0 * initial.joint_velocities[1];
    const double desired = -2.0 * constraint.damping_ratio * constraint.frequency * speed -
        constraint.frequency * constraint.frequency * error;
    const double multiplier = (desired - torques[0] / 1.4 + 2.0 * torques[1] / 2.0) /
        (1.0 / 1.4 + 4.0 / 2.0);
    const auto result = dynamics.forward_dynamics(initial.joint_positions, initial.joint_velocities, torques);
    require(result.constraint_forces.size() == 1, "constraint multiplier shape");
    near(result.constraint_forces[0], multiplier, 1e-11, "analytic multiplier");
    near(result.joint_accelerations[0], (torques[0] + multiplier) / 1.4, 1e-11, "constrained acceleration0");
    near(result.joint_accelerations[1], (torques[1] - 2.0 * multiplier) / 2.0, 1e-11, "constrained acceleration1");
    arm::ArmState state = initial;
    for (std::size_t step = 0; step < 300; ++step) {
        const auto acceleration = dynamics.forward_dynamics(state.joint_positions, state.joint_velocities, torques);
        const double position_error = state.joint_positions[0] - 2.0 * state.joint_positions[1] - constraint.target;
        const double velocity_error = state.joint_velocities[0] - 2.0 * state.joint_velocities[1];
        near(acceleration.joint_accelerations[0] - 2.0 * acceleration.joint_accelerations[1] +
             2.0 * constraint.damping_ratio * constraint.frequency * velocity_error +
             constraint.frequency * constraint.frequency * position_error, 0.0, 1e-10, "constraint residual");
        state = rollout.step(state, torques, 0.01);
    }
    require(std::abs(state.joint_positions[0] - 2.0 * state.joint_positions[1] - constraint.target) < 1e-7,
            "stabilized position drift");
    require(std::abs(state.joint_velocities[0] - 2.0 * state.joint_velocities[1]) < 1e-6,
            "stabilized velocity drift");
    state = {{0.32, 0.10}, {0.24, 0.12}};
    for (std::size_t step = 0; step < 100; ++step) {
        state = rollout.step(state, torques, 0.01);
        near(state.joint_positions[0] - 2.0 * state.joint_positions[1], constraint.target, 1e-11,
             "feasible rollout position drift");
        near(state.joint_velocities[0] - 2.0 * state.joint_velocities[1], 0.0, 1e-11,
             "feasible rollout velocity drift");
    }
    auto duplicate = options;
    duplicate.constraints.push_back(duplicate.constraints[0]);
    const arm::DynamicsSolver singular(model, {}, duplicate);
    throws<std::runtime_error>([&] {
        singular.forward_dynamics(initial.joint_positions, initial.joint_velocities, torques);
    }, "duplicate constraints must throw");
}

arm::DynamicsOptions contact_options() {
    arm::DynamicsOptions options;
    options.contacts.push_back({0, {}, {0.0, 0.0, 1.0}, 0.0, 120.0, 0.015});
    return options;
}

double contact_potential(double gap, double stiffness, double smoothing) {
    const double root = std::sqrt(gap * gap + smoothing * smoothing);
    return 0.25 * stiffness * (gap * gap - gap * root -
                              smoothing * smoothing * std::asinh(gap / smoothing));
}

void contact_reference() {
    const arm::ArmModel model(slider_joints(1));
    const auto options = contact_options();
    const arm::DynamicsSolver dynamics(model, {}, options);
    const arm::RolloutSolver rollout(dynamics);
    const auto& contact = options.contacts[0];
    double previous_force = std::numeric_limits<double>::infinity();
    for (double gap : {-0.2, -0.08, -0.01, 0.0, 0.03, 0.2, 100.0}) {
        const auto result = dynamics.forward_dynamics({gap}, {0.37}, {0.0});
        const double force = 1.4 * result.joint_accelerations[0];
        const double root = std::hypot(gap, contact.smoothing);
        const double expected = gap > 0.0 ?
            0.5 * contact.stiffness * contact.smoothing * contact.smoothing / (root + gap) :
            0.5 * contact.stiffness * (root - gap);
        near(force, expected, 1e-11, "smooth contact closed form");
        require(force > 0.0 && force < previous_force, "contact monotonic penetration");
        near(result.bias_forces[0], 0.0, 1e-12, "contact not included in rigid bias");
        previous_force = force;
    }
    require(previous_force < 1e-4, "separated tiny positive force");
    const auto energy = [&](const arm::ArmState& state) {
        return 0.7 * state.joint_velocities[0] * state.joint_velocities[0] +
            contact_potential(state.joint_positions[0], contact.stiffness, contact.smoothing);
    };
    arm::ArmState state{{-0.06}, {-0.18}};
    const double initial_energy = energy(state);
    for (std::size_t step = 0; step < 200; ++step) {
        state = rollout.step(state, {0.0}, 0.001);
        near(energy(state), initial_energy, 1e-8, "conservative contact energy");
    }
}

struct Scene {
    std::string name;
    std::vector<arm::RevoluteJoint> joints;
    arm::Vec3 gravity;
    arm::DynamicsOptions options;
    arm::ArmState state;
    double dt;
};

arm::TorqueSequence controls_for(std::size_t steps, std::size_t dof) {
    arm::TorqueSequence controls(steps, Vector(dof));
    for (std::size_t step = 0; step < steps; ++step) {
        for (std::size_t coordinate = 0; coordinate < dof; ++coordinate) {
            controls[step][coordinate] = 0.23 * std::sin(0.7 * static_cast<double>(step + 1) +
                                                       0.4 * static_cast<double>(coordinate)) + 0.09;
        }
    }
    return controls;
}

arm::RolloutObjective objective_for(const Scene& scene, std::size_t steps, bool cartesian) {
    arm::RolloutObjective objective;
    objective.cartesian = cartesian;
    objective.control_weight = 0.013;
    if (cartesian) {
        objective.link = scene.joints.size() - 1;
        objective.local_point = {0.38, -0.11, 0.09};
        objective.target_point = {0.45, 0.17, -0.21};
        objective.terminal_weight = 1.3;
    } else {
        objective.target_positions = Vector(scene.joints.size(), -0.14);
        objective.terminal_weight = 0.0;
        objective.trajectory_weight = 0.6 / static_cast<double>(steps);
        objective.trajectory_targets.assign(steps, Vector(scene.joints.size()));
        for (std::size_t step = 0; step < steps; ++step) {
            for (std::size_t coordinate = 0; coordinate < scene.joints.size(); ++coordinate) {
                objective.trajectory_targets[step][coordinate] = -0.12 +
                    0.015 * static_cast<double>(step) - 0.03 * static_cast<double>(coordinate);
            }
        }
    }
    return objective;
}

Vector flatten(const arm::RolloutGradient& gradient) {
    Vector result = gradient.initial_state.joint_positions;
    result.insert(result.end(), gradient.initial_state.joint_velocities.begin(), gradient.initial_state.joint_velocities.end());
    for (const auto& control : gradient.controls) {
        result.insert(result.end(), control.begin(), control.end());
    }
    result.insert(result.end(), gradient.masses.begin(), gradient.masses.end());
    for (const auto& inertia : gradient.inertias) {
        result.insert(result.end(), inertia.begin(), inertia.end());
    }
    return result;
}

arm::RolloutGradient unflatten(const Vector& entries, std::size_t dof, std::size_t steps) {
    require(entries.size() == dof * (9 + steps), "flat direction shape");
    arm::RolloutGradient result{};
    std::size_t cursor = 0;
    const auto take = [&](std::size_t count) {
        Vector values(entries.begin() + static_cast<std::ptrdiff_t>(cursor),
                      entries.begin() + static_cast<std::ptrdiff_t>(cursor + count));
        cursor += count;
        return values;
    };
    result.initial_state.joint_positions = take(dof);
    result.initial_state.joint_velocities = take(dof);
    for (std::size_t step = 0; step < steps; ++step) {
        result.controls.push_back(take(dof));
    }
    result.masses = take(dof);
    result.inertias.resize(dof);
    for (auto& inertia : result.inertias) {
        for (auto& entry : inertia) {
            entry = entries[cursor++];
        }
    }
    return result;
}

Vector variables(const Scene& scene, const arm::TorqueSequence& controls) {
    arm::RolloutGradient values{};
    values.initial_state = scene.state;
    values.controls = controls;
    for (const auto& link : scene.joints) {
        values.masses.push_back(link.mass);
        std::array<double, 6> inertia{};
        for (std::size_t entry = 0; entry < inertia.size(); ++entry) {
            inertia[entry] = link.inertia_com.values[inertia_indices[entry][0]];
        }
        values.inertias.push_back(inertia);
    }
    return flatten(values);
}

double perturbed_loss(const Scene& scene, const arm::TorqueSequence& controls,
                      const arm::RolloutObjective& objective, const Vector& values) {
    const auto input = unflatten(values, scene.joints.size(), controls.size());
    auto joints = scene.joints;
    for (std::size_t link = 0; link < joints.size(); ++link) {
        joints[link].mass = input.masses[link];
        for (std::size_t entry = 0; entry < inertia_indices.size(); ++entry) {
            for (const auto index : inertia_indices[entry]) {
                joints[link].inertia_com.values[index] = input.inertias[link][entry];
            }
        }
    }
    const arm::ArmModel model(std::move(joints));
    const arm::DynamicsSolver dynamics(model, scene.gravity, scene.options);
    const arm::RolloutSolver rollout(dynamics);
    return rollout.evaluate_objective(input.initial_state, input.controls, scene.dt, objective);
}

double inner_product(const Vector& first, const Vector& second) {
    require(first.size() == second.size(), "dot shape");
    double result = 0.0;
    for (std::size_t index = 0; index < first.size(); ++index) {
        result += first[index] * second[index];
    }
    return result;
}

void compare_vectors(const Vector& actual, const Vector& reference, std::size_t begin,
                     std::size_t end, double floor, const std::string& label) {
    double difference_squared = 0.0;
    double actual_squared = 0.0;
    double reference_squared = 0.0;
    for (std::size_t index = begin; index < end; ++index) {
        require(std::isfinite(actual[index]) && std::isfinite(reference[index]), label + " finite");
        const double difference = actual[index] - reference[index];
        difference_squared += difference * difference;
        actual_squared += actual[index] * actual[index];
        reference_squared += reference[index] * reference[index];
    }
    const double denominator = std::max({floor, std::sqrt(actual_squared), std::sqrt(reference_squared)});
    require(std::sqrt(difference_squared) / denominator < 3e-4, label + " relative norm");
}

void gradient_scene(const Scene& scene) {
    const std::size_t dof = scene.joints.size();
    const arm::ArmModel model(scene.joints);
    const arm::DynamicsSolver dynamics(model, scene.gravity, scene.options);
    const arm::RolloutSolver rollout(dynamics);
    for (const std::size_t steps : {std::size_t(5), std::size_t(40)}) {
        const auto controls = controls_for(steps, dof);
        const auto values = variables(scene, controls);
        const std::size_t masses_begin = dof * (2 + steps);
        const std::size_t inertia_begin = masses_begin + dof;
        for (const bool cartesian : {true, false}) {
            const auto objective = objective_for(scene, steps, cartesian);
            const std::string label = scene.name + (cartesian ? " Cartesian " : " tracking ") + std::to_string(steps);
            const auto gradient = rollout.automatic_gradient(scene.state, controls, scene.dt, objective);
            const auto analytic = flatten(gradient);
            require(analytic.size() == values.size(), label + " full gradient shape");
            near(gradient.loss, rollout.evaluate_objective(scene.state, controls, scene.dt, objective), 1e-12,
                 label + " primal loss");
            Vector direction(values.size());
            for (std::size_t index = 0; index < direction.size(); ++index) {
                const double scale = index >= inertia_begin ? std::max(std::abs(values[index]), 0.01) : 1.0;
                direction[index] = scale * (0.3 * std::sin(0.63 * static_cast<double>(index + 1)) + 0.17);
            }
            const double dot_gradient = inner_product(analytic, direction);
            const double jvp = rollout.automatic_directional_derivative(
                scene.state, controls, scene.dt, objective, unflatten(direction, dof, steps));
            near(jvp, dot_gradient, 2e-10, label + " one-dimensional JVP dot full gradient");
            for (double epsilon : {1e-3, 1e-4, 1e-5}) {
                auto plus = values;
                auto minus = values;
                for (std::size_t index = 0; index < values.size(); ++index) {
                    plus[index] += epsilon * direction[index];
                    minus[index] -= epsilon * direction[index];
                }
                const double directional_fd = (perturbed_loss(scene, controls, objective, plus) -
                                               perturbed_loss(scene, controls, objective, minus)) / (2.0 * epsilon);
                require(std::abs(directional_fd - jvp) / std::max({1e-6, std::abs(directional_fd), std::abs(jvp)}) < 3e-4,
                        label + " directional FD epsilon=" + std::to_string(epsilon));
                if (steps != 5) {
                    continue;
                }
                Vector finite_difference(values.size());
                for (std::size_t index = 0; index < values.size(); ++index) {
                    const double delta = epsilon * (index >= inertia_begin ? std::max(std::abs(values[index]), 0.01) : 1.0);
                    plus = values;
                    minus = values;
                    plus[index] += delta;
                    minus[index] -= delta;
                    finite_difference[index] = (perturbed_loss(scene, controls, objective, plus) -
                                                perturbed_loss(scene, controls, objective, minus)) / (2.0 * delta);
                }
                const std::string sweep = label + " epsilon=" + std::to_string(epsilon);
                compare_vectors(analytic, finite_difference, 0, dof, 1e-7, sweep + " initial q");
                compare_vectors(analytic, finite_difference, dof, 2 * dof, 1e-7, sweep + " initial v");
                compare_vectors(analytic, finite_difference, 2 * dof, masses_begin, 1e-7, sweep + " controls");
                compare_vectors(analytic, finite_difference, masses_begin, inertia_begin, 1e-7, sweep + " masses");
                compare_vectors(analytic, finite_difference, inertia_begin, values.size(), 1e-7, sweep + " inertias");
                compare_vectors(analytic, finite_difference, 0, values.size(), 1e-7, sweep + " full");
                const double analytic_norm = std::sqrt(inner_product(analytic, analytic));
                const double fd_norm = std::sqrt(inner_product(finite_difference, finite_difference));
                require(analytic_norm > 1e-8 && fd_norm > 1e-8, sweep + " nonzero full vector");
                require(inner_product(analytic, finite_difference) / (analytic_norm * fd_norm) > 0.999999,
                        sweep + " full cosine");
            }
            auto penalty = objective;
            penalty.terminal_weight = 0.0;
            penalty.trajectory_weight = 0.0;
            const auto penalty_gradient = rollout.automatic_gradient(scene.state, controls, scene.dt, penalty);
            const auto penalty_flat = flatten(penalty_gradient);
            double expected_loss = 0.0;
            for (std::size_t index = 0; index < values.size(); ++index) {
                const bool control = index >= 2 * dof && index < masses_begin;
                near(penalty_flat[index], control ? penalty.control_weight * values[index] : 0.0, 1e-12,
                     label + " isolated control penalty gradient");
                if (control) {
                    expected_loss += 0.5 * penalty.control_weight * values[index] * values[index];
                }
            }
            near(penalty_gradient.loss, expected_loss, 1e-12, label + " isolated penalty loss");
        }
    }
}

void invalid_inputs() {
    const Scene scene{"validation", slider_joints(1), {}, {}, {{0.24}, {-0.13}}, 0.01};
    const arm::ArmModel model(scene.joints);
    const arm::DynamicsSolver dynamics(model, {});
    const arm::RolloutSolver rollout(dynamics);
    const auto controls = controls_for(5, 1);
    const auto objective = objective_for(scene, 5, false);
    const auto direction = unflatten(Vector(14, 0.1), 1, 5);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double infinity = std::numeric_limits<double>::infinity();
    const auto reject = [&](const arm::ArmState& state, const arm::TorqueSequence& sequence,
                            double dt, const arm::RolloutObjective& target) {
        throws<std::invalid_argument>([&] { rollout.evaluate_objective(state, sequence, dt, target); }, "invalid primal");
        throws<std::invalid_argument>([&] { rollout.automatic_gradient(state, sequence, dt, target); }, "invalid gradient");
        throws<std::invalid_argument>([&] {
            rollout.automatic_directional_derivative(state, sequence, dt, target, direction);
        }, "invalid JVP input");
    };
    for (double dt : {0.0, -0.01, nan, infinity}) {
        reject(scene.state, controls, dt, objective);
    }
    reject({{}, {-0.13}}, controls, scene.dt, objective);
    reject({{0.24}, {}}, controls, scene.dt, objective);
    reject({{nan}, {-0.13}}, controls, scene.dt, objective);
    reject({{0.24}, {infinity}}, controls, scene.dt, objective);
    reject(scene.state, {{}, {0.1}}, scene.dt, objective);
    auto bad_controls = controls;
    bad_controls[2] = {0.1, 0.2};
    reject(scene.state, bad_controls, scene.dt, objective);
    bad_controls = controls;
    bad_controls[1][0] = nan;
    reject(scene.state, bad_controls, scene.dt, objective);
    for (std::size_t variant = 0; variant < 13; ++variant) {
        auto bad = objective;
        switch (variant) {
        case 0: bad.target_positions.clear(); break;
        case 1: bad.target_positions[0] = nan; break;
        case 2: bad.terminal_weight = -1.0; break;
        case 3: bad.trajectory_weight = infinity; break;
        case 4: bad.control_weight = nan; break;
        case 5: bad.trajectory_targets.pop_back(); break;
        case 6: bad.trajectory_targets[0].push_back(0.0); break;
        case 7: bad.trajectory_targets[0][0] = nan; break;
        case 8: bad.local_point.x = infinity; break;
        case 9: bad.target_point.z = nan; break;
        case 10: bad.cartesian = true; bad.link = 1; break;
        case 11: bad.cartesian = true; break;
        case 12: bad.cartesian = true; bad.target_positions = {0.0, 0.0}; break;
        }
        reject(scene.state, controls, scene.dt, bad);
    }
    for (std::size_t variant = 0; variant < 9; ++variant) {
        auto bad = direction;
        switch (variant) {
        case 0: bad.initial_state.joint_positions.clear(); break;
        case 1: bad.initial_state.joint_velocities.clear(); break;
        case 2: bad.controls.pop_back(); break;
        case 3: bad.controls[0].clear(); break;
        case 4: bad.masses.clear(); break;
        case 5: bad.inertias.clear(); break;
        case 6: bad.inertias[0][4] = nan; break;
        case 7: bad.masses[0] = infinity; break;
        case 8: bad.controls[0][0] = nan; break;
        }
        throws<std::invalid_argument>([&] {
            rollout.automatic_directional_derivative(scene.state, controls, scene.dt, objective, bad);
        }, "invalid direction");
    }
    throws<std::invalid_argument>([] { arm::ArmModel empty({}); }, "empty model");
    for (std::size_t variant = 0; variant < 9; ++variant) {
        auto bad = scene.joints;
        switch (variant) {
        case 0: bad[0].mass = 0.0; break;
        case 1: bad[0].mass = nan; break;
        case 2: bad[0].axis = {}; break;
        case 3: bad[0].parent = 0; break;
        case 4: bad[0].joint_offset.x = nan; break;
        case 5: bad[0].joint_rotation.values[0] = 2.0; break;
        case 6: bad[0].joint_rotation.values[0] = -1.0; break;
        case 7: bad[0].inertia_com.values[0] = -0.1; break;
        case 8: bad[0].inertia_com.values[1] += 0.1; break;
        }
        throws<std::invalid_argument>([&] { arm::ArmModel invalid(bad); }, "invalid physical model");
    }
    throws<std::invalid_argument>([&] { arm::DynamicsSolver invalid(model, {nan, 0.0, 0.0}); }, "invalid gravity");
    for (std::size_t variant = 0; variant < 11; ++variant) {
        auto bad = contact_options();
        switch (variant) {
        case 0: bad.contacts[0].link = 1; break;
        case 1: bad.contacts[0].normal = {0.0, 0.0, 2.0}; break;
        case 2: bad.contacts[0].smoothing = 0.0; break;
        case 3: bad.contacts[0].stiffness = -1.0; break;
        case 4: bad.contacts[0].local_point.x = nan; break;
        case 5: bad.contacts[0].plane_offset = infinity; break;
        case 6: bad.constraints.push_back({{1.0, 2.0}, 0.0, 1.0, 1.0}); break;
        case 7: bad.constraints.push_back({{1.0}, 0.0, -1.0, 1.0}); break;
        case 8: bad.constraints.push_back({{1.0}, 0.0, 1.0, -1.0}); break;
        case 9: bad.constraints.push_back({{1.0}, nan, 1.0, 1.0}); break;
        case 10: bad.constraints.push_back({{nan}, 0.0, 1.0, 1.0}); break;
        }
        throws<std::invalid_argument>([&] { arm::DynamicsSolver invalid(model, {}, bad); }, "invalid dynamics option");
    }
    arm::RolloutObjective empty_objective;
    empty_objective.target_positions = {-0.2};
    const auto empty_gradient = rollout.automatic_gradient(scene.state, {}, scene.dt, empty_objective);
    near(empty_gradient.loss, 0.5 * 0.44 * 0.44, 1e-12, "empty rollout terminal loss");
    near(empty_gradient.initial_state.joint_positions[0], 0.44, 1e-12, "empty rollout q gradient");
    require(empty_gradient.controls.empty(), "empty rollout control gradient");
    near(empty_gradient.initial_state.joint_velocities[0], 0.0, 1e-12, "empty rollout v gradient");
    near(empty_gradient.masses[0], 0.0, 1e-12, "empty rollout mass gradient");
    for (const auto entry : empty_gradient.inertias[0]) {
        near(entry, 0.0, 1e-12, "empty rollout inertia gradient");
    }
    const auto unchanged = rollout.rollout(scene.state, {}, scene.dt);
    require(unchanged.joint_positions == scene.state.joint_positions &&
            unchanged.joint_velocities == scene.state.joint_velocities, "empty rollout identity");
    near(rollout.automatic_directional_derivative(scene.state, {}, scene.dt, empty_objective,
                                                 unflatten(Vector(9, 0.1), 1, 0)),
         0.044, 1e-12, "empty rollout JVP");
    reject(scene.state, {}, scene.dt, objective);
}

}

int main() {
    const auto run = [](const std::string& name, const auto& function) {
        function();
        std::cout << "PASS " << name << '\n';
    };
    try {
        run("1 planar analytical mass, bias, acceleration and SPD", planar_reference);
        run("2 branched sibling decoupling and rotated prismatic dynamics", tree_reference);
        run("3 analytical constraints, residual, drift and rank deficiency", constraint_reference);
        run("4 smooth contact, monotonicity and conservative energy", contact_reference);
        run("5 pendulum gradients", [] {
            auto pendulum = joint("pendulum", {0.0, 1.0, 0.0}, 1.6, -1);
            gradient_scene({"pendulum", {pendulum}, {0.0, 0.0, -9.81}, {}, {{0.43}, {-0.27}}, 0.008});
        });
        run("6 two-link chain gradients", [] {
            gradient_scene({"chain", planar_joints(), {0.0, -9.81, 0.0}, {}, {{0.31, -0.42}, {0.23, -0.19}}, 0.006});
        });
        run("7 constrained and contact slider gradients", [] {
            gradient_scene({"constraints", slider_joints(2), {}, constraint_options(),
                            {{0.29, 0.085}, {0.17, 0.085}}, 0.008});
            gradient_scene({"contact", slider_joints(1), {0.0, 0.0, -1.2}, contact_options(),
                            {{-0.023}, {-0.12}}, 0.004});
        });
        run("8 spatial mixed-tree gradients and invalid inputs", [] {
            gradient_scene({"mixed tree", branched_joints(), {0.7, -1.3, -4.0}, {},
                            {{0.28, 0.11, -0.36}, {0.21, -0.14, 0.18}}, 0.006});
            invalid_inputs();
        });
        std::cout << "All eight stages passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL " << error.what() << '\n';
        return 1;
    }
}