// Interactive ImGui front end for the differentiable articulated-arm simulator.
// Lets the user edit the model (joints, masses, inertias), configure bilateral
// coordinate constraints and smooth plane contacts, feed initial states and
// torques, and watch live RK4 rollouts in a 3D wireframe viewport.

#include "arm/arm_model.hpp"
#include "arm/dynamics_solver.hpp"
#include "arm/rollout.hpp"
#include "arm/simulation_core.hpp"

#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {

constexpr double kPi = 3.14159265358979323846;

struct JointEdit {
    char name[64] = "joint";
    int type = 0;  // 0 = Revolute, 1 = Prismatic
    int parent = -2;
    double axis[3] = {0.0, 1.0, 0.0};
    double joint_offset[3] = {0.0, 0.0, 0.0};
    double rest_rpy_deg[3] = {0.0, 0.0, 0.0};  // roll(X), pitch(Y), yaw(Z)
    double link_offset[3] = {0.5, 0.0, 0.0};
    double mass = 1.0;
    double com_offset[3] = {0.25, 0.0, 0.0};
    double inertia_diag[3] = {0.01, 0.01, 0.01};
    double inertia_offdiag[3] = {0.0, 0.0, 0.0};  // xy, xz, yz
};

struct ConstraintEdit {
    std::vector<double> coefficients;
    double target = 0.0;
    double frequency = 10.0;
    double damping_ratio = 1.0;
};

struct ContactEdit {
    int link = 0;
    double local_point[3] = {0.0, 0.0, 0.0};
    double normal[3] = {0.0, 0.0, 1.0};
    double plane_offset = 0.0;
    double stiffness = 1000.0;
    double smoothing = 0.001;
};

struct Camera {
    float yaw = 0.7f;
    float pitch = 0.45f;
    float distance = 3.5f;
    arm::Vec3 target{0.0, 0.0, 0.3};
};

struct AppState {
    std::vector<JointEdit> joints;
    std::vector<ConstraintEdit> constraints;
    std::vector<ContactEdit> contacts;
    double gravity[3] = {0.0, 0.0, -9.81};

    // Declaration order matters: dynamics references model, rollout references dynamics.
    std::unique_ptr<arm::ArmModel> model;
    std::unique_ptr<arm::DynamicsSolver> dynamics;
    std::unique_ptr<arm::RolloutSolver> rollout;

    bool model_dirty = true;
    std::string model_error;
    std::string sim_error;

    std::vector<double> initial_positions;
    std::vector<double> initial_velocities;
    std::vector<double> torques;
    double torque_limit = 20.0;

    arm::ArmState current;
    double timestep = 0.01;
    double speed = 1.0;
    double time_accumulator = 0.0;
    double sim_time = 0.0;
    bool playing = false;

    std::vector<std::vector<float>> position_history;
    std::vector<std::vector<float>> velocity_history;

    Camera camera;
    bool show_contact_planes = true;
    bool show_com_markers = true;
};

arm::Mat3 rotation_from_rpy_degrees(const double rpy_deg[3]) {
    const double roll = rpy_deg[0] * kPi / 180.0;
    const double pitch = rpy_deg[1] * kPi / 180.0;
    const double yaw = rpy_deg[2] * kPi / 180.0;
    const arm::Mat3 rx = arm::rotation_matrix({1.0, 0.0, 0.0}, roll);
    const arm::Mat3 ry = arm::rotation_matrix({0.0, 1.0, 0.0}, pitch);
    const arm::Mat3 rz = arm::rotation_matrix({0.0, 0.0, 1.0}, yaw);
    return rz * (ry * rx);
}

void sync_dimensions(AppState& state) {
    const std::size_t dof = state.joints.size();
    state.initial_positions.resize(dof, 0.0);
    state.initial_velocities.resize(dof, 0.0);
    state.torques.resize(dof, 0.0);
    for (ConstraintEdit& constraint : state.constraints) {
        constraint.coefficients.resize(dof, 0.0);
    }
    for (ContactEdit& contact : state.contacts) {
        contact.link = std::clamp(contact.link, 0,
                                  dof == 0 ? 0 : static_cast<int>(dof) - 1);
    }
}

void reset_simulation(AppState& state) {
    state.current.joint_positions = state.initial_positions;
    state.current.joint_velocities = state.initial_velocities;
    state.sim_time = 0.0;
    state.time_accumulator = 0.0;
    state.sim_error.clear();
    const std::size_t dof = state.joints.size();
    state.position_history.assign(dof, {});
    state.velocity_history.assign(dof, {});
}

void rebuild_solvers(AppState& state) {
    state.model_error.clear();
    const std::size_t previous_dof = state.model ? state.model->degrees_of_freedom() : 0;
    sync_dimensions(state);
    if (state.joints.empty()) {
        state.rollout.reset();
        state.dynamics.reset();
        state.model.reset();
        reset_simulation(state);
        return;
    }
    try {
        std::vector<arm::RevoluteJoint> joints;
        joints.reserve(state.joints.size());
        for (const JointEdit& edit : state.joints) {
            arm::RevoluteJoint joint;
            joint.name = edit.name;
            joint.type = edit.type == 0 ? arm::Revolute : arm::Prismatic;
            joint.parent = edit.parent;
            joint.axis = {edit.axis[0], edit.axis[1], edit.axis[2]};
            joint.joint_offset = {edit.joint_offset[0], edit.joint_offset[1], edit.joint_offset[2]};
            joint.joint_rotation = rotation_from_rpy_degrees(edit.rest_rpy_deg);
            joint.link_offset = {edit.link_offset[0], edit.link_offset[1], edit.link_offset[2]};
            joint.mass = edit.mass;
            joint.com_offset = {edit.com_offset[0], edit.com_offset[1], edit.com_offset[2]};
            joint.inertia_com = {{edit.inertia_diag[0], edit.inertia_offdiag[0], edit.inertia_offdiag[1],
                                  edit.inertia_offdiag[0], edit.inertia_diag[1], edit.inertia_offdiag[2],
                                  edit.inertia_offdiag[1], edit.inertia_offdiag[2], edit.inertia_diag[2]}};
            joints.push_back(std::move(joint));
        }
        auto model = std::make_unique<arm::ArmModel>(std::move(joints));

        arm::DynamicsOptions options;
        for (const ConstraintEdit& edit : state.constraints) {
            arm::CoordinateConstraint constraint;
            constraint.coefficients = edit.coefficients;
            constraint.target = edit.target;
            constraint.frequency = edit.frequency;
            constraint.damping_ratio = edit.damping_ratio;
            options.constraints.push_back(std::move(constraint));
        }
        for (const ContactEdit& edit : state.contacts) {
            arm::SmoothPlaneContact contact;
            contact.link = static_cast<std::size_t>(std::max(edit.link, 0));
            contact.local_point = {edit.local_point[0], edit.local_point[1], edit.local_point[2]};
            contact.normal = {edit.normal[0], edit.normal[1], edit.normal[2]};
            contact.plane_offset = edit.plane_offset;
            contact.stiffness = edit.stiffness;
            contact.smoothing = edit.smoothing;
            options.contacts.push_back(contact);
        }

        const arm::Vec3 gravity{state.gravity[0], state.gravity[1], state.gravity[2]};
        auto dynamics = std::make_unique<arm::DynamicsSolver>(*model, gravity, std::move(options));
        auto rollout = std::make_unique<arm::RolloutSolver>(*dynamics);

        state.rollout.reset();
        state.dynamics.reset();
        state.model.reset();
        state.model = std::move(model);
        state.dynamics = std::move(dynamics);
        state.rollout = std::move(rollout);

        if (state.model->degrees_of_freedom() != previous_dof ||
            state.current.joint_positions.size() != state.model->degrees_of_freedom()) {
            reset_simulation(state);
        } else {
            state.sim_error.clear();
        }
    } catch (const std::exception& error) {
        state.model_error = error.what();
    }
}

void load_pendulum_preset(AppState& state) {
    state.joints.clear();
    state.constraints.clear();
    state.contacts.clear();
    JointEdit joint;
    std::snprintf(joint.name, sizeof(joint.name), "pendulum");
    joint.type = 0;
    joint.parent = -2;
    joint.axis[0] = 0.0; joint.axis[1] = 1.0; joint.axis[2] = 0.0;
    joint.link_offset[0] = 1.0; joint.link_offset[1] = 0.0; joint.link_offset[2] = 0.0;
    joint.mass = 2.0;
    joint.com_offset[0] = 0.5; joint.com_offset[1] = 0.0; joint.com_offset[2] = 0.0;
    joint.inertia_diag[0] = 0.01; joint.inertia_diag[1] = 1.0 / 6.0; joint.inertia_diag[2] = 1.0 / 6.0;
    state.joints.push_back(joint);
    sync_dimensions(state);
    state.initial_positions[0] = 0.7;
    state.initial_velocities[0] = 0.2;
    state.model_dirty = true;
    state.current.joint_positions.clear();  // force reset on rebuild
}

void load_three_link_preset(AppState& state) {
    state.joints.clear();
    state.constraints.clear();
    state.contacts.clear();

    JointEdit yaw;
    std::snprintf(yaw.name, sizeof(yaw.name), "shoulder_yaw");
    yaw.axis[0] = 0.0; yaw.axis[1] = 0.0; yaw.axis[2] = 1.0;
    yaw.link_offset[0] = 0.0; yaw.link_offset[1] = 0.0; yaw.link_offset[2] = 0.30;
    yaw.mass = 0.4;
    yaw.com_offset[0] = 0.0; yaw.com_offset[1] = 0.0; yaw.com_offset[2] = 0.15;
    yaw.inertia_diag[0] = 0.002; yaw.inertia_diag[1] = 0.002; yaw.inertia_diag[2] = 0.001;

    JointEdit pitch;
    std::snprintf(pitch.name, sizeof(pitch.name), "shoulder_pitch");
    pitch.axis[0] = 0.0; pitch.axis[1] = 1.0; pitch.axis[2] = 0.0;
    pitch.link_offset[0] = 0.30; pitch.link_offset[1] = 0.0; pitch.link_offset[2] = 0.0;
    pitch.mass = 1.5;
    pitch.com_offset[0] = 0.15; pitch.com_offset[1] = 0.0; pitch.com_offset[2] = 0.0;
    pitch.inertia_diag[0] = 0.012; pitch.inertia_diag[1] = 0.012; pitch.inertia_diag[2] = 0.003;

    JointEdit elbow;
    std::snprintf(elbow.name, sizeof(elbow.name), "elbow_pitch");
    elbow.axis[0] = 0.0; elbow.axis[1] = 1.0; elbow.axis[2] = 0.0;
    elbow.link_offset[0] = 0.25; elbow.link_offset[1] = 0.0; elbow.link_offset[2] = 0.0;
    elbow.mass = 0.9;
    elbow.com_offset[0] = 0.125; elbow.com_offset[1] = 0.0; elbow.com_offset[2] = 0.0;
    elbow.inertia_diag[0] = 0.005; elbow.inertia_diag[1] = 0.005; elbow.inertia_diag[2] = 0.0015;

    state.joints = {yaw, pitch, elbow};
    sync_dimensions(state);
    state.initial_positions = {0.2, -0.5, 0.8};
    state.initial_velocities = {0.0, 0.0, 0.0};
    state.model_dirty = true;
    state.current.joint_positions.clear();
}

void push_history(AppState& state) {
    const std::size_t dof = state.current.joint_positions.size();
    if (state.position_history.size() != dof) {
        state.position_history.assign(dof, {});
        state.velocity_history.assign(dof, {});
    }
    for (std::size_t index = 0; index < dof; ++index) {
        state.position_history[index].push_back(
            static_cast<float>(state.current.joint_positions[index]));
        state.velocity_history[index].push_back(
            static_cast<float>(state.current.joint_velocities[index]));
        if (state.position_history[index].size() > 6000) {
            state.position_history[index].erase(state.position_history[index].begin(),
                                                state.position_history[index].begin() + 3000);
            state.velocity_history[index].erase(state.velocity_history[index].begin(),
                                                state.velocity_history[index].begin() + 3000);
        }
    }
}

void advance_simulation(AppState& state, double frame_seconds) {
    if (!state.playing || !state.rollout ||
        state.current.joint_positions.size() != state.joints.size()) {
        return;
    }
    state.time_accumulator += frame_seconds * state.speed;
    int guard = 0;
    while (state.time_accumulator >= state.timestep && guard < 400) {
        try {
            state.current = state.rollout->step(state.current, state.torques, state.timestep);
            state.sim_time += state.timestep;
            push_history(state);
        } catch (const std::exception& error) {
            state.sim_error = error.what();
            state.playing = false;
            break;
        }
        state.time_accumulator -= state.timestep;
        ++guard;
    }
    if (guard >= 400) {
        state.time_accumulator = 0.0;  // drop backlog instead of stalling the UI
    }
}

// ---------------------------------------------------------------------------
// 3D wireframe viewport
// ---------------------------------------------------------------------------

struct Projector {
    arm::Vec3 camera_position;
    arm::Vec3 forward;
    arm::Vec3 right;
    arm::Vec3 up;
    ImVec2 center;
    float focal = 600.0f;

    bool project(const arm::Vec3& point, ImVec2& out) const {
        const arm::Vec3 relative = point - camera_position;
        const double depth = arm::dot(relative, forward);
        if (depth < 0.05) {
            return false;
        }
        const double horizontal = arm::dot(relative, right);
        const double vertical = arm::dot(relative, up);
        out.x = center.x + static_cast<float>(focal * horizontal / depth);
        out.y = center.y - static_cast<float>(focal * vertical / depth);
        return true;
    }
};

Projector make_projector(const Camera& camera, ImVec2 canvas_center, float canvas_height) {
    const double yaw = camera.yaw;
    const double pitch = std::clamp(static_cast<double>(camera.pitch), -1.5, 1.5);
    const arm::Vec3 offset{std::cos(pitch) * std::cos(yaw),
                           std::cos(pitch) * std::sin(yaw),
                           std::sin(pitch)};
    Projector projector;
    projector.camera_position = camera.target + offset * camera.distance;
    projector.forward = arm::normalized(camera.target - projector.camera_position);
    projector.right = arm::normalized(arm::cross(projector.forward, arm::Vec3{0.0, 0.0, 1.0}));
    projector.up = arm::cross(projector.right, projector.forward);
    projector.center = canvas_center;
    projector.focal = 0.5f * canvas_height / std::tan(25.0f * 3.14159265f / 180.0f);
    return projector;
}

void draw_line3(ImDrawList* draw_list, const Projector& projector,
                const arm::Vec3& from, const arm::Vec3& to, ImU32 color, float thickness) {
    ImVec2 a, b;
    if (projector.project(from, a) && projector.project(to, b)) {
        draw_list->AddLine(a, b, color, thickness);
    }
}

void draw_viewport(AppState& state) {
    ImGui::SetNextWindowPos(ImVec2(450, 10), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(700, 600), ImGuiCond_FirstUseEver);
    ImGui::Begin("Viewport");
    ImGui::TextDisabled("Left-drag: orbit | Right-drag: pan | Wheel: zoom");

    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    const ImVec2 canvas_origin = ImGui::GetCursorScreenPos();
    ImVec2 canvas_size = ImGui::GetContentRegionAvail();
    canvas_size.x = std::max(canvas_size.x, 64.0f);
    canvas_size.y = std::max(canvas_size.y, 64.0f);

    ImGui::InvisibleButton("viewport_canvas", canvas_size,
                           ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight);
    const bool hovered = ImGui::IsItemHovered();
    ImGuiIO& io = ImGui::GetIO();
    if (ImGui::IsItemActive()) {
        if (ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
            state.camera.yaw -= io.MouseDelta.x * 0.01f;
            state.camera.pitch = std::clamp(state.camera.pitch + io.MouseDelta.y * 0.01f,
                                            -1.5f, 1.5f);
        }
    }
    const ImVec2 canvas_center(canvas_origin.x + canvas_size.x * 0.5f,
                               canvas_origin.y + canvas_size.y * 0.5f);
    Projector projector = make_projector(state.camera, canvas_center, canvas_size.y);
    if (ImGui::IsItemActive() && ImGui::IsMouseDragging(ImGuiMouseButton_Right)) {
        const double pan = 0.0025 * state.camera.distance;
        state.camera.target = state.camera.target -
            projector.right * (io.MouseDelta.x * pan) +
            projector.up * (io.MouseDelta.y * pan);
        projector = make_projector(state.camera, canvas_center, canvas_size.y);
    }
    if (hovered && io.MouseWheel != 0.0f) {
        state.camera.distance = std::clamp(
            state.camera.distance * std::pow(0.9f, io.MouseWheel), 0.3f, 50.0f);
        projector = make_projector(state.camera, canvas_center, canvas_size.y);
    }

    draw_list->PushClipRect(canvas_origin,
                            ImVec2(canvas_origin.x + canvas_size.x, canvas_origin.y + canvas_size.y),
                            true);
    draw_list->AddRectFilled(canvas_origin,
                             ImVec2(canvas_origin.x + canvas_size.x, canvas_origin.y + canvas_size.y),
                             IM_COL32(24, 26, 32, 255));

    // Ground grid on z = 0.
    const ImU32 grid_color = IM_COL32(70, 72, 80, 120);
    for (int line = -4; line <= 4; ++line) {
        const double coordinate = line * 0.5;
        draw_line3(draw_list, projector, {coordinate, -2.0, 0.0}, {coordinate, 2.0, 0.0},
                   grid_color, 1.0f);
        draw_line3(draw_list, projector, {-2.0, coordinate, 0.0}, {2.0, coordinate, 0.0},
                   grid_color, 1.0f);
    }
    // World axes.
    draw_line3(draw_list, projector, {0, 0, 0}, {0.4, 0, 0}, IM_COL32(220, 70, 70, 255), 2.0f);
    draw_line3(draw_list, projector, {0, 0, 0}, {0, 0.4, 0}, IM_COL32(80, 200, 80, 255), 2.0f);
    draw_line3(draw_list, projector, {0, 0, 0}, {0, 0, 0.4}, IM_COL32(90, 130, 240, 255), 2.0f);

    // Contact planes.
    if (state.show_contact_planes) {
        for (const ContactEdit& contact : state.contacts) {
            arm::Vec3 normal{contact.normal[0], contact.normal[1], contact.normal[2]};
            const double length = arm::norm(normal);
            if (length < 1e-9) {
                continue;
            }
            normal = normal * (1.0 / length);
            const arm::Vec3 helper = std::abs(normal.z) < 0.9 ?
                arm::Vec3{0.0, 0.0, 1.0} : arm::Vec3{1.0, 0.0, 0.0};
            const arm::Vec3 tangent1 = arm::normalized(arm::cross(normal, helper));
            const arm::Vec3 tangent2 = arm::cross(normal, tangent1);
            const arm::Vec3 center = normal * contact.plane_offset;
            const double extent = 1.2;
            const ImU32 plane_color = IM_COL32(230, 180, 60, 90);
            for (int line = -3; line <= 3; ++line) {
                const double along = line * extent / 3.0;
                draw_line3(draw_list, projector,
                           center + tangent1 * along - tangent2 * extent,
                           center + tangent1 * along + tangent2 * extent, plane_color, 1.0f);
                draw_line3(draw_list, projector,
                           center + tangent2 * along - tangent1 * extent,
                           center + tangent2 * along + tangent1 * extent, plane_color, 1.0f);
            }
        }
    }

    // Links from the library's own forward kinematics.
    if (state.model &&
        state.current.joint_positions.size() == state.model->degrees_of_freedom()) {
        static const ImU32 palette[] = {
            IM_COL32(86, 180, 233, 255),  IM_COL32(230, 159, 0, 255),
            IM_COL32(0, 158, 115, 255),   IM_COL32(204, 121, 167, 255),
            IM_COL32(240, 228, 66, 255),  IM_COL32(213, 94, 0, 255),
        };
        try {
            const std::vector<double> zeros(state.model->degrees_of_freedom(), 0.0);
            const auto links = arm::detail::kinematics(
                *state.model, state.current.joint_positions, zeros, zeros);
            const auto to_vec3 = [](const arm::detail::Vector3<double>& value) {
                return arm::Vec3{value.x, value.y, value.z};
            };
            for (std::size_t index = 0; index < links.size(); ++index) {
                const auto& joint = state.model->joints()[index];
                const arm::Vec3 parent_endpoint = joint.parent >= 0 ?
                    to_vec3(links[static_cast<std::size_t>(joint.parent)].endpoint) :
                    arm::Vec3{0.0, 0.0, 0.0};
                const arm::Vec3 origin = to_vec3(links[index].origin);
                const arm::Vec3 endpoint = to_vec3(links[index].endpoint);
                const ImU32 color = palette[index % (sizeof(palette) / sizeof(palette[0]))];

                draw_line3(draw_list, projector, parent_endpoint, origin,
                           IM_COL32(130, 130, 140, 180), 1.5f);
                draw_line3(draw_list, projector, origin, endpoint, color, 4.0f);

                ImVec2 origin_screen;
                if (projector.project(origin, origin_screen)) {
                    if (joint.type == arm::Revolute) {
                        draw_list->AddCircleFilled(origin_screen, 5.0f, color);
                        draw_list->AddCircle(origin_screen, 5.0f, IM_COL32(255, 255, 255, 200));
                    } else {
                        draw_list->AddRectFilled(
                            ImVec2(origin_screen.x - 4.5f, origin_screen.y - 4.5f),
                            ImVec2(origin_screen.x + 4.5f, origin_screen.y + 4.5f), color);
                    }
                }
                if (state.show_com_markers) {
                    ImVec2 com_screen;
                    if (projector.project(to_vec3(links[index].center), com_screen)) {
                        draw_list->AddCircleFilled(com_screen, 2.5f, IM_COL32(255, 255, 255, 220));
                    }
                }
            }
            // Contact attachment points.
            for (const ContactEdit& contact : state.contacts) {
                if (contact.link < 0 || static_cast<std::size_t>(contact.link) >= links.size()) {
                    continue;
                }
                const auto& link = links[static_cast<std::size_t>(contact.link)];
                const arm::detail::Vector3<double> local{
                    contact.local_point[0], contact.local_point[1], contact.local_point[2]};
                const arm::Vec3 world = to_vec3(link.origin + link.orientation * local);
                ImVec2 point_screen;
                if (projector.project(world, point_screen)) {
                    draw_list->AddCircle(point_screen, 6.0f, IM_COL32(230, 180, 60, 255), 0, 2.0f);
                }
            }
        } catch (const std::exception&) {
            // Transiently invalid state while editing; skip drawing this frame.
        }
    }

    draw_list->PopClipRect();
    ImGui::End();
}

// ---------------------------------------------------------------------------
// Editor panels
// ---------------------------------------------------------------------------

bool input_vec3(const char* label, double* values, const char* format = "%.4f") {
    return ImGui::InputScalarN(label, ImGuiDataType_Double, values, 3, nullptr, nullptr, format);
}

void draw_model_editor(AppState& state) {
    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(430, 880), ImGuiCond_FirstUseEver);
    ImGui::Begin("Model");

    ImGui::TextUnformatted("Presets:");
    ImGui::SameLine();
    if (ImGui::SmallButton("Pendulum")) {
        load_pendulum_preset(state);
    }
    ImGui::SameLine();
    if (ImGui::SmallButton("Three-link arm")) {
        load_three_link_preset(state);
    }
    ImGui::Separator();

    if (input_vec3("Gravity (m/s^2)", state.gravity)) {
        state.model_dirty = true;
    }

    if (ImGui::CollapsingHeader("Joints", ImGuiTreeNodeFlags_DefaultOpen)) {
        int remove_index = -1;
        for (int index = 0; index < static_cast<int>(state.joints.size()); ++index) {
            JointEdit& joint = state.joints[static_cast<std::size_t>(index)];
            ImGui::PushID(index);
            const std::string header =
                "[" + std::to_string(index) + "] " + joint.name;
            if (ImGui::TreeNode("joint_node", "%s", header.c_str())) {
                bool changed = false;
                changed |= ImGui::InputText("Name", joint.name, sizeof(joint.name));
                changed |= ImGui::Combo("Type", &joint.type, "Revolute\0Prismatic\0");
                changed |= ImGui::InputInt("Parent (-2 prev, -1 world)", &joint.parent);
                changed |= input_vec3("Axis (rest frame)", joint.axis);
                changed |= input_vec3("Joint offset (parent, m)", joint.joint_offset);
                changed |= input_vec3("Rest rotation RPY (deg)", joint.rest_rpy_deg, "%.2f");
                changed |= input_vec3("Link offset (local, m)", joint.link_offset);
                changed |= ImGui::InputDouble("Mass (kg)", &joint.mass, 0.0, 0.0, "%.4f");
                changed |= input_vec3("COM offset (local, m)", joint.com_offset);
                changed |= input_vec3("Inertia diag xx,yy,zz", joint.inertia_diag);
                changed |= input_vec3("Inertia offdiag xy,xz,yz", joint.inertia_offdiag);
                if (changed) {
                    state.model_dirty = true;
                }
                if (ImGui::Button("Remove joint")) {
                    remove_index = index;
                }
                ImGui::TreePop();
            }
            ImGui::PopID();
        }
        if (remove_index >= 0) {
            state.joints.erase(state.joints.begin() + remove_index);
            // Re-point explicit parents that referenced removed/later joints.
            for (std::size_t index = 0; index < state.joints.size(); ++index) {
                int& parent = state.joints[index].parent;
                if (parent == remove_index) {
                    parent = -1;
                } else if (parent > remove_index) {
                    --parent;
                }
            }
            state.model_dirty = true;
        }
        if (ImGui::Button("Add joint")) {
            JointEdit joint;
            std::snprintf(joint.name, sizeof(joint.name), "joint_%d",
                          static_cast<int>(state.joints.size()));
            state.joints.push_back(joint);
            state.model_dirty = true;
        }
    }

    if (ImGui::CollapsingHeader("Coordinate constraints")) {
        ImGui::TextDisabled("Rows of A*q = target with stabilization");
        int remove_index = -1;
        for (int index = 0; index < static_cast<int>(state.constraints.size()); ++index) {
            ConstraintEdit& constraint = state.constraints[static_cast<std::size_t>(index)];
            ImGui::PushID(1000 + index);
            if (ImGui::TreeNode("constraint_node", "Constraint %d", index)) {
                bool changed = false;
                constraint.coefficients.resize(state.joints.size(), 0.0);
                if (!constraint.coefficients.empty()) {
                    changed |= ImGui::InputScalarN(
                        "Coefficients", ImGuiDataType_Double, constraint.coefficients.data(),
                        static_cast<int>(constraint.coefficients.size()), nullptr, nullptr, "%.3f");
                }
                changed |= ImGui::InputDouble("Target", &constraint.target, 0.0, 0.0, "%.4f");
                changed |= ImGui::InputDouble("Frequency w (1/s)", &constraint.frequency, 0.0, 0.0, "%.3f");
                changed |= ImGui::InputDouble("Damping ratio zeta", &constraint.damping_ratio, 0.0, 0.0, "%.3f");
                if (changed) {
                    state.model_dirty = true;
                }
                if (ImGui::Button("Remove constraint")) {
                    remove_index = index;
                }
                ImGui::TreePop();
            }
            ImGui::PopID();
        }
        if (remove_index >= 0) {
            state.constraints.erase(state.constraints.begin() + remove_index);
            state.model_dirty = true;
        }
        if (ImGui::Button("Add constraint")) {
            ConstraintEdit constraint;
            constraint.coefficients.assign(state.joints.size(), 0.0);
            if (!constraint.coefficients.empty()) {
                constraint.coefficients[0] = 1.0;
            }
            state.constraints.push_back(std::move(constraint));
            state.model_dirty = true;
        }
    }

    if (ImGui::CollapsingHeader("Smooth plane contacts")) {
        ImGui::TextDisabled("Frictionless smooth normal forces");
        ImGui::Checkbox("Draw contact planes", &state.show_contact_planes);
        int remove_index = -1;
        for (int index = 0; index < static_cast<int>(state.contacts.size()); ++index) {
            ContactEdit& contact = state.contacts[static_cast<std::size_t>(index)];
            ImGui::PushID(2000 + index);
            if (ImGui::TreeNode("contact_node", "Contact %d (link %d)", index, contact.link)) {
                bool changed = false;
                changed |= ImGui::InputInt("Link index", &contact.link);
                changed |= input_vec3("Local point (m)", contact.local_point);
                changed |= input_vec3("World normal", contact.normal);
                changed |= ImGui::InputDouble("Plane offset (m)", &contact.plane_offset, 0.0, 0.0, "%.4f");
                changed |= ImGui::InputDouble("Stiffness (N/m)", &contact.stiffness, 0.0, 0.0, "%.1f");
                changed |= ImGui::InputDouble("Smoothing (m)", &contact.smoothing, 0.0, 0.0, "%.5f");
                if (changed) {
                    state.model_dirty = true;
                }
                if (ImGui::Button("Remove contact")) {
                    remove_index = index;
                }
                ImGui::TreePop();
            }
            ImGui::PopID();
        }
        if (remove_index >= 0) {
            state.contacts.erase(state.contacts.begin() + remove_index);
            state.model_dirty = true;
        }
        if (ImGui::Button("Add contact")) {
            state.contacts.emplace_back();
            state.model_dirty = true;
        }
    }

    if (!state.model_error.empty()) {
        ImGui::Separator();
        ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255, 90, 90, 255));
        ImGui::TextWrapped("Model error: %s", state.model_error.c_str());
        ImGui::PopStyleColor();
    } else if (state.model) {
        ImGui::Separator();
        ImGui::Text("Model OK: %d DOF, %d constraint(s), %d contact(s)",
                    static_cast<int>(state.model->degrees_of_freedom()),
                    static_cast<int>(state.constraints.size()),
                    static_cast<int>(state.contacts.size()));
    } else {
        ImGui::Separator();
        ImGui::TextDisabled("Add a joint or load a preset to build a model.");
    }

    ImGui::End();
}

void draw_simulation_panel(AppState& state) {
    ImGui::SetNextWindowPos(ImVec2(1160, 10), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(430, 600), ImGuiCond_FirstUseEver);
    ImGui::Begin("Simulation");

    const std::size_t dof = state.joints.size();
    const bool ready = state.model != nullptr && state.model_error.empty();

    ImGui::InputDouble("Timestep dt (s)", &state.timestep, 0.0, 0.0, "%.5f");
    state.timestep = std::clamp(state.timestep, 1e-5, 0.1);
    ImGui::SliderScalar("Playback speed", ImGuiDataType_Double, &state.speed,
                        [] { static const double value = 0.05; return &value; }(),
                        [] { static const double value = 5.0; return &value; }(), "%.2fx");

    if (!ready) {
        ImGui::TextDisabled("Fix the model to enable simulation.");
    }
    ImGui::BeginDisabled(!ready);
    if (ImGui::Button(state.playing ? "Pause" : "Play", ImVec2(80, 0))) {
        state.playing = !state.playing;
        state.time_accumulator = 0.0;
    }
    ImGui::SameLine();
    if (ImGui::Button("Step", ImVec2(80, 0)) && state.rollout &&
        state.current.joint_positions.size() == dof) {
        try {
            state.current = state.rollout->step(state.current, state.torques, state.timestep);
            state.sim_time += state.timestep;
            push_history(state);
            state.sim_error.clear();
        } catch (const std::exception& error) {
            state.sim_error = error.what();
        }
    }
    ImGui::SameLine();
    if (ImGui::Button("Reset", ImVec2(80, 0))) {
        reset_simulation(state);
        state.playing = false;
    }
    ImGui::EndDisabled();
    ImGui::Text("Simulated time: %.3f s", state.sim_time);
    if (!state.sim_error.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255, 90, 90, 255));
        ImGui::TextWrapped("Simulation error: %s", state.sim_error.c_str());
        ImGui::PopStyleColor();
    }

    if (ImGui::CollapsingHeader("Initial state", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::TextDisabled("Applied on Reset");
        for (std::size_t index = 0; index < dof; ++index) {
            ImGui::PushID(static_cast<int>(3000 + index));
            ImGui::Text("%s", state.joints[index].name);
            ImGui::InputDouble("q0", &state.initial_positions[index], 0.0, 0.0, "%.4f");
            ImGui::SameLine();
            ImGui::InputDouble("v0", &state.initial_velocities[index], 0.0, 0.0, "%.4f");
            ImGui::PopID();
        }
    }

    if (ImGui::CollapsingHeader("Joint efforts (held constant)", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::InputDouble("Effort limit", &state.torque_limit, 0.0, 0.0, "%.1f");
        state.torque_limit = std::max(state.torque_limit, 0.1);
        const double limit = state.torque_limit;
        const double negative_limit = -limit;
        for (std::size_t index = 0; index < dof; ++index) {
            ImGui::PushID(static_cast<int>(4000 + index));
            const bool prismatic = state.joints[index].type == 1;
            const std::string label = std::string(state.joints[index].name) +
                (prismatic ? " (N)" : " (N-m)");
            ImGui::SliderScalar(label.c_str(), ImGuiDataType_Double, &state.torques[index],
                                &negative_limit, &limit, "%.3f");
            ImGui::PopID();
        }
        if (ImGui::Button("Zero all efforts")) {
            std::fill(state.torques.begin(), state.torques.end(), 0.0);
        }
    }

    if (ImGui::CollapsingHeader("Current state", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (state.current.joint_positions.size() == dof) {
            for (std::size_t index = 0; index < dof; ++index) {
                ImGui::Text("%s: q = %+.4f, v = %+.4f", state.joints[index].name,
                            state.current.joint_positions[index],
                            state.current.joint_velocities[index]);
            }
        } else {
            ImGui::TextDisabled("No state (reset required).");
        }
        ImGui::Checkbox("Show COM markers", &state.show_com_markers);
    }

    ImGui::End();
}

void draw_plots(AppState& state) {
    ImGui::SetNextWindowPos(ImVec2(450, 620), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(1140, 270), ImGuiCond_FirstUseEver);
    ImGui::Begin("Trajectories");
    const std::size_t dof = state.position_history.size();
    if (dof == 0 || state.position_history[0].empty()) {
        ImGui::TextDisabled("Play the simulation to record joint trajectories.");
    } else {
        const float width = std::max(120.0f, ImGui::GetContentRegionAvail().x /
                                     static_cast<float>(dof) - 10.0f);
        for (std::size_t index = 0; index < dof; ++index) {
            const char* name = index < state.joints.size() ? state.joints[index].name : "?";
            ImGui::BeginGroup();
            ImGui::Text("%s", name);
            char plot_id[64];
            std::snprintf(plot_id, sizeof(plot_id), "##pos%d", static_cast<int>(index));
            ImGui::PlotLines(plot_id, state.position_history[index].data(),
                             static_cast<int>(state.position_history[index].size()),
                             0, "position", FLT_MAX, FLT_MAX, ImVec2(width, 80));
            std::snprintf(plot_id, sizeof(plot_id), "##vel%d", static_cast<int>(index));
            ImGui::PlotLines(plot_id, state.velocity_history[index].data(),
                             static_cast<int>(state.velocity_history[index].size()),
                             0, "velocity", FLT_MAX, FLT_MAX, ImVec2(width, 80));
            ImGui::EndGroup();
            if (index + 1 < dof) {
                ImGui::SameLine();
            }
        }
    }
    ImGui::End();
}

void glfw_error_callback(int error, const char* description) {
    std::fprintf(stderr, "GLFW error %d: %s\n", error, description);
}

}  // namespace

int main() {
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit()) {
        return 1;
    }
#ifdef __APPLE__
    const char* glsl_version = "#version 150";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#else
    const char* glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
#endif
    GLFWwindow* window = glfwCreateWindow(
        1600, 900, "Differentiable Articulated Dynamics - Interactive", nullptr, nullptr);
    if (window == nullptr) {
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    AppState state;
    load_pendulum_preset(state);

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        if (state.model_dirty) {
            rebuild_solvers(state);
            if (state.model_error.empty()) {
                state.model_dirty = false;
            } else {
                state.model_dirty = false;  // wait for the next edit before retrying
            }
        }
        advance_simulation(state, ImGui::GetIO().DeltaTime);

        draw_model_editor(state);
        draw_simulation_panel(state);
        draw_viewport(state);
        draw_plots(state);

        ImGui::Render();
        int display_width = 0;
        int display_height = 0;
        glfwGetFramebufferSize(window, &display_width, &display_height);
        glViewport(0, 0, display_width, display_height);
        glClearColor(0.06f, 0.06f, 0.08f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
