#include "vehicle_model/maneuver_reference.hpp"

#include <fstream>
#include <iomanip>
#include <iostream>

namespace {

using vehicle_model::DoubleTrackModel;
using vehicle_model::Maneuver;
using State = DoubleTrackModel::State<double>;
using Input = DoubleTrackModel::Input<double>;
constexpr double pi = 3.14159265358979323846;

struct Options {
    vehicle_model::TrackingMpcConfig mpc;
    vehicle_model::ManeuverConfig maneuver;
    std::string scenario = "all";
    std::string output = ".";
    double max_lateral_error = 0.5;
    double max_rms_lateral_error = 0.25;
    double max_speed_error = 1.0;
    double max_yaw_error = 0.15;
};

void help() {
    std::cout <<
        "Closed-loop vehicle MPC maneuvers (illustrative, not standards-certified)\n"
        "Usage: mpc_maneuvers [--option value ...]\n"
        "  --scenario all|step-steer|double-lane-change   (all)\n"
        "  --output-dir EXISTING_DIRECTORY               (current directory)\n"
        "  --speed M/S                                   (10)\n"
        "  --start-time S                                (1)\n"
        "  --step-angle-deg DEG                          (1.4324)\n"
        "  --step-duration S                             (6)\n"
        "  --lane-width M                                (3)\n"
        "  --lane-change-time S                          (2.5 per transition)\n"
        "  --lane-hold-time S                            (1)\n"
        "  --settle-time S                               (2)\n"
        "  --initial-y-error M                           (0)\n"
        "  --sample-time S                               (0.05)\n"
        "  --horizon STEPS                               (12)\n"
        "  --rk4-substeps COUNT                          (50)\n"
        "  --steering-limit-deg DEG                      (8.5944)\n"
        "  --steering-rate-deg-s DEG/S                    (17.1887)\n"
        "  --torque-limit NM                             (200 per wheel)\n"
        "  --torque-rate NM/S                            (1500 per wheel)\n"
        "  --max-iterations COUNT                        (100)\n"
        "  --jit true|false                              (true)\n"
        "  --jit-compiler COMMAND                        (gcc)\n"
        "  --max-lateral-error M                         (0.5)\n"
        "  --max-rms-lateral-error M                     (0.25)\n"
        "  --max-speed-error M/S                         (1)\n"
        "  --max-yaw-error RAD                           (0.15)\n"
        "Fish-hook is excluded. Solver/finite-state/limit/tracking failures exit nonzero.\n";
}

double number(const std::string& text) {
    std::size_t used = 0;
    const double value = std::stod(text, &used);
    if (used != text.size() || !std::isfinite(value))
        throw std::invalid_argument("Expected a finite number, got: " + text);
    return value;
}

int integer(const std::string& text) {
    const double value = number(text);
    if (value < 1.0 || value > 10000.0 || std::floor(value) != value)
        throw std::invalid_argument("Expected an integer in [1,10000], got: " + text);
    return static_cast<int>(value);
}

Options parse(int argc, char** argv) {
    Options result;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--help") {
            help();
            std::exit(EXIT_SUCCESS);
        }
        if (i + 1 == argc) throw std::invalid_argument("Missing value for " + key);
        const std::string value = argv[++i];
        if (key == "--scenario") result.scenario = value;
        else if (key == "--output-dir") result.output = value;
        else if (key == "--speed") result.maneuver.speed = number(value);
        else if (key == "--start-time") result.maneuver.start_time = number(value);
        else if (key == "--step-angle-deg") result.maneuver.step_angle = number(value) * pi / 180.0;
        else if (key == "--step-duration") result.maneuver.step_duration = number(value);
        else if (key == "--lane-width") result.maneuver.lane_width = number(value);
        else if (key == "--lane-change-time") result.maneuver.lane_change_time = number(value);
        else if (key == "--lane-hold-time") result.maneuver.lane_hold_time = number(value);
        else if (key == "--settle-time") result.maneuver.settle_time = number(value);
        else if (key == "--initial-y-error") result.maneuver.initial_y_error = number(value);
        else if (key == "--sample-time") result.mpc.sample_time = number(value);
        else if (key == "--horizon") result.mpc.horizon = integer(value);
        else if (key == "--rk4-substeps") result.mpc.rk4_substeps = integer(value);
        else if (key == "--steering-limit-deg") result.mpc.steering_limit = number(value) * pi / 180.0;
        else if (key == "--steering-rate-deg-s") result.mpc.steering_rate_limit = number(value) * pi / 180.0;
        else if (key == "--torque-limit") result.mpc.torque_limit = number(value);
        else if (key == "--torque-rate") result.mpc.torque_rate_limit = number(value);
        else if (key == "--max-iterations") result.mpc.max_iterations = integer(value);
        else if (key == "--jit") {
            if (value != "true" && value != "false")
                throw std::invalid_argument("--jit expects true or false");
            result.mpc.jit = value == "true";
        }
        else if (key == "--jit-compiler") result.mpc.jit_compiler = value;
        else if (key == "--max-lateral-error") result.max_lateral_error = number(value);
        else if (key == "--max-rms-lateral-error") result.max_rms_lateral_error = number(value);
        else if (key == "--max-speed-error") result.max_speed_error = number(value);
        else if (key == "--max-yaw-error") result.max_yaw_error = number(value);
        else throw std::invalid_argument("Unknown option: " + key);
    }
    result.mpc.validate();
    result.maneuver.validate();
    if (result.output.empty())
        throw std::invalid_argument("Output directory must not be empty");
    if (result.scenario != "all" && result.scenario != "step-steer" &&
        result.scenario != "double-lane-change")
        throw std::invalid_argument("Scenario must be all, step-steer, or double-lane-change");
    for (double limit : {result.max_lateral_error, result.max_rms_lateral_error,
                         result.max_speed_error, result.max_yaw_error}) {
        if (limit <= 0.0) throw std::invalid_argument("Tracking thresholds must be positive");
    }
    if (result.scenario != "double-lane-change" &&
        std::abs(result.maneuver.step_angle) > result.mpc.steering_limit)
        throw std::invalid_argument("Step angle exceeds the MPC steering limit");
    return result;
}

struct Metrics {
    double max_y = 0.0, sum_y_squared = 0.0, max_speed = 0.0, max_yaw = 0.0;
    double max_delta = 0.0, max_torque = 0.0, max_rate = 0.0, max_torque_rate = 0.0;
    double max_violation = 0.0, solve_seconds = 0.0;
    double min_load = std::numeric_limits<double>::infinity();
    int samples = 0;

    void observe(const State& x, const State& r) {
        const double ey = x[DoubleTrackModel::IY] - r[DoubleTrackModel::IY];
        const double psi = x[DoubleTrackModel::IPSI] - r[DoubleTrackModel::IPSI];
        max_y = std::max(max_y, std::abs(ey));
        sum_y_squared += ey * ey;
        max_speed = std::max(max_speed,
            std::abs(x[DoubleTrackModel::IVX] - r[DoubleTrackModel::IVX]));
        max_yaw = std::max(max_yaw, std::abs(std::atan2(std::sin(psi), std::cos(psi))));
        ++samples;
    }

    double rms() const { return std::sqrt(sum_y_squared / samples); }
};

void run(const Options& options, Maneuver maneuver, const std::string& name) {
    const DoubleTrackModel model;
    const auto reference = vehicle_model::make_maneuver_reference(
        model, maneuver, options.maneuver, options.mpc);
    std::cout << name << ": " << reference.simulation_steps << " control intervals\n" << std::flush;
    State x = reference.states.front();
    x[DoubleTrackModel::IY] += options.maneuver.initial_y_error;
    const double torque = vehicle_model::detail::cruise_torque(model.params(), options.maneuver.speed);
    Input previous{0.0, torque, torque, torque, torque};
    if (std::abs(torque) > options.mpc.torque_limit)
        throw std::invalid_argument("Cruise torque exceeds configured actuator limit");

#ifdef _WIN32
    const std::string path = options.output + "\\" + name + ".csv";
#else
    const std::string path = options.output + "/" + name + ".csv";
#endif
    std::ofstream csv;
    csv.exceptions(std::ios::failbit | std::ios::badbit);
    try {
        csv.open(path);
    } catch (const std::ios_base::failure& error) {
        throw std::runtime_error("Cannot open trajectory " + path + ": " + error.what());
    }
    csv << std::setprecision(12);
    csv << "t,X,Y,psi,vx,vy,r,w_fl,w_fr,w_rl,w_rr,"
           "ref_X,ref_Y,ref_psi,ref_vx,ref_vy,ref_r,ref_w_fl,ref_w_fr,ref_w_rl,ref_w_rr,"
           "delta,T_fl,T_fr,T_rl,T_rr,ax,ay,"
           "kappa_fl,kappa_fr,kappa_rl,kappa_rr,alpha_fl,alpha_fr,alpha_rl,alpha_rr,"
           "Fz_fl,Fz_fr,Fz_rl,Fz_rr,Fx_fl,Fx_fr,Fx_rl,Fx_rr,Fy_fl,Fy_fr,Fy_rl,Fy_rr,"
           "lateral_error,yaw_error,speed_error,solver_iterations,solve_seconds,constraint_violation\n";
    vehicle_model::TrackingMpc controller(model, options.mpc);
    Metrics metrics;
    auto row = [&](int k, const Input& applied, const vehicle_model::TrackingMpcResult& result) {
        DoubleTrackModel::Diagnostics<double> d;
        model.dynamics(x, applied, d);
        if (!std::isfinite(d.ax) || !std::isfinite(d.ay))
            throw std::runtime_error(name + ": nonfinite acceleration");
        for (double load : d.Fz) metrics.min_load = std::min(metrics.min_load, load);
        const auto& r = reference.states[k];
        csv << k * options.mpc.sample_time;
        for (double value : x) csv << ',' << value;
        for (double value : r) csv << ',' << value;
        for (double value : applied) csv << ',' << value;
        csv << ',' << d.ax << ',' << d.ay;
        for (const auto* values : {&d.kappa, &d.alpha, &d.Fz, &d.Fx_w, &d.Fy_w})
            for (double value : *values) {
                if (!std::isfinite(value)) throw std::runtime_error(name + ": nonfinite diagnostics");
                csv << ',' << value;
            }
        const double epsi = x[DoubleTrackModel::IPSI] - r[DoubleTrackModel::IPSI];
        csv << ',' << x[DoubleTrackModel::IY] - r[DoubleTrackModel::IY]
            << ',' << std::atan2(std::sin(epsi), std::cos(epsi))
            << ',' << x[DoubleTrackModel::IVX] - r[DoubleTrackModel::IVX]
            << ',' << result.iterations << ',' << result.solve_seconds
            << ',' << result.constraint_violation << '\n';
    };
    metrics.observe(x, reference.states.front());
    row(0, previous, {previous, 0, 0.0, 0.0});
    for (int k = 0; k < reference.simulation_steps; ++k) {
        vehicle_model::TrackingMpcResult result;
        try {
            result = controller.solve(x, previous, reference.window(k, options.mpc.horizon));
        } catch (const std::exception& error) {
            throw std::runtime_error(name + " at t=" +
                std::to_string(k * options.mpc.sample_time) + ": " + error.what());
        }
        const auto& u = result.input;
        metrics.max_delta = std::max(metrics.max_delta, std::abs(u[0]));
        metrics.max_rate = std::max(metrics.max_rate,
            std::abs(u[0] - previous[0]) / options.mpc.sample_time);
        for (int i = 1; i < DoubleTrackModel::NU; ++i) {
            metrics.max_torque = std::max(metrics.max_torque, std::abs(u[i]));
            metrics.max_torque_rate = std::max(metrics.max_torque_rate,
                std::abs(u[i] - previous[i]) / options.mpc.sample_time);
        }
        metrics.max_violation = std::max(metrics.max_violation, result.constraint_violation);
        metrics.solve_seconds += result.solve_seconds;
        for (int i = 0; i < options.mpc.rk4_substeps; ++i) {
            x = vehicle_model::rk4_step<DoubleTrackModel, double>(
                model, x, u, options.mpc.sample_time / options.mpc.rk4_substeps);
            for (double value : x) {
                if (!std::isfinite(value))
                    throw std::runtime_error(name + ": plant integration became nonfinite");
            }
        }
        previous = u;
        metrics.observe(x, reference.states[k + 1]);
        row(k + 1, u, result);
    }
    csv.close();
    const double bound_tolerance = options.mpc.feasibility_tolerance;
    const bool passed = metrics.max_y <= options.max_lateral_error &&
        metrics.rms() <= options.max_rms_lateral_error &&
        metrics.max_speed <= options.max_speed_error && metrics.max_yaw <= options.max_yaw_error &&
        metrics.max_delta <= options.mpc.steering_limit + bound_tolerance &&
        metrics.max_torque <= options.mpc.torque_limit + bound_tolerance &&
        metrics.max_rate <= options.mpc.steering_rate_limit + bound_tolerance / options.mpc.sample_time &&
        metrics.max_torque_rate <= options.mpc.torque_rate_limit + bound_tolerance / options.mpc.sample_time;
    std::ofstream summary;
    summary.exceptions(std::ios::failbit | std::ios::badbit);
#ifdef _WIN32
    summary.open(options.output + "\\" + name + "_summary.csv");
#else
    summary.open(options.output + "/" + name + "_summary.csv");
#endif
    summary << std::setprecision(12)
            << "passed,speed,sample_time,horizon,rk4_substeps,start_time,step_angle_rad,"
               "step_duration,lane_width,lane_change_time,lane_hold_time,settle_time,"
               "initial_y_error,max_lateral_error,rms_lateral_error,max_yaw_error,"
               "max_speed_error,max_steer_rad,max_torque_nm,max_steer_rate_rad_s,"
               "max_torque_rate_nm_s,min_normal_load_n,max_constraint_violation,"
               "mean_solve_seconds,lateral_threshold,rms_lateral_threshold,"
               "yaw_threshold,speed_threshold,steering_limit_rad,torque_limit_nm,"
               "steering_rate_limit_rad_s,torque_rate_limit_nm_s\n"
            << passed << ',' << options.maneuver.speed << ',' << options.mpc.sample_time
            << ',' << options.mpc.horizon << ',' << options.mpc.rk4_substeps
            << ',' << options.maneuver.start_time << ',' << options.maneuver.step_angle
            << ',' << options.maneuver.step_duration << ',' << options.maneuver.lane_width
            << ',' << options.maneuver.lane_change_time << ',' << options.maneuver.lane_hold_time
            << ',' << options.maneuver.settle_time << ',' << options.maneuver.initial_y_error
            << ',' << metrics.max_y << ',' << metrics.rms() << ',' << metrics.max_yaw
            << ',' << metrics.max_speed << ',' << metrics.max_delta << ',' << metrics.max_torque
            << ',' << metrics.max_rate << ',' << metrics.max_torque_rate
            << ',' << metrics.min_load << ',' << metrics.max_violation
            << ',' << metrics.solve_seconds / reference.simulation_steps
            << ',' << options.max_lateral_error << ',' << options.max_rms_lateral_error
            << ',' << options.max_yaw_error << ',' << options.max_speed_error
            << ',' << options.mpc.steering_limit << ',' << options.mpc.torque_limit
            << ',' << options.mpc.steering_rate_limit << ',' << options.mpc.torque_rate_limit << '\n';
    summary.close();
    std::cout << std::fixed << std::setprecision(4) << name
              << ": max lateral=" << metrics.max_y << " m, RMS lateral=" << metrics.rms()
              << " m, max yaw=" << metrics.max_yaw << " rad, max speed=" << metrics.max_speed
              << " m/s\n  max steer=" << metrics.max_delta << " rad, max torque="
              << metrics.max_torque << " Nm, max steer rate=" << metrics.max_rate
              << " rad/s, max torque rate=" << metrics.max_torque_rate
              << " Nm/s, constraint violation=" << metrics.max_violation
              << ", min normal load=" << metrics.min_load << " N"
              << "\n  mean solve=" << metrics.solve_seconds / reference.simulation_steps
              << " s, CSV=" << path << '\n';
    if (!passed) throw std::runtime_error(name + ": tracking or actuator acceptance thresholds exceeded");
    std::cout << "[PASS] " << name << '\n';
}

} // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse(argc, argv);
        if (options.scenario == "all" || options.scenario == "step-steer")
            run(options, Maneuver::StepSteer, "step_steer_mpc");
        if (options.scenario == "all" || options.scenario == "double-lane-change")
            run(options, Maneuver::DoubleLaneChange, "double_lane_change_mpc");
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
