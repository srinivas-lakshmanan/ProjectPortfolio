#include "vehicle_model/maneuver_reference.hpp"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>

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
    std::string output = "mpc_training_data.csv";
    int episodes = 1;
    unsigned int seed = 1;
    double speed_min = 6.0;
    double speed_max = 14.0;
    double step_angle_min = 0.5 * pi / 180.0;
    double step_angle_max = 2.5 * pi / 180.0;
    double lane_width_min = 1.5;
    double lane_width_max = 3.5;
    double initial_y_error_max = 0.25;
};

void help() {
    std::cout <<
        "Generate MPC expert training data (illustrative, not standards-certified)\n"
        "Usage: mpc_maneuver_data_dump [--option value ...]\n"
        "  --scenario all|step-steer|double-lane-change   (all)\n"
        "  --episodes COUNT                              (1)\n"
        "  --seed INTEGER                                 (1)\n"
        "  --output-file CSV_PATH                         (mpc_training_data.csv)\n"
        "  --speed-min M/S                               (6)\n"
        "  --speed-max M/S                               (14)\n"
        "  --step-angle-min-deg DEG                      (0.5)\n"
        "  --step-angle-max-deg DEG                      (2.5)\n"
        "  --lane-width-min M                            (1.5)\n"
        "  --lane-width-max M                            (3.5)\n"
        "  --initial-y-error-max M                       (0.25; sampled symmetrically)\n"
        "  --start-time S                                (1)\n"
        "  --step-duration S                             (6)\n"
        "  --lane-change-time S                          (2.5 per transition)\n"
        "  --lane-hold-time S                            (1)\n"
        "  --settle-time S                               (2)\n"
        "  --sample-time S                               (0.05)\n"
        "  --horizon STEPS                               (12)\n"
        "  --rk4-substeps COUNT                          (50)\n"
        "  --max-iterations COUNT                        (100)\n"
        "  --jit true|false                              (true)\n"
        "  --jit-compiler COMMAND                        (gcc)\n"
        "Each row contains the measured state, prior input, full reference preview,\n"
        "the first optimized MPC input, and solver metadata. Split train/test by episode.\n";
}

double number(const std::string& text) {
    std::size_t used = 0;
    const double value = std::stod(text, &used);
    if (used != text.size() || !std::isfinite(value))
        throw std::invalid_argument("Expected a finite number, got: " + text);
    return value;
}

int positive_integer(const std::string& text) {
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
        else if (key == "--episodes") result.episodes = positive_integer(value);
        else if (key == "--seed") {
            const double parsed = number(value);
            if (parsed < 0.0 || parsed > 4294967295.0 || std::floor(parsed) != parsed)
                throw std::invalid_argument("--seed must be an integer in [0,4294967295]");
            result.seed = static_cast<unsigned int>(parsed);
        }
        else if (key == "--output-file") result.output = value;
        else if (key == "--speed-min") result.speed_min = number(value);
        else if (key == "--speed-max") result.speed_max = number(value);
        else if (key == "--step-angle-min-deg") result.step_angle_min = number(value) * pi / 180.0;
        else if (key == "--step-angle-max-deg") result.step_angle_max = number(value) * pi / 180.0;
        else if (key == "--lane-width-min") result.lane_width_min = number(value);
        else if (key == "--lane-width-max") result.lane_width_max = number(value);
        else if (key == "--initial-y-error-max") result.initial_y_error_max = number(value);
        else if (key == "--start-time") result.maneuver.start_time = number(value);
        else if (key == "--step-duration") result.maneuver.step_duration = number(value);
        else if (key == "--lane-change-time") result.maneuver.lane_change_time = number(value);
        else if (key == "--lane-hold-time") result.maneuver.lane_hold_time = number(value);
        else if (key == "--settle-time") result.maneuver.settle_time = number(value);
        else if (key == "--sample-time") result.mpc.sample_time = number(value);
        else if (key == "--horizon") result.mpc.horizon = positive_integer(value);
        else if (key == "--rk4-substeps") result.mpc.rk4_substeps = positive_integer(value);
        else if (key == "--max-iterations") result.mpc.max_iterations = positive_integer(value);
        else if (key == "--jit") {
            if (value != "true" && value != "false")
                throw std::invalid_argument("--jit expects true or false");
            result.mpc.jit = value == "true";
        }
        else if (key == "--jit-compiler") result.mpc.jit_compiler = value;
        else throw std::invalid_argument("Unknown option: " + key);
    }

    result.mpc.validate();
    result.maneuver.validate();
    if (result.output.empty()) throw std::invalid_argument("Output file must not be empty");
    if (result.scenario != "all" && result.scenario != "step-steer" &&
        result.scenario != "double-lane-change")
        throw std::invalid_argument("Scenario must be all, step-steer, or double-lane-change");
    if (!std::isfinite(result.speed_min) || !std::isfinite(result.speed_max) ||
        result.speed_min <= 0.0 || result.speed_max < result.speed_min)
        throw std::invalid_argument("Speed range must be finite, positive, and ordered");
    if (!std::isfinite(result.step_angle_min) || !std::isfinite(result.step_angle_max) ||
        result.step_angle_min <= 0.0 || result.step_angle_max < result.step_angle_min ||
        result.step_angle_max > result.mpc.steering_limit)
        throw std::invalid_argument("Step-angle range must be positive, ordered, and within steering limits");
    if (!std::isfinite(result.lane_width_min) || !std::isfinite(result.lane_width_max) ||
        result.lane_width_min <= 0.0 || result.lane_width_max < result.lane_width_min)
        throw std::invalid_argument("Lane-width range must be finite, positive, and ordered");
    if (!std::isfinite(result.initial_y_error_max) || result.initial_y_error_max < 0.0)
        throw std::invalid_argument("Initial lateral error maximum must be finite and nonnegative");
    return result;
}

double uniform(std::mt19937& rng, double minimum, double maximum) {
    return std::uniform_real_distribution<double>(minimum, maximum)(rng);
}

void write_header(std::ostream& csv, int horizon) {
    csv << "episode_id,maneuver,time_s,speed_m_s,step_angle_rad,lane_width_m,initial_y_error_m";
    constexpr const char* state_names[] = {
        "X", "Y", "psi", "vx", "vy", "r", "w_fl", "w_fr", "w_rl", "w_rr"};
    constexpr const char* input_names[] = {"delta", "T_fl", "T_fr", "T_rl", "T_rr"};
    for (const char* name : state_names) csv << ",state_" << name;
    for (const char* name : input_names) csv << ",previous_" << name;
    for (int k = 0; k <= horizon; ++k)
        for (const char* name : state_names)
            csv << ",reference_" << k << '_' << name;
    for (int k = 0; k < horizon; ++k)
        for (const char* name : input_names)
            csv << ",reference_input_" << k << '_' << name;
    for (const char* name : input_names) csv << ",expert_" << name;
    csv << ",solver_iterations,solve_seconds,constraint_violation\n";
}

void write_row(std::ostream& csv, int episode, const std::string& maneuver_name,
               double time, double speed, double step_angle, double lane_width,
               double initial_y_error, const State& state, const Input& previous,
               const vehicle_model::TrackingReference& reference,
               const vehicle_model::TrackingMpcResult& result) {
    csv << episode << ',' << maneuver_name << ',' << time << ',' << speed << ','
        << step_angle << ',' << lane_width << ',' << initial_y_error;
    for (double value : state) csv << ',' << value;
    for (double value : previous) csv << ',' << value;
    for (const auto& reference_state : reference.states)
        for (double value : reference_state) csv << ',' << value;
    for (const auto& reference_input : reference.inputs)
        for (double value : reference_input) csv << ',' << value;
    for (double value : result.input) csv << ',' << value;
    csv << ',' << result.iterations << ',' << result.solve_seconds
        << ',' << result.constraint_violation << '\n';
}

void run_episode(int episode, const Options& options, std::mt19937& rng,
                 const DoubleTrackModel& model, vehicle_model::TrackingMpc& controller,
                 std::ostream& csv) {
    const bool step_steer = options.scenario == "step-steer" ||
        (options.scenario == "all" && std::uniform_int_distribution<int>(0, 1)(rng) == 0);
    const Maneuver maneuver = step_steer ? Maneuver::StepSteer : Maneuver::DoubleLaneChange;
    const std::string maneuver_name = step_steer ? "step-steer" : "double-lane-change";
    auto settings = options.maneuver;
    settings.speed = uniform(rng, options.speed_min, options.speed_max);
    settings.initial_y_error = uniform(rng, -options.initial_y_error_max,
                                       options.initial_y_error_max);
    settings.step_angle = uniform(rng, options.step_angle_min, options.step_angle_max) *
        (std::uniform_int_distribution<int>(0, 1)(rng) == 0 ? -1.0 : 1.0);
    settings.lane_width = uniform(rng, options.lane_width_min, options.lane_width_max) *
        (std::uniform_int_distribution<int>(0, 1)(rng) == 0 ? -1.0 : 1.0);

    const double cruise_torque = vehicle_model::detail::cruise_torque(model.params(), settings.speed);
    if (std::abs(cruise_torque) > options.mpc.torque_limit)
        throw std::invalid_argument("Sampled speed requires cruise torque outside configured torque limits");
    const auto reference = vehicle_model::make_maneuver_reference(
        model, maneuver, settings, options.mpc);
    State state = reference.states.front();
    state[DoubleTrackModel::IY] += settings.initial_y_error;
    const double torque = vehicle_model::detail::cruise_torque(model.params(), settings.speed);
    Input previous{0.0, torque, torque, torque, torque};
    controller.reset_warm_start();
    std::cout << "Episode " << episode << '/' << options.episodes << ": "
              << maneuver_name << ", " << reference.simulation_steps
              << " decisions at " << settings.speed << " m/s\n" << std::flush;

    for (int k = 0; k < reference.simulation_steps; ++k) {
        const auto preview = reference.window(k, options.mpc.horizon);
        vehicle_model::TrackingMpcResult result;
        try {
            result = controller.solve(state, previous, preview);
        } catch (const std::exception& error) {
            throw std::runtime_error("Episode " + std::to_string(episode) + " at t=" +
                std::to_string(k * options.mpc.sample_time) + ": " + error.what());
        }
        write_row(csv, episode, maneuver_name, k * options.mpc.sample_time,
                  settings.speed, settings.step_angle, settings.lane_width,
                  settings.initial_y_error, state, previous, preview, result);
        for (int i = 0; i < options.mpc.rk4_substeps; ++i) {
            state = vehicle_model::rk4_step<DoubleTrackModel, double>(
                model, state, result.input, options.mpc.sample_time / options.mpc.rk4_substeps);
            for (double value : state) {
                if (!std::isfinite(value))
                    throw std::runtime_error("Episode " + std::to_string(episode) +
                        ": plant integration became nonfinite");
            }
        }
        previous = result.input;
    }
    csv.flush();
    std::cout << "  wrote " << reference.simulation_steps << " labeled decisions\n" << std::flush;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse(argc, argv);
        std::mt19937 rng(options.seed);
        const DoubleTrackModel model;
        vehicle_model::TrackingMpc controller(model, options.mpc);
        std::ofstream csv;
        csv.exceptions(std::ios::failbit | std::ios::badbit);
        csv.open(options.output, std::ios::out | std::ios::trunc);
        csv << std::setprecision(12);
        write_header(csv, options.mpc.horizon);
        for (int episode = 1; episode <= options.episodes; ++episode)
            run_episode(episode, options, rng, model, controller, csv);
        csv.close();
        std::cout << "Dataset saved to " << options.output << '\n';
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
