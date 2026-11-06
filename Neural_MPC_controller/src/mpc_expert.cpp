#include "vehicle_model/tracking_mpc.hpp"

#include <iomanip>
#include <iostream>
#include <sstream>

namespace {

double number(const std::string& text) {
    std::size_t used = 0;
    const double value = std::stod(text, &used);
    if (used != text.size() || !std::isfinite(value))
        throw std::invalid_argument("Expected a finite number: " + text);
    return value;
}

int integer(const std::string& text) {
    const double value = number(text);
    if (value < 1.0 || value > 10000.0 || value != std::floor(value))
        throw std::invalid_argument("Expected an integer in [1,10000]: " + text);
    return static_cast<int>(value);
}

vehicle_model::TrackingMpcConfig parse(int argc, char** argv) {
    vehicle_model::TrackingMpcConfig config;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (i + 1 == argc) throw std::invalid_argument("Missing value for " + key);
        const std::string value = argv[++i];
        if (key == "--sample-time") config.sample_time = number(value);
        else if (key == "--horizon") config.horizon = integer(value);
        else if (key == "--rk4-substeps") config.rk4_substeps = integer(value);
        else if (key == "--max-iterations") config.max_iterations = integer(value);
        else if (key == "--jit") {
            if (value != "true" && value != "false")
                throw std::invalid_argument("--jit expects true or false");
            config.jit = value == "true";
        } else throw std::invalid_argument("Unknown option: " + key);
    }
    config.validate();
    return config;
}

template <std::size_t N>
void read(std::istringstream& stream, std::array<double, N>& values) {
    for (double& value : values) {
        if (!(stream >> value) || !std::isfinite(value))
            throw std::invalid_argument("Request contains missing or nonfinite values");
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const auto config = parse(argc, argv);
        const vehicle_model::DoubleTrackModel model;
        vehicle_model::TrackingMpc controller(model, config);
        std::cout << std::setprecision(17) << "READY\n" << std::flush;
        std::string line;
        while (std::getline(std::cin, line)) {
            if (line == "RESET") {
                controller.reset_warm_start();
                std::cout << "OK\n" << std::flush;
                continue;
            }
            std::istringstream stream(line);
            std::string command;
            stream >> command;
            if (command != "SOLVE") throw std::invalid_argument("Expected SOLVE or RESET");
            vehicle_model::TrackingMpc::State state;
            vehicle_model::TrackingMpc::Input previous;
            read(stream, state);
            read(stream, previous);
            vehicle_model::TrackingReference reference;
            reference.states.resize(config.horizon + 1);
            reference.inputs.resize(config.horizon);
            for (auto& value : reference.states) read(stream, value);
            for (auto& value : reference.inputs) read(stream, value);
            std::string extra;
            if (stream >> extra) throw std::invalid_argument("Request has extra fields");
            const auto result = controller.solve(state, previous, reference);
            std::cout << "OK";
            for (double value : result.input) std::cout << ' ' << value;
            std::cout << ' ' << result.iterations << ' ' << result.solve_seconds
                      << ' ' << result.constraint_violation << '\n' << std::flush;
        }
        if (!std::cin.eof()) throw std::runtime_error("Failed reading expert requests");
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        std::cout << "ERROR " << error.what() << '\n' << std::flush;
        return EXIT_FAILURE;
    }
}
