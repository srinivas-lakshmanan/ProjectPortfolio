#pragma once

#include <cmath>
#include <cstddef>
#include <vector>

namespace arm {

class Dual {
public:
    double value = 0.0;
    std::vector<double> derivative;

    Dual() = default;
    Dual(double input) : value(input) {}
    Dual(double input, std::size_t dimensions)
        : value(input), derivative(dimensions, 0.0) {}

    static Dual variable(double input, std::size_t dimensions, std::size_t index) {
        Dual result(input, dimensions);
        result.derivative.at(index) = 1.0;
        return result;
    }
};

inline double derivative_at(const Dual& value, std::size_t index) {
    return index < value.derivative.size() ? value.derivative[index] : 0.0;
}

inline Dual operator+(const Dual& lhs, const Dual& rhs) {
    Dual result(lhs.value + rhs.value,
                lhs.derivative.size() > rhs.derivative.size()
                    ? lhs.derivative.size() : rhs.derivative.size());
    for (std::size_t index = 0; index < result.derivative.size(); ++index) {
        result.derivative[index] = derivative_at(lhs, index) + derivative_at(rhs, index);
    }
    return result;
}

inline Dual operator-(const Dual& lhs, const Dual& rhs) {
    Dual result(lhs.value - rhs.value,
                lhs.derivative.size() > rhs.derivative.size()
                    ? lhs.derivative.size() : rhs.derivative.size());
    for (std::size_t index = 0; index < result.derivative.size(); ++index) {
        result.derivative[index] = derivative_at(lhs, index) - derivative_at(rhs, index);
    }
    return result;
}

inline Dual operator-(const Dual& value) {
    Dual result(-value.value, value.derivative.size());
    for (std::size_t index = 0; index < result.derivative.size(); ++index) {
        result.derivative[index] = -value.derivative[index];
    }
    return result;
}

inline Dual operator*(const Dual& lhs, const Dual& rhs) {
    Dual result(lhs.value * rhs.value,
                lhs.derivative.size() > rhs.derivative.size()
                    ? lhs.derivative.size() : rhs.derivative.size());
    for (std::size_t index = 0; index < result.derivative.size(); ++index) {
        result.derivative[index] = derivative_at(lhs, index) * rhs.value +
            lhs.value * derivative_at(rhs, index);
    }
    return result;
}

inline Dual operator/(const Dual& lhs, const Dual& rhs) {
    Dual result(lhs.value / rhs.value,
                lhs.derivative.size() > rhs.derivative.size()
                    ? lhs.derivative.size() : rhs.derivative.size());
    const double denominator = rhs.value * rhs.value;
    for (std::size_t index = 0; index < result.derivative.size(); ++index) {
        result.derivative[index] =
            (derivative_at(lhs, index) * rhs.value - lhs.value * derivative_at(rhs, index)) /
            denominator;
    }
    return result;
}

inline Dual sin(const Dual& input) {
    Dual result(std::sin(input.value), input.derivative.size());
    const double scale = std::cos(input.value);
    for (std::size_t index = 0; index < result.derivative.size(); ++index) {
        result.derivative[index] = scale * input.derivative[index];
    }
    return result;
}

inline Dual cos(const Dual& input) {
    Dual result(std::cos(input.value), input.derivative.size());
    const double scale = -std::sin(input.value);
    for (std::size_t index = 0; index < result.derivative.size(); ++index) {
        result.derivative[index] = scale * input.derivative[index];
    }
    return result;
}

}  // namespace arm