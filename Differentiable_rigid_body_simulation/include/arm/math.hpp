#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace arm {

struct Vec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;

    Vec3 operator+(const Vec3& other) const {
        return {x + other.x, y + other.y, z + other.z};
    }

    Vec3 operator-(const Vec3& other) const {
        return {x - other.x, y - other.y, z - other.z};
    }

    Vec3 operator*(double scalar) const {
        return {x * scalar, y * scalar, z * scalar};
    }
};

inline double dot(const Vec3& lhs, const Vec3& rhs) {
    return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
}

inline Vec3 cross(const Vec3& lhs, const Vec3& rhs) {
    return {
        lhs.y * rhs.z - lhs.z * rhs.y,
        lhs.z * rhs.x - lhs.x * rhs.z,
        lhs.x * rhs.y - lhs.y * rhs.x
    };
}

inline double norm(const Vec3& value) {
    return std::sqrt(dot(value, value));
}

inline Vec3 normalized(const Vec3& value) {
    const double length = norm(value);
    if (length <= 1e-12) {
        throw std::invalid_argument("Joint axis must be non-zero");
    }
    return value * (1.0 / length);
}

struct Mat3 {
    std::array<double, 9> values{
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0
    };

    static Mat3 zero() {
        return {{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
    }

    Vec3 operator*(const Vec3& value) const {
        return {
            values[0] * value.x + values[1] * value.y + values[2] * value.z,
            values[3] * value.x + values[4] * value.y + values[5] * value.z,
            values[6] * value.x + values[7] * value.y + values[8] * value.z
        };
    }

    Mat3 operator*(const Mat3& other) const {
        Mat3 result{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
        for (std::size_t row = 0; row < 3; ++row) {
            for (std::size_t column = 0; column < 3; ++column) {
                for (std::size_t inner = 0; inner < 3; ++inner) {
                    result.values[row * 3 + column] +=
                        values[row * 3 + inner] * other.values[inner * 3 + column];
                }
            }
        }
        return result;
    }

    Mat3 transpose() const {
        return {{
            values[0], values[3], values[6],
            values[1], values[4], values[7],
            values[2], values[5], values[8]
        }};
    }
};

inline Mat3 rotation_matrix(const Vec3& unit_axis, double angle) {
    const double x = unit_axis.x;
    const double y = unit_axis.y;
    const double z = unit_axis.z;
    const double cosine = std::cos(angle);
    const double sine = std::sin(angle);
    const double one_minus_cosine = 1.0 - cosine;

    return {{
        cosine + x * x * one_minus_cosine,
        x * y * one_minus_cosine - z * sine,
        x * z * one_minus_cosine + y * sine,
        y * x * one_minus_cosine + z * sine,
        cosine + y * y * one_minus_cosine,
        y * z * one_minus_cosine - x * sine,
        z * x * one_minus_cosine - y * sine,
        z * y * one_minus_cosine + x * sine,
        cosine + z * z * one_minus_cosine
    }};
}

}  // namespace arm