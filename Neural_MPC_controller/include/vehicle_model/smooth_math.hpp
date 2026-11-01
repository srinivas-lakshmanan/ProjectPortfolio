#pragma once

#include <cmath>

namespace vehicle_model {

// Smooth, everywhere-differentiable helpers. All functions are templated on
// the scalar type and use unqualified calls (with std:: fallbacks via ADL)
// so they work with double today and casadi::SX / casadi::MX later.

/// Smooth approximation of |x|: sqrt(x^2 + eps^2).
template <typename Scalar>
Scalar smooth_abs(const Scalar& x, double eps) {
    using std::sqrt;
    return sqrt(x * x + eps * eps);
}

/// Smooth sign(x) ~ x / sqrt(x^2 + eps^2).
template <typename Scalar>
Scalar smooth_sign(const Scalar& x, double eps) {
    return x / smooth_abs(x, eps);
}

/// Division with a strictly positive, smooth denominator: num / sqrt(den^2 + eps^2).
template <typename Scalar>
Scalar smooth_div(const Scalar& num, const Scalar& den, double eps) {
    return num / smooth_abs(den, eps);
}

} // namespace vehicle_model
