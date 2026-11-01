#pragma once

#include <cmath>

#include "smooth_math.hpp"
#include "vehicle_params.hpp"

namespace vehicle_model {

/// Tire forces in the wheel frame [N].
template <typename Scalar>
struct TireForces {
    Scalar Fx;  ///< longitudinal force (wheel frame)
    Scalar Fy;  ///< lateral force (wheel frame)
};

/// Simplified Pacejka "magic formula" with combined-slip friction ellipse.
///
/// Pure-slip forces:
///   Fx0 = mu * Fz * sin(Cx * atan(Bx*k - Ex*(Bx*k - atan(Bx*k))))
///   Fy0 = mu * Fz * sin(Cy * atan(By*a - Ey*(By*a - atan(By*a))))
/// Combined slip is handled by scaling both forces onto the friction ellipse.
///
/// @param kappa  longitudinal slip ratio [-]
/// @param alpha  slip angle [rad]
/// @param Fz     vertical load [N] (clamped smoothly at 0 for wheel lift-off)
template <typename Scalar>
TireForces<Scalar> pacejka_combined(const Scalar& kappa, const Scalar& alpha,
                                    const Scalar& Fz, const VehicleParams& p) {
    using std::atan;
    using std::sin;
    using std::sqrt;

    // Smoothly clamp Fz at zero (wheel lift-off) while staying differentiable.
    const Scalar Fz_pos = 0.5 * (Fz + smooth_abs(Fz, 1.0));

    // Pure longitudinal slip
    const Scalar xk = p.Bx * kappa;
    const Scalar Fx0 = p.mu * Fz_pos * sin(p.Cx * atan(xk - p.Ex * (xk - atan(xk))));

    // Pure lateral slip
    const Scalar xa = p.By * alpha;
    const Scalar Fy0 = p.mu * Fz_pos * sin(p.Cy * atan(xa - p.Ey * (xa - atan(xa))));

    // Combined slip: scale pure-slip forces by relative slip magnitudes so the
    // resultant stays on/inside the friction ellipse (smooth approximation).
    const double eps = 1e-4;
    const Scalar s = sqrt(kappa * kappa + alpha * alpha + eps * eps);

    TireForces<Scalar> f;
    f.Fx = Fx0 * smooth_abs(kappa, eps) / s;
    f.Fy = Fy0 * smooth_abs(alpha, eps) / s;
    return f;
}

} // namespace vehicle_model
