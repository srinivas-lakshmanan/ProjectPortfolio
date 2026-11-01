#pragma once

#include <array>
#include <cmath>

#include "smooth_math.hpp"
#include "tire_model.hpp"
#include "vehicle_params.hpp"

namespace vehicle_model {

/// Full 4-wheel (double-track) vehicle model with 7 dynamic DOF:
/// planar body motion (vx, vy, r) + 4 wheel spin dynamics, plus global
/// pose kinematics (X, Y, psi).
///
/// Differential vehicle dynamics are supported natively: each wheel takes an
/// independent drive/brake torque, so differential drive, torque vectoring,
/// and asymmetric braking all fall out of the model.
///
/// State  x (NX = 10): [X, Y, psi, vx, vy, r, w_fl, w_fr, w_rl, w_rr]
/// Input  u (NU = 5):  [delta, T_fl, T_fr, T_rl, T_rr]
///
/// The dynamics are written as a smooth ODE x_dot = f(x, u), templated on the
/// scalar type: instantiate with double for simulation or with casadi::SX /
/// casadi::MX to build an MPC (all math is eps-regularized and
/// branch-free, so it is symbolically differentiable).
class DoubleTrackModel {
public:
    static constexpr int NX = 10;  ///< number of states
    static constexpr int NU = 5;   ///< number of inputs
    static constexpr int NW = 4;   ///< number of wheels

    /// State vector indices.
    enum StateIdx {
        IX = 0,   ///< global x position [m]
        IY,       ///< global y position [m]
        IPSI,     ///< yaw angle [rad]
        IVX,      ///< body-frame longitudinal velocity [m/s]
        IVY,      ///< body-frame lateral velocity [m/s]
        IR,       ///< yaw rate [rad/s]
        IW_FL,    ///< wheel speed front-left [rad/s]
        IW_FR,    ///< wheel speed front-right [rad/s]
        IW_RL,    ///< wheel speed rear-left [rad/s]
        IW_RR     ///< wheel speed rear-right [rad/s]
    };

    /// Input vector indices.
    enum InputIdx {
        IDELTA = 0,  ///< front road-wheel steering angle [rad]
        IT_FL,       ///< drive(+)/brake(-) torque front-left [Nm]
        IT_FR,       ///< drive(+)/brake(-) torque front-right [Nm]
        IT_RL,       ///< drive(+)/brake(-) torque rear-left [Nm]
        IT_RR        ///< drive(+)/brake(-) torque rear-right [Nm]
    };

    template <typename Scalar>
    using State = std::array<Scalar, NX>;
    template <typename Scalar>
    using Input = std::array<Scalar, NU>;

    /// Per-wheel diagnostic quantities (wheel order: FL, FR, RL, RR).
    template <typename Scalar>
    struct Diagnostics {
        std::array<Scalar, NW> kappa;    ///< slip ratios [-]
        std::array<Scalar, NW> alpha;    ///< slip angles [rad]
        std::array<Scalar, NW> Fz;       ///< vertical loads [N]
        std::array<Scalar, NW> Fx_w;     ///< longitudinal tire force, wheel frame [N]
        std::array<Scalar, NW> Fy_w;     ///< lateral tire force, wheel frame [N]
        Scalar ax;                       ///< body-frame longitudinal acceleration [m/s^2]
        Scalar ay;                       ///< body-frame lateral acceleration [m/s^2]
    };

    explicit DoubleTrackModel(VehicleParams params = VehicleParams{})
        : p_(params) {}

    const VehicleParams& params() const { return p_; }
    VehicleParams& params() { return p_; }

    /// Continuous-time dynamics x_dot = f(x, u).
    template <typename Scalar>
    State<Scalar> dynamics(const State<Scalar>& x, const Input<Scalar>& u) const {
        Diagnostics<Scalar> unused;
        return dynamics(x, u, unused);
    }

    /// Continuous-time dynamics with per-wheel diagnostics output.
    template <typename Scalar>
    State<Scalar> dynamics(const State<Scalar>& x, const Input<Scalar>& u,
                           Diagnostics<Scalar>& diag) const {
        using std::cos;
        using std::sin;

        const Scalar& psi = x[IPSI];
        const Scalar& vx  = x[IVX];
        const Scalar& vy  = x[IVY];
        const Scalar& r   = x[IR];
        const Scalar& delta = u[IDELTA];

        // Wheel contact-patch positions in the body frame (FL, FR, RL, RR).
        const double px[NW] = { p_.lf,  p_.lf, -p_.lr, -p_.lr};
        const double py[NW] = { p_.tf / 2.0, -p_.tf / 2.0,
                                p_.tr / 2.0, -p_.tr / 2.0};

        // --- Slip quantities (independent of vertical load) ---
        std::array<Scalar, NW> kappa, alpha, cos_d, sin_d;
        for (int i = 0; i < NW; ++i) {
            // Contact-point velocity in body frame: v_i = v_cg + r * k x p_i
            const Scalar vxi = vx - r * py[i];
            const Scalar vyi = vy + r * px[i];

            // Front wheels are steered by delta, rear wheels are not.
            const bool front = (i < 2);
            cos_d[i] = front ? Scalar(cos(delta)) : Scalar(1.0);
            sin_d[i] = front ? Scalar(sin(delta)) : Scalar(0.0);

            // Velocity expressed in the wheel frame.
            const Scalar vwx =  cos_d[i] * vxi + sin_d[i] * vyi;
            const Scalar vwy = -sin_d[i] * vxi + cos_d[i] * vyi;

            const Scalar& omega = x[IW_FL + i];
            kappa[i] = smooth_div(p_.Re * omega - vwx, vwx, p_.v_eps);
            alpha[i] = -atan_smooth(vwy, vwx);
        }

        // --- Vertical loads with quasi-static load transfer ---
        // Fixed-point style: pass 1 with static loads to estimate (ax, ay),
        // pass 2 with transferred loads. Smooth and CasADi-compatible.
        const double L = p_.wheelbase();
        const double Fzf_stat = p_.m * p_.g * p_.lr / (2.0 * L);
        const double Fzr_stat = p_.m * p_.g * p_.lf / (2.0 * L);

        std::array<Scalar, NW> Fz{Scalar(Fzf_stat), Scalar(Fzf_stat),
                                  Scalar(Fzr_stat), Scalar(Fzr_stat)};

        std::array<Scalar, NW> Fx_w, Fy_w;   // wheel-frame tire forces
        Scalar sumFx{0}, sumFy{0}, sumMz{0};

        for (int pass = 0; pass < 2; ++pass) {
            sumFx = Scalar(0);
            sumFy = Scalar(0);
            sumMz = Scalar(0);
            for (int i = 0; i < NW; ++i) {
                const TireForces<Scalar> f =
                    pacejka_combined(kappa[i], alpha[i], Fz[i], p_);
                Fx_w[i] = f.Fx;
                Fy_w[i] = f.Fy;

                // Rotate wheel-frame force into the body frame.
                const Scalar Fx_b = cos_d[i] * f.Fx - sin_d[i] * f.Fy;
                const Scalar Fy_b = sin_d[i] * f.Fx + cos_d[i] * f.Fy;

                sumFx += Fx_b;
                sumFy += Fy_b;
                sumMz += px[i] * Fy_b - py[i] * Fx_b;
            }

            if (pass == 0) {
                // Resistances enter the longitudinal acceleration estimate.
                const Scalar F_res = resistance_force(vx);
                const Scalar ax = (sumFx - F_res) / p_.m;
                const Scalar ay = sumFy / p_.m;

                // Longitudinal transfer (per wheel on each axle).
                const Scalar dFz_long = p_.m * p_.h_cg / (2.0 * L) * ax;
                // Lateral transfer split by roll stiffness distribution.
                const Scalar dFz_lat_f =
                    p_.roll_dist_f * p_.m * p_.h_cg / p_.tf * ay;
                const Scalar dFz_lat_r =
                    (1.0 - p_.roll_dist_f) * p_.m * p_.h_cg / p_.tr * ay;

                Fz[0] = Fzf_stat - dFz_long - dFz_lat_f;  // FL
                Fz[1] = Fzf_stat - dFz_long + dFz_lat_f;  // FR
                Fz[2] = Fzr_stat + dFz_long - dFz_lat_r;  // RL
                Fz[3] = Fzr_stat + dFz_long + dFz_lat_r;  // RR
            }
        }

        // --- Resistances (aero drag + rolling resistance), smooth-signed ---
        const Scalar F_res = resistance_force(vx);

        // --- Rigid-body dynamics ---
        const Scalar ax = (sumFx - F_res) / p_.m;
        const Scalar ay = sumFy / p_.m;

        State<Scalar> xdot;
        xdot[IX]   = vx * cos(psi) - vy * sin(psi);
        xdot[IY]   = vx * sin(psi) + vy * cos(psi);
        xdot[IPSI] = r;
        xdot[IVX]  = ax + r * vy;
        xdot[IVY]  = ay - r * vx;
        xdot[IR]   = sumMz / p_.Iz;

        // --- Wheel spin dynamics: Iw * w_dot = T - Re * Fx_wheel ---
        for (int i = 0; i < NW; ++i) {
            xdot[IW_FL + i] = (u[IT_FL + i] - p_.Re * Fx_w[i]) / p_.Iw;
        }

        diag.kappa = kappa;
        diag.alpha = alpha;
        diag.Fz = Fz;
        diag.Fx_w = Fx_w;
        diag.Fy_w = Fy_w;
        diag.ax = ax;
        diag.ay = ay;
        return xdot;
    }

    /// Convenience: state of a vehicle driving straight at speed v with
    /// wheels rolling freely (zero slip).
    template <typename Scalar = double>
    State<Scalar> straight_driving_state(double v) const {
        State<Scalar> x{};
        x[IVX] = Scalar(v);
        for (int i = 0; i < NW; ++i) {
            x[IW_FL + i] = Scalar(v / p_.Re);
        }
        return x;
    }

private:
    /// Aerodynamic drag + rolling resistance, signed smoothly with vx.
    template <typename Scalar>
    Scalar resistance_force(const Scalar& vx) const {
        const Scalar F_aero =
            0.5 * p_.rho_air * p_.Cd * p_.A_front * vx * smooth_abs(vx, p_.v_eps);
        const Scalar F_roll =
            p_.c_rr * p_.m * p_.g * smooth_sign(vx, p_.v_eps);
        return F_aero + F_roll;
    }

    /// atan(num/den) with a smooth, strictly positive denominator.
    template <typename Scalar>
    Scalar atan_smooth(const Scalar& num, const Scalar& den) const {
        using std::atan;
        return atan(smooth_div(num, den, p_.v_eps));
    }

    VehicleParams p_;
};

} // namespace vehicle_model
