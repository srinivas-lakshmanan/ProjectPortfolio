#pragma once

#include <array>

namespace vehicle_model {

/// Classic 4th-order Runge-Kutta step for any model exposing
/// `State dynamics(State, Input)`. Templated on the scalar type so the same
/// integrator can be used for double simulation or for building a discrete
/// CasADi dynamics function x_next = F(x, u) inside an MPC transcription.
template <typename Model, typename Scalar>
typename Model::template State<Scalar> rk4_step(
    const Model& model,
    const typename Model::template State<Scalar>& x,
    const typename Model::template Input<Scalar>& u,
    double dt) {
    using State = typename Model::template State<Scalar>;
    constexpr int NX = Model::NX;

    const State k1 = model.dynamics(x, u);

    State x2;
    for (int i = 0; i < NX; ++i) x2[i] = x[i] + 0.5 * dt * k1[i];
    const State k2 = model.dynamics(x2, u);

    State x3;
    for (int i = 0; i < NX; ++i) x3[i] = x[i] + 0.5 * dt * k2[i];
    const State k3 = model.dynamics(x3, u);

    State x4;
    for (int i = 0; i < NX; ++i) x4[i] = x[i] + dt * k3[i];
    const State k4 = model.dynamics(x4, u);

    State xn;
    for (int i = 0; i < NX; ++i) {
        xn[i] = x[i] + dt / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
    }
    return xn;
}

} // namespace vehicle_model
