#pragma once

namespace vehicle_model {

/// Physical parameters of the double-track vehicle.
/// Defaults roughly correspond to a mid-size passenger car.
struct VehicleParams {
    // --- Mass / inertia ---
    double m = 1500.0;    ///< total vehicle mass [kg]
    double Iz = 2500.0;   ///< yaw moment of inertia [kg m^2]
    double Iw = 1.2;      ///< spin inertia of one wheel [kg m^2]

    // --- Geometry ---
    double lf = 1.2;      ///< CoG -> front axle distance [m]
    double lr = 1.4;      ///< CoG -> rear axle distance [m]
    double tf = 1.6;      ///< front track width [m]
    double tr = 1.6;      ///< rear track width [m]
    double h_cg = 0.55;   ///< CoG height above ground [m]
    double Re = 0.31;     ///< effective wheel rolling radius [m]

    // --- Tire (Pacejka magic formula, identical front/rear by default) ---
    double mu = 1.0;      ///< road friction coefficient [-]
    double Bx = 12.0;     ///< longitudinal stiffness factor
    double Cx = 1.65;     ///< longitudinal shape factor
    double Ex = -0.5;     ///< longitudinal curvature factor
    double By = 9.0;      ///< lateral stiffness factor
    double Cy = 1.30;     ///< lateral shape factor
    double Ey = -1.0;     ///< lateral curvature factor

    // --- Load transfer ---
    double roll_dist_f = 0.55;  ///< fraction of lateral load transfer on front axle [-]

    // --- Resistances ---
    double rho_air = 1.225;   ///< air density [kg/m^3]
    double Cd = 0.32;         ///< aero drag coefficient [-]
    double A_front = 2.2;     ///< frontal area [m^2]
    double c_rr = 0.012;      ///< rolling resistance coefficient [-]

    // --- Environment / regularization ---
    double g = 9.81;      ///< gravity [m/s^2]
    double v_eps = 0.5;   ///< speed regularization for slip computation [m/s]

    double wheelbase() const { return lf + lr; }
};

} // namespace vehicle_model
