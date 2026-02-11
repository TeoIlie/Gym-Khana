"""
MPC dynamics models in CasADi for nonlinear model predictive control.

This module provides symbolic dynamics models for the F1TENTH vehicle that can be
used with CasADi optimizers. Starting with kinematic bicycle model, with planned
upgrade to full STD dynamics model.
"""

import casadi as ca


def kinematic_bicycle_dynamics_casadi(x, u, params):
    """
    Kinematic bicycle model dynamics in CasADi symbolic form.

    Pure dynamics without constraint saturation (constraints handled in NLP).
    Direct translation of vehicle_dynamics_ks from kinematic.py.

    State vector x: [X, Y, delta, V, psi] (5 states)
        X: x position in global coordinates [m]
        Y: y position in global coordinates [m]
        delta: steering angle of front wheels [rad]
        V: longitudinal velocity [m/s]
        psi: yaw angle [rad]

    Control vector u: [steering_velocity, acceleration] (2 inputs)
        steering_velocity: steering angle rate [rad/s]
        acceleration: longitudinal acceleration [m/s²]

    Args:
        x: CasADi SX/MX variable (5,) - state vector
        u: CasADi SX/MX variable (2,) - control input vector
        params: dict - vehicle parameters (lf, lr, etc.)

    Returns:
        f: CasADi SX/MX (5,) - state derivatives dx/dt
    """
    # Extract states
    X = x[0]
    Y = x[1]
    delta = x[2]
    V = x[3]
    psi = x[4]

    # Extract controls (pure, no constraints applied here)
    steering_velocity = u[0]
    acceleration = u[1]

    # Vehicle parameters
    lwb = params["lf"] + params["lr"]  # wheelbase

    # Kinematic bicycle model ODEs (from kinematic.py:82-90)
    f = ca.vertcat(
        V * ca.cos(psi),  # dX/dt
        V * ca.sin(psi),  # dY/dt
        steering_velocity,  # d(delta)/dt
        acceleration,  # dV/dt
        (V / lwb) * ca.tan(delta),  # d(psi)/dt
    )

    return f


def rk4_step_casadi(dynamics_fn, x, u, params, dt):
    """
    Single RK4 (Runge-Kutta 4th order) integration step in CasADi.

    RK4 is a numerical integration method that provides good accuracy for
    nonlinear dynamics. It's the standard choice for MPC because it balances
    accuracy with computational cost.

    Args:
        dynamics_fn: function(x, u, params) -> dx/dt
        x: CasADi variable (n_states,) - current state
        u: CasADi variable (n_controls,) - control input
        params: dict - vehicle parameters
        dt: float - time step [s]

    Returns:
        x_next: CasADi variable (n_states,) - state at next time step
    """
    # RK4 integration: x_{k+1} = x_k + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
    k1 = dynamics_fn(x, u, params)
    k2 = dynamics_fn(x + (dt / 2) * k1, u, params)
    k3 = dynamics_fn(x + (dt / 2) * k2, u, params)
    k4 = dynamics_fn(x + dt * k3, u, params)

    x_next = x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    return x_next
