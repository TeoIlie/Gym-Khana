"""Single Track (ST) vehicle dynamics model."""

import numpy as np
from numba import njit

from .utils import accl_constraints, steering_constraint


def vehicle_dynamics_st(x: np.ndarray, u_init: np.ndarray, params: dict):
    """Compute Single Track vehicle dynamics.

    Reference: CommonRoad vehicle models, section 7.

    Args:
        x: State vector of shape ``(7,)``:
            ``[x_pos, y_pos, steering_angle, velocity, yaw_angle, yaw_rate, slip_angle]``.
        u_init: Control input ``[steering_velocity, acceleration]``.
        params: Vehicle parameters dict. Uses ``mu``, ``C_Sf``, ``C_Sr``,
            ``lf``, ``lr``, ``h``, ``m``, ``I``, ``s_min``, ``s_max``,
            ``sv_min``, ``sv_max``, ``v_switch``, ``a_max``, ``v_min``,
            ``v_max``. See :mod:`gymkhana.envs.dynamic_models` for full
            parameter descriptions.

    Returns:
        Time derivatives of the state vector, shape ``(7,)``.
    """
    # Implementation notes (vs. CommonRoad vehiclemodels/vehicle_dynamics_st.py):
    # 1. Framework: parameters in a dict (not a p object), state as np array
    # 2. Kinematic threshold: switch at V < 0.5 instead of x[3] < 0.1
    # 3. Slip angle derivatives (BETA_HAT, BETA_DOT) computed differently from
    #    commonroad (d_beta, dd_psi) — BETA_HAT performs a modulus operation
    # 4. Same dynamic equations, restructured for readability; np replaces math
    # 5. np array returned instead of list

    # States
    X = x[0]  # x1
    Y = x[1]  # x2
    DELTA = x[2]  # x3
    V = x[3]  # x4?
    PSI = x[4]  # yaw angle
    PSI_DOT = x[5]  # yaw rate
    BETA = x[6]  # slip angle
    # We have to wrap the slip angle to [-pi, pi]
    # BETA = np.arctan2(np.sin(BETA), np.cos(BETA))

    # gravity constant m/s^2
    g = 9.81

    # constraints
    u = np.array(
        [
            steering_constraint(
                DELTA,
                u_init[0],
                params["s_min"],  # p.min  - steering min
                params["s_max"],  # p.max  - steering max
                params["sv_min"],  # p.v_min  - steering velocity min
                params["sv_max"],  # p.v_max  - steering velocity max
            ),
            accl_constraints(
                V,
                u_init[1],
                params["v_switch"],
                params["a_max"],
                params["v_min"],  # p.v_min
                params["v_max"],  #
            ),
        ]
    )
    # Controls
    STEER_VEL = u[0]
    ACCL = u[1]

    # switch to kinematic model for small velocities
    if V < 0.5:
        # wheelbase
        lwb = params["lf"] + params["lr"]  # p.a + p.b
        BETA_HAT = np.arctan(np.tan(DELTA) * params["lr"] / lwb)  # this is used to modulate
        BETA_DOT = (
            (1 / (1 + (np.tan(DELTA) * (params["lr"] / lwb)) ** 2))
            * (params["lr"] / (lwb * np.cos(DELTA) ** 2))
            * STEER_VEL
        )
        f = np.array(
            [
                V * np.cos(PSI + BETA_HAT),  # X_DOT
                V * np.sin(PSI + BETA_HAT),  # Y_DOT
                STEER_VEL,  # DELTA_DOT
                ACCL,  # V_DOT
                V * np.cos(BETA_HAT) * np.tan(DELTA) / lwb,  # PSI_DOT
                (1 / lwb)
                * (
                    ACCL * np.cos(BETA) * np.tan(DELTA)
                    - V * np.sin(BETA) * np.tan(DELTA) * BETA_DOT
                    + ((V * np.cos(BETA) * STEER_VEL) / (np.cos(DELTA) ** 2))
                ),  # PSI_DOT_DOT
                BETA_DOT,  # BETA_DOT
            ]
        )
    else:
        # system dynamics
        glr = g * params["lr"] - ACCL * params["h"]  # rear load transfer
        glf = g * params["lf"] + ACCL * params["h"]  # front load transfer
        f = np.array(
            [
                V * np.cos(PSI + BETA),  # X_DOT
                V * np.sin(PSI + BETA),  # Y_DOT
                STEER_VEL,  # DELTA_DOT
                ACCL,  # V_DOT
                PSI_DOT,  # PSI_DOT
                ((params["mu"] * params["m"]) / (params["I"] * (params["lf"] + params["lr"])))
                * (
                    params["lf"] * params["C_Sf"] * (glr) * DELTA
                    + (params["lr"] * params["C_Sr"] * (glf) - params["lf"] * params["C_Sf"] * (glr)) * BETA
                    - (
                        params["lf"] * params["lf"] * params["C_Sf"] * (glr)
                        + params["lr"] * params["lr"] * params["C_Sr"] * (glf)
                    )
                    * (PSI_DOT / V)
                ),  # PSI_DOT_DOT
                (params["mu"] / (V * (params["lr"] + params["lf"])))
                * (
                    params["C_Sf"] * (glr) * DELTA
                    - (params["C_Sr"] * (glf) + params["C_Sf"] * (glr)) * BETA
                    + (params["C_Sr"] * (glf) * params["lr"] - params["C_Sf"] * (glr) * params["lf"]) * (PSI_DOT / V)
                )
                - PSI_DOT,  # BETA_DOT
            ]
        )

    # return f is the time derivatives of the input state x
    return f


@njit(cache=True)
def get_standardized_state_st(x: np.ndarray) -> dict:
    """Extract standardized state dict from ST model state vector.

    Args:
        x: ST state vector ``[x_pos, y_pos, steering_angle, velocity,
           yaw_angle, yaw_rate, slip_angle]``.

    Returns:
        Dict with keys: ``x``, ``y``, ``delta``, ``v_x``, ``v_y``,
        ``yaw``, ``yaw_rate``, ``slip``.
    """
    d = dict()
    d["x"] = x[0]
    d["y"] = x[1]
    d["delta"] = x[2]
    d["v_x"] = x[3] * np.cos(x[6])
    d["v_y"] = x[3] * np.sin(x[6])
    d["yaw"] = x[4]
    d["yaw_rate"] = x[5]
    d["slip"] = x[6]
    return d
