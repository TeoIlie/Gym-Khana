"""Kinematic Single Track (KS) vehicle dynamics model."""

import numpy as np
from numba import njit

from .utils import accl_constraints, steering_constraint


def vehicle_dynamics_ks(x: np.ndarray, u_init: np.ndarray, params: dict):
    """Compute Kinematic Single Track vehicle dynamics.

    Reference: CommonRoad vehicle models, section 5.

    Args:
        x: State vector of shape ``(5,)``:
            ``[x_pos, y_pos, steering_angle, velocity, yaw_angle]``.
        u_init: Control input ``[steering_velocity, acceleration]``.
        params: Vehicle parameters dict. Uses ``lf``, ``lr``, ``s_min``,
            ``s_max``, ``sv_min``, ``sv_max``, ``v_switch``, ``a_max``,
            ``v_min``, ``v_max``. See :mod:`gymkhana.envs.dynamic_models`
            for full parameter descriptions.

    Returns:
        Time derivatives of the state vector, shape ``(5,)``.
    """
    # Controls
    X = x[0]
    Y = x[1]
    DELTA = x[2]
    V = x[3]
    PSI = x[4]
    # Raw Actions
    RAW_STEER_VEL = u_init[0]
    RAW_ACCL = u_init[1]
    # wheelbase
    lwb = params["lf"] + params["lr"]

    # constraints
    u = np.array(
        [
            steering_constraint(
                DELTA,
                RAW_STEER_VEL,
                params["s_min"],
                params["s_max"],
                params["sv_min"],
                params["sv_max"],
            ),
            accl_constraints(
                V,
                RAW_ACCL,
                params["v_switch"],
                params["a_max"],
                params["v_min"],
                params["v_max"],
            ),
        ]
    )
    # Corrected Actions
    STEER_VEL = u[0]
    ACCL = u[1]

    # system dynamics
    f = np.array(
        [
            V * np.cos(PSI),  # X_DOT
            V * np.sin(PSI),  # Y_DOT
            STEER_VEL,  # DELTA_DOT
            ACCL,  # V_DOT
            (V / lwb) * np.tan(DELTA),  # PSI_DOT
        ]
    )
    return f


def vehicle_dynamics_ks_cog(x: np.ndarray, u_init: np.ndarray, params: dict):
    """Compute Kinematic Single Track dynamics referenced at the centre of gravity.

    Unlike :func:`vehicle_dynamics_ks` (which references the rear axle), this
    variant computes position derivatives at the vehicle's centre of gravity by
    incorporating the kinematic slip angle ``beta``. Used internally by the STD
    model for the low-speed kinematic blending regime.

    Reference: CommonRoad vehicle models, section 5.

    Args:
        x: State vector of shape ``(5,)``:
            ``[x_pos, y_pos, steering_angle, velocity, yaw_angle]``.
        u_init: Control input ``[steering_velocity, acceleration]``.
        params: Vehicle parameters dict. Uses ``lf``, ``lr``, ``s_min``,
            ``s_max``, ``sv_min``, ``sv_max``, ``v_switch``, ``a_max``,
            ``v_min``, ``v_max``. See :mod:`gymkhana.envs.dynamic_models`
            for full parameter descriptions.

    Returns:
        Time derivatives of the state vector, shape ``(5,)``.
    """
    # Controls
    X = x[0]
    Y = x[1]
    DELTA = x[2]
    V = x[3]
    PSI = x[4]
    # Raw Actions
    RAW_STEER_VEL = u_init[0]
    RAW_ACCL = u_init[1]
    # wheelbase
    lwb = params["lf"] + params["lr"]
    # constraints
    u = np.array(
        [
            steering_constraint(
                DELTA,
                RAW_STEER_VEL,
                params["s_min"],
                params["s_max"],
                params["sv_min"],
                params["sv_max"],
            ),
            accl_constraints(
                V,
                RAW_ACCL,
                params["v_switch"],
                params["a_max"],
                params["v_min"],
                params["v_max"],
            ),
        ]
    )
    # slip angle (beta) from vehicle kinematics
    beta = np.arctan(np.tan(x[2]) * params["lr"] / lwb)

    # system dynamics
    f = [
        V * np.cos(beta + PSI),
        V * np.sin(beta + PSI),
        u[0],
        u[1],
        V * np.cos(beta) * np.tan(DELTA) / lwb,
    ]

    return f


@njit(cache=True)
def get_standardized_state_ks(x: np.ndarray) -> dict:
    """Extract standardized state dict from KS model state vector.

    Args:
        x: KS state vector ``[x_pos, y_pos, steering_angle, velocity, yaw_angle]``.

    Returns:
        Dict with keys: ``x``, ``y``, ``delta``, ``v_x``, ``v_y``,
        ``yaw``, ``yaw_rate``, ``slip``.
        ``v_y``, ``yaw_rate``, and ``slip`` are zero (kinematic model).
    """
    d = dict()
    d["x"] = x[0]
    d["y"] = x[1]
    d["delta"] = x[2]
    d["v_x"] = x[3]
    d["v_y"] = 0.0
    d["yaw"] = x[4]
    d["yaw_rate"] = x[5]
    d["slip"] = 0.0
    return d
