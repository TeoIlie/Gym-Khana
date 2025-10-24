import numpy as np
from numba import njit
from numba.typed import Dict

from .utils import steering_constraint, accl_constraints


def vehicle_dynamics_st(x: np.ndarray, u_init: np.ndarray, params: dict):
    """
    Single Track Vehicle Dynamics.
    From https://gitlab.lrz.de/tum-cps/commonroad-vehicle-models/-/blob/master/vehicleModels_commonRoad.pdf, section 7

    Difference from commonroad imp (commonroad/vehiclemodels/vehicle_dynamics_st.py):
    1. Framework conversion (Python classes -> NumPy/Numba)
        a. parameters are stored in a params dict instead of a p object
        b. state vector x stored as a np array instead of a list, and gives them clearer names
    2. Kinematic threshold change
        a. switch to kinematic/dynamics happens for V < 0.5, instead of x[3] < 0.1
    3. slip angle derivatives (BETA_HAT, BETA_DOT) are computed differently from commonroad (d_beta, dd_psi)
        a. All the BETA_HAT variable is doing is performing a modulus operation to bound the value
    4. Dynamic model equations
        a. these are the same but restructured
        b. glr and glf are precomputed
        c. Readability is improved
        d. np array returned instead of list
    5. Added method get_standardized_state_st to extract state info
    6. Math libraries - all math function calls are replaced with np function calls (for ex. np.cos replaces math.cos)

        Args:
            x (numpy.ndarray (7, )): vehicle state vector (x0, x1, x2, x3, x4, x5, x6)
                x0: x position in global coordinates
                x1: y position in global coordinates
                x2: steering angle of front wheels
                x3: velocity in x direction
                x4:yaw angle
                x5: yaw rate
                x6: slip angle at vehicle center
            u (numpy.ndarray (2, )): control input vector (u1, u2)
                u1: steering angle velocity of front wheels
                u2: longitudinal acceleration
            params (dict): dictionary containing the following parameters:
                mu (float): friction coefficient
                C_Sf (float): cornering stiffness of front wheels
                C_Sr (float): cornering stiffness of rear wheels
                lf (float): distance from center of gravity to front axle
                lr (float): distance from center of gravity to rear axle
                h (float): height of center of gravity
                m (float): mass of vehicle
                I (float): moment of inertia of vehicle, about Z axis
                s_min (float): minimum steering angle
                s_max (float): maximum steering angle
                sv_min (float): minimum steering velocity
                sv_max (float): maximum steering velocity
                v_switch (float): velocity above which the acceleration is no longer able to create wheel spin
                a_max (float): maximum allowed acceleration
                v_min (float): minimum allowed velocity
                v_max (float): maximum allowed velocity

        Returns:
            f (numpy.ndarray): right hand side of differential equations
    """
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
                (
                    (params["mu"] * params["m"])
                    / (params["I"] * (params["lf"] + params["lr"]))
                )
                * (
                    params["lf"] * params["C_Sf"] * (glr) * DELTA
                    + (
                        params["lr"] * params["C_Sr"] * (glf)
                        - params["lf"] * params["C_Sf"] * (glr)
                    )
                    * BETA
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
                    + (
                        params["C_Sr"] * (glf) * params["lr"]
                        - params["C_Sf"] * (glr) * params["lf"]
                    )
                    * (PSI_DOT / V)
                )
                - PSI_DOT,  # BETA_DOT
            ]
        )

    # return f is the time derivatives of the input state x
    return f


@njit(cache=True)
def get_standardized_state_st(x: np.ndarray) -> dict:
    """[X,Y,DELTA,V_X, V_Y,YAW,YAW_RATE,SLIP]"""
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
