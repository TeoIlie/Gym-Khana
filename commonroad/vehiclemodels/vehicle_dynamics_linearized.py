import math
from typing import List
import numpy as np

from commonroad.vehiclemodels.utils.steering_constraints import kappa_dot_dot_constraints
from commonroad.vehiclemodels.utils.acceleration_constraints import jerk_dot_constraints


__author__ = "Gerald Würsching, Xiao Wang"
__copyright__ = "TUM Cyber-Physical Systems Group"
__version__ = "2020a"
__maintainer__ = "Gerald Würsching"
__email__ = "commonroad@lists.lrz.de"
__status__ = "Released"


def _make_valid_orientation(angle: float) -> float:
    two_pi = 2 * math.pi
    while angle > two_pi:
        angle = angle - two_pi
    while angle < -two_pi:
        angle = angle + two_pi
    return angle


def _interpolate_angle(x: float, x1: float, x2: float, y1: float, y2: float) -> float:
    """
    Interpolates an angle value between two angles according to the minimal value of the absolute difference
    :param x: value of other dimension to interpolate
    :param x1: lower bound of the other dimension
    :param x2: upper bound of the other dimension
    :param y1: lower bound of angle to interpolate
    :param y2: upper bound of angle to interpolate
    :return: interpolated angular value (in rad)
    """
    delta = y2 - y1
    return _make_valid_orientation(delta * (x - x1) / (x2 - x1) + y1)


def vehicle_dynamics_linearized(x: List, u_init: List, p, ref_pos, ref_theta):
    """
    linearized kinematic single-track model: separation of longitudinal and lateral motion
    see: Pek, C. and Althoff, M. "Computationally Efficient Fail-safe Trajectory Planning for Self-driving Vehicles
    Using Convex Optimization", ITSC, 2018

    Inputs:
        :param x: combined (lon and lat) vehicle state vector [s, v, a, j, d, theta, kappa, kappa_dot]
        :param u_init: vehicle input vector [j_dot, kappa_dot_dot]
        :param p: vehicle parameters
        :param ref_pos: reference path position array (s coordinate)
        :param ref_theta: reference path orientation array (theta)

    Outputs:
        :return f: right-hand side of differential equations
    """
    # longitudinal states
    # x1 = s: longitudinal position along reference
    # x2 = v: longitudinal velocity
    # x3 = a: longitudinal acceleration
    # x4 = j: jerk

    # lateral states
    # x5 = d: lateral deviation from reference
    # x6 = theta: global orientation
    # x7 = kappa: curvature
    # x8 = kappa_dot: curvature rate

    # inputs
    # u1 = jerk_dot change of jerk
    # u2 = kappa_dot_dot curvature rate rate

    assert len(ref_pos) == len(ref_theta), "reference path position and orientation arrays must have the same length"

    # input constraints
    u = list()
    u.append(jerk_dot_constraints(u_init[0], x[3], p.longitudinal))
    u.append(kappa_dot_dot_constraints(u_init[1], x[7], p.steering))

    # longitudinal and lateral state vector
    x_long = x[0:4]
    x_lat = x[4:]

    # longitudinal dynamics
    f_long = [x_long[1], x_long[2], x_long[3], u[0]]

    # lon position and velocity
    s = x_long[0]
    v = x_long[1]

    # interpolate theta_s from ref_theta at x_long[0]
    s_idx = np.argmax(ref_pos > s) - 1
    theta_s = _interpolate_angle(s, ref_pos[s_idx], ref_pos[s_idx + 1], ref_theta[s_idx], ref_theta[s_idx + 1])

    # lateral dynamics
    f_lat = [v * x_lat[1] - v * theta_s, v * x_lat[2], x_lat[3], u[1]]

    return f_long + f_lat
