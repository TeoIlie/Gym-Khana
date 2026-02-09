"""
Simple PD-like Controller for Centerline Tracking
"""
import numpy as np
from typing import Any

from examples.controllers.base import Controller
from train.config.env_config import get_drift_test_config

# Path-tracking controller
FRENET_N_GAIN = 1.0  # Lateral deviation gain
FRENET_U_GAIN = 0.5  # Heading error gain

# Stability controller gains
BETA_GAIN = 1.0  # Sideslip angle gain
R_GAIN = 0.5  # Yaw rate gain

TARGET_SPEED = 2.0  # m/s

# config constants
LOOKAHEAD_N_POINTS = 10
LOOKAHEAD_DS = 0.3
OBS_TYPE = "drift"

# obs indices for state extraction
FRENET_U_I = 2
FRENET_N_I = 3
R_I = 4
BETA_I = 6


class PDSteerController(Controller):
    """
    Path tracking controller.

    This controller minimized lateral deviation and heading error to track a path.
    """

    def __init__(
        self, Kn: float = FRENET_N_GAIN, Ku: float = FRENET_U_GAIN, target_speed: float = TARGET_SPEED, map: str = "IMS"
    ):
        self.Kn = Kn
        self.Ku = Ku
        self.target_speed = target_speed
        self.map = map

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        frenet_u = obs[FRENET_U_I]
        frenet_n = obs[FRENET_N_I]
        steering_angle = self.compute_steering(frenet_n, frenet_u)
        action = np.array([[steering_angle, self.target_speed]], dtype=np.float32)

        return action

    def get_env_config(self) -> dict[str, Any]:
        return get_config(map=self.map)

    def compute_steering(self, frenet_n, frenet_u):
        steering_angle = -self.Kn * frenet_n - self.Ku * frenet_u
        return steering_angle


class PDStabilityController(Controller):
    """
    Pure Stability Controller for Vehicle Recovery

    This controller directly minimizes beta (sideslip angle) and r (yaw rate)
    to stabilize the vehicle.
    """

    def __init__(
        self, Kbeta: float = BETA_GAIN, Kr: float = R_GAIN, target_speed: float = TARGET_SPEED, map: str = "IMS"
    ):
        """
        Initialize the stability controller.

        Args:
            Kbeta: Gain for sideslip angle correction
            Kr: Gain for yaw rate correction
            target_speed: Constant target speed [m/s]
        """
        self.Kbeta = Kbeta
        self.Kr = Kr
        self.target_speed = target_speed
        self.map = map

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        beta = obs[BETA_I]
        r = obs[R_I]
        steering_angle = self.compute_steering(beta, r)
        action = np.array([[steering_angle, self.target_speed]], dtype=np.float32)

        return action

    def get_env_config(self) -> dict[str, Any]:
        return get_config(map=self.map)

    def compute_steering(self, beta: float, r: float) -> float:
        steering_angle = -self.Kbeta * beta - self.Kr * r
        return steering_angle


def get_config(obs_type=OBS_TYPE, lookahead_n_points=LOOKAHEAD_N_POINTS, lookahead_ds=LOOKAHEAD_DS, map="Drift_large"):
    """
    Helper function to create steering controlelrs
    """
    config = get_drift_test_config()
    config["map"] = map
    config["control_input"] = ["speed", "steering_angle"]
    config["observation_config"] = {"type": obs_type}
    config["normalize_act"] = False
    config["normalize_obs"] = False
    config["predictive_collision"] = False
    config["wall_deflection"] = False
    config["render_lookahead_curvatures"] = True
    config["render_track_lines"] = True
    config["debug_frenet_projection"] = False
    config["lookahead_n_points"] = lookahead_n_points
    config["lookahead_ds"] = lookahead_ds
    config["render_arc_length_annotations"] = True
    config["track_direction"] = "normal"
    return config
