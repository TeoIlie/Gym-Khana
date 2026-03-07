"""Controller wrapper for the Single Track MPC (STMPC) controller."""

from typing import Any

import numpy as np

from examples.controllers.base import Controller
from examples.controllers.mpc.gym_bridge import STMPCGymBridge
from train.config.env_config import get_drift_test_config


class STMPCController(Controller):
    """Controller ABC wrapper around STMPCGymBridge.

    Constructor is factory-compatible (no track needed).
    Call initialize(env) after gym.make() and on_reset(obs) after every env.reset().
    """

    def __init__(self, ref_speed: float = 4.0, map: str = "IMS"):
        self.ref_speed = ref_speed
        self.map = map
        self.bridge = None

    def initialize(self, env) -> None:
        self.bridge = STMPCGymBridge(env, ref_speed=self.ref_speed)

    def on_reset(self, obs) -> None:
        self.bridge.init_from_obs(obs)

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        return self.bridge.get_action(obs)

    def get_env_config(self) -> dict[str, Any]:
        config = get_drift_test_config()
        config["model"] = "std"
        config["control_input"] = ["speed", "steering_angle"]
        config["observation_config"] = {"type": "frenet_dynamic_state"}
        config["normalize_act"] = False
        config["normalize_obs"] = False
        config["map"] = self.map
        config["record_obs_min_max"] = False
        config["track_direction"] = "normal"
        return config
