"""Wrapper for learned PPO policies to match controller interface."""

from typing import Any

import numpy as np
from stable_baselines3 import PPO

from train.config.env_config import get_drift_test_config

from .base import Controller


class LearnedController(Controller):
    """Wrapper for learned PPO models."""

    def __init__(self, model_path: str, map: str = "IMS"):
        """
        Initialize learned controller.

        Args:
            model_path: Path to saved PPO model (.zip file)
            map: Map name for environment config
        """
        self.model = PPO.load(model_path, print_system_info=True, device="auto")
        self.model_path = model_path
        self.map = map

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Predict action using learned policy. Ignores state_vars."""
        action, _ = self.model.predict(obs, deterministic=True)
        return action

    def get_env_config(self) -> dict[str, Any]:
        """Return config for learned policy evaluation."""
        config = get_drift_test_config()
        config["track_direction"] = "normal"
        config["map"] = self.map
        config["record_obs_min_max"] = False
        return config
