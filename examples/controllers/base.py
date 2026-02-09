from abc import ABC, abstractmethod
from typing import Any
import numpy as np


class Controller(ABC):
    """Abstract base class defining the interface for all controllers."""

    @abstractmethod
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """
        Compute control action from observation
        Args:
            obs: Full observation array from environment
        Returns:
            Action array [[steering_angle/steering_velocity, speed/acceleration]]
        """
        pass

    @abstractmethod
    def get_env_config(self) -> dict[str, Any]:
        """
        Return environment configuration required by this controller.
        """
        pass
