__version__ = "1.1.0"

import gymnasium as gym

gym.register(
    id="gymkhana-v0",
    entry_point="gymkhana.envs:GKEnv",
)
