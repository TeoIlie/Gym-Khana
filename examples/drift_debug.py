"""
File for debugging gym observations
"""

import gymnasium as gym
import numpy as np

from examples.examples_utils import display_drift_obs
from train.config.env_config import LOOKAHEAD_N_POINTS, get_drift_test_config, get_env_id

if __name__ == "__main__":
    config = get_drift_test_config()
    config["map"] = "Spielberg"

    # create env
    env = gym.make(get_env_id(), config=config, render_mode="human")

    # print observation info
    print(f"Drifting observation space: {env.observation_space}")

    obs, info = env.reset()
    print("Initial observation after env reset:")
    display_drift_obs(0, obs, None, LOOKAHEAD_N_POINTS)

    # For single agent, action should be 2D array: shape (1, 2)
    action = np.array([[0.0, 0.2]])  # action format: normalized steering target, normalized acceleration

    for step in range(10000):  # Reduced for testing
        obs, reward, done, truncated, info = env.step(action)
        display_drift_obs(step, obs, reward, LOOKAHEAD_N_POINTS)

        env.render()

    env.close()
