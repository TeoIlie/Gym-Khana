"""Runner script for the Kinematic MPC controller on the F1TENTH gym."""

from pathlib import Path

import gymnasium as gym
import numpy as np
from controllers.mpc.gym_bridge import KMPCGymBridge
from examples_utils import display_kinematic_state_obs

from train.config.env_config import get_drift_test_config, get_env_id

MPC_CONFIG = Path(__file__).parent / "controllers" / "mpc" / "config" / "kinematic_mpc_params.yaml"
CAR_CONFIG = Path(__file__).parent / "controllers" / "mpc" / "config" / "car_model.yaml"
REF_SPEED = 4.0


def main():
    config = get_drift_test_config()
    config["model"] = "ks"
    config["control_input"] = ["speed", "steering_angle"]
    config["observation_config"] = {"type": "kinematic_state"}
    config["normalize_act"] = False
    config["normalize_obs"] = False
    config["track_direction"] = "normal"

    env = gym.make(
        get_env_id(),
        config=config,
        render_mode="human",
    )

    track = env.unwrapped.track
    bridge = KMPCGymBridge(track, MPC_CONFIG, CAR_CONFIG, ref_speed=REF_SPEED)

    x0, y0, yaw0 = bridge.get_start_pose()
    obs, info = env.reset(options={"poses": np.array([[x0, y0, yaw0]])})

    done = False
    step = 0
    total_reward = 0.0
    env.render()

    while not done:
        action = bridge.get_action(obs)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        display_kinematic_state_obs(step, obs, reward, total_reward)
        step += 1
        env.render()

    print("Done")


if __name__ == "__main__":
    main()
