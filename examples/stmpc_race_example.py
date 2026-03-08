"""Runner script for the Single Track MPC controller on the F1TENTH gym (race mode)."""

import gymnasium as gym
import numpy as np
from controllers.mpc.gym_bridge import STMPCGymBridge

from examples.examples_utils import display_frenet_dynamic_state_obs
from train.config.env_config import get_drift_test_config, get_env_id

REF_SPEED = 4.0


def main():
    config = get_drift_test_config()
    config["model"] = "std"
    config["control_input"] = ["speed", "steering_angle"]
    config["observation_config"] = {"type": "frenet_dynamic_state"}
    config["normalize_act"] = False
    config["normalize_obs"] = False
    config["track_direction"] = "normal"

    env = gym.make(
        get_env_id(),
        config=config,
        render_mode="human",
    )

    bridge = STMPCGymBridge(env, ref_speed=REF_SPEED)

    # Build initial state
    # State format: [x, y, delta, v, yaw, yaw_rate, slip_angle]
    track = env.unwrapped.track
    x, y, yaw = track.frenet_to_cartesian(0, ey=0, ephi=0)
    init_states = np.array([[x, y, 0.0, REF_SPEED, yaw, 0.0, 0.0]])

    obs, info = env.reset(options={"states": init_states})
    bridge.init_from_obs(obs)

    env.render()

    for step in range(10000):
        action = bridge.get_action(obs)
        obs, reward, done, truncated, info = env.step(action)
        display_frenet_dynamic_state_obs(step, obs, reward)
        env.render()

        if done or truncated:
            print(f"Episode ended at step {step} (recovered={info.get('recovered', False)})")
            obs, info = env.reset(options={"states": init_states})
            bridge.init_from_obs(obs)

    print("Done")


if __name__ == "__main__":
    main()
