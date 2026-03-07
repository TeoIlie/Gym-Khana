"""Runner script for the Single Track MPC controller on the F1TENTH gym (recovery mode)."""

from pathlib import Path

import gymnasium as gym
from controllers.mpc.gym_bridge import STMPCGymBridge

from examples.examples_utils import display_frenet_dynamic_state_obs
from train.config.env_config import get_drift_test_config, get_env_id

STMPC_CONFIG = Path(__file__).parent / "controllers" / "mpc" / "config" / "single_track_mpc_params.yaml"
CAR_CONFIG = Path(__file__).parent / "controllers" / "mpc" / "config" / "car_model.yaml"
TIRE_CONFIG = Path(__file__).parent / "controllers" / "mpc" / "config" / "pacejka_tire_params.yaml"
REF_SPEED = 4.0


def main():
    config = get_drift_test_config()
    config["model"] = "std"
    config["control_input"] = ["speed", "steering_angle"]
    config["observation_config"] = {"type": "frenet_dynamic_state"}
    config["normalize_act"] = False
    config["normalize_obs"] = False
    config["track_direction"] = "normal"
    config["training_mode"] = "recover"

    env = gym.make(
        get_env_id(),
        config=config,
        render_mode="human",
    )

    track = env.unwrapped.track
    bridge = STMPCGymBridge(track, STMPC_CONFIG, CAR_CONFIG, TIRE_CONFIG, ref_speed=REF_SPEED)

    # Recovery mode: env.reset() generates a random perturbed state at recovery_s_init
    obs, info = env.reset()
    bridge.init_from_obs(obs)

    env.render()

    for step in range(10000):
        action = bridge.get_action(obs)
        obs, reward, done, truncated, info = env.step(action)
        display_frenet_dynamic_state_obs(step, obs, reward)
        env.render()

        if done or truncated:
            print(f"Episode ended at step {step} (recovered={info.get('recovered', False)})")
            obs, info = env.reset()
            bridge.init_from_obs(obs)

    print("Done")


if __name__ == "__main__":
    main()
