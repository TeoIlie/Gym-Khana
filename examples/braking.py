import gymnasium as gym
import numpy as np

from gymkhana.envs.gymkhana_env import GKEnv
from train.config.env_config import get_drift_test_config, get_env_id


def make_env_with_action(long_act_type):
    config = get_drift_test_config()
    config["map"] = "Spielberg_blank"
    config["normalize_act"] = False
    config["normalize_obs"] = False
    config["debug_frenet_projection"] = False
    config["control_input"] = [long_act_type, "steering_angle"]

    env = gym.make(
        get_env_id(),
        config=config,
        render_mode="human",
    )
    obs, info = env.reset()
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    return env


def accl_and_brake(env, target_vel, max_throttle, min_throttle):
    curr_vel = 0
    accl_steps = 0

    while curr_vel < target_vel:
        action = np.array([[0.0, max_throttle]])
        obs, reward, done, truncated, info = env.step(action)
        curr_vel = obs[0]
        accl_steps += 1
        env.render()

    print(f"\nMax velocity reached after {accl_steps} steps")

    brake_steps = 0
    while curr_vel > 0:
        action = np.array([[0.0, min_throttle]])
        obs, reward, done, truncated, info = env.step(action)
        curr_vel = obs[0]
        brake_steps += 1
        env.render()

    print(f"Brake to a halt after additional {brake_steps} steps")

    total_steps = accl_steps + brake_steps
    print(f"Total steps taken to reach target velocity {target_vel} and brake to 0: \n{total_steps} steps\n")

    return total_steps


def main():
    target_vel = 2

    print("==============================")
    print("Brake test")
    print("==============================")
    print(f"Target velocity: {target_vel}")

    print("Test 1: Target speed longitudinal action type + braking")

    max_vel = GKEnv.f1tenth_std_vehicle_params()["v_max"]
    min_vel = GKEnv.f1tenth_std_vehicle_params()["v_min"]
    env = make_env_with_action("speed")

    total_speed_braking_steps = accl_and_brake(
        env=env, target_vel=target_vel, max_throttle=max_vel, min_throttle=min_vel
    )

    print("Test 2: Accl longitudinal action type + braking")

    max_accl = GKEnv.f1tenth_std_vehicle_params()["a_max"]
    min_accl = -max_accl
    env = make_env_with_action("accl")

    total_accl_braking_steps = accl_and_brake(
        env=env, target_vel=target_vel, max_throttle=max_accl, min_throttle=min_accl
    )

    print("Test 3: Target speed longitudinal action type + coasting")

    max_vel = GKEnv.f1tenth_std_vehicle_params()["v_max"]
    min_vel = 0
    env = make_env_with_action("speed")

    total_speed_coast_steps = accl_and_brake(env=env, target_vel=target_vel, max_throttle=max_vel, min_throttle=min_vel)

    print("==============================")
    print("Results")
    print("==============================")
    print(f"Speed + braking test: {total_speed_braking_steps} steps")
    print(f"Accl + braking test: {total_accl_braking_steps} steps")
    assert total_speed_braking_steps == total_accl_braking_steps, "Why in tarnation ain't the model behave the same?"

    print(f"Speed + coasting test: {total_speed_coast_steps} steps")
    assert total_speed_coast_steps > total_speed_braking_steps, "Why in tarnation ain't the model behave the same?"


if __name__ == "__main__":
    main()
