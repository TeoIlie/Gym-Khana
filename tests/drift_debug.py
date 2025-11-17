"""
File for debugging gym observations
"""

import gymnasium as gym
import numpy as np
from train.config.env_config import get_drift_test_config, get_env_id

lookahead_n_points = 10


def format_float(x):
    if abs(x) < 1e-5:
        return "Approx 0"
    else:
        return f"{x:.4f}"


def display_step_info(obs, reward):
    # Extract observations from obs
    vx = obs[0]
    vy = obs[1]
    heading_error_radians = obs[2]
    heading_error_degrees = np.degrees(heading_error_radians)
    lateral_dist = obs[3]
    yaw_rate = obs[4]
    delta = obs[5]
    prev_steer_cmd = obs[6]
    prev_accl_cmd = obs[7]
    prev_avg_wheel_omega = obs[8]
    curr_vel_cmd = obs[9]
    curvatures = obs[10 : 10 + lookahead_n_points]
    widths = obs[10 + lookahead_n_points : 10 + (2 * lookahead_n_points)]

    print(
        f"  vx={vx:6.2f}, vy={vy:6.2f}, yaw_rate={yaw_rate:6.2f}, delta={delta:6.4f}\n"
        f"  heading error (degrees)={heading_error_degrees:6.2f}, lateral distance={lateral_dist:6.2f}\n"
        f"  previous steering cmd={prev_steer_cmd:6.4f}\n"
        f"  previous accl cmd={prev_accl_cmd:6.4f}\n"
        f"  previous average wheel ang speed={prev_avg_wheel_omega:6.4f}\n"
        f"  current velocity command (integrated from accl)={curr_vel_cmd:6.4f}\n"
        f"  curvature lookahead:"
    )
    for i, value in enumerate(curvatures, start=1):
        print(f"    Point {i} = {format_float(value)}")

    print(f"  widths lookahead:")
    for i, value in enumerate(widths, start=1):
        print(f"    Point {i} = {format_float(value)}")

    print(f"\n  Reward = {reward}\n")


if __name__ == "__main__":
    # create env
    env = gym.make(get_env_id(), config=get_drift_test_config(), render_mode="human")

    # print observation info
    print(f"Drifting observation space: {env.observation_space}")

    obs, info = env.reset()
    print(f"Initial observation after env reset:")
    display_step_info(obs, None)

    # For single agent, action should be 2D array: shape (1, 2)
    action = np.array([[0.0, 0.2]])  # action format: normalized steering target, normalized acceleration

    for step in range(10000):  # Reduced for testing
        obs, reward, done, truncated, info = env.step(action)
        print(f"\n=====================\nStep {step+1}:\n=====================\n")
        display_step_info(obs, reward)
        # if done:
        #     print("\n=====================\nDONE!\n=====================\n")
        #     break

        # render
        env.render()

    env.close()
