import gymnasium as gym
import numpy as np
from f1tenth_gym.envs.f110_env import F110Env

# Shared params
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
    env = gym.make(
        "f1tenth_gym:f1tenth-v0",
        config={
            "map": "Drift",  # Open area for drift practice
            "num_agents": 1,  # Single agent for focused learning
            "timestep": 0.01,  # High-frequency control (100Hz)
            "integrator": "rk4",  # Accurate physics integration
            "model": "std",  # Single Track dynamic bicycle model with tire slip
            "control_input": ["accl", "steering_angle"],
            "observation_config": {"type": "drift"},  # 6D drift state: [vx, vy, yaw_rate, delta, frenet_u, frenet_n]
            "reset_config": {"type": "cl_random_static"},
            "render_lookahead_curvatures": True,  # Enable lookahead curvature visualization
            "lookahead_n_points": lookahead_n_points,  # Number of lookahead points
            "lookahead_ds": 0.3,  # Spacing between points (meters)
            "debug_frenet_projection": True,  # Enable Frenet projection debug visualization
            "params": F110Env.f1tenth_std_vehicle_params(),
            "render_track_lines": True,
            "normalize_obs": True,
            "record_obs_min_max": True,
            "predictive_collision": False,
            "normalize_act": True,
            "wall_deflection": False,
        },
        render_mode="human",
    )

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
