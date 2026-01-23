"""
Simple PD-like Controller for Centerline Tracking

This script demonstrates a simple controller that uses Frenet frame observations
(frenet_n and frenet_u) to keep the car following the centerline of the track
with the correct heading.

    steering_angle = -Kn * frenet_n - Ku * frenet_u

"""

import gymnasium as gym
import numpy as np

from examples.examples_utils import display_drift_obs
from f1tenth_gym.envs.f110_env import F110Env
from train.config.env_config import get_drift_test_config


class PDSteerController:
    def __init__(self, Kn: float, Ku: float, target_speed: float, frenet_u_i: int, frenet_n_i: int):
        self.Kn = Kn
        self.Ku = Ku
        self.target_speed = target_speed
        self.frenet_u_i = frenet_u_i
        self.frenet_n_i = frenet_n_i

    def compute_steering(self, frenet_n, frenet_u):
        steering_angle = -self.Kn * frenet_n - self.Ku * frenet_u

        return steering_angle

    def get_action(self, obs):
        frenet_u = obs[self.frenet_u_i]  # heading error
        frenet_n = obs[self.frenet_n_i]  # lateral deviation

        # Compute steering angle
        steering_angle = self.compute_steering(frenet_n, frenet_u)

        # Return action: [[speed, steering_angle]] with shape (1, 2)
        action = np.array([[steering_angle, self.target_speed]], dtype=np.float32)

        return action


def get_config(obs_type, lookahead_n_points, lookahead_ds):
    config = get_drift_test_config()
    config["map"] = "Drift_large"
    config["control_input"] = ["speed", "steering_angle"]
    config["observation_config"] = {"type": obs_type}
    config["normalize_act"] = False
    config["normalize_obs"] = False
    config["predictive_collision"] = False
    config["wall_deflection"] = False
    config["render_lookahead_curvatures"] = True
    config["render_track_lines"] = True
    config["debug_frenet_projection"] = False
    config["lookahead_n_points"] = lookahead_n_points
    config["lookahead_ds"] = lookahead_ds
    config["render_arc_length_annotations"] = True
    return config


def main():
    # Create controller with tunable gains
    FRENET_N_GAIN = 1.0  # Lateral deviation gain
    FRENET_K_GAIN = 0.5  # Heading error gain
    TARGET_SPEED = 2.0  # m/s

    # config constants
    LOOKAHEAD_N_POINTS = 10
    LOOKAHEAD_DS = 0.3
    OBS_TYPE = "drift"

    NUM_STEPS = 20_000

    # initial state
    S = 96

    env = gym.make(
        "f1tenth_gym:f1tenth-v0",
        config=get_config(OBS_TYPE, LOOKAHEAD_N_POINTS, LOOKAHEAD_DS),
        render_mode="human",
    )
    x, y, yaw = env.unwrapped.track.frenet_to_cartesian(S, ey=0, ephi=0)

    init_state = np.array(
        [
            [
                x,  # x position (Cartesian coordinates)
                y,  # y position (Cartesian coordinates)
                0.0,  # delta (steering angle)
                TARGET_SPEED,  # v (velocity)
                yaw,  # yaw angle (Cartesian orientation)
                0.0,  # yaw_rate
                0.0,  # slip_angle
            ]
        ]
    )
    obs, info = env.reset(options={"states": init_state})

    # Set frenet indices based on obs type
    if OBS_TYPE == "drift":
        FRENET_U_I = 2
        FRENET_N_I = 3
    elif OBS_TYPE == "frenet":
        FRENET_U_I = 0
        FRENET_N_I = 1
    else:
        raise ValueError("Please specify frenet observation indices.")

    controller = PDSteerController(
        Kn=FRENET_N_GAIN, Ku=FRENET_K_GAIN, target_speed=TARGET_SPEED, frenet_u_i=FRENET_U_I, frenet_n_i=FRENET_N_I
    )

    print(f"Centerline Tracking Controller")
    print(f"==============================")
    print(f"Controller gains: Kn={FRENET_N_GAIN}, Ku={FRENET_K_GAIN}")
    print(f"Target speed: {TARGET_SPEED} m/s")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print()

    # Run simulation
    done = False
    step_count = 0
    total_lateral_error = 0.0
    total_heading_error = 0.0
    total_reward = 0.0

    for step in range(NUM_STEPS):
        action = controller.get_action(obs)
        # Step environment
        obs, reward, done, truncated, info = env.step(action)

        # Track metrics
        frenet_u = obs[FRENET_U_I]  # heading error
        frenet_n = obs[FRENET_N_I]  # lateral deviation

        total_lateral_error += abs(frenet_n)
        total_heading_error += abs(frenet_u)
        total_reward += reward

        display_drift_obs(step, obs, reward, LOOKAHEAD_N_POINTS, total_reward)

        env.render()

        step_count += 1

    # Print final statistics
    avg_lateral = total_lateral_error / step_count
    avg_heading = total_heading_error / step_count
    avg_reward = total_reward / step_count
    print(f"\nFinal Statistics:")
    print(f"  Total steps: {step_count}")
    print(f"  Average lateral error: {avg_lateral:.4f} m")
    print(f"  Average heading error: {avg_heading:.4f} rad ({np.degrees(avg_heading):.2f} deg)")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Average reward per step: {avg_reward:.4f}")

    env.close()


if __name__ == "__main__":
    main()
