"""
Simple PD-like Controller for Centerline Tracking

This script demonstrates a simple controller that uses Frenet frame observations
(frenet_n and frenet_u) to keep the car following the centerline of the track
with the correct heading.

    steering_angle = -Kn * frenet_n - Ku * frenet_u

"""
import numpy as np
import gymnasium as gym

from f1tenth_gym.envs.f110_env import F110Env


class PDSteerController:
    def __init__(self, Kn: float, Ku: float, target_speed: float):
        self.Kn = Kn
        self.Ku = Ku
        self.target_speed = target_speed

    def compute_steering(self, frenet_n, frenet_u):
        steering_angle = -self.Kn * frenet_n - self.Ku * frenet_u

        return steering_angle

    def get_action(self, obs):
        frenet_u = obs[0]  # heading error
        frenet_n = obs[1]  # lateral deviation

        # Compute steering angle
        steering_angle = self.compute_steering(frenet_n, frenet_u)

        # Return action: [[speed, steering_angle]] with shape (1, 2)
        action = np.array([[steering_angle, self.target_speed]], dtype=np.float32)

        return action


def main():
    # Create controller with tunable gains
    Kn = 1.0  # Lateral deviation gain
    Ku = 0.5  # Heading error gain
    target_speed = 4.0  # m/s

    controller = PDSteerController(Kn=Kn, Ku=Ku, target_speed=target_speed)

    env = gym.make(
        "f1tenth_gym:f1tenth-v0",
        config={
            "map": "Spielberg",
            "model": "std",
            "params": F110Env.f1tenth_std_vehicle_params(),
            "num_agents": 1,
            "timestep": 0.01,
            "num_beams": 36,
            "integrator": "rk4",
            "control_input": ["speed", "steering_angle"],
            "observation_config": {"type": "frenet"},
            "reset_config": {"type": "rl_random_static"},
            "normalize_act": False,
            "normalize_obs": False,
            "predictive_collision": True,
            "wall_deflection": True,
            "render_lookahead_curvatures": True,
            "render_track_lines": True
            # "debug_frenet_projection": True,
        },
        render_mode="human",
    )

    print(f"Centerline Tracking Controller")
    print(f"==============================")
    print(f"Controller gains: Kn={Kn}, Ku={Ku}")
    print(f"Target speed: {target_speed} m/s")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print()

    # Reset environment
    obs, info = env.reset()

    # Run simulation
    done = False
    step_count = 0
    total_lateral_error = 0.0
    total_heading_error = 0.0
    total_reward = 0.0

    try:
        while not done:
            # Get action from controller
            action = controller.get_action(obs)

            # Step environment
            obs, reward, done, truncated, info = env.step(action)
            done = done or truncated

            env.render()

            # Track metrics
            frenet_u = obs[0]  # heading error
            frenet_n = obs[1]  # lateral deviation
            total_lateral_error += abs(frenet_n)
            total_heading_error += abs(frenet_u)
            total_reward += reward
            step_count += 1

            # Print periodic status
            if step_count % 100 == 0:
                avg_lateral = total_lateral_error / step_count
                avg_heading = total_heading_error / step_count
                print(
                    f"Step {step_count}: "
                    f"frenet_n={frenet_n:.4f}m, "
                    f"frenet_u={frenet_u:.4f}rad, "
                    f"reward={reward:.4f}"
                )
                print(
                    f"  Avg errors - lateral: {avg_lateral:.4f}m, "
                    f"heading: {avg_heading:.4f}rad, "
                    f"total_reward: {total_reward:.2f}"
                )

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")

    finally:
        # Print final statistics
        if step_count > 0:
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
