import gymnasium as gym
import numpy as np
from f1tenth_gym.envs.f110_env import F110Env

# create env
env = gym.make(
    "f1tenth_gym:f1tenth-v0",
    config={
        "map": "IMS",  # Open area for drift practice
        "num_agents": 1,  # Single agent for focused learning
        "timestep": 0.01,  # High-frequency control (100Hz)
        "integrator": "rk4",  # Accurate physics integration
        "model": "st",  # Single Track dynamic bicycle model with tire slip
        "control_input": ["speed", "steering_angle"],
        "observation_config": {"type": "drift"},  # 6D drift state: [vx, vy, yaw_rate, delta, frenet_u, frenet_n]
        "reset_config": {"type": "rl_random_static"},
    },
    render_mode="human",
)

# print observation info
print(f"Drifting observation space: {env.observation_space}")

obs, info = env.reset()
print(f"Observation type: {type(obs)}")
print(f"Initial observation after env reset: {obs}")

# Try to render the initial state
try:
    env.render()
    print("✅ Rendering enabled - you should see the track window")
except Exception as e:
    print(f"❌ Rendering failed: {e}")
    print("Running without visualization...")

# Test with action to see non-zero values
# For single agent, action should be 2D array: shape (1, 2)
action = np.array([[0.0, 0.1]])  # steering_angle, target velocity
# Take multiple steps to see if steering catches up
for step in range(10000):  # Reduced for testing
    obs, reward, done, truncated, info = env.step(action)
    heading_error_radians = obs[4]
    heading_error_degrees = np.degrees(heading_error_radians)
    print(
        f"Step {step+1:6d}: vx={obs[0]:6.2f}, vy={obs[1]:6.2f}, yaw_rate={obs[2]:6.2f}, delta={obs[3]:6.2f}, heading error (degrees)={heading_error_degrees:6.2f}, lateral distance={obs[5]:6.2f}"
    )

    # Try to render each step
    try:
        env.render()
    except Exception:
        pass  # Continue without rendering

    if abs(obs[3] - 0.1) < 0.001:  # Check if delta approaches commanded value
        print(f"✅ Steering reached target at step {step+1}")
        break

env.close()
