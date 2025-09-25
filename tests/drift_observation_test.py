import gymnasium as gym
import numpy as np

env = gym.make("f1tenth_gym:f1tenth-v0", config={"observation_type": "drift"})

print(f"Observation space: {env.observation_space}")

obs, info = env.reset()
print(f"Observation type: {type(obs)}")
print(f"Observation keys: {obs.keys()}")

# Inspect each component of the observation
for key, value in obs.items():
    if isinstance(value, (int, np.integer)):
        print(f"{key}: value={value}, type={type(value)}")
    elif hasattr(value, "shape"):  # numpy array or similar
        print(f"{key}: shape={value.shape}, dtype={value.dtype}")
        print(f"  Values: {value}")
    else:
        print(f"{key}: value={value}, type={type(value)}")
    print()
