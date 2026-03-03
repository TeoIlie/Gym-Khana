# RL Integration Guide: Normalized Actions

## Overview

This guide explains how to integrate the F1TENTH Gym normalized action space (`normalize_act=True`) with reinforcement learning algorithms. The focus is on ensuring RL networks correctly output actions in the `[-1, 1]` range while maintaining proper exploration and gradient flow.

## Key Concept: Action Space Bounds

When `normalize_act=True`, the F1TENTH Gym environment exposes a bounded action space:

```python
env.action_space = Box(low=[-1, -1], high=[1, 1], shape=(2,), dtype=float32)
```

This tells RL algorithms that:

1. Actions must be in the range `[-1, 1]`
2. The environment will handle scaling to physical units
3. Gradient-based methods should respect these bounds

## Methods for Constraining Actions to [-1, 1]

### Method 1: Tanh Squashing (Recommended)

**Description**: Apply `tanh` activation to the neural network's output to squash unbounded values into `[-1, 1]`.

**Advantages**:

- ✅ Smooth, differentiable constraint
- ✅ Always produces valid actions (no clipping needed)
- ✅ Good gradient flow throughout training
- ✅ Standard practice in modern RL (used in SAC, TD3)
- ✅ Compatible with all policy gradient methods

**Disadvantages**:

- ⚠️ Requires log-probability correction for some algorithms
- ⚠️ Saturation near bounds (gradient vanishes when |tanh input| > 3)

**When to Use**: Default choice for continuous control RL algorithms (PPO, SAC, TD3, DDPG).

---

### Method 2: Action Space-Aware Policy (Automatic)

**Description**: Let the RL library automatically handle bounded action spaces based on the Gymnasium `Box` definition.

**Advantages**:

- ✅ No manual intervention needed
- ✅ Library handles implementation details
- ✅ Often uses tanh internally
- ✅ Correct log-probability calculations

**Disadvantages**:

- ⚠️ Behavior varies by library
- ⚠️ Less control over implementation

**When to Use**: When using established RL libraries (Stable-Baselines3, RLlib, CleanRL) that support bounded continuous action spaces.

---

### Method 3: Manual Clipping (Not Recommended)

**Description**: Apply `clip(action, -1, 1)` after network output.

**Advantages**:

- ✅ Simple to implement
- ✅ Guarantees valid actions

**Disadvantages**:

- ❌ Poor gradient flow (zero gradient when clipped)
- ❌ Exploration issues (actions stuck at bounds)
- ❌ Violates action distribution assumptions
- ❌ Can hurt learning performance

**When to Use**: Avoid for training. Only use for inference/deployment if necessary.

---

## Implementation Examples

### Example 1: PPO with Stable-Baselines3 (Recommended for Beginners)

**Stable-Baselines3** automatically handles bounded action spaces with tanh squashing.

```python
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Create environment with normalized actions
env = make_vec_env(
    'f1tenth_gym:f1tenth-v0',
    n_envs=4,  # Parallel environments
    env_kwargs={
        'config': {
            'normalize_act': True,
            'control_input': ['accl', 'steering_angle'],
            'num_agents': 1,
            'map': 'Spielberg',
            'timestep': 0.01,
        }
    }
)

# Create PPO agent
# The library automatically applies tanh since action_space is Box([-1,1], [1,1])
model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,  # No entropy bonus initially
    verbose=1,
)

# Train
model.learn(total_timesteps=1_000_000)

# Save
model.save("f1tenth_ppo_normalized")

# Load and test
model = PPO.load("f1tenth_ppo_normalized")
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()
```

**Key Points**:

- ✅ No manual tanh needed - SB3 handles it automatically
- ✅ Action bounds respected during exploration
- ✅ Proper log-probability calculations built-in

---

## Comparison of Methods

| Method | Gradient Flow | Exploration | Ease of Use | Recommended For |
|--------|---------------|-------------|-------------|-----------------|
| **Tanh Squashing** | ✅ Excellent | ✅ Good | ⚠️ Moderate | PPO, SAC, TD3, DDPG |
| **Library Auto-Handle** | ✅ Excellent | ✅ Good | ✅ Easy | Beginners, rapid prototyping |
| **Manual Clipping** | ❌ Poor | ❌ Poor | ✅ Easy | Inference only (not training) |

## My implementation

I will begin by using the Library Auto-Handle, and use a different approach only if this fails
