#!/usr/bin/env python3
"""
PPO drift training script
View tensorboard live with: 
    tensorboard --logdir outputs/tensorboard
"""

import os
import gymnasium as gym
import torch.nn as nn
import torch.cuda as cuda
import torch.backends.cudnn as cudnn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from config.env_config import (
    get_env_id,
    get_drift_train_config,
    N_ENVS,
    TOTAL_TIMESTEPS,
    N_STEPS,
    BATCH_SIZE,
    GAMMA,
    END_LEARNING_RATE,
    START_LEARNING_RATE,
    SEED,
)


def make_env(seed: int, rank: int):
    """
    Create a single F1TENTH gym environment.
    Args:
        rank: Unique ID for for seeding
    Returns:
        Callable that creates the environment
    """

    def _init():
        env = gym.make(get_env_id(), config=get_drift_train_config())
        env.reset(seed=seed + rank)  # Seed each env differently for diverse experiences
        return env

    return _init


def linear_schedule(initial_value: float, final_value: float):
    """
    Linear learning rate schedule from initial_value to final_value.
    """

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0 (end)
        """
        return final_value + progress_remaining * (initial_value - final_value)

    return func


def main():
    print("=" * 40)
    print("PPO Drift Training")
    print("=" * 40)
    print(f"Number of parallel environments: {N_ENVS}")
    print(f"Total timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"Rollout steps per env: {N_STEPS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Gamma (discount): {GAMMA}")
    print(f"Learning rate schedule: {START_LEARNING_RATE} → {END_LEARNING_RATE}")
    print("=" * 40)

    # Create output directories
    os.makedirs("outputs/tensorboard", exist_ok=True)
    os.makedirs("outputs/models", exist_ok=True)

    # Learning rate decay function
    learning_rate = linear_schedule(START_LEARNING_RATE, END_LEARNING_RATE)

    # Set up seeding for numpy, random, and torch
    set_random_seed(SEED)
    # Optional for full reproducibility
    # cudnn.deterministic = True
    # cudnn.benchmark = False

    # Create vectorized training environments
    print(f"Creating {N_ENVS} parallel environments...")
    env = SubprocVecEnv([make_env(seed=SEED, rank=i) for i in range(N_ENVS)])

    # Create PPO model
    print("Initializing PPO model...")
    model = PPO(
        policy="MlpPolicy",
        env=env,
        seed=SEED,
        learning_rate=learning_rate,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        verbose=1,
        tensorboard_log="outputs/tensorboard",
        device="cuda" if cuda.is_available() else "cpu",  # Use GPU if available
    )
    print("\nStarting training...")

    # Train the model
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        progress_bar=True,  # Use custom progress bar instead
    )

    # Save final model
    final_model_path = "outputs/models/ppo_drift_final"
    model.save(final_model_path)
    print(f"\nTraining completed! 🚀 \nFinal model saved to: {final_model_path}")

    # Close environments
    env.close()


if __name__ == "__main__":
    main()
