#!/usr/bin/env python3
"""
PPO drift training script
View tensorboard live with:
    tensorboard --logdir outputs/tensorboard
Or online with the wandb link provided during a run
"""

import os

import gymnasium as gym
import torch.backends.cudnn as cudnn
import torch.cuda as cuda
import torch.nn as nn
import wandb
from config.env_config import (
    ACT_FUNC_NEG_SLOPE,
    ACTOR_LAYER_SIZE,
    BATCH_SIZE,
    CRITIC_LAYER_SIZE,
    END_LEARNING_RATE,
    GAMMA,
    N_ENVS,
    N_STEPS,
    SEED,
    START_LEARNING_RATE,
    TOTAL_TIMESTEPS,
    PROJECT_NAME,
    get_drift_train_config,
    get_env_id,
)
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv

from train.training_utils import CustomLeakyReLU, get_output_dirs, make_output_dirs


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
        env = Monitor(env)
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
    # Create tensorboard, wandb output dirs
    proj_root, output_root = get_output_dirs()

    # Init wandb
    run = wandb.init(project=PROJECT_NAME, sync_tensorboard=True, monitor_gym=True, dir=proj_root)
    run_id = run.id

    print("=" * 40)
    print("PPO Drift Training")
    print("=" * 40)
    print(f"Run ID: {run_id}")
    print(f"Number of parallel environments: {N_ENVS}")
    print(f"Total timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"Rollout steps per env: {N_STEPS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Gamma (discount): {GAMMA}")
    print(f"Actor: 2 hidden layers, each of size {ACTOR_LAYER_SIZE}")
    print(f"Critic: 2 hidden layers, each of size {CRITIC_LAYER_SIZE}")
    print(f"Actor, critic use LeakyReLU activation with negative slope -{ACT_FUNC_NEG_SLOPE}")
    print(f"Learning rate schedule: {START_LEARNING_RATE} → {END_LEARNING_RATE}")
    print("=" * 40)

    # Create output dirs using the run ID
    tensorboard_dir, models_dir, videos_dir = make_output_dirs(run.id, output_root)

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

    # Configure network architecture
    policy_kwargs = dict(
        net_arch=dict(
            pi=[ACTOR_LAYER_SIZE, ACTOR_LAYER_SIZE],  # Actor
            vf=[CRITIC_LAYER_SIZE, CRITIC_LAYER_SIZE],  # Critic
        ),
        activation_fn=CustomLeakyReLU,
    )

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
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=tensorboard_dir,
        device="cuda" if cuda.is_available() else "cpu",  # Use GPU if available
    )
    print("\nStarting training...")

    # Train the model
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        progress_bar=True,
    )

    # Save final model
    final_model_path = f"{models_dir}/ppo_drift_final_{run_id}"
    model.save(final_model_path)
    print(f"\nTraining completed! 🚀")
    print(f"Final model saved to: {final_model_path}")
    print(f"Run ID: {run_id}")

    # Close environments
    env.close()

    # Finish wandb run
    run.finish()


if __name__ == "__main__":
    main()
