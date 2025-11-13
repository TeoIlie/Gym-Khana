import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
import torch.nn as nn
import torch
import os
import numpy as np

# Wandb and tensorboard integration
from wandb.integration.sb3 import WandbCallback
import wandb

from f1tenth_gym.envs.f110_env import F110Env

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def linear_schedule(initial_value: float, final_value: float):
    """
    Linear learning rate schedule that decays from initial_value to final_value.

    Args:
        initial_value: Starting learning rate (1e-3)
        final_value: Ending learning rate (1e-4)

    Returns:
        Function that takes progress (0 to 1) and returns current learning rate
    """

    def func(progress_remaining: float) -> float:
        # progress_remaining goes from 1 (start) to 0 (end)
        # We want to go from initial_value to final_value
        return final_value + (initial_value - final_value) * progress_remaining

    return func


def make_env(rank: int, seed: int = 0):
    """
    Factory function to create a single environment instance.
    Each parallel environment gets a unique seed for diversity.

    Args:
        rank: Index of the environment (0 to 399)
        seed: Base random seed

    Returns:
        Function that creates and returns a monitored environment
    """

    def _init():
        env = gym.make(
            "f1tenth_gym:f1tenth-v0",
            config={
                "map": "Drift",  # Open area for drift practice
                "num_agents": 1,  # Single agent for focused learning
                "timestep": 0.01,  # High-frequency control (100Hz)
                "integrator": "rk4",  # Accurate physics integration
                "model": "std",  # Single Track dynamic bicycle model with tire slip
                "control_input": ["accl", "steering_angle"],
                "observation_config": {
                    "type": "drift"
                },  # 6D drift state: [vx, vy, yaw_rate, delta, frenet_u, frenet_n]
                "reset_config": {"type": "cl_random_static"},
                "render_lookahead_curvatures": True,  # Enable lookahead curvature visualization
                "lookahead_n_points": 10,  # Number of lookahead points
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
        )

        # Monitor wrapper tracks episode statistics
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env

    return _init


# REQUIREMENT 8-11: Custom actor-critic network architecture
class CustomActorCriticPolicy(nn.Module):
    """
    Custom policy network with specific architecture requirements:
    - Actor (policy): 2 hidden layers of 256 neurons
    - Critic (value): 2 hidden layers of 512 neurons
    - Both use LeakyReLU with negative slope 0.2
    """

    pass  # We'll configure this through policy_kwargs instead


def main():
    # ========== CONFIGURATION ==========

    # REQUIREMENT 2: 400 parallel environments
    N_ENVS = 400

    # REQUIREMENT 3: 1024 steps per rollout
    N_STEPS = 1024

    # REQUIREMENT 4: Batch size of 1024
    BATCH_SIZE = 1024

    # REQUIREMENT 5: Discount factor gamma
    GAMMA = 0.99

    # REQUIREMENT 6: Learning rate schedule (1e-3 to 1e-4)
    LEARNING_RATE = linear_schedule(1e-3, 1e-4)

    # REQUIREMENT 7: 120 million timesteps
    TOTAL_TIMESTEPS = 120_000_000

    # REQUIREMENT 8-11: Network architecture
    # Actor: 2 layers of 256 neurons
    # Critic: 2 layers of 512 neurons
    # LeakyReLU with negative slope 0.2
    policy_kwargs = dict(
        activation_fn=nn.LeakyReLU,
        net_arch=dict(pi=[256, 256], vf=[512, 512]),  # Actor (policy) network  # Critic (value function) network
        # LeakyReLU with negative slope 0.2
        activation_fn_kwargs=dict(negative_slope=0.2),
    )

    # Random seed for reproducibility
    SEED = 42

    # ========== SETUP ==========

    # Initialize Wandb for experiment tracking
    run = wandb.init(
        project="f1tenth_drift_racing_ppo",
        config={
            "algorithm": "PPO",
            "n_envs": N_ENVS,
            "n_steps": N_STEPS,
            "batch_size": BATCH_SIZE,
            "gamma": GAMMA,
            "learning_rate_start": 1e-3,
            "learning_rate_end": 1e-4,
            "total_timesteps": TOTAL_TIMESTEPS,
            "actor_layers": [256, 256],
            "critic_layers": [512, 512],
            "activation": "LeakyReLU(0.2)",
            "timestep": 0.05,
        },
        sync_tensorboard=True,  # Sync with tensorboard
        save_code=True,
        monitor_gym=True,
    )

    # Set directories
    tensorboard_dir = os.path.join(SCRIPT_DIR, "runs", run.id)
    model_dir = os.path.join(SCRIPT_DIR, "models", run.id)
    os.makedirs(model_dir, exist_ok=True)

    print(f"Creating {N_ENVS} parallel environments...")

    # REQUIREMENT 2: Create vectorized environment with 400 parallel instances
    # SubprocVecEnv runs each environment in a separate process for true parallelization
    env = SubprocVecEnv([make_env(i, SEED) for i in range(N_ENVS)])

    # VecMonitor adds monitoring capabilities to vectorized environments
    env = VecMonitor(env)

    print("Environment created successfully!")

    # ========== MODEL CREATION ==========

    print("Initializing PPO model...")

    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")

    # REQUIREMENT 1: PPO algorithm with all custom parameters
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=LEARNING_RATE,  # Decaying learning rate
        n_steps=N_STEPS,  # Steps per rollout
        batch_size=BATCH_SIZE,  # Minibatch size
        gamma=GAMMA,  # Discount factor
        policy_kwargs=policy_kwargs,  # Custom network architecture
        verbose=1,
        tensorboard_log=tensorboard_dir,
        device=device,  # Automatically use GPU if available
        seed=SEED,
    )

    print("Model initialized!")
    print(f"\nTraining configuration:")
    print(f"  Total timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"  Parallel environments: {N_ENVS}")
    print(f"  Steps per rollout: {N_STEPS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Timesteps per update: {N_ENVS * N_STEPS:,}")
    print(f"  Total updates: {TOTAL_TIMESTEPS // (N_ENVS * N_STEPS):,}")
    print(f"  Learning rate: 1e-3 → 1e-4 (linear decay)")

    # ========== CALLBACKS ==========

    # Checkpoint callback: saves model periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=max(100_000 // N_ENVS, 1),  # Save every ~100k timesteps
        save_path=model_dir,
        name_prefix="ppo_drift_model",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    # Wandb callback: logs metrics to Wandb and Tensorboard
    wandb_callback = WandbCallback(
        gradient_save_freq=0,
        model_save_path=model_dir,
        verbose=2,
    )

    # Combine callbacks
    callbacks = [checkpoint_callback, wandb_callback]

    # ========== TRAINING ==========

    print("\nStarting training...")
    print("=" * 60)

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callbacks,
        progress_bar=True,
    )

    print("=" * 60)
    print("Training complete!")

    # Save final model
    final_model_path = os.path.join(model_dir, "final_model")
    model.save(final_model_path)
    print(f"Final model saved to: {final_model_path}")

    # Cleanup
    env.close()
    run.finish()

    print("\nYou can view training progress in:")
    print(f"  - Tensorboard: tensorboard --logdir {tensorboard_dir}")
    print(f"  - Wandb: {run.get_url()}")


if __name__ == "__main__":
    main()
