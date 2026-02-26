"""
Helper functions for model training, storage, downloading, and evaluation
"""

import os
import traceback
from collections import Counter
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch.nn as nn
import yaml
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

import wandb
from f1tenth_gym.envs.track import Track
from f1tenth_gym.envs.track.track_utils import get_min_max_curvature, get_min_max_track_width
from train.config.env_config import (
    ACT_FUNC_NEG_SLOPE,
    BEST_MODEL,
    CKPT_SAVE_FREQ,
    N_EVAL_EPISODES,
    get_env_id,
)


def make_subprocvecenv(seed: int, config: dict, n_envs: int, track_pool: list[str] | None = None) -> SubprocVecEnv:
    """
    Create a SubprocVecEnv parallelized environment.
    Args:
        seed: Seed for reproducibility
        config: Gym env config
        n_envs: How many parallel envs to create
        track_pool: Optional list of maps to distribute across envs.
                   If provided, envs will cycle through these maps.
                   Example: ["Drift", "Drift2", "Drift_large"]
    Returns:
        SubprocVecEnv parallelized gym env distributed across track pool (if provided)
    """
    if track_pool is not None:
        # Multi-map

        # Validate track pool
        if not isinstance(track_pool, list) or len(track_pool) == 0:
            raise ValueError("track_pool must be a non-empty list")

        # Validate all track names exist before creating subprocesses
        # This provides better error messages than subprocess failures
        for track_name in track_pool:
            try:
                Track.from_track_name(track_name)
            except FileNotFoundError as e:
                raise ValueError(
                    f"Invalid track name '{track_name}' in track_pool. "
                    f"Please check available tracks in the 'maps/' directory."
                ) from e

        # Create environments with different maps
        env_fns = []
        for i in range(n_envs):
            # Cycle through track pool
            map_name = track_pool[i % len(track_pool)]
            env_config = config.copy()
            env_config["map"] = map_name
            env_fns.append(make_env(seed=seed, rank=i, config=env_config))

        env = SubprocVecEnv(env_fns)

        print(f"✅ Successfully created {n_envs} parallel multi-map environments as SubProcVecEnv with seed {seed}")

        distribution = Counter(track_pool[i % len(track_pool)] for i in range(n_envs))
        print(f"Multi-map Track distribution: {dict(distribution)}")

        return env
    else:
        # Single-map
        env = SubprocVecEnv([make_env(seed=seed, rank=i, config=config) for i in range(n_envs)])
        print(f"✅ Successfully created {n_envs} parallel single-map environments as SubProcVecEnv with seed {seed}")
        return env


def compute_global_track_bounds(track_pool: list[str], track_scale: float = 1.0) -> dict:
    """
    Compute global normalization bounds across all tracks in a pool.

    This is a utility for generating hard-coded constants in utils.py
    when new tracks are added. It is not called at runtime.

    Usage:
        python maps/extract_global_track_norm_bounds.py
        # Or directly:
        from train.training_utils import compute_global_track_bounds
        bounds = compute_global_track_bounds(["Drift", "Drift2", "Austin", ...])

    Args:
        track_pool: List of track names to compute bounds across
        track_scale: Scale factor for track loading (default 1.0)

    Returns:
        Dictionary with keys: track_max_curv, track_min_width, track_max_width
    """

    max_curvatures = []
    min_widths = []
    max_widths = []

    print_header("Track bounds")
    print(f"{'Track':<20} {'Max Curv':>12} {'Min Width':>12} {'Max Width':>12}")
    print("-" * 70)

    for track_name in track_pool:
        try:
            track = Track.from_track_name(track_name, track_scale=track_scale)
        except FileNotFoundError as e:
            raise ValueError(f"Invalid track name '{track_name}' in track_pool. ") from e

        # get_min_max_curvature returns symmetric bounds (-max, +max)
        _, max_curv = get_min_max_curvature(track)
        max_curvatures.append(max_curv)

        min_width, max_width = get_min_max_track_width(track)
        min_widths.append(min_width)
        max_widths.append(max_width)

        # Print per-track values
        print(f"{track_name:<20} {max_curv:>12.4f} {min_width:>12.4f} {max_width:>12.4f}")

    # Compute global bounds
    global_max_curv = max(max_curvatures)
    global_min_width = min(min_widths)
    global_max_width = max(max_widths)

    # Print global bounds
    print("=" * 70)
    print(f"{'GLOBAL':<20} {global_max_curv:>12.4f} {global_min_width:>12.4f} {global_max_width:>12.4f}")
    print()
    print("Update these values in f1tenth_gym/envs/utils.py:")
    print(f"  GLOBAL_MAX_CURVATURE = {global_max_curv:.4f}")
    print(f"  GLOBAL_MIN_WIDTH = {global_min_width:.4f}")
    print(f"  GLOBAL_MAX_WIDTH = {global_max_width:.4f}")
    print()

    return {
        "track_max_curv": global_max_curv,
        "track_min_width": global_min_width,
        "track_max_width": global_max_width,
    }


def make_env(seed: int, rank: int, config: dict):
    """
    Create a single F1TENTH gym environment, wrapped in Monitor.
    Args:
        rank: Unique ID for for seeding
    Returns:
        Callable that creates the environment
    """

    def _init():
        try:
            env = gym.make(get_env_id(), config=config)
            env = Monitor(env)
            env.reset(seed=seed + rank)  # Seed each env differently for diverse experiences
            return env
        except Exception as e:
            print(f"[Worker rank={rank}] env creation failed: {e}", flush=True)
            traceback.print_exc()
            raise

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


class CustomLeakyReLU(nn.LeakyReLU):
    """
    Custom implementation of LeakyReLU
    """

    def __init__(self):
        super().__init__(negative_slope=ACT_FUNC_NEG_SLOPE)


def make_output_dirs(run_id: str, root_dir: str) -> tuple[str, str, str]:
    """
    Create output directories for a training run.
    """
    tensorboard_dir = f"{root_dir}/tensorboard/{run_id}"
    models_dir = f"{root_dir}/models/{run_id}"
    config_dir = f"{root_dir}/config/{run_id}"

    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(config_dir, exist_ok=True)

    return tensorboard_dir, models_dir, config_dir


def get_output_dirs() -> tuple[str, str]:
    """
    Return project root Path
    """
    proj_root = str(Path(__file__).parent.parent.resolve())
    return proj_root, f"{proj_root}/outputs"


def get_ckpt_callback(models_dir: str, save_freq: int = CKPT_SAVE_FREQ) -> CheckpointCallback:
    """
    Checkpoint callback for periodic model saving
    """
    return CheckpointCallback(
        save_freq=save_freq,
        save_path=f"{models_dir}/checkpoints",
        name_prefix="ckpt",
        save_replay_buffer=False,
        save_vecnormalize=True,
        verbose=1,
    )


def get_eval_callback(
    eval_env,
    models_dir: str,
    eval_freq: int = CKPT_SAVE_FREQ,
    n_eval_episodes: int = N_EVAL_EPISODES,
) -> EvalCallback:
    """
    Create evaluation callback for periodic model evaluation during training.

    Args:
        eval_env: Single evaluation environment (must be Monitor-wrapped)
        models_dir: Base directory for model outputs
        eval_freq: Evaluation frequency in steps (default: CKPT_SAVE_FREQ)
        n_eval_episodes: Number of episodes per evaluation (default: 5)
    """
    return EvalCallback(
        eval_env=eval_env,
        best_model_save_path=f"{models_dir}/{BEST_MODEL}",
        log_path=f"{models_dir}/eval_logs",
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
        verbose=1,
    )


def log_best_eval_timestep(models_dir: str):
    """Log the timestep at which the best eval/mean_reward occurred."""
    eval_log_path = f"{models_dir}/eval_logs/evaluations.npz"
    if not os.path.exists(eval_log_path):
        print("No eval logs found, skipping best timestep summary.")
        return
    data = np.load(eval_log_path)
    timesteps = data["timesteps"]
    mean_rewards = data["results"].mean(axis=1)
    best_idx = np.argmax(mean_rewards)
    print(f"\nBest eval/mean_reward: {mean_rewards[best_idx]:.2f} at timestep {timesteps[best_idx]}")


def make_eval_env(seed: int, config: dict):
    """
    Create a single evaluation environment for EvalCallback.
    """
    env = gym.make(get_env_id(), config=config)
    env = Monitor(env)
    env.reset(seed=seed)
    return env


def save_config(config: dict, config_dir: str, filename: str) -> None:
    """
    Save config to YAML file for future reference.

    Args:
        config: Configuration dictionary to save
        config_dir: Directory path where config will be saved
        filename: Name of the config file (default: config.yaml)
    """
    config_file_path = os.path.join(config_dir, filename)
    with open(config_file_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Config file {filename} saved to {config_file_path}")


def extract_rl_config(model: object, total_timesteps: int, n_envs: int) -> dict:
    """
    Extract RL training configuration from trained PPO model.

    Args:
        model: Trained PPO model
        total_timesteps: Total timesteps used in training
        n_envs: Number of parallel environments

    Returns:
        Dictionary containing RL hyperparameters
    """
    config = {
        "n_steps": model.n_steps,
        "batch_size": model.batch_size,
        "gamma": model.gamma,
        "seed": model.seed,
        "total_timesteps": total_timesteps,
        "n_envs": n_envs,
        "ckpt_save_freq": CKPT_SAVE_FREQ,
        "n_eval_episodes": N_EVAL_EPISODES,
    }

    # Handle learning rate schedule (callable) vs constant (float)
    if callable(model.learning_rate):
        # Extract start and end learning rates from schedule
        config["start_learning_rate"] = float(model.learning_rate(1.0))
        config["end_learning_rate"] = float(model.learning_rate(0.0))
    else:
        config["learning_rate"] = float(model.learning_rate)

    return config


def download_model_from_wandb(run_id: str, download_dir: str, model_prefix: str, project_name: str) -> str:
    """
    Download model from wandb and return the path to cached model.

    Args:
        run_id: Wandb run ID to download from
        download_dir: Directory to cache the downloaded model
        model_prefix: Model filename prefix (e.g., "ppo_race", "ppo_drift")

    Returns:
        Path to the cached model file
    """
    os.makedirs(download_dir, exist_ok=True)

    api = wandb.Api()
    run_path = f"{api.default_entity}/{project_name}/runs/{run_id}"

    run = api.run(run_path)
    print(f"Found run: {run.name} ({run.state})")

    # Find and download model file (prioritize best model)
    print("Downloading model file...")
    best_model_files = [f for f in run.files() if f.name.endswith(".zip") and BEST_MODEL in f.name]
    if best_model_files:
        model_files = best_model_files
    else:
        model_files = [f for f in run.files() if f.name.endswith(".zip") and f"{model_prefix}_" in f.name]

    if not model_files:
        raise FileNotFoundError(
            f"No model file found (searched for '{BEST_MODEL}.zip' or '{model_prefix}_*.zip') in run {run_id}"
        )

    model_file = model_files[0]
    model_file.download(root=download_dir, replace=False)

    # Rename to standardized name
    model_cache_path = os.path.join(download_dir, "model.zip")
    downloaded_path = os.path.join(download_dir, model_file.name)

    if downloaded_path != model_cache_path:
        os.rename(downloaded_path, model_cache_path)
        # Remove empty subdirectory left by WandB's download
        subdir = os.path.join(download_dir, os.path.dirname(model_file.name))

        if subdir != download_dir and os.path.isdir(subdir):
            os.rmdir(subdir)

    return model_cache_path


def print_header(title: str) -> None:
    """
    Print a formatted header for better console readability.
    """
    print("=" * 45)
    print(f"  {title}")
    print("=" * 45)
