import os
from pathlib import Path
import gymnasium as gym
import torch.nn as nn
import yaml
import wandb
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

from train.config.env_config import (
    ACT_FUNC_NEG_SLOPE,
    CKPT_SAVE_FREQ,
    PROJECT_NAME,
    get_env_id,
)


def make_subprocvecenv(seed: int, config: dict, n_envs: int):
    """
    Create a SubprocVecEnv parallelized environment.
    Args:
        seed: Seed for reproducibility
        config: Gym env config
        n_envs: How many parallel envs to create
    Returns:
        SubprocVecEnc parallelized gym env
    """
    print(f"Creating {n_envs} parallel environments...")

    env = SubprocVecEnv([make_env(seed=seed, rank=i, config=config) for i in range(n_envs)])

    print(f"✅ Successfully created {n_envs} parallel environments as SubProcVecEnv with seed {seed}")

    return env


def make_env(seed: int, rank: int, config: dict):
    """
    Create a single F1TENTH gym environment, wrapped in Monitor.
    Args:
        rank: Unique ID for for seeding
    Returns:
        Callable that creates the environment
    """

    def _init():
        env = gym.make(get_env_id(), config=config)
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
    return {
        "n_steps": model.n_steps,
        "batch_size": model.batch_size,
        "gamma": model.gamma,
        "learning_rate": float(model.learning_rate),
        "seed": model.seed,
        "total_timesteps": total_timesteps,
        "n_envs": n_envs,
    }


def download_model_from_wandb(run_id: str, download_dir: str, model_prefix: str) -> str:
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
    run_path = f"{api.default_entity}/{PROJECT_NAME}/runs/{run_id}"

    run = api.run(run_path)
    print(f"Found run: {run.name} ({run.state})")

    # Find and download model file
    print("Downloading model file...")
    model_files = [f for f in run.files() if f.name.endswith(".zip") and f"{model_prefix}_" in f.name]
    if not model_files:
        raise FileNotFoundError(f"No model file found with prefix '{model_prefix}' in run {run_id}")

    model_file = model_files[0]
    model_file.download(root=download_dir, replace=False)

    # Rename to standardized name
    model_cache_path = os.path.join(download_dir, "model.zip")
    downloaded_path = os.path.join(download_dir, model_file.name)
    if downloaded_path != model_cache_path:
        os.rename(downloaded_path, model_cache_path)

    return model_cache_path


def print_header(title: str) -> None:
    """
    Print a formatted header for better console readability.
    """
    print("=" * 45)
    print(f"  {title}")
    print("=" * 45)
