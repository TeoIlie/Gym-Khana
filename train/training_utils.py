import os
from pathlib import Path
import gymnasium as gym
import torch.nn as nn
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

from train.config.env_config import (
    ACT_FUNC_NEG_SLOPE,
    CKPT_SAVE_FREQ,
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

    print(f"Created {n_envs} parallel environments as SubProcVecEnv with seed {seed}")
    print(f"Config: {config}")

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
    videos_dir = f"{root_dir}/videos/{run_id}"

    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(videos_dir, exist_ok=True)

    return tensorboard_dir, models_dir, videos_dir


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
