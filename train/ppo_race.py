"""
PPO Race Training and Evaluation Script

Usage:
    # Train a new model
    python train/ppo_race.py --m t

    # Evaluate a local model (uses latest wandb run if --path not specified)
    python train/ppo_race.py --m e
    python train/ppo_race.py --m e --path /path/to/model.zip

    # Download model from wandb and evaluate (uses cache if already downloaded)
    python train/ppo_race.py --m d --run_id <wandb_run_id>

    # Continue training from existing model
    python train/ppo_race.py --m c --path /path/to/model.zip --additional_timesteps 10000000
"""

import wandb
import os
import argparse
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from wandb.integration.sb3 import WandbCallback
from train.config.env_config import (
    BEST_MODEL,
    CKPT_SAVE_FREQ,
    END_LEARNING_RATE,
    EVAL_SEED,
    N_ENVS,
    N_STEPS,
    PROJECT_NAME,
    SEED,
    START_LEARNING_RATE,
    TOTAL_TIMESTEPS,
    TRACK_POOL,
    get_drift_test_config,
    get_drift_train_config,
    get_env_id,
)
from train.training_utils import (
    get_output_dirs,
    linear_schedule,
    make_output_dirs,
    get_ckpt_callback,
    get_eval_callback,
    make_eval_env,
    make_subprocvecenv,
    save_config,
    extract_rl_config,
    download_model_from_wandb,
    print_header,
)


# Extract model prefix from filename (e.g., "ppo_race" from "ppo_race.py")
MODEL_PREFIX = os.path.splitext(os.path.basename(__file__))[0]
TRAIN_CONFIG = get_drift_train_config()
TEST_CONFIG = get_drift_test_config()


def train_ppo_race():
    print_header("PPO Race Training")

    proj_root, output_root = get_output_dirs()

    run = wandb.init(project=PROJECT_NAME, sync_tensorboard=True, monitor_gym=True, dir=proj_root, save_code=True)
    run_id = run.id

    tensorboard_dir, models_dir, config_dir = make_output_dirs(run.id, output_root)
    save_config(TRAIN_CONFIG, config_dir, "gym_config.yaml")

    env = make_subprocvecenv(SEED, TRAIN_CONFIG, N_ENVS, TRACK_POOL)
    eval_env = make_eval_env(EVAL_SEED, TRAIN_CONFIG)

    learning_rate = linear_schedule(START_LEARNING_RATE, END_LEARNING_RATE)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        n_steps=N_STEPS,
        verbose=1,
        tensorboard_log=tensorboard_dir,
        device="auto",
        seed=SEED,
        learning_rate=learning_rate,
    )

    rl_config = extract_rl_config(model, TOTAL_TIMESTEPS, N_ENVS)
    save_config(rl_config, config_dir, "rl_config.yaml")

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[
            WandbCallback(gradient_save_freq=0, verbose=2),
            get_ckpt_callback(models_dir=models_dir, save_freq=CKPT_SAVE_FREQ),
            get_eval_callback(eval_env=eval_env, models_dir=models_dir),
        ],
        progress_bar=True,
    )

    final_model_path = f"{models_dir}/{MODEL_PREFIX}_{run_id}"
    model.save(final_model_path)

    # Save best model
    best_model_path = f"{models_dir}/{BEST_MODEL}/{BEST_MODEL}"
    run.save(f"{best_model_path}.zip", base_path=models_dir)

    env.close()
    eval_env.close()

    run.finish()


def evaluate_ppo_race(model_path: str = ""):
    print_header("PPO Race Evaluation")

    proj_root, _ = get_output_dirs()

    if model_path == "":
        model_path = os.path.join(proj_root, "wandb", "latest-run", "files", "model.zip")

    model = PPO.load(model_path, print_system_info=True, device="cpu")
    print(f"Loaded model from {model_path}")

    eval_env = gym.make(
        get_env_id(),
        config=TEST_CONFIG,
        render_mode="human",
    )
    np.random.seed()
    obs, info = eval_env.reset()
    done, trunc = False, False
    total_reward = 0.0

    while not (done or trunc):
        action, _states = model.predict(obs, deterministic=True)

        obs, reward, done, trunc, info = eval_env.step(action)
        total_reward += reward
        eval_env.render()

        # VecEnv resets automatically
        # if done:
        #   obs = env.reset()
    eval_env.close()
    print(f"Total reward: {total_reward}")


def continue_training_ppo_race(model_path: str, additional_timesteps: int):
    """
    Continue training from a saved model checkpoint with a new wandb run.

    Args:
        model_path: Path to saved model.zip file
        additional_timesteps: additional steps to train
    """
    print_header("PPO Race - Continue Training")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if not model_path.endswith(".zip"):
        raise ValueError(f"Model path must be a .zip file, got: {model_path}")

    if additional_timesteps <= 0:
        raise ValueError(f"additional_timesteps must be positive, got: {additional_timesteps}")

    print(f"Loading model from: {model_path}")
    print(f"Additional timesteps: {additional_timesteps:,}")

    proj_root, output_root = get_output_dirs()

    run = wandb.init(
        project=PROJECT_NAME,
        sync_tensorboard=True,
        monitor_gym=True,
        dir=proj_root,
        save_code=True,
    )
    new_run_id = run.id

    print(f"New run ID: {new_run_id}")

    # Create output directories for this continuation run
    tensorboard_dir, models_dir, config_dir = make_output_dirs(new_run_id, output_root)

    # Uses current env_config.py
    env = make_subprocvecenv(SEED, TRAIN_CONFIG, N_ENVS)
    eval_env = make_eval_env(EVAL_SEED, TRAIN_CONFIG)

    model = PPO.load(model_path, env=env, device="auto")

    model.tensorboard_log = tensorboard_dir

    print(f"Model loaded successfully")
    print(f"Continuing from checkpoint's timestep count")

    save_config(TRAIN_CONFIG, config_dir, "gym_config.yaml")

    rl_config = extract_rl_config(model, additional_timesteps, N_ENVS)
    save_config(rl_config, config_dir, "rl_config.yaml")

    print("\nContinuing training...")

    model.learn(
        total_timesteps=additional_timesteps,
        callback=[
            WandbCallback(gradient_save_freq=0, verbose=2),
            get_ckpt_callback(models_dir=models_dir, save_freq=CKPT_SAVE_FREQ),
            get_eval_callback(eval_env=eval_env, models_dir=models_dir),
        ],
        progress_bar=True,
        reset_num_timesteps=False,  # False to continue from checkpoint
    )

    final_model_path = f"{models_dir}/{MODEL_PREFIX}_{new_run_id}"
    model.save(final_model_path)

    # Save best model
    best_model_path = f"{models_dir}/{BEST_MODEL}/{BEST_MODEL}"
    run.save(f"{best_model_path}.zip", base_path=models_dir)

    print(f"\nContinued training completed!")
    print(f"Final model saved: {final_model_path}.zip")
    print(f"New run ID: {new_run_id}")

    env.close()
    eval_env.close()
    run.finish()


def download_and_evaluate(run_id: str):
    """Download model from wandb and evaluate it."""
    print_header("Downloading and Evaluating PPO Model")

    _, output_root = get_output_dirs()
    download_dir = os.path.join(output_root, "downloads", run_id)
    model_cache_path = os.path.join(download_dir, "model.zip")

    # Use cached model if available, otherwise download
    if os.path.exists(model_cache_path):
        print(f"Using cached model from {download_dir}")
    else:
        print(f"Downloading model from wandb run: {run_id}")
        model_cache_path = download_model_from_wandb(run_id, download_dir, MODEL_PREFIX)
        print(f"Model cached to {download_dir}")

    evaluate_ppo_race(model_path=model_cache_path)


def main():
    parser = argparse.ArgumentParser(description="Train or evaluate a PPO model for autonomous racing")
    parser.add_argument(
        "--m",
        choices=["t", "e", "d", "c"],
        default="t",
        help="Run mode: 't' to train a new model, 'e' to evaluate, 'd' to download and evaluate, 'c' to continue training",
    )
    parser.add_argument(
        "--path",
        type=str,
        default="",
        help="Path to trained model for evaluation or continue training (uses latest if not specified for mode 'e')",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default="",
        help="Wandb run ID to download model from (required for mode 'd')",
    )
    parser.add_argument(
        "--additional_timesteps",
        type=int,
        default=10_000_000,
        help="Additional timesteps to train for when continuing training (mode 'c'). Default: 10,000,000",
    )

    args = parser.parse_args()

    if args.m == "t":
        train_ppo_race()
    elif args.m == "e":
        evaluate_ppo_race(model_path=args.path)
    elif args.m == "d":
        if not args.run_id:
            parser.error("--run_id is required when using mode 'd' (download)")
        download_and_evaluate(run_id=args.run_id)
    elif args.m == "c":
        if not args.path:
            parser.error("--path is required when using mode 'c' (continue training)")
        continue_training_ppo_race(model_path=args.path, additional_timesteps=args.additional_timesteps)


if __name__ == "__main__":
    main()
