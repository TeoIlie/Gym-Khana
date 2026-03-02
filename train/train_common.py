"""
Orchestration methods for model training, downloading, and evaluation
"""

import argparse
import os
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from wandb.integration.sb3 import WandbCallback

import wandb
from train.callbacks import make_curriculum_callback
from train.config.env_config import (
    ADDITIONAL_TIMESTEPS,
    BEST_MODEL,
    CKPT_SAVE_FREQ,
    END_LEARNING_RATE,
    EVAL_SEED,
    N_ENVS,
    N_STEPS,
    SEED,
    START_LEARNING_RATE,
    TOTAL_TIMESTEPS,
    TRANSFER_RESET_LOG_STD,
    get_curriculum_config,
    get_env_id,
)
from train.train_utils import (
    download_model_from_wandb,
    extract_rl_config,
    get_ckpt_callback,
    get_eval_callback,
    get_output_dirs,
    linear_schedule,
    log_best_eval_timestep,
    make_eval_env,
    make_output_dirs,
    make_subprocvecenv,
    print_header,
    save_config,
)


@dataclass
class TrainingProfile:
    project_name: str  # wandb project name
    track_pool: list[str] | None  # track pool for multi-map training
    train_config: dict  # from get_*_train_config()
    test_config: dict  # from get_*_test_config()
    display_name: str  # "PPO Race" or "PPO Recover" (for print headers)
    model_prefix: str  # "ppo_race" or "ppo_recover" (for save paths)


def train(profile: TrainingProfile):
    print_header(profile.display_name + " Training")

    proj_root, output_root = get_output_dirs()

    run = wandb.init(
        project=profile.project_name, sync_tensorboard=True, monitor_gym=True, dir=proj_root, save_code=True
    )
    run_id = run.id

    tensorboard_dir, models_dir, config_dir = make_output_dirs(run.id, output_root)
    save_config(profile.train_config, config_dir, "gym_config.yaml")

    env = make_subprocvecenv(SEED, profile.train_config, N_ENVS, profile.track_pool)
    eval_env = make_eval_env(EVAL_SEED, profile.train_config)

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

    curriculum_config = get_curriculum_config()
    save_config(curriculum_config, config_dir, "curriculum_config.yaml")

    callbacks = [
        WandbCallback(gradient_save_freq=0, verbose=2),
        get_ckpt_callback(models_dir=models_dir, save_freq=CKPT_SAVE_FREQ),
        get_eval_callback(eval_env=eval_env, models_dir=models_dir),
    ]
    curriculum_cb = make_curriculum_callback(
        curriculum_config, training_mode=profile.train_config.get("training_mode", "")
    )
    if curriculum_cb is not None:
        callbacks.append(curriculum_cb)

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callbacks,
        progress_bar=True,
    )

    final_model_path = f"{models_dir}/{profile.model_prefix}_{run_id}"
    model.save(final_model_path)

    # Save best model
    best_model_path = f"{models_dir}/{BEST_MODEL}/{BEST_MODEL}"
    run.save(f"{best_model_path}.zip", base_path=models_dir)

    env.close()
    eval_env.close()

    log_best_eval_timestep(models_dir)

    run.finish()


def evaluate(profile: TrainingProfile, model_path: str = ""):
    print_header(profile.display_name + " Evaluation")

    proj_root, _ = get_output_dirs()

    if model_path == "":
        model_path = os.path.join(proj_root, "wandb", "latest-run", "files", "model.zip")

    model = PPO.load(model_path, print_system_info=True, device="cpu")
    print(f"Loaded model from {model_path}")

    eval_env = gym.make(
        get_env_id(),
        config=profile.test_config,
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


def continue_training(profile: TrainingProfile, model_path: str, additional_timesteps: int = ADDITIONAL_TIMESTEPS):
    """
    Continue training from a saved model checkpoint with a new wandb run.

    Args:
        model_path: Path to saved model.zip file
        additional_timesteps: additional steps to train
    """
    print_header(profile.display_name + " - Continue Training")

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
        project=profile.project_name,
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
    env = make_subprocvecenv(SEED, profile.train_config, N_ENVS, profile.track_pool)
    eval_env = make_eval_env(EVAL_SEED, profile.train_config)

    model = PPO.load(model_path, env=env, device="auto")

    model.tensorboard_log = tensorboard_dir

    print("Model loaded successfully")
    print("Continuing from checkpoint's timestep count")

    save_config(profile.train_config, config_dir, "gym_config.yaml")

    rl_config = extract_rl_config(model, additional_timesteps, N_ENVS)
    save_config(rl_config, config_dir, "rl_config.yaml")

    curriculum_config = get_curriculum_config()
    save_config(curriculum_config, config_dir, "curriculum_config.yaml")

    callbacks = [
        WandbCallback(gradient_save_freq=0, verbose=2),
        get_ckpt_callback(models_dir=models_dir, save_freq=CKPT_SAVE_FREQ),
        get_eval_callback(eval_env=eval_env, models_dir=models_dir),
    ]
    curriculum_cb = make_curriculum_callback(
        curriculum_config, training_mode=profile.train_config.get("training_mode", "")
    )
    if curriculum_cb is not None:
        callbacks.append(curriculum_cb)

    print("\nContinuing training...")

    model.learn(
        total_timesteps=additional_timesteps,
        callback=callbacks,
        progress_bar=True,
        reset_num_timesteps=False,  # False to continue from checkpoint
    )

    final_model_path = f"{models_dir}/{profile.model_prefix}_{new_run_id}"
    model.save(final_model_path)

    # Save best model
    best_model_path = f"{models_dir}/{BEST_MODEL}/{BEST_MODEL}"
    run.save(f"{best_model_path}.zip", base_path=models_dir)

    print("\nContinued training completed!")
    print(f"Final model saved: {final_model_path}.zip")
    print(f"New run ID: {new_run_id}")

    env.close()
    eval_env.close()

    log_best_eval_timestep(models_dir)

    run.finish()


def transfer_train(
    profile: TrainingProfile,
    model_path: str,
    additional_timesteps: int = ADDITIONAL_TIMESTEPS,
    reset_log_std: float | None = TRANSFER_RESET_LOG_STD,
):
    """
    Transfer a trained model to a new task with fresh optimizer, LR schedule, and optional log_std reset.

    Loads pretrained weights (preserving learned dynamics knowledge), but resets training state
    so the model adapts to the new task's reward function from a clean optimization starting point.

    Args:
        profile: Target task's training profile (may differ from source model's task)
        model_path: Path to source model .zip file
        additional_timesteps: Total timesteps for the transfer training run
        reset_log_std: Value to reset log_std to (None to keep source model's value)
    """
    print_header(profile.display_name + " - Transfer Training")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if not model_path.endswith(".zip"):
        raise ValueError(f"Model path must be a .zip file, got: {model_path}")

    if additional_timesteps <= 0:
        raise ValueError(f"additional_timesteps must be positive, got: {additional_timesteps}")

    print(f"Loading model from: {model_path}")
    print(f"Target task: {profile.display_name}")
    print(f"Additional timesteps: {additional_timesteps:,}")
    print(f"Reset log_std: {reset_log_std}")

    proj_root, output_root = get_output_dirs()

    run = wandb.init(
        project=profile.project_name,
        sync_tensorboard=True,
        monitor_gym=True,
        dir=proj_root,
        save_code=True,
    )
    new_run_id = run.id

    print(f"New run ID: {new_run_id}")

    # Create output directories for this transfer run
    tensorboard_dir, models_dir, config_dir = make_output_dirs(new_run_id, output_root)

    # Uses current env_config.py
    env = make_subprocvecenv(SEED, profile.train_config, N_ENVS, profile.track_pool)
    eval_env = make_eval_env(EVAL_SEED, profile.train_config)

    model = PPO.load(model_path, env=env, device="auto")
    model.tensorboard_log = tensorboard_dir

    # Fresh LR schedule - must set both learning_rate (source of truth) and lr_schedule (cached callable used at each update)
    model.learning_rate = linear_schedule(START_LEARNING_RATE, END_LEARNING_RATE)
    model.lr_schedule = model.learning_rate

    print(f"\nReset LR schedule: {model.lr_schedule(1.0)} -> {model.lr_schedule(0.0)}")

    # Fresh Adam optimizer - use optimizer_class/optimizer_kwargs to match SB3's original construction
    model.policy.optimizer = model.policy.optimizer_class(
        model.policy.parameters(),
        lr=model.learning_rate(1.0),
        **model.policy.optimizer_kwargs,
    )

    pg = model.policy.optimizer.param_groups[0]
    print(f"Reset optimizer: {model.policy.optimizer.__class__.__name__}(lr={pg['lr']}, eps={pg['eps']})")

    # Reset update counter for clean logging (cosmetic — not used in any schedule or logic)
    model._n_updates = 0

    # Reset log_std for fresh exploration in the new task, if set in config file
    if reset_log_std is not None:
        if not hasattr(model.policy, "log_std"):
            raise AttributeError("Policy has no log_std parameter (not a continuous action distribution)")
        model.policy.log_std.data.fill_(reset_log_std)
        print(f"Reset log_std to {reset_log_std}")

    print("\nModel loaded successfully")

    save_config(profile.train_config, config_dir, "gym_config.yaml")

    rl_config = extract_rl_config(model, additional_timesteps, N_ENVS)
    save_config(rl_config, config_dir, "rl_config.yaml")

    transfer_config = {
        "original_model_path": model_path,
        "additional_timesteps": additional_timesteps,
        "reset_log_std": reset_log_std,
    }
    save_config(transfer_config, config_dir, "transfer_config.yaml")

    curriculum_config = get_curriculum_config()
    save_config(curriculum_config, config_dir, "curriculum_config.yaml")

    callbacks = [
        WandbCallback(gradient_save_freq=0, verbose=2),
        get_ckpt_callback(models_dir=models_dir, save_freq=CKPT_SAVE_FREQ),
        get_eval_callback(eval_env=eval_env, models_dir=models_dir),
    ]
    curriculum_cb = make_curriculum_callback(
        curriculum_config, training_mode=profile.train_config.get("training_mode", "")
    )
    if curriculum_cb is not None:
        callbacks.append(curriculum_cb)

    print("\nStarting transfer training...")

    model.learn(
        total_timesteps=additional_timesteps,
        callback=callbacks,
        progress_bar=True,
        reset_num_timesteps=True,  # New task = fresh step counter and LR schedule
    )

    final_model_path = f"{models_dir}/{profile.model_prefix}_{new_run_id}"
    model.save(final_model_path)

    best_model_path = f"{models_dir}/{BEST_MODEL}/{BEST_MODEL}"
    run.save(f"{best_model_path}.zip", base_path=models_dir)

    print("\nTransfer training completed!")
    print(f"Final model saved: {final_model_path}.zip")
    print(f"New run ID: {new_run_id}")

    env.close()
    eval_env.close()

    log_best_eval_timestep(models_dir)

    run.finish()


def download_and_evaluate(profile: TrainingProfile, run_id: str):
    """Download model from wandb and evaluate it."""
    print_header("Downloading and Evaluating Model from WandB")

    _, output_root = get_output_dirs()
    download_dir = os.path.join(output_root, "downloads", run_id)
    model_cache_path = os.path.join(download_dir, "model.zip")

    # Use cached model if available, otherwise download
    if os.path.exists(model_cache_path):
        print(f"Using cached model from {download_dir}")
    else:
        print(f"Downloading model from wandb run: {run_id}")
        model_cache_path = download_model_from_wandb(run_id, download_dir, profile.model_prefix, profile.project_name)
        print(f"Model cached to {download_dir}")

    evaluate(profile=profile, model_path=model_cache_path)


def main(profile: TrainingProfile):
    """Parse user argument and reroute to correct method"""
    parser = argparse.ArgumentParser(description="Train or evaluate a model")
    parser.add_argument(
        "--m",
        choices=["t", "e", "d", "c", "f"],
        default="t",
        help="Run mode: 't' train, 'e' evaluate, 'd' download+evaluate, 'c' continue training, 'f' transfer/fine-tune",
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
    args = parser.parse_args()

    if args.m == "t":
        train(profile=profile)
    elif args.m == "e":
        evaluate(profile=profile, model_path=args.path)
    elif args.m == "d":
        if not args.run_id:
            parser.error("--run_id is required when using mode 'd' (download)")
        download_and_evaluate(profile=profile, run_id=args.run_id)
    elif args.m == "c":
        if not args.path:
            parser.error("--path is required when using mode 'c' (continue training)")
        continue_training(profile=profile, model_path=args.path)
    elif args.m == "f":
        if not args.path:
            parser.error("--path is required when using mode 'f' (transfer/fine-tune)")
        transfer_train(profile=profile, model_path=args.path)
