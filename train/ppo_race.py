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
"""

import wandb
import os
import argparse
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from wandb.integration.sb3 import WandbCallback
from train.config.env_config import N_ENVS, N_STEPS, PROJECT_NAME, SEED
from train.training_utils import (
    get_output_dirs,
    make_output_dirs,
    get_ckpt_callback,
    make_subprocvecenv,
    save_config,
    extract_rl_config,
    download_model_from_wandb,
    print_header,
)
from f1tenth_gym.envs.f110_env import F110Env


# Extract model prefix from filename (e.g., "ppo_race" from "ppo_race.py")
MODEL_PREFIX = os.path.splitext(os.path.basename(__file__))[0]
TOTAL_TIMESTEPS = 10_000
# config for train and test
CONFIG = {
    "map": "Drift_large",
    "num_agents": 1,
    "model": "std",
    "params": F110Env.f1tenth_std_vehicle_params(),  # new vehicle params with rear weight bias
    "timestep": 0.01,
    "num_beams": 2,
    "integrator": "rk4",
    "control_input": ["accl", "steering_angle"],
    "observation_config": {"type": "drift"},
    "reset_config": {"type": "cl_random_static"},
    "normalize_act": True,
    "normalize_obs": True,
    "predictive_collision": False,
    "wall_deflection": False,
    "lookahead_n_points": 3,
    "lookahead_ds": 0.5,
}


def train_ppo_race():
    print_header("PPO Race Training")

    proj_root, output_root = get_output_dirs()

    run = wandb.init(project=PROJECT_NAME, sync_tensorboard=True, monitor_gym=True, dir=proj_root, save_code=True)
    run_id = run.id

    tensorboard_dir, models_dir, config_dir = make_output_dirs(run.id, output_root)
    save_config(CONFIG, config_dir, "gym_config.yaml")

    env = make_subprocvecenv(SEED, CONFIG, N_ENVS)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        n_steps=N_STEPS,
        verbose=1,
        tensorboard_log=tensorboard_dir,
        device="auto",
        seed=SEED,
    )

    rl_config = extract_rl_config(model, TOTAL_TIMESTEPS, N_ENVS)
    save_config(rl_config, config_dir, "rl_config.yaml")

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[
            WandbCallback(gradient_save_freq=0, verbose=2),
            get_ckpt_callback(models_dir=models_dir, save_freq=10000),
        ],
        progress_bar=True,
    )

    final_model_path = f"{models_dir}/{MODEL_PREFIX}_{run_id}"
    model.save(final_model_path)
    run.save(f"{final_model_path}.zip", base_path=models_dir)

    env.close()

    run.finish()


def evaluate_ppo_race(model_path: str = ""):
    print_header("PPO Race Evaluation")

    proj_root, _ = get_output_dirs()

    if model_path == "":
        model_path = os.path.join(proj_root, "wandb", "latest-run", "files", "model.zip")

    model = PPO.load(model_path, print_system_info=True, device="cpu")
    print(f"Loaded model from {model_path}")

    CONFIG["debug_frenet_projection"] = True
    CONFIG["render_track_lines"] = True
    CONFIG["render_lookahead_curvatures"] = True

    eval_env = gym.make(
        "f1tenth_gym:f1tenth-v0",
        config=CONFIG,
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
        choices=["t", "e", "d"],
        default="t",
        help="Run mode: 't' to train a new model, 'e' to evaluate an existing model, or 'd' to download and evaluate from wandb",
    )
    parser.add_argument(
        "--path",
        type=str,
        default="",
        help="Path to trained model for evaluation (uses latest if not specified)",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default="",
        help="Wandb run ID to download model from (required for mode 'd')",
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


if __name__ == "__main__":
    main()
