import wandb
import os
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from wandb.integration.sb3 import WandbCallback
from train.config.env_config import N_ENVS, PROJECT_NAME, SEED
from train.training_utils import get_output_dirs, make_env, make_output_dirs, get_ckpt_callback

# toggle this to train or evaluate
TRAIN = True

# config for train and test
CONFIG = {
    "map": "Drift_large",
    "num_agents": 1,
    "timestep": 0.01,
    "num_beams": 36,
    "integrator": "rk4",
    "control_input": ["speed", "steering_angle"],
    "observation_config": {"type": "race"},
    "reset_config": {"type": "rl_random_static"},
    "normalize_act": True,
    "normalize_obs": True,
    "predictive_collision": False,
    "wall_deflection": False,
    "lookahead_n_points": 3,
    "lookahead_ds": 0.5,
}


def main():
    if TRAIN:
        proj_root, output_root = get_output_dirs()

        run = wandb.init(project=PROJECT_NAME, sync_tensorboard=True, monitor_gym=True, dir=proj_root, save_code=True)
        run_id = run.id

        print("=" * 40)
        print("PPO Race Training")
        print("=" * 40)

        tensorboard_dir, models_dir, videos_dir = make_output_dirs(run.id, output_root)

        env = SubprocVecEnv([make_env(seed=SEED, rank=i, config=CONFIG) for i in range(N_ENVS)])

        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=tensorboard_dir, device="auto", seed=SEED)
        model.learn(
            total_timesteps=10_000_000,
            callback=[
                WandbCallback(gradient_save_freq=0, model_save_path=models_dir, verbose=2),
                get_ckpt_callback(models_dir=models_dir, save_freq=200000),
            ],
            progress_bar=True,
        )

        final_model_path = f"{models_dir}/ppo_drift_final_{run_id}"
        model.save(final_model_path)

        env.close()

        run.finish()

    else:
        proj_root, _ = get_output_dirs()
        run_id = "q81h6jga"  # replace with your run ID
        model_path = os.path.join(proj_root, "outputs", "models", "wlbjsjbv", "checkpoints", f"ckpt_3500000_steps.zip")
        model = PPO.load(model_path, print_system_info=True, device="cpu")
        eval_env = gym.make(
            "f1tenth_gym:f1tenth-v0",
            config=CONFIG,
            render_mode="human",
        )
        np.random.seed()
        obs, info = eval_env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, trunc, info = eval_env.step(action)
            eval_env.render()

            # VecEnv resets automatically
            # if done:
            #   obs = env.reset()
        eval_env.close()


if __name__ == "__main__":
    main()
