import wandb
import os
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from wandb.integration.sb3 import WandbCallback
from train.config.env_config import N_ENVS, N_STEPS, PROJECT_NAME, SEED
from train.training_utils import get_output_dirs, make_output_dirs, get_ckpt_callback, make_subprocvecenv
from f1tenth_gym.envs.f110_env import F110Env

# toggle this to train or evaluate
TRAIN = False

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
    "reset_config": {"type": "rl_random_static"},
    "normalize_act": True,
    "normalize_obs": True,
    "predictive_collision": False,
    "wall_deflection": False,
    "lookahead_n_points": 7,
    "lookahead_ds": 0.7,
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
        model.learn(
            total_timesteps=100_000_000,
            callback=[
                WandbCallback(gradient_save_freq=0, model_save_path=models_dir, verbose=2),
                get_ckpt_callback(models_dir=models_dir, save_freq=10000),
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
        model_path = os.path.join(proj_root, "wandb", "latest-run", "files", "model.zip")
        model = PPO.load(model_path, print_system_info=True, device="cpu")

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
        done = False
        total_reward = 0.0

        while not done:
            action, _states = model.predict(obs, deterministic=True)

            obs, reward, done, trunc, info = eval_env.step(action)
            total_reward += reward
            eval_env.render()

            # VecEnv resets automatically
            # if done:
            #   obs = env.reset()
        eval_env.close()
        print(f"Total reward: {total_reward}")


if __name__ == "__main__":
    main()
