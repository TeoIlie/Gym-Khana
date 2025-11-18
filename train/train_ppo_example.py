import wandb
import os
import gymnasium as gym
from stable_baselines3 import PPO
from wandb.integration.sb3 import WandbCallback
from train.config.env_config import PROJECT_NAME
from train.training_utils import get_output_dirs, make_output_dirs

# toggle this to train or evaluate
train = True


def main():
    if train:
        proj_root, output_root = get_output_dirs()

        run = wandb.init(project=PROJECT_NAME, sync_tensorboard=True, monitor_gym=True, dir=proj_root)
        run_id = run.id
        tensorboard_dir, models_dir, videos_dir = make_output_dirs(run.id, output_root)

        env = gym.make(
            "f1tenth_gym:f1tenth-v0",
            config={
                "map": "Spielberg",
                "num_agents": 1,
                "timestep": 0.01,
                "num_beams": 36,
                "integrator": "rk4",
                "control_input": ["speed", "steering_angle"],
                "observation_config": {"type": "rl"},
                "reset_config": {"type": "rl_random_static"},
            },
        )

        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=tensorboard_dir, device="auto", seed=42)
        model.learn(
            total_timesteps=2_000,
            callback=WandbCallback(gradient_save_freq=0, model_save_path=models_dir, verbose=2),
            progress_bar=True,
        )

        final_model_path = f"{models_dir}/ppo_drift_final_{run_id}"
        model.save(final_model_path)

        # If debugging observation normalization, explicitly close the environment to
        # trigger observation statistics printing -> env is wrapped in DummyVecEnv, so we need to close the underlying env
        # if debug_obs_norm:
        #     if hasattr(env, "envs"):
        #         # VecEnv wrapper - close the underlying environments
        #         for underlying_env in env.envs:
        #             underlying_env.close()
        #     else:
        #         # Regular env
        #         env.close()
        env.close()
        run.finish()

    else:
        model_path = os.path.join(os.path.dirname(__file__), "models", "70ftjvia", "model.zip")
        model = PPO.load(model_path, print_system_info=True, device="cpu")
        eval_env = gym.make(
            "f1tenth_gym:f1tenth-v0",
            config={
                "map": "Spielberg",
                "num_agents": 1,
                "timestep": 0.01,
                "num_beams": 36,
                "integrator": "rk4",  # this is the Runge-Kutta method Dimitria mentioned!
                "control_input": ["speed", "steering_angle"],
                "observation_config": {"type": "rl"},
                "reset_config": {"type": "rl_random_static"},
            },
            render_mode="human",
        )
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
