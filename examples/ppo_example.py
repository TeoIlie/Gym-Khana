import gymnasium as gym
from stable_baselines3 import PPO
import os

# if using wandb (recommended):
from wandb.integration.sb3 import WandbCallback
import wandb

from f1tenth_gym.envs.f110_env import F110Env

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# toggle this to train or evaluate
train = True
debug_obs_norm = True

if train:
    run = wandb.init(
        project="f1tenth_gym_ppo",
        sync_tensorboard=True,
        save_code=True,
    )

    env = gym.make(
        "f1tenth_gym:f1tenth-v0",
        config={
            "map": "IMS",  # Open area for drift practice
            "num_agents": 1,  # Single agent for focused learning
            "timestep": 0.01,  # High-frequency control (100Hz)
            "integrator": "rk4",  # Accurate physics integration
            "model": "std",  # Single Track dynamic bicycle model with tire slip
            "control_input": ["accl", "steering_angle"],
            "observation_config": {"type": "drift"},  # 6D drift state: [vx, vy, yaw_rate, delta, frenet_u, frenet_n]
            "reset_config": {"type": "rl_random_static"},
            "params": F110Env.f1tenth_std_vehicle_params(),
            "normalize": True,
            "record_obs_min_max": debug_obs_norm,
        },
    )

    # set directories to /f1tenth_gym/examples/... regardless of where this is run from
    tensorboard_dir = os.path.join(SCRIPT_DIR, "runs", run.id)
    model_dir = os.path.join(SCRIPT_DIR, "models", run.id)

    # will be faster on cpu (for now)
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=tensorboard_dir, device="cpu", seed=42)
    model.learn(
        total_timesteps=2_000_000,
        callback=WandbCallback(gradient_save_freq=0, model_save_path=model_dir, verbose=2),
    )

    # If debugging observation normalization, explicitly close the environment to
    # trigger observation statistics printing -> env is wrapped in DummyVecEnv, so we need to close the underlying env
    if debug_obs_norm:
        if hasattr(env, "envs"):
            # VecEnv wrapper - close the underlying environments
            for underlying_env in env.envs:
                underlying_env.close()
        else:
            # Regular env
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
