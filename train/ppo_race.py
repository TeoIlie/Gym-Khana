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

from train.config.env_config import PROJECT_NAME, TRACK_POOL, get_drift_train_config, get_drift_test_config
from train.train_common import TrainingProfile, main

profile = TrainingProfile(
    project_name=PROJECT_NAME,
    track_pool=TRACK_POOL,
    train_config=get_drift_train_config(),
    test_config=get_drift_test_config(),
    display_name="PPO Race",
    model_prefix="ppo_race",
)

if __name__ == "__main__":
    main(profile)
