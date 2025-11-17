# PPO Implementation Plan

To train my drifting controller, I am planning to use **PPO** from SB3 with `wandb` and `tensorboard` integration for model training visualization, and model saving.

## Core functionality

### Overview

Implement training script `training/train_ppo_drift.py` that uses PPO from SB3 to implement a controller that learns how to drift around a track.

### Tasks

| Status | Task |
|----|----|
| [ ] | PPO from SB3 |
| [ ] |  Parallelized with 400 simulated instances running concurrently |
| [ ] |  Each rollout consists of 1024 steps with a time step of 0.05s |
| [ ] | Correct reset config - perhaps `cl_random_random`?
| [ ] |  Training with batch size of 1024 |
| [ ] |  Sum of discounted reward is calculated with discount factor gamma=0.99 |
| [ ] |  Learning rate decays from `1 x 10^{-3}` to `1 x 10^-4` over the course of training |
| [ ] |  120 million time steps |
| [ ] |  Actor and critic networks implemented as MLPs |
| [ ] |  Actor has 2 hidden layers of 256 neurons |
| [ ] |  Critic has two hidden layers of 512 neurons |
| [] |  Both actor, critic networks use Leaky Rectified Linear Unit activation function with negative slope of 0.2 |

## Code organization and debugging

### Overview

1. Train with a single command

    ```bash
    python training/train_ppo_drift.py
    ```

2. Import environment from a single source of truth in `training/env_config.py`
3. Directories organized and clear
    - Models → `outputs/models/{wandb_run_id}/`
    - Videos → `outputs/videos/{wandb_run_id}/`
    - Logs → `outputs/tensorboard/{wandb_run_id}/`
    - Wandb → `outputs/wandb`, using `wandb.init(dir="outputs/wandb")` or similar
4. Live visualization with **wandb** live metrics and videos, and **tensorboard**

### Tasks

| Status | Task |
|------|-------|
| [ ] | Integration with **wandb** video recording with `monitor_gym=True` to debug training, recording only samples of training to save memory space |
| [ ] | Save config params with `wandb.init(config={..})` to save params for comparison |
| [ ] | Save training config as `config.json` alongside each model |
| [ ] | Integration with **tensorboard** with `sync_tensorboard=True` |
| [ ] | Organized file structure for training, outputs, and **wandb** auto-generated outputs (see below) |
| [ ] | Live visualization of training, including key RL metrics such as reward convergence, and NN metrics like `policy_loss` and `value_loss` |
| [ ] | Centralized environment config (see below) |
| [ ] | `.gitignore` to exclude `outputs/` and `wandb/ |
| [ ] | Periodic checkpoint saving |
| [ ] | Best model tracking with evaluation callback |
| [ ] | Resume capability from checkpoint|
| [ ] | Create a separate evaluation environment for period evaluation of the current policu with full episodes, using SB3 `EvalCallback` |
| [ ] | Enable early stopping with reward plateau detection |

### Folder Structure

```
F1TENTH_Gym/
├── training/
|   ├── configs/
|   │   ├── gym_config.yaml     # Gym environment configs
|   │   ├── ppo_config.yaml     # RL-specific hyperparameters
|   │   └── env_config.py       # Centralized gym environment configs, using yaml files
│   ├── training_utils.py       # Utility functions for use by training scripts, for ex.
│   │                           #    directory setup, callback setup, etc.
│   └── train_ppo_drift.py      # Main training script with all integrations
│   ... train_sac_drift.py      # Possibly other training scripts  
│
└── outputs/                    # All training outputs (gitignored)
    ├── models/{run_id}/        # Model checkpoints
    ├── videos/{run_id}/        # Training videos
    ├── tensorboard/{run_id}/   # TensorBoard logs
    └── wandb/                  # WandB runs (auto-created, gitignored)

```
