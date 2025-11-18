# PPO Implementation Plan

To train my drifting controller, I am planning to use **PPO** from SB3 with `wandb` and `tensorboard` integration for model training visualization, and model saving.

## Core functionality

### Overview

Implement training script `train/train_ppo_drift.py` that uses PPO from SB3 to implement a controller that learns how to drift around a track.

### Tasks

| Status | Task |
|----|----|
| [ ] |  Time step is 0.05s → *currently behaving erratically for timestep !+ `0.01`*|
| [ ] |  Parallelized with 400 simulated instances running concurrently → *currently crashing with more than ~150 envs* |
| [ ] |  Unit testing suite |
| [X] |  PPO from SB3 |
| [X] |  Each rollout consists of 1024 steps |
| [X] |  Correct reset config `cl_random_static` (`cl_random_random` also works) |
| [X] |  Training with batch size of 1024 |
| [X] |  Sum of discounted reward is calculated with discount factor gamma=0.99 |
| [X] |  Learning rate decays from `1 x 10^{-3}` to `1 x 10^-4` over the course of training |
| [X] |  120 million time steps |
| [X] |  Actor and critic networks implemented as MLPs |
| [X] |  Actor has 2 hidden layers of 256 neurons |
| [X] |  Critic has two hidden layers of 512 neurons |
| [X] |  Both actor, critic networks use Leaky Rectified Linear Unit activation function with negative slope of 0.2 |

## Code organization and debugging

### Overview

1. Train with a single command

    ```bash
    python training/train_ppo_drift.py
    ```

2. Import environment from a single source of truth in `training/env_config.py`
3. Directories clearly organized
4. Live visualization with **wandb** live metrics and videos, and **tensorboard**

### Tasks

| Status | Task |
|------|-------|
| [ ] | Integration with **wandb** video recording with `monitor_gym=True` to debug training, recording only samples of training to save memory space |
| [ ] | Save config params with `wandb.init(config={..})` to save params for comparison |
| [ ] | Save training config as `config.json` alongside each model |
| [ ] | Periodic checkpoint saving |
| [ ] | Best model tracking with evaluation callback |
| [ ] | Resume capability from checkpoint|
| [ ] | Create a separate evaluation environment for period evaluation of the current policu with full episodes, using SB3 `EvalCallback` |
| [ ] | Enable early stopping with reward plateau detection |
| [X] | Live visualization of training, including key RL metrics such as reward convergence, and NN metrics like `policy_loss` and `value_loss` |
| [X] | Integration with **tensorboard** with `sync_tensorboard=True` |
| [X] | Organized file structure for training, outputs, and **wandb** auto-generated outputs (see below) |
| [X] | `.gitignore` to exclude output directories|
| [X] | Centralized environment config (see below) |

### Folder Structure

```plain
F1TENTH_Gym/
├── train/
|   ├── config/
|   │   ├── gym_config.yaml     # Gym environment configs
|   │   ├── rl_config.yaml     # RL-specific hyperparameters
|   │   └── env_config.py       # Centralized gym environment configs, using yaml files
│   ├── training_utils.py       # Utility functions for use by training scripts, for ex.
│   │                           #    directory setup, callback setup, etc.
│   └── train_ppo_drift.py      # Main training script with all integrations
│   ... train_sac_drift.py      # Possibly other training scripts  
│
├── outputs/                    # All training outputs (gitignored)
│   ├── models/{run_id}/        # Model checkpoints
│   ├── videos/{run_id}/        # Training videos
│   └── tensorboard/{run_id}/   # TensorBoard logs
└── wandb/                      # WandB runs (auto-created, gitignored)

```
