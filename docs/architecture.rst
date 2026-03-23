.. image:: assets/logo.png
   :width: 20
   :align: left
   :alt: logo

Architecture
============

Package structure
-----------------

::

   gymkhana/                  Main package (Gymnasium RL environment)
   ├── envs/
   │   ├── gymkhana_env.py    GKEnv — main environment class
   │   ├── base_classes.py    RaceCar, Simulator core classes
   │   ├── observation.py     Observation factory (rl, drift types)
   │   ├── action.py          Action types and normalization
   │   ├── laser_models.py    LiDAR ray-casting simulation
   │   ├── collision_models.py  Collision detection (TTC, Frenet)
   │   ├── dynamic_models/    Vehicle dynamics models
   │   ├── track/             Track loading, Frenet conversion
   │   ├── reset/             Reset strategies for RL training
   │   └── rendering/         Pygame/PyQt visualization
   examples/                  Reference implementations (Pure Pursuit, etc.)
   train/                     RL training scripts and configuration
   ├── config/                env_config.py, rl_config.yaml, gym_config.yaml
   ├── ppo_race.py            Racing training script
   ├── ppo_recover.py         Recovery training script
   ├── train_common.py        Shared training workflow
   └── train_utils.py         Shared utilities
   tests/                     Test suite including model validation
   maps/                      Racing track definitions (git submodule)

Important files
---------------

- ``gymkhana/envs/base_classes.py:503`` defines the ``step`` method
- See :doc:`api/action` for action space details
- ``gymkhana/envs/dynamic_models/single_track_drift.py`` — STD model with PAC2002 tire model, recommended for drift RL training
- ``gymkhana/envs/dynamic_models/single_track.py`` — basic single-track dynamics without explicit tire model
- ``gymkhana/envs/dynamic_models/multi_body.py`` — most detailed model, but parameters only available for full-scale vehicles

Key dependencies
----------------

- **gymnasium** — RL environment interface
- **stable-baselines3** — PPO and other RL algorithms
- **wandb** — experiment tracking and model logging
- **numba** — JIT compilation for real-time physics
- **pygame / PyQt6** — real-time visualization
- **numpy / scipy** — control and estimation math
