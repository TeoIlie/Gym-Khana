Training
========

Gym-Khana integrates with `Stable Baselines3 <https://stable-baselines3.readthedocs.io/>`_ for PPO training and `Weights & Biases <https://wandb.ai/>`_ for experiment tracking.

Training scripts
----------------

The main racing training script is ``train/ppo_race.py``. The recovery training script is ``train/ppo_recover.py``. Both support the following modes:

1. **Train** (``--m t``): Train a new model with parallel environments using ``SubprocVecEnv``
2. **Evaluate** (``--m e``): Evaluate a trained model with visualization
3. **Download** (``--m d``): Fetch a model from wandb and evaluate it
4. **Continue** (``--m c``): Continue training from an existing checkpoint
5. **Transfer** (``--m f``): Transfer a pretrained model to a new task, preserving network weights but resetting optimizer, LR schedule, and optionally resetting ``log_std`` for fresh exploration and critic network for fresh value approximation

Examples:

.. code:: bash

   # Train a new racing model
   python3 train/ppo_race.py --m t

   # Evaluate a local model (uses latest wandb run if --path not specified)
   python3 train/ppo_race.py --m e
   python3 train/ppo_race.py --m e --path /path/to/model.zip

   # Download from wandb and evaluate
   python3 train/ppo_race.py --m d --run_id <wandb_run_id>

   # Continue training
   python3 train/ppo_race.py --m c --path /path/to/model.zip --additional_timesteps 10000000

Detailed usage guidelines are at the top of each training script file.

Callbacks
---------

Default SB3 callbacks used during training:

- ``WandbCallback`` — log metrics to wandb
- ``CheckpointCallback`` — save periodic checkpoints
- ``EvalCallback`` — evaluate during training

Curriculum learning
-------------------

A custom ``CurriculumLearningCallback`` is available for recovery training. It gradually expands the recovery state initialization ranges as the agent's success rate improves.

CL is configured in ``train/config/gym_config.yaml`` under the ``curriculum`` heading:

.. code:: yaml

   curriculum:
     enabled: true
     n_stages: ...
     success_threshold: ...
     v_range: [...]
     beta_range: [...]

.. note::

   Curriculum learning is only supported for recovery training (``training_mode: "recover"``), accessed through ``train/ppo_recover.py``.

Wandb integration
-----------------

Models and training runs are logged to: https://wandb.ai/teo-altum-quinque-queen-s-university/projects

Formatting and linting
----------------------

The project uses ``ruff`` for formatting and linting:

.. code:: bash

   ruff check --fix .  # lint + auto-fix (unused imports, import sorting)
   ruff format .       # format (Black-compatible)

Pre-commit hooks run both automatically on ``git commit`` (configured in ``.pre-commit-config.yaml``).
