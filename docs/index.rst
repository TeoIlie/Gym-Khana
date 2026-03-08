Gym-Khana Documentation
========================

Gym-Khana is a fast, deterministic Gymnasium environment for training autonomous 1/10th scale racing agents to race, drift, and recover. It supports RL training with SB3 and wandb, as well as classical controllers like MPC and Pure Pursuit.

Built on top of `f1tenth_gym <https://github.com/f1tenth/f1tenth_gym>`_ from UPenn (`original docs <https://f1tenth-gym.readthedocs.io>`_).

.. image:: assets/demo_drift.gif
   :width: 400
   :align: center
   :alt: PPO agent drifting on the Spielberg circuit

Features
--------

- **Realistic vehicle dynamics**: Kinematic, single-track, multi-body, and single-track drift (PAC2002 tire) models
- **Fast simulation**: Numba JIT-compiled dynamics for faster-than-real-time execution
- **RL integration**: SB3 and wandb support with parallel training via SubprocVecEnv
- **Drift training**: STD model with Pacejka tire physics for learning aggressive driving
- **Recovery training**: Train policies to stabilize an out-of-control vehicle with curriculum learning
- **Deterministic**: Seeded randomness and simultaneous agent stepping for reproducible experiments
- **Multi-agent support**: Multiple competing vehicles in the same environment
- **LiDAR simulation**: Accurate ray-casting for perception research
- **Configurable observations & actions**: Multiple types with normalization support
- **Visualization**: Real-time rendering with debug overlays (Frenet projection, lookahead curvatures, track lines)

GitHub: https://github.com/TeoIlie/Gym-Khana

See :doc:`installation` to get started.

.. toctree::
   :caption: GETTING STARTED
   :maxdepth: 2

   installation
   quickstart

.. toctree::
   :caption: USER GUIDE
   :maxdepth: 2

   configuration
   training
   architecture
   controllers
   analysis
   known_issues

.. toctree::
   :caption: API REFERENCE
   :maxdepth: 2

   api/env
   api/base_classes
   api/dynamic_models
   api/observation
   api/action
   api/track
