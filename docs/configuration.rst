Configuration
=============

All configuration is passed through the ``config`` dictionary to ``gym.make()``.

Default configurations
----------------------

Default configurations for RL training and testing are in ``train/config/env_config.py``, with parameters loaded from:

- ``train/config/rl_config.yaml`` — RL hyperparameters (n_envs, learning rate, batch size, etc.)
- ``train/config/gym_config.yaml`` — Gym environment parameters (map, model, timestep, etc.)

Convenience functions return ready-made configs:

- ``get_drift_test_config()`` / ``get_drift_train_config()``
- ``get_recovery_test_config()`` / ``get_recovery_train_config()``

Vehicle model and control
-------------------------

.. code:: python

   env = gym.make('gymkhana:gymkhana-v0', config={
       'model': 'std',                              # STD for drifting with PAC2002 tire model
       'params': GKEnv.f1tenth_std_vehicle_params(),  # Drift parameters for 1/10 scale
   })

Available models: ``ks`` (kinematic), ``st`` (single-track), ``mb`` (multi-body), ``std`` (single-track drift). See :doc:`api/dynamic_models` for details.

For action space configuration (``control_input``, ``normalize_act``), see :doc:`api/action`.

Training mode
-------------

Set ``training_mode`` to define the training goal. This modifies the reset, initialization, track, and reward settings:

- ``"race"`` (default) — used by ``train/ppo_race.py`` for training racing policies
- ``"recover"`` — used by ``train/ppo_recover.py`` to train policies for stabilizing an out-of-control vehicle

Track direction
---------------

Set ``track_direction`` to control which direction to drive around the track:

- ``"normal"`` (default) — follow waypoint direction (may be CW or CCW depending on the map)
- ``"reverse"`` — drive in the opposite direction
- ``"random"`` — randomly choose direction at each reset (50/50), useful for learning both left and right cornering

Observation configuration
-------------------------

.. code:: python

   config = {
       'observation_config': {'type': 'drift'},  # or 'rl' for standard observations
       'lookahead_n_points': 10,                  # Number of lookahead points (default: 10)
       'lookahead_ds': 0.3,                       # Spacing between points in meters (default: 0.3m)
       'sparse_width_obs': False,                 # False: all widths, True: only 1st and last
       'normalize_obs': True,                     # Enable observation normalization (only 'drift' obs)
   }

See :doc:`api/observation` for details.

Collision and safety
--------------------

.. code:: python

   config = {
       'predictive_collision': True,   # True: TTC collision checking, False: Frenet-based
       'wall_deflection': False,       # False: edges are boundaries, True: edges cause collision
   }

Note that ``predictive_collision`` also modifies the reward function.

Reward configuration
--------------------

.. code:: python

   config = {
       'progress_gain': 1.0,          # Gain multiplier for forward progress reward (>= 1)
       'out_of_bounds_penalty': ...,   # Penalty for driving off the track boundary
       'negative_vel_penalty': ...,    # Penalty for driving backward
       'max_episode_steps': ...,       # Maximum number of episode steps
   }

Reset options
-------------

Vehicles can be initialized at specific configurations via ``env.reset(options=...)``.

**Poses** — reset agents at a specific ``[x, y, yaw]`` in Cartesian coordinates:

.. code:: python

   # Single agent
   poses = np.array([[x, y, yaw]])
   env.reset(options={"poses": poses})

   # Multiple agents
   poses = np.array([[x1, y1, yaw1],
                     [x2, y2, yaw2]])
   env.reset(options={"poses": poses})

**States** — reset agents to a full 7-d state (only for ``model='std'``):

.. code:: python

   # [x, y, delta, v, yaw, yaw_rate, slip_angle]
   states = np.array([[x, y, delta, v, yaw, yaw_rate, slip_angle]])
   env.reset(options={"states": states})

   # Front & rear angular wheel velocities are automatically initialized
   # to form the full 9-d state for the STD model type

Only one of ``poses`` or ``states`` can be used per reset call. To use Frenet coordinates, convert first using ``frenet_to_cartesian()`` in ``gymkhana/envs/track/track.py``.

Debugging and visualization
---------------------------

.. code:: python

   config = {
       'render_mode': 'human',                    # Enable visualization window
       'render_track_lines': True,                # Centerline (green) and raceline (red)
       'render_lookahead_curvatures': True,       # Lookahead curvature points (yellow)
       'render_arc_length_annotations': True,     # Arc-length points along centerline (orange)
       'arc_length_annotation_interval': 2.0,     # Spacing in metres (default: 2.0)
       'debug_frenet_projection': True,           # Visualize Frenet coordinate accuracy
       'record_obs_min_max': True,                # Record min/max obs for normalization tuning
   }

Debug with breakpoints by looping through environment steps (see ``tests/drift_debug.py``).

Custom maps
-----------

Maps follow the ROS map convention: a ``.yaml`` metadata file and a single-channel black-and-white image (black = obstacles, white = free space).

The ``resolution`` field (m/pixel) and the first two values of ``origin`` (bottom-left corner in world coordinates) are used by the environment. Place both files in the same directory with the same base name, then:

.. code:: python

   env = gym.make('gymkhana:gymkhana-v0', config={
       'map': '/your/path/to/map',  # without extension
       'map_ext': '.png',
   })

Maps are managed through a two-tier system:

- **Local** ``maps/`` **directory**: git submodule from https://github.com/TeoIlie/F1TENTH_Racetracks — used for development
- **User cache** ``~/.gymkhana/maps/``: auto-downloaded from GitHub releases for pip-installed users

To update the maps submodule: ``git pull --recurse-submodules``
