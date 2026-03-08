Quickstart
==========

After :doc:`installing <installation>` Gym-Khana, try the included examples.

Running examples
----------------

.. code:: bash

   cd examples
   python3 waypoint_follow.py        # Pure Pursuit waypoint following
   python3 controller_example.py     # P controller centerline following
   python3 drift_debug.py            # Drift debugging with visualization

Environment basics
------------------

An environment can be instantiated without extra arguments. By default it spawns two agents on the Spielberg map using the single-track model. The ego agent index is 0.

``reset(options)`` returns ``(obs, info)``. Pass ``"poses"`` as a ``(num_agents, 3)`` array of ``[x, y, yaw]`` per agent.

``step(action)`` returns ``(obs, reward, terminated, truncated, info)``. The action is a ``(num_agents, 2)`` array of control inputs per agent (see :doc:`api/action`). The reward is the physics timestep. ``terminated`` flips on collision or after 2 laps; ``truncated`` flips at ``max_episode_steps``.

A working example is in ``examples/waypoint_follow.py``.

Simulation loop
---------------

.. code:: python

   import gymnasium as gym
   import numpy as np
   from your_custom_policy import planner  # your policy/motion planner

   env = gym.make('gymkhana:gymkhana-v0', render_mode='human')

   # reset with initial poses for 2 agents
   poses = np.array([[0.0, 0.0, 0.0],   # ego agent
                      [2.0, 0.0, 0.0]])   # 2nd agent
   obs, info = env.reset(options={'poses': poses})

   my_planner = planner()
   done = False

   while not done:
       actions = my_planner.plan(obs)
       obs, reward, terminated, truncated, info = env.step(actions)
       done = terminated or truncated

   env.close()
