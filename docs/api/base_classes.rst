Base Classes
============

The base classes handle physical simulation and interaction between vehicles.

Source: ``gymkhana/envs/base_classes.py``

RaceCar
-------

Handles the simulation of a single vehicle's dynamics model and laser scan generation.

Each ``RaceCar`` instance:

- Maintains the vehicle's state (position, velocity, orientation, etc.)
- Steps the selected dynamics model forward in time
- Generates simulated LiDAR scans via the laser model

Simulator
---------

Orchestrates the multi-agent simulation, including:

- Managing multiple ``RaceCar`` instances
- Collision checking between agents and with track boundaries
- Stepping all agents simultaneously for deterministic simulation

The ``step`` method (line 503) is the core simulation loop. All agents' physics are stepped simultaneously, and all randomness is seeded for reproducibility. The explicit stepping enables faster-than-real-time execution.

API reference
-------------

.. automodule:: gymkhana.envs.base_classes
   :members:
   :undoc-members:
