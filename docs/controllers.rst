Controllers
===========

In addition to training Deep Reinforcement Learning policies, Gym-Khana includes several example classical controllers for path tracking and vehicle stabilization.

Pure Pursuit
------------

A Pure Pursuit waypoint follower that tracks the raceline using a lookahead circle intersection method.

Source: ``examples/waypoint_follow.py``

.. code:: bash

   cd examples
   python3 waypoint_follow.py

The ``PurePursuitPlanner`` class finds the nearest point on the raceline trajectory, projects a lookahead circle forward, and computes the steering angle from the Ackermann geometry. It also includes render callbacks for visualizing the lookahead point and local plan.

Stanley and PD Controllers
--------------------------

Source: ``examples/controllers/steer_controller.py``

Three steering controllers are provided, all operating on Frenet-frame observations:

``PDSteerController``
   Path tracking controller that minimizes lateral deviation (``frenet_n``) and heading error (``frenet_u``) with proportional gains.

``PDStabilityController``
   Recovery controller that directly minimizes sideslip angle (beta) and yaw rate (r) to stabilize an out-of-control vehicle. Useful as a baseline for comparing against learned recovery policies.

``StanleyController``
   Stanley path tracking controller with speed-adaptive cross-track error correction. Combines heading error correction with a nonlinear cross-track term that provides aggressive low-speed corrections and gentle high-speed corrections:

   .. code::

      delta = k_heading * theta_e + arctan(k * e / (|v| + k_soft))

MPC Controllers
---------------

Model Predictive Control implementations.

Source: ``examples/controllers/mpc/``

.. note::

   MPC controllers require ``acados`` built from source. See :ref:`additional-dependencies` in the Installation guide for setup instructions.

**Kinematic MPC** (``examples/controllers/mpc/kmpc/``)
   Uses the kinematic single-track model for trajectory optimization. Suitable for low-speed scenarios.

   Example usage:

   .. code:: bash

      cd examples
      python3 kmpc_race_example.py

**Single-Track MPC** (``examples/controllers/mpc/stmpc/``)
   Uses the single-track dynamic model for trajectory optimization. Handles higher speeds and tire slip dynamics.

   Example usage:

   .. code:: bash

      cd examples
      python3 stmpc_race_example.py

Both MPC controllers use a ``GymBridge`` interface (``examples/controllers/mpc/gym_bridge.py``) that translates between the Gym observation/action format and the MPC solver's state/input representation.
