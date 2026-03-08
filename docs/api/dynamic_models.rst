Dynamic Models
==============

Gym-Khana provides four vehicle dynamics models of increasing complexity. All use numba JIT compilation for real-time performance.

Source: ``gymkhana/envs/dynamic_models/``

KS — Kinematic Single Track
----------------------------

``gymkhana/envs/dynamic_models/kinematic.py``

Simplest model using pure kinematics without tire forces. Suitable for low-speed scenarios where tire slip is negligible.

ST — Single Track
-----------------

``gymkhana/envs/dynamic_models/single_track.py``

Single-track dynamics model with basic tire modeling. Models lateral dynamics but without an explicit tire force model.

MB — Multi-Body
----------------

``gymkhana/envs/dynamic_models/multi_body/``

Most detailed model with full tire modeling and multi-body dynamics. Provides the highest fidelity but parameters are only available for a full-scale vehicle (not 1/10 scale).

STD — Single Track Drift
-------------------------

``gymkhana/envs/dynamic_models/single_track_drift/``

Single-track dynamics with PAC2002 (Pacejka Magic Formula) tire model. **Recommended for RL drift training.** This model captures realistic tire force saturation and slip behavior needed for drifting.

Use with:

.. code:: python

   config = {
       'model': 'std',
       'control_input': ['accl', 'steering_angle'],
       'params': GKEnv.f1tenth_std_vehicle_params(),
   }

Vehicle parameters
------------------

The dynamic model's physical parameters include:

- **mu**: surface friction coefficient
- **C_Sf** / **C_Sr**: cornering stiffness coefficient, front/rear [1/rad]
- **lf** / **lr**: distance from CG to front/rear axle [m]
- **h**: height of center of gravity [m]
- **m**: total vehicle mass [kg]
- **I**: moment of inertia about z axis [kg m^2]
- **s_min** / **s_max**: steering angle constraints [rad]
- **sv_min** / **sv_max**: steering velocity constraints [rad/s]
- **v_switch**: velocity at which acceleration can no longer create wheel spin [m/s]
- **a_max**: maximum longitudinal acceleration [m/s^2]
- **v_min** / **v_max**: longitudinal velocity bounds [m/s]
- **width** / **length**: vehicle dimensions [m]

Parameters can be passed via the ``params`` key in the config dict, or updated at runtime with ``env.update_params()``.
