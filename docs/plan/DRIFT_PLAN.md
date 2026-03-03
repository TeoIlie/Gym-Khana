# Drifting Controller Implementation Plan

## Overview

I am aiming to use this simulation to create a custom implementation of a drifting controller for racing. Similar to the `examples/ppo_example.py` file, I will be training my DRL algorithm with a SB3 implementation, in a Gym environment, to train the model to learn how to drift autonomously.

In order to achieve this using this existing simulation environment, there are several features I need to implement:

 1. Create Drift-Optimized Observation

- Extend the RL observation to include drift-critical state information. Existing observations are marked [X] and To-Do observations are marked [ ]:

   [X] 1) Linear velocities in vehicle frame: longitudinal vx and lateral vy

   [X] 2) Yaw rate: r

   [X] 3) The angle between the car’s heading and the track heading, expressed in Frenet coordinates u

   [X] 4) Lateral distance from the centerline, in Frenet coordinates: n

   [X] 5) Measured steering angle δ and the previous control input δ_ref

   [X] 6) Measured wheel speed ω, commanded wheel speed ω_ref, and the last control input ω̇ref

   [X] 7) Track information, represented by N points, with curvature c ∈ R^N and width w ∈ R^N . These N points are sampled uniformly in front of the vehicle at 30-centimeter intervals.
- Location: Modify f1tenth_gym/envs/observation.py or create custom observation type

 2. Implement Drift-Reward Function

- Design reward function that encourages fast driving and penalize track boundary departure
    r_t =
    1) -1 if track boundary exceeded
    2) s_t - s_{t-1} where s_t denotes the progress along the centerline expressed in Frenet coordinates
- Location: Override _get_reward() method in custom environment

 3. Create Pacejka Drift Dynamics Model

- Implement more realistic tire model using the Pacejka Magic Formula
- Location: f1tenth_gym/envs/dynamic_models/pacejka_drift.py

 4. Develop Drift Training Script

- Create examples/drift.py based on ppo_example.py template
- Configure environment with bicycle kinematics model - this is already defined and can be used as `gym.make("model": "st",...)`

 5. Model the actuator dynamics as a first-order lag system

- Actuator dynamics must be accurately modeled for subsequent sim-to-real transfer
- The equations to use are these (discretized with the Euler method based on the ODE)

```
# Steering actuator dynamics
self.delta += (self.dt / self.T_delta) * (delta_ref - self.delta)

# Wheel speed actuator dynamics
self.omega += (self.dt / self.T_omega) * (omega_ref - self.omega)                               
```
