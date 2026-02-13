# Recovery Training Plan

## Overview

Currently this gym env is setup for training racing policies using `train/ppo_race.py`. The goal is to implement a new feature to be able to train a policy that recovers control of a vehicle on a straight-line path. This will later be extended to recover control on an arbitrary curved path. This new training script will be written at `train/ppo_recover.py`, and requires several new features.

## New features

1. **Configuration**. Some will be shared with `ppo_race.py`, and some will be different. It is necessary to figure out how to properly organize the config so that it is easy to maintain. For example, shared configs can be stored in one location, and then custom differences between the race and recover training scripts can be stored elsewhere

2. **Initialization**. The vehicle, of dynamic model type `STD` and observation type `drift`, will be initialized at the start of a straight segment of track, specifically at the bottom-right of the IMS track (arc-length `S = 96`) travelling CCW (`config["track_direction"] = "normal"). It will be given a random initial perturbation of some of its parameters:

    | State variable | Range | Units |
    | --- | --- | --- |
    | `beta` (sideslip angle) | [-pi/3, pi/3] | radians |
    | `r` (yaw rate) | [-pi/2, pi/2] | radians/s |
    | `v` (velocity) | [2, 20] | m/s |
    | `yaw` (yaw in Cartesian coords) | [-pi/3, pi/3] | radians |

    Later on, I will experiment with also perturbing other parameters, such as steering angle, lateral deviation (given by Cartesian coordinate x), and so on, but for now these parameters are sufficient.

3. **Reset condition** The vehicle must be reset when either it crashes, it succeeds to recover, or it runs of out time

    1. **Crash**: The vehicle should be reset when it exceeds the track boundaries (conceptually, this is like exceeding its lane on a road). The episode is done
    2. **Success**: Successful recovery is defined as all the following holding true:
        1. `delta` steering angle is less than `DELTA_RECOVERY_THRESHOLD` away from `0`
        2. `beta` is less than `BETA_RECOVERY_THRESHOLD` away from `0`
        3. `r` is less than `YAW_RECOVERY_THRESHOLD` away from `0`
        4. derivative of `beta` is less than `D_BETA_RECOVERY_THRESHOLD` away from `0`
        5. derivative of `r` is less than `D_R_RECOVERY_THRESHOLD` away from `0`
        6. `frenet_u` heading error is less than `FRENET_U_RECOVERY_THRESHOLD` away from `0` - Note that this is not the same as sideslip `beta`. The sideslip tells us the velocity vector of the car is aligned with its position, but that does not mean it is also aligned with the track
        7. The car is within the track bounds (this is already handled by the reset condition)

        Some possible initial values for the thresholds:

        | Threshold param | Value | Reasoning |
        | --- | --- | --- |
        | DELTA_RECOVERY | 0.05 rad (~3 deg) | Small steering input |
        | BETA_RECOVERY | 0.05 rad (~3 deg) | Nearly zero sideslip |
        | YAW_RATE_RECOVERY | 0.1 rad/s | Minimal rotation |
        | D_BETA_RECOVERY | 0.1 rad/s | Beta is settling |
        | D_R_RECOVERY | 0.2 rad/s^2 | Yaw acceleration near zero |
        | FRENET_U_RECOVERY | 0.05 rad (~3 deg) | Aligned with track |

    3. **Time exceeded**: If the vehicle reaches arc-length `S=140`, or the total time steps exceeds `2048`, the episode is truncated

4. **Reward function**. The reward function is composed of several components summed together. $ K_r, K_e, K_c $ represent tunable gains.

    $$ R_{total} = - K_eR_{Euclid} + R_{col} + R_{rec} - R_{const} $$

    1. **Euclidean distance of `(beta, r)` to `(0, 0)`**: reward distance from current `(beta, r)` to `(0, 0)` as a Eucliden distance to give a dense, shaped signal of how close to recovery the state is, approximately

        $ R_{Euclid} = \sqrt{\beta^2 + r^2}$

    2. **Penalty for collision**: when the car exits the track bounds, there is a constant penalty applied

        $R_{col} =
        \begin{cases}
        -50, & \text{if track bounds exceeded} \\
        0, & \text{otherwise}
        \end{cases}$

    3. **Reward for successful recovery**: when the car successfully recovers, according to the aforementioned definition of recovery, a constant positive reward is applied

        $R_{rec} =
        \begin{cases}
        100, & \text{if successfully recovered} \\
        0, & \text{otherwise}
        \end{cases}$

    4. **Constant timestep penalty**: There is a constant penalty at each timestep of $ -K_cdt $. This encourages the policy to learn the minimum-time sequence that achieves recovery

        $ R_{const} = K_cdt $

    5. **Optional reward modifications** If the training is failing, or if the policy is unsatisfactory, there are some possible improvements to make, following the first implementation:
        1. **Add forward progress**: reward forward progress along centerline, in Frenet-frame, if the vehicle is not learning to drive straight

            $ R_{prog} = K_p * (s_t - s_{t-1})$

        2. **Add action penalty**: add action penalties if the action inputs are too large, something like:

            $ R_{act} = -K_a * (\sqrt{ {d\_delta}^2 + {accl}^2}) $

        3. **Negative velocity penalty** add a penalty for negative velocity if the car is not learning to drive forwards

            $ R_{\text{reverse}} =
            \begin{cases}
            -1, & \text{if longitudinal vehicle velocity is} < 0 \\
            0, & \text{otherwise}
            \end{cases} $

## Implementation details

Many of the necessary features are already implemented, and need only be used correctly. Others must be implemented from scratch:

| Feature | Existing code | New implementation |
| --- | ---- | --- |
| Configuration | There exists a config structure already in `/train/config` | Additional customization must be implemented, possibly sharing come configuration options with the race implementation, and creating new ones |
| Initialization | There is already a method for initializing a car with a specific initial state using `eval_env.reset(options={"states": init_state})` | Depending on how I decide to select the intial state params from a distribution, it will be necessary to implement a feature to initialize a state with parameters selected from a given distribution at the start of each episode |
| Reset condition: crash | There is already a feature for checking when the vehicle hits track boundaries. This can be used with `predictive_collision: false` and `wall_deflection: false`, as defined in `/train/config/gym_config.yaml`. Note that this will enable the car outer tires to exceed the track, checking only if the center of the car exceeds the track, but this is sufficient for this use case. | None |
| Reset condition: Success | None | This must be implemented fully |
| Reset condition: Time exceeded | None | This must be implemented fully |
| Reward function | This is partially implemented, with forward progress and collision penalty implemented. | Additional elements $ R_{Euclid},  R_{rec}, R_{const} $ must be implemented. |
| `d_beta` and `d_r` calculation | None | the derivative values of `beta` and `r` must be computed for use in the recovery condition |

## Open questions

There are a few design questions I address separately:

1. How do I perturb the parameters in the range do I want? Should it be randomly chosen with equal probability in the specified ranges? Should it be chosen from a Gaussian distribution, or some other distribution?
    - Initially I will use a uniform distribution. This can be modified in future, for example, but adding curriculum learning to gradually increase perturbation over the course of learning
2. Do I need all the elements of my recovery condition? Is it necessary to include that derivatives of `beta` and `r` are also 0; is that even possible when `beta` and `r` are 0 from the phyiscs? How do I set reasonable thresholds?
    - For now, I will use this definition, and experiment later with removing the derivatives requirements
3. Is my reward formulation meaningful?
    - I will begin with my current reward, and implement optional enhancements as needed
4. How do I configure a gym environment to use my new required features (initialization, reset, reward, etc) when I create it? Should I create a new configuration parameter, something like `"goal"` with setting `race` (default) and `recover` (this new implementation)? How else is it best managed, to create a clear separation between racing and recovering training, enabling both to work in this repo?
    1. Config layer: Create `gym_config_recover.yaml` with recovery-specific defaults. Add `get_recovery_train_config()` and `get_recovery_test_config()` factory functions to `env_config.py`. Shared constants (model, timestep, vehicle params, integrator) stay as Python constants in `env_config.py` — no duplication.
    2. Environment layer: Add a `training_mode` config key (`"race"` default, or `"recover"`). This switches behavior in two methods:
        - `_get_reward()` — uses recovery reward formula instead of racing reward
        - `_check_done()` — adds recovery success and arc-length truncation conditions
    3. Training script: `ppo_recover.py` imports from the same `training_utils.py` and `env_config.py`, calling `get_recovery_train_config()` instead of `get_drift_train_config()`. The custom initialization (random perturbation at `S=96`) lives in the training script's reset logic.
