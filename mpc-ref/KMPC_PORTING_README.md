# Kinematic MPC (KMPC) Controller - Porting Context

This document summarizes the KMPC controller from the ForzaETH race stack, intended to provide full context for porting it to a standalone F1TENTH Gym simulator (no ROS).

## Architecture Overview

The KMPC controller uses **acados** to solve a nonlinear optimal control problem (OCP) at each timestep. The vehicle is modeled as a **kinematic bicycle model in Frenet coordinates** (progress `s`, lateral deviation `n`, heading error `alpha`, velocity `v`, steering angle `delta`). The OCP minimizes a cost that balances track advancement, velocity tracking, lateral deviation, heading error, and input smoothness, subject to track boundary constraints, lateral acceleration limits, and actuator limits.

### Control Flow (ROS version)

1. **`controller_manager.py`** orchestrates everything. It subscribes to ROS topics for car state (pose, speed, frenet coordinates, IMU), local/global waypoints, opponent info, and state machine commands.
2. When `ctrl_algo == "KMPC"`, it instantiates `Kinematic_MPC_Controller` and calls `kmpc_cycle()` at 40 Hz.
3. `kmpc_cycle()` calls `kmpc_controller.main_loop(state, position_in_map, waypoint_array_in_map, speed_now, opponent, position_in_map_frenet, acc_now, track_length, compute_time)`.
4. `main_loop()` returns `(speed, acceleration, jerk, steering_angle, states)`.
5. The manager publishes an `AckermannDriveStamped` message with these values.

### What needs porting

To remove ROS, you need to:
- Replace `rospy.wait_for_message("/global_waypoints", WpntArray)` with direct waypoint loading (e.g., from a CSV/numpy file).
- Replace `FrenetConverter` initialization from ROS waypoints with direct array initialization.
- Replace `pbl_config` load functions (`load_KMPC_config_ros`, `load_car_config_ros`, `load_trailing_config_ros`) with direct YAML loading or hardcoded configs (no `rospkg` dependency).
- Remove all `rospy` calls (logging, params, subscribers, publishers).
- Provide the `main_loop()` inputs directly from your simulator state.

## Files to Copy

### Core KMPC files (copy these)

| File | Purpose |
|------|---------|
| `controller/mpc/src/kinematic_mpc/kinematic_mpc.py` | Main MPC controller class (`Kinematic_MPC_Controller`) |
| `controller/mpc/src/kinematic_mpc/bicycle_model.py` | CasADi kinematic bicycle model in Frenet frame + cost function |
| `controller/mpc/src/kinematic_mpc/acados_settings.py` | Acados OCP setup (solver config, constraints, slack variables) |
| `controller/mpc/src/kinematic_mpc/utils/splinify.py` | `SplineTrack` / `EnhancedSpline` classes for track representation |
| `controller/mpc/src/kinematic_mpc/utils/indicies.py` | Enum index definitions for states/inputs/parameters |

### Dependencies to copy/adapt

| File | Purpose |
|------|---------|
| `f110_utils/libs/frenet_conversion/src/frenet_converter/frenet_converter.py` | `FrenetConverter` class (Cartesian <-> Frenet coordinate conversion) |
| `stack_master/pbl_config/src/pbl_config/controller/mpc/KMPCConfig.py` | `KMPCConfig` Pydantic model (MPC parameters) |
| `stack_master/pbl_config/src/pbl_config/CarConfig.py` | `CarConfig` Pydantic model (car physical parameters) |
| `stack_master/pbl_config/src/pbl_config/TrailingConfig.py` | `TrailingConfig` Pydantic model (trailing/head-to-head parameters) |

### Config files (copy these)

| File | Purpose |
|------|---------|
| `stack_master/config/DEFAULT/kinematic_mpc_params.yaml` | KMPC tuning parameters |
| `stack_master/config/NUC2/car_model.yaml` | Car physical parameters (mass, inertia, wheelbase, etc.) |
| `stack_master/config/NUC2/trailing_params.yaml` | Trailing controller PID parameters |

### Reference files (for understanding integration, don't need to copy)

| File | Purpose |
|------|---------|
| `controller/controller_manager.py` | Shows how KMPC is instantiated and called in the control loop |
| `stack_master/launch/time_trials.launch` | Launch file showing KMPC node configuration |
| `stack_master/launch/base_system.launch` | Launch file for the base system (sim/real) |

## Key Data Structures

### `main_loop()` Input Arguments

```python
def main_loop(self, state, position_in_map, waypoint_array_in_map, speed_now,
              opponent, position_in_map_frenet, acc_now, track_length, compute_time):
```

| Argument | Type | Description |
|----------|------|-------------|
| `state` | `str` | State machine state: `"TRAILING"`, `"OVERTAKE"`, or other (e.g., `"GB"` for go-brrrr / time trial) |
| `position_in_map` | `np.array` shape `(1, 3)` | `[x, y, theta]` in map frame |
| `waypoint_array_in_map` | `np.array` shape `(N_wpnts, 8)` | Local waypoints: `[x, y, vx, d_m, s_m, kappa, psi, ax]` |
| `speed_now` | `float` | Current longitudinal speed [m/s] |
| `opponent` | `list` or `None` | `[s, d, vs, is_static, is_visible]` or `None` if no opponent |
| `position_in_map_frenet` | `np.array` shape `(4,)` | `[s, d, vs, vd]` in Frenet frame (note: `[2]` gets overwritten with `alpha` by the manager) |
| `acc_now` | `np.array` shape `(5,)` | Last 5 longitudinal acceleration values |
| `track_length` | `float` | Total track length [m] |
| `compute_time` | `float` | Previous loop compute time (currently unused, set to 0.0) |

### `main_loop()` Output

```python
return speed, acceleration, jerk, steering_angle, states
```

| Output | Type | Description |
|--------|------|-------------|
| `speed` | `float` | Commanded speed [m/s] |
| `acceleration` | `float` | Commanded acceleration (currently 0) |
| `jerk` | `float` | Commanded jerk (currently 0) |
| `steering_angle` | `float` | Commanded steering angle [rad] |
| `states` | `list` | Flattened predicted states over horizon (for visualization) |

### Global Waypoints Format

The MPC is initialized with **global waypoints** (the full track raceline). Each waypoint has:
- `x_m, y_m`: Cartesian position
- `vx_mps`: Reference velocity
- `s_m`: Arc-length position along centerline
- `kappa_radpm`: Curvature
- `d_left, d_right`: Distance to left/right track boundary
- `d_m`: Lateral offset from centerline
- `psi_rad`: Heading angle

## Initialization Sequence

1. Load configs: `KMPCConfig`, `CarConfig`, `TrailingConfig` from YAML files
2. Create `Kinematic_MPC_Controller(racecar_version, kmpc_config, car_config, trailing_config)`
3. Inside constructor:
   - `mpc_init_params()` sets up internal state buffers
   - `mpc_initialize_solver()`:
     - Waits for global waypoints (this is what needs replacing)
     - Creates `FrenetConverter` from waypoint x/y arrays
     - Creates `SplineTrack` from waypoint coordinates + boundaries
     - Calls `acados_settings()` which:
       - Creates the CasADi bicycle model (`bicycle_model()`)
       - Sets up the acados OCP with DISCRETE integrator, SQP_RTI solver, PARTIAL_CONDENSING_HPIPM QP solver
       - Returns `constraint, model, acados_solver, params`

## Model Details

### States (5): `[s, n, alpha, v, delta]`
- `s`: progress along centerline [m]
- `n`: lateral deviation from centerline [m]
- `alpha`: heading error w.r.t. centerline tangent [rad]
- `v`: velocity [m/s]
- `delta`: steering angle [rad]

### Inputs (2): `[der_v, derDelta]`
- `der_v`: acceleration (velocity change rate) [m/s^2]
- `derDelta`: steering angle change rate [rad/s]

### Dynamics (kinematic bicycle in Frenet frame)
```
s_dot = v * cos(alpha) / (1 - kappa(s) * n)
n_dot = v * sin(alpha)
alpha_dot = v / (lr + lf) * tan(delta) - kappa(s) * s_dot
v_dot = der_v        (input)
delta_dot = derDelta  (input)
```

Discretization: Custom RK2 (Heun's method) implemented in CasADi, used as DISCRETE dynamics in acados.

### Cost Function
```
cost = -qadv * s_dot/freq
     + qn * (n - overtake_d)^2
     + qalpha * alpha^2
     + qv * (v - V_target)^2
     + qac * der_v^2
     + qddelta * derDelta^2
```

### Constraints
- **Track boundaries**: soft constraints on `n` using interpolated left/right boundary distances
- **Lateral acceleration**: `a_lat = v^2 * tan(delta) / (lr + lf)`, bounded by `alat_max`
- **State bounds**: `v_min <= v <= v_max`, `delta_min <= delta <= delta_max`
- **Input bounds**: `a_min <= der_v <= a_max`, `ddelta_min <= derDelta <= ddelta_max`

### Online Parameters (10)
Set per shooting node to allow varying reference speed and weights along the horizon:
```
[V_target, qadv, qv, qn, qalpha, qac, qddelta, alat_max, bound_inflation, overtake_d]
```

## Default Parameter Values (from kinematic_mpc_params.yaml)

```yaml
N: 40                    # prediction horizon steps
MPC_freq: 20             # Hz
t_delay: 0.125           # s
steps_delay: 3
track_safety_margin: 0.3 # m
track_max_width: 1000    # m (effectively unconstrained)
overtake_d: 1            # m

# Cost weights
qac: 0.01
qddelta: 0.1
qadv: 0.0
qn: 20
qalpha: 7
qv: 10

# Slack penalties
Zl: 100
Zu: 100
zl: 10
zu: 10

# Steering constraints
delta_min: -0.40  # rad
delta_max: 0.40   # rad

# Velocity constraints
v_min: 2.0   # m/s
v_max: 12.0  # m/s

# Input bounds
ddelta_min: -0.2  # rad/s
ddelta_max: 0.2   # rad/s
a_min: -10.0      # m/s^2
a_max: 10.0       # m/s^2

# Lateral acceleration
alat_max: 10.0    # m/s^2
```

## Car Parameters (NUC2)

```yaml
m: 3.54           # kg
Iz: 0.05797       # kg*m^2
lf: 0.162         # m (CG to front axle)
lr: 0.145         # m (CG to rear axle)
wheelbase: 0.307  # m
h_cg: 0.014       # m
```

## Python Dependencies

- `acados_template` (acados Python interface)
- `casadi` (symbolic math / automatic differentiation)
- `numpy`
- `scipy` (for `InterpolatedUnivariateSpline` in splinify.py and `CubicSpline` in frenet_converter.py)
- `pydantic` (for config validation)
- `yaml` (PyYAML, for config loading)

## Porting Checklist

- [ ] Copy core KMPC files and reorganize imports
- [ ] Copy and adapt `FrenetConverter` (already ROS-free, just needs import path fix)
- [ ] Copy and adapt `KMPCConfig`, `CarConfig`, `TrailingConfig` (remove `rospkg` dependency, load YAML directly)
- [ ] Replace `rospy.wait_for_message("/global_waypoints", WpntArray)` in `kinematic_mpc.py:mpc_initialize_solver()` with direct waypoint array input
- [ ] Replace `rospy.loginfo` calls with `print()` or Python `logging`
- [ ] Remove `WpntArray` / `f110_msgs` imports; pass waypoint data as numpy arrays or dicts
- [ ] Create a wrapper that maps your simulator's state to the `main_loop()` input format
- [ ] Create a wrapper that maps `main_loop()` output to your simulator's actuator commands
- [ ] Install `acados` and `casadi` in your environment
- [ ] For time-trial mode only (no opponent), set `state="GB"` and `opponent=None`
