# Plan: Port KMPC Controller to F1TENTH Gym

## Context

Port the Kinematic MPC (KMPC) controller from the ForzaETH ROS-based race stack (`mpc-ref/`) into this pure Python F1TENTH Gym simulator. The KMPC uses acados + CasADi to solve a nonlinear OCP each timestep, modeling the vehicle as a kinematic bicycle in Frenet coordinates. Goal: run KMPC in time-trial mode (no opponent) with the existing simulator.

## Known Compatibility Issues & Decisions

### Track Width Data
The gym's raceline CSV has no width data (`w_lefts`/`w_rights` are `None`). Only the centerline has widths. The KMPC needs `d_left`/`d_right` for boundary constraints. **Solution**: For each raceline waypoint, find nearest centerline point, compute the raceline's lateral offset from centerline, and adjust widths accordingly:
- `d_right_raceline = d_right_centerline + lateral_offset` (raceline is offset left → more room on right)
- `d_left_raceline = d_left_centerline - lateral_offset`

### Vehicle Model Choice
Use `model="ks"` (kinematic single-track) as the best match for KMPC's internal kinematic bicycle model. Using `"st"` or `"std"` is possible (MPC re-solves each step and adapts to actual state), but introduces model mismatch — the KMPC doesn't model tire slip or Pacejka forces.

### Control Interface
The KMPC `main_loop()` outputs `(speed, steering_angle)` as commands. Use `control_input=["speed", "steering_angle"]`. This mirrors the real robot interface (VESC speed controller + steering servo) that KMPC was designed for. The gym applies a P-controller for speed and bang-bang for steering — similar to real hardware actuators.

### Steering Delay
The sim has a 2-step steering delay buffer (~20ms). The KMPC has `t_delay=0.125s` for latency compensation. May need tuning if tracking is poor.

### Vehicle Parameters
KMPC uses NUC2 params (`lf=0.162, lr=0.145`). Gym defaults differ (`lf=0.15875, lr=0.17145`). Update `car_model.yaml` to match gym params for consistency, since the sim dynamics use gym params.

---

## Step 1: Install Dependencies

```bash
pip install casadi

# acados (build from source)
git clone https://github.com/acados/acados.git --recurse-submodules ~/software/acados
cd ~/software/acados && mkdir build && cd build
cmake -DACADOS_WITH_QPOASES=ON ..
make install -j$(nproc)

# Environment variables (add to shell profile)
export ACADOS_SOURCE_DIR=~/software/acados
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/software/acados/lib

# Python interface
pip install -e ~/software/acados/interfaces/acados_template
```

## Step 2: Create Directory Structure

```
examples/controllers/mpc/
├── __init__.py
├── kinematic_mpc.py          # Adapted from mpc-ref (ROS removed)
├── bicycle_model.py          # Copied, import paths fixed
├── acados_settings.py        # Copied, import paths fixed
├── splinify.py               # Copied as-is (pure numpy/scipy)
├── frenet_converter.py       # Copied as-is (pure numpy/scipy)
├── config.py                 # NEW: merged KMPCConfig + CarConfig
├── gym_bridge.py             # NEW: maps gym obs ↔ main_loop() interface
└── config/
    ├── kinematic_mpc_params.yaml
    └── car_model.yaml              # Updated to match gym vehicle params

examples/mpc_runner.py              # NEW: runner script
```

## Step 3: Copy Pure Files (No Changes Needed)

- `mpc-ref/controller/mpc/src/kinematic_mpc/utils/splinify.py` → `examples/controllers/mpc/splinify.py`
- `mpc-ref/f110_utils/libs/frenet_conversion/src/frenet_converter/frenet_converter.py` → `examples/controllers/mpc/frenet_converter.py`
- 2 YAML config files → `examples/controllers/mpc/config/`

## Step 4: Create `config.py` — Merged Config Classes

Merge `KMPCConfig` and `CarConfig` into one file. Changes:
- Remove `import rospkg` and all `load_*_ros()` functions
- Add `from_yaml(path)` classmethod on each model
- Keep all field definitions identical
- `TrailingConfig` is intentionally excluded — it's only used for head-to-head trailing behind opponents, not needed for time-trial mode
- Keep `overtake_d` in `KMPCConfig` (used by `acados_settings.py` for initial parameter vector), set to `0.0` in YAML

Source files:
- `mpc-ref/stack_master/pbl_config/src/pbl_config/controller/mpc/KMPCConfig.py`
- `mpc-ref/stack_master/pbl_config/src/pbl_config/CarConfig.py`

## Step 5: Adapt `bicycle_model.py` — Fix Imports Only

```python
from .config import KMPCConfig, CarConfig  # was: from pbl_config import ...
```

Source: `mpc-ref/controller/mpc/src/kinematic_mpc/bicycle_model.py`

Note: This file handles the 2-lap data extension internally (prepend + append arrays over `[-L, 2L]` range) and creates CasADi B-spline interpolants. No changes needed to this logic.

## Step 6: Adapt `acados_settings.py` — Fix Imports Only

```python
from .config import KMPCConfig, CarConfig       # was: from pbl_config import ...
from .bicycle_model import bicycle_model         # was: from kinematic_mpc.bicycle_model import ...
```

Source: `mpc-ref/controller/mpc/src/kinematic_mpc/acados_settings.py`

## Step 7: Adapt `kinematic_mpc.py` — ROS & Head-to-Head Removal ✅

Source: `mpc-ref/controller/mpc/src/kinematic_mpc/kinematic_mpc.py`

### 7a. Remove ROS imports and TrailingConfig
Delete: `import rospy`, `from f110_msgs.msg import WpntArray`, `from scipy.integrate import solve_ivp`.
Remove `TrailingConfig` from constructor — it's a PID gap controller for following opponents, unused in time-trial mode. Constructor is now `__init__(self, kmpc_config, car_config)`.

### 7b. Fix local imports
```python
from .frenet_converter import FrenetConverter
from .splinify import SplineTrack
from .config import KMPCConfig, CarConfig
from .acados_settings import acados_settings
```

### 7c. Replace `rospy.loginfo()` → `print()`

### 7d. Rewrite `mpc_initialize_solver()` to accept arrays directly
Replace ROS `wait_for_message` with direct array input:
```python
def mpc_initialize_solver(self, xs, ys, vx_ref, kappa_ref, s_ref, d_left, d_right):
```
Build `FrenetConverter` and `SplineTrack` from passed arrays. Call `acados_settings()`.

### 7e. Remove auto-call to `mpc_initialize_solver()` in `__init__`
Let the bridge call it explicitly after construction with waypoint data.

### 7f. Delete ROS-specific helpers
Remove `_transform_waypoints_to_cartesian()` and `_transform_waypoints_to_coords()`.

### 7g. Remove trailing/overtake logic from `main_loop()`
The `TRAILING` state used a PID controller (`trailing_controller()`) to modulate target speed for gap-keeping behind an opponent. The `OVERTAKE` state set a lateral offset (`overtake_d`) in the cost function to steer around opponents. Both are dead code in time-trial (guarded by `opponent is not None`).

Removed from `main_loop`:
- Parameters: `state`, `speed_now`, `opponent`, `acc_now`, `track_length` (all unused in time-trial path)
- `TRAILING` and `OVERTAKE` branches — only the default raceline-tracking branch remains
- `overtake_d` hardcoded to `0.0` in online parameters (keeps acados parameter vector compatible with `bicycle_model.py`)

New signature:
```python
def main_loop(self, position_in_map, waypoint_array_in_map, position_in_map_frenet, compute_time):
```

New return value: `(speed, steering_angle)` — removed `acceleration` (always 0), `jerk` (always 0), `states` (ROS visualization only).

### 7h. Remove dead time-delay propagation
`propagate_time_delay()` result was immediately overwritten (`propagated_x = x0`), so it was effectively disabled. Removed the method and `solve_ivp` import. Kept `_dynamics_of_car()` since `get_warm_start()` still uses it.

### 7i. Clean up `mpc_init_params()`
Removed trailing/overtake state variables: `self.overtake_d`, `self.gap*`, `self.v_diff`, `self.i_gap`, `self.loop_rate`.

### 7j. Delete `trailing_controller()` method entirely

## Step 8: Create `gym_bridge.py` — Simulator ↔ KMPC Bridge ✅

### Changes from original plan:
- **Centerline instead of raceline**: Uses `track.centerline` directly instead of `track.raceline`. Avoids the raceline→centerline width projection complexity. Centerline already has `w_lefts`/`w_rights` natively — no offset computation needed.
- **Uniform reference speed**: Takes a `ref_speed` parameter and creates a uniform `vx_ref = np.full(n, ref_speed)` array instead of using raceline speed profile. Simpler for initial testing.
- **Recomputed arc-length**: Computes `s_ref` from cumulative `hypot(diff(xs), diff(ys))` to match the domains of `FrenetConverter` and `SplineTrack`, rather than using the track's stored `ss`.
- **MPC's own Frenet converter**: Uses `controller.fren_conv.get_frenet()` and `controller.spline.get_derivative()` for Frenet conversion instead of the gym's `track.cartesian_to_frenet()`. Ensures the Frenet coordinates are consistent with what the MPC solver expects internally.
- **Heading deviation computed manually**: `alpha` computed as `arctan2(sin(θ - θ_track), cos(θ - θ_track))` from `SplineTrack` derivative, since `FrenetConverter.get_frenet()` only returns `(s, d)`.
- **Vehicle state injection**: Feeds actual `linear_vel_x` and `delta` from observations into `controller.speed` and `controller.steering_angle_buf` each step, so the MPC uses real vehicle state instead of its own predictions.
- **Constructor signature**: `KMPCGymBridge(track, kmpc_config_path, car_config_path, ref_speed)` — takes a `Track` object and config file paths, not the full env.
- **`get_start_pose()` helper**: Returns `(x, y, yaw)` from centerline start point for env reset.

### Initialization:
1. Load `KMPCConfig` and `CarConfig` from YAML paths
2. Extract centerline data: `xs, ys, ks, yaws, w_lefts, w_rights` (all cast to `float64`)
3. Create uniform `vx_ref` from `ref_speed` parameter
4. Recompute `s_ref` from waypoint distances
5. Build `waypoint_array` (Nx8): `[x, y, vx, 0, s_m, kappa, yaw, 0]` — only columns 2 (vx) and 4 (s_m) used at runtime
6. Construct `Kinematic_MPC_Controller(kmpc_config, car_config)` and call `mpc_initialize_solver(xs, ys, vx_ref, ks, s_ref, w_lefts, w_rights)`

### Per-step `get_action(obs)`:
1. Extract `pose_x, pose_y, pose_theta` from `obs["agent_0"]`
2. Compute `(s, d)` via `controller.fren_conv.get_frenet()`
3. Compute `alpha` (heading deviation) via `controller.spline.get_derivative(s)`
4. Inject actual vehicle state: `linear_vel_x` → `controller.speed`, `delta` → `controller.steering_angle_buf`
5. Call `controller.main_loop(position_in_map, waypoint_array, position_in_map_frenet, compute_time)`
6. Return `np.array([[steering_angle, speed]])`

## Step 9: Create `mpc_runner.py` ✅

Uses `get_drift_test_config()` as a base config (provides map, num_agents, timestep, integrator, render settings) with overrides for MPC-specific needs.

### Changes from original plan:
- **Config base**: Uses `get_drift_test_config()` instead of inline dict — reuses existing config infrastructure and inherits render/track settings
- **Vehicle model**: `model="std"` instead of `"ks"` — STD model used for consistency with drift training infrastructure; MPC re-solves each step so model mismatch is acceptable
- **Normalization disabled**: Explicitly sets `normalize_act=False`, `normalize_obs=False` since MPC expects raw values
- **Bridge constructor**: `KMPCGymBridge(track, MPC_CONFIG, CAR_CONFIG, ref_speed=REF_SPEED)` — takes track object and config file paths separately instead of the full env. `ref_speed` parameter controls uniform reference speed (default 2.0 m/s for initial testing)
- **Start pose**: Uses `bridge.get_start_pose()` to get initial pose from centerline start point
- **Fixed step count**: Runs for 10,000 steps instead of `while not done` — avoids early termination during debugging
- **Config paths**: `MPC_CONFIG` and `CAR_CONFIG` defined as module-level `Path` constants pointing to YAML files

## Step 10: Handle acados Generated Code ✅

Set `ocp.code_export_directory` and `json_file` in `acados_settings.py` to write generated C code and solver JSON into the mpc package directory (`examples/controllers/mpc/c_generated_code/` and `examples/controllers/mpc/acados_ocp.json`). Uses `_MPC_DIR = Path(__file__).resolve().parent` to resolve absolute paths regardless of CWD.

Added `acados_ocp.json` to `.gitignore` (alongside the existing `c_generated_code/` entry). Removed stale `acados_ocp.json` that was previously generated in the project root.

## Verification

1. **Solver init**: Verify `acados_settings()` completes without error
2. **Frenet consistency**: Compare gym's `cartesian_to_frenet()` vs KMPC's internal Frenet — should be close
3. **Solver convergence**: Check `acados_solver.solve()` returns status 0
4. **Visual**: Render with `render_mode="human"`, verify car follows raceline smoothly
5. **Lap completion**: Car completes a full lap on Spielberg without crashing

## Key Files

| File | Purpose |
|------|---------|
| `mpc-ref/controller/mpc/src/kinematic_mpc/kinematic_mpc.py` | Core controller to adapt |
| `mpc-ref/controller/mpc/src/kinematic_mpc/bicycle_model.py` | CasADi model (import fix) |
| `mpc-ref/controller/mpc/src/kinematic_mpc/acados_settings.py` | Solver setup (import fix) |
| `mpc-ref/controller/mpc/src/kinematic_mpc/utils/splinify.py` | Track splines (copy as-is) |
| `mpc-ref/f110_utils/.../frenet_converter.py` | Frenet conversion (copy as-is) |
| `mpc-ref/stack_master/pbl_config/src/pbl_config/controller/mpc/KMPCConfig.py` | Config model |
| `mpc-ref/stack_master/pbl_config/src/pbl_config/CarConfig.py` | Car params model |
| `examples/waypoint_follow.py` | Runner script pattern |
| `f1tenth_gym/envs/track/raceline.py` | Raceline data structure |
