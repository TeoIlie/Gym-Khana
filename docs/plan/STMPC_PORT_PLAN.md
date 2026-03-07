# Plan: Port STMPC Controller to F1TENTH Gym

## Context

The Kinematic MPC (KMPC) controller has already been ported from the ForzaETH ROS race stack (`mpc-ref/`) into `examples/controllers/mpc/`. Now we port the Single Track MPC (STMPC) controller — a more advanced MPC that uses Pacejka tire forces, lateral velocity, yaw rate, and jerk-based control — into the same package, reusing shared utilities.

As part of this, we reorganize the flat `mpc/` directory into subpackages: `utils/` for shared utilities, `kmpc/` for kinematic MPC files, and `stmpc/` for single-track MPC files.

**Key differences STMPC vs KMPC:**
- 8-state model (adds v_y, yaw_rate, acceleration) vs KMPC's 5-state
- Pacejka Magic Formula tire forces vs pure kinematics
- Jerk control input (d/dt acceleration) vs direct acceleration
- Combined acceleration constraints (ellipse/diamond) and load transfer
- ERK integrator (acados handles integration) vs DISCRETE (manual RK4)
- 12 online parameters vs 10 (adds a_min, a_max separately)

## Directory Structure

```
examples/controllers/mpc/
├── __init__.py
├── config.py                            # Shared: KMPCConfig, CarConfig, STMPCConfig, PacejkaTireConfig
├── gym_bridge.py                        # Shared: KMPCGymBridge + STMPCGymBridge
├── utils/
│   ├── __init__.py
│   ├── frenet_converter.py              # MOVED from mpc/
│   └── splinify.py                      # MOVED from mpc/
├── kmpc/
│   ├── __init__.py
│   ├── bicycle_model.py                 # MOVED from mpc/bicycle_model.py
│   ├── acados_settings.py               # MOVED from mpc/acados_settings.py
│   ├── kinematic_mpc.py                 # MOVED from mpc/kinematic_mpc.py
│   └── c_generated_code/               # MOVED from mpc/c_generated_code/
├── stmpc/
│   ├── __init__.py
│   ├── bicycle_model.py                 # NEW: Pacejka bicycle model
│   ├── acados_settings.py               # NEW: STMPC solver setup
│   ├── single_track_mpc.py              # NEW: Main controller
│   ├── indicies.py                      # NEW: StateIndex enum
│   └── st_c_generated_code/             # NEW: generated solver
├── config/
│   ├── kinematic_mpc_params.yaml
│   ├── car_model.yaml
│   ├── single_track_mpc_params.yaml     # NEW
│   └── pacejka_tire_params.yaml         # NEW

examples/
├── kmpc_example.py                       # Updated import path
└── stmpc_example.py                     # NEW
```

## Step 0: Reorganize Existing Files into Subpackages

### 0a. Create subpackage directories and `__init__.py` files
```
mkdir -p examples/controllers/mpc/{utils,kmpc,stmpc}
touch examples/controllers/mpc/{utils,kmpc,stmpc}/__init__.py
```

### 0b. Move shared utilities to `utils/`
```
git mv examples/controllers/mpc/frenet_converter.py examples/controllers/mpc/utils/
git mv examples/controllers/mpc/splinify.py examples/controllers/mpc/utils/
```

### 0c. Move KMPC files to `kmpc/`
```
git mv examples/controllers/mpc/bicycle_model.py examples/controllers/mpc/kmpc/
git mv examples/controllers/mpc/acados_settings.py examples/controllers/mpc/kmpc/
git mv examples/controllers/mpc/kinematic_mpc.py examples/controllers/mpc/kmpc/
```
Also move `c_generated_code/` and `acados_ocp.json`:
```
git mv examples/controllers/mpc/c_generated_code examples/controllers/mpc/kmpc/
git mv examples/controllers/mpc/acados_ocp.json examples/controllers/mpc/kmpc/
```

### 0d. Update imports in moved KMPC files

**`kmpc/acados_settings.py`:**
- `from .bicycle_model import bicycle_model` — unchanged (still same subpackage)
- `from .config import CarConfig, KMPCConfig` → `from ..config import CarConfig, KMPCConfig`
- `_MPC_DIR` path: now resolves to `kmpc/` — `c_generated_code/` is already there, so no change needed

**`kmpc/kinematic_mpc.py`:**
- `from .acados_settings import acados_settings` — unchanged
- `from .config import CarConfig, KMPCConfig` → `from ..config import CarConfig, KMPCConfig`
- `from .frenet_converter import FrenetConverter` → `from ..utils.frenet_converter import FrenetConverter`
- `from .splinify import SplineTrack` → `from ..utils.splinify import SplineTrack`

**`kmpc/bicycle_model.py`:**
- `from .config import CarConfig, KMPCConfig` → `from ..config import CarConfig, KMPCConfig`

**`gym_bridge.py`:**
- `from .kinematic_mpc import Kinematic_MPC_Controller` → `from .kmpc.kinematic_mpc import Kinematic_MPC_Controller`

**`kmpc_example.py`** — no change needed (imports from `controllers.mpc.gym_bridge`)

### 0e. Verify KMPC still works
Run `python examples/kmpc_example.py` to confirm nothing broke.

## Step 1: Add `STMPCConfig` and `PacejkaTireConfig` to `config.py`

**File:** `examples/controllers/mpc/config.py`

Append two new Pydantic classes after existing `CarConfig`:

**`STMPCConfig`** — from `mpc-ref/stack_master/pbl_config/src/pbl_config/controller/mpc/STMPCConfig.py`:
- Remove `import rospkg` and `load_STMPC_config_ros()` function
- Add `from_yaml(path)` classmethod (same pattern as `KMPCConfig`)
- Fields: `N`, `t_delay`, `steps_delay`, `MPC_freq`, `track_safety_margin`, `track_max_width`, `overtake_d`, `qjerk`, `qddelta`, `qadv`, `qn`, `qalpha`, `qv`, `Zl/Zu/zl/zu`, `delta_min/max`, `v_min/max`, `a_min/max`, `ddelta_min/max`, `jerk_min/max`, `alat_max`, `vy_minimization` (bool), `adv_maximization` (bool), `combined_constraints` (str), `load_transfer` (bool), `correct_v_y_dot` (bool)

**`PacejkaTireConfig`** — from `mpc-ref/stack_master/pbl_config/src/pbl_config/PacejkaTireConfig.py`:
- Remove `import rospkg`, `load_pacejka_tire_config_ros()`, `save_pacejka_tire_config()`, `NotExistingFloor`
- Add `from_yaml(path)` classmethod
- Fields: `friction_coeff`, `Bf/Cf/Df/Ef`, `Br/Cr/Dr/Er`, `floor`

## Step 2: Create Config YAML Files

**`examples/controllers/mpc/config/single_track_mpc_params.yaml`** — Copy from `mpc-ref/stack_master/config/NUC2/single_track_mpc_params.yaml` with adjustments:
- Set `overtake_d: 0.0` (time-trial, no overtaking)
- Set `v_min: 0.5` (avoid division-by-zero in slip angle computation)
- Keep all other NUC2 values as-is

**`examples/controllers/mpc/config/pacejka_tire_params.yaml`** — Copy from `mpc-ref/stack_master/config/DEFAULT/pacejka/DEFAULT/default.yaml`:
```yaml
Bf: 7.016
Br: 5.4485
Cf: 0.4626
Cr: 1.2871
Df: 1.392
Dr: 0.8068
Ef: 0.5023
Er: 1.3117
floor: dubi
friction_coeff: 1.0
```

## Step 3: Create `stmpc/indicies.py`

**Source:** `mpc-ref/controller/mpc/src/single_track_mpc/utils/indicies.py`
**Dest:** `examples/controllers/mpc/stmpc/indicies.py`

Copy as-is. Contains `StateIndex`, `Input`, and `Parameter` IntEnums.

## Step 4: Create `stmpc/bicycle_model.py` — Pacejka Bicycle Model

**Source:** `mpc-ref/controller/mpc/src/single_track_mpc/bicycle_model.py`

**Fix imports only:**
```python
from ..config import STMPCConfig, CarConfig, PacejkaTireConfig  # was: from pbl_config import ...
```

No other changes needed. Function signature stays:
```python
def bicycle_model(s0, kapparef, d_left, d_right,
                  stmpc_config, car_config, tire_config)
```

## Step 5: Create `stmpc/acados_settings.py` — STMPC Solver Setup

**Source:** `mpc-ref/controller/mpc/src/single_track_mpc/acados_settings.py`

**5a. Fix imports:**
```python
from pathlib import Path
from .bicycle_model import bicycle_model
from ..config import STMPCConfig, CarConfig, PacejkaTireConfig
from .indicies import StateIndex
```

**5b. Add `_MPC_DIR` and output path handling:**
```python
_MPC_DIR = Path(__file__).resolve().parent
```
Set `ocp.code_export_directory = str(_MPC_DIR / "st_c_generated_code")` and `json_file=str(_MPC_DIR / "st_acados_ocp.json")`.

**5c.** Uses `integrator_type = "ERK"` with `sim_method_num_stages=4, sim_method_num_steps=3` (acados handles integration, not manual RK4).

**5d.** Only sets `f_expl_expr` on acados model (no `f_impl_expr` or `disc_dyn_expr`).

## Step 6: Create `stmpc/single_track_mpc.py` — Main Controller

**Source:** `mpc-ref/controller/mpc/src/single_track_mpc/single_track_mpc.py`

### 6a. Remove ROS imports and TrailingConfig
Delete: `import rospy`, `from f110_msgs.msg import WpntArray`, `from scipy.integrate import solve_ivp`.

### 6b. Fix local imports
```python
from .acados_settings import acados_settings
from ..config import STMPCConfig, CarConfig, PacejkaTireConfig
from ..utils.frenet_converter import FrenetConverter
from ..utils.splinify import SplineTrack
from .indicies import StateIndex
```

### 6c. Replace `rospy.loginfo()` / `rospy.logwarn()` → `print()`

### 6d. Simplify constructor
```python
def __init__(self, stmpc_config: STMPCConfig, car_config: CarConfig,
             tire_config: PacejkaTireConfig) -> None:
```
Remove: `pose_frenet`, `racecar_version`, `trailing_config`, `controller_frequency`, `using_gokart`.
Do NOT auto-call `mpc_initialize_solver()` in `__init__` — let the bridge call it.

### 6e. Rewrite `mpc_initialize_solver()` to accept arrays directly
Same pattern as KMPC:
```python
def mpc_initialize_solver(self, xs, ys, vx_ref, kappa_ref, s_ref, d_left, d_right):
```
Build `FrenetConverter` and `SplineTrack` from arrays. Call `acados_settings()` with `tire_config`.

### 6f. Remove `_transform_waypoints_to_cartesian()` and `_transform_waypoints_to_coords()`

### 6g. Remove trailing/overtake logic from `main_loop()`
Remove parameters: `state`, `speed_now`, `opponent`, `track_length`.
Hardcode `overtake_d = 0.0`.

New signature:
```python
def main_loop(self, position_in_map, waypoint_array_in_map,
              position_in_map_frenet, single_track_state, compute_time):
```

`single_track_state`: `[v_y, yaw_rate, measured_acc, measured_steer]` — use f110 path (not gokart): `steering_angle_buf[-1]` for measured_steer, `prev_acc` for measured_acc.

New return: `(speed, steering_angle, status)`.

### 6h. Remove `propagate_time_delay()` — disabled in reference. Keep `_dynamics_of_car()` for `get_warm_start()`.

### 6i. Clean up `mpc_init_params()` — remove trailing/overtake state vars.

### 6j. Delete `trailing_controller()` and `update_warm_start()`.

## Step 7: Add `STMPCGymBridge` to `gym_bridge.py`

**File:** `examples/controllers/mpc/gym_bridge.py` — add new class alongside existing `KMPCGymBridge`.

Add import:
```python
from .stmpc.single_track_mpc import Single_track_MPC_Controller
from .config import STMPCConfig, PacejkaTireConfig
```

### `STMPCGymBridge`

```python
class STMPCGymBridge:
    def __init__(self, track: Track, stmpc_config_path, car_config_path,
                 tire_config_path, ref_speed=4.0):
```

**Initialization** — same pattern as `KMPCGymBridge` with additions:
1. Load `STMPCConfig`, `CarConfig`, `PacejkaTireConfig` from YAML paths
2. Extract centerline, build `vx_ref`, `s_ref`, `waypoint_array` — same as KMPC
3. Construct `Single_track_MPC_Controller(stmpc_config, car_config, tire_config)`
4. Call `controller.mpc_initialize_solver(xs, ys, vx_ref, ks, s_ref, w_lefts, w_rights)`

**`get_action(obs)`** — similar to KMPC but feeds additional state:
1. Extract `pose_x, pose_y, pose_theta, linear_vel_x, linear_vel_y, ang_vel_z, delta` from `obs["agent_0"]`
2. Compute `(s, d)` and heading deviation `alpha` — same as KMPC
3. Build `single_track_state = [linear_vel_y, ang_vel_z, prev_acc, delta]`
4. Inject vehicle state: `controller.speed = linear_vel_x`, `controller.steering_angle_buf[:] = delta`
5. Call `controller.main_loop(...)`, return `np.array([[steering_angle, speed]])`

**`get_start_pose()`** — same as KMPC.

## Step 8: Create `stmpc_example.py` Runner

**File:** `examples/stmpc_example.py`

Same pattern as `kmpc_example.py` with:
- Import `STMPCGymBridge` instead of `KMPCGymBridge`
- Config paths: `STMPC_CONFIG`, `CAR_CONFIG`, `TIRE_CONFIG`
- `config["model"] = "std"`
- `config["observation_config"] = {"type": "frenet_dynamic_state"}`
- `config["control_input"] = ["speed", "steering_angle"]`

## Step 9: Update `.gitignore`

Add:
```
st_c_generated_code/
st_acados_ocp.json
```

## Files Summary

| File | Action | Details |
|------|--------|---------|
| `mpc/utils/__init__.py` | **Create** | Empty |
| `mpc/utils/frenet_converter.py` | **Move** | `git mv` from `mpc/` |
| `mpc/utils/splinify.py` | **Move** | `git mv` from `mpc/` |
| `mpc/kmpc/__init__.py` | **Create** | Empty |
| `mpc/kmpc/bicycle_model.py` | **Move** | `git mv` from `mpc/`, update imports |
| `mpc/kmpc/acados_settings.py` | **Move** | `git mv` from `mpc/`, update imports |
| `mpc/kmpc/kinematic_mpc.py` | **Move** | `git mv` from `mpc/`, update imports |
| `mpc/kmpc/c_generated_code/` | **Move** | `git mv` from `mpc/` |
| `mpc/kmpc/acados_ocp.json` | **Move** | `git mv` from `mpc/` |
| `mpc/config.py` | **Edit** | Add `STMPCConfig` + `PacejkaTireConfig` |
| `mpc/gym_bridge.py` | **Edit** | Update KMPC import + add `STMPCGymBridge` |
| `mpc/stmpc/__init__.py` | **Create** | Empty |
| `mpc/stmpc/bicycle_model.py` | **Create** | From `mpc-ref/.../single_track_mpc/bicycle_model.py` |
| `mpc/stmpc/acados_settings.py` | **Create** | From `mpc-ref/.../single_track_mpc/acados_settings.py` |
| `mpc/stmpc/single_track_mpc.py` | **Create** | From `mpc-ref/.../single_track_mpc/single_track_mpc.py` |
| `mpc/stmpc/indicies.py` | **Create** | From `mpc-ref/.../single_track_mpc/utils/indicies.py` |
| `mpc/config/single_track_mpc_params.yaml` | **Create** | From NUC2 YAML |
| `mpc/config/pacejka_tire_params.yaml` | **Create** | From DEFAULT YAML |
| `examples/stmpc_example.py` | **Create** | STMPC runner |
| `.gitignore` | **Edit** | Add `st_c_generated_code/`, `st_acados_ocp.json` |

**Unchanged:** `mpc/config.py` (KMPCConfig, CarConfig), `mpc/__init__.py`, `examples/kmpc_example.py`

## Verification

1. **KMPC regression**: Run `python examples/kmpc_example.py` after Step 0 to confirm reorganization didn't break anything
2. **Solver init**: Run `python examples/stmpc_example.py` — verify C code generates into `stmpc/st_c_generated_code/`
3. **Solver convergence**: Check `acados_solver.solve()` returns status 0
4. **Visual**: Render with `render_mode="human"` — car follows centerline
5. **Lap completion**: Full lap without crashing
6. **Comparison**: Run both runners on same map to compare KMPC vs STMPC tracking
