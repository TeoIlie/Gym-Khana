# Plan: Implement Step 6 — `stmpc/single_track_mpc.py`

## Context

We are porting the Single Track MPC (STMPC) controller from the ForzaETH ROS race stack into F1TENTH Gym. Steps 0–5 are done (directory reorganization, configs, indicies, bicycle_model, acados_settings). Step 6 creates the main controller class by adapting `mpc-ref/controller/mpc/src/single_track_mpc/single_track_mpc.py`, removing ROS/trailing/overtake/gokart logic, following the same pattern as the already-ported KMPC at `examples/controllers/mpc/kmpc/kinematic_mpc.py`.

## Critical Files

- **Reference:** `mpc-ref/controller/mpc/src/single_track_mpc/single_track_mpc.py` (source to port)
- **Pattern to follow:** `examples/controllers/mpc/kmpc/kinematic_mpc.py` (already-ported KMPC)
- **Target:** `examples/controllers/mpc/stmpc/single_track_mpc.py` (new file)
- **Already ported (Step 5):** `examples/controllers/mpc/stmpc/acados_settings.py`
- **Configs:** `examples/controllers/mpc/config.py` (STMPCConfig, CarConfig, PacejkaTireConfig)
- **Indices:** `examples/controllers/mpc/stmpc/indicies.py` (StateIndex enum)
- **Shared utils:** `examples/controllers/mpc/utils/frenet_converter.py`, `examples/controllers/mpc/utils/splinify.py`

## Key Differences: STMPC vs KMPC

| Aspect | KMPC | STMPC |
|--------|------|-------|
| State vector | 5: [s, n, alpha, v, delta] | 8: [s, n, theta, v_x, v_y, delta, yaw_rate, accel] |
| Controls | [acceleration, ddelta] | [jerk, ddelta] |
| Online params | 10 | 12 (adds a_min, a_max separately) |
| State bounds | 2: [v, delta] | 3: [v_x, delta, accel] |
| Input bounds | [a_min/max, ddelta_min/max] | [jerk hardcoded ±50, ddelta_min/max] — but dynamically set each step |
| Warm start dims | 7 (5 states + 2 inputs) | 10 (8 states + 2 inputs) |
| `_dynamics_of_car` | 7 args | 10 args (8 states + jerk + derDelta) |
| Delay mechanism | Fixed `time_delay_step=3` | Uses `stmpc_config.steps_delay` + qddelta multiplier |
| Lap counting | `fre_s // track_length` based | `previous_frenet_s` delta comparison |
| Return | `(speed, steering_angle)` | `(speed, steering_angle, status)` |

## State Feedback Loop (Bridge → Controller → Bridge)

Each gym step, the bridge reads observations and feeds them to the controller:

```
Gym obs (agent_0)          →  Bridge builds  →  Controller x0 (8 states)
─────────────────────────     ──────────────     ─────────────────────────
pose_x, pose_y             →  FrenetConv     →  x0[0] s    (arc-length)
                           →  FrenetConv     →  x0[1] n    (lateral dev)
pose_theta + track deriv   →  alpha calc     →  x0[2] theta (heading dev)
linear_vel_x               →  controller.speed = ...  →  x0[3] v_x
linear_vel_y               →  single_track_state[0]   →  x0[4] v_y
delta                      →  steering_angle_buf[:]=  →  x0[5] delta
ang_vel_z                  →  single_track_state[1]   →  x0[6] yaw_rate
(not from obs)             →  self.prev_acc           →  x0[7] accel  ← INTERNAL ESTIMATE
```

**Acceleration feedback (internal estimate):** After each solve, acceleration is updated via Euler integration of the MPC's jerk output: `prev_acc += jerk / MPC_freq`. This is fed back as `x0[7]` on the next step. No gym observation is used — this matches the reference's f110 car behavior (no accelerometer).

**Controller outputs → Bridge → Gym:**
```
Controller solve  →  pred_x at delayed_index  →  Bridge action
────────────────     ────────────────────────     ──────────────
pred_x[3] v_x    →  speed                     →  action[1] speed
pred_x[5] delta  →  steering_angle            →  action[0] steering_angle
```

The gym receives `control_input = ["speed", "steering_angle"]` — position-level commands, NOT jerk/acceleration. Jerk is only the MPC's internal optimization variable.

## Implementation Plan

### 1. Imports
```python
import numpy as np
from .acados_settings import acados_settings
from ..config import STMPCConfig, CarConfig, PacejkaTireConfig
from ..utils.frenet_converter import FrenetConverter
from ..utils.splinify import SplineTrack
from .indicies import StateIndex
```
Remove: `rospy`, `solve_ivp`, `WpntArray`, `TrailingConfig`.

### 2. Constructor — `__init__(self, stmpc_config, car_config, tire_config)`
- Store configs only. Call `mpc_init_params()`.
- Do NOT call `mpc_initialize_solver()` (let bridge call it, same as KMPC pattern).
- Remove: `pose_frenet`, `racecar_version`, `trailing_config`, `controller_frequency`, `using_gokart`.

### 3. `mpc_init_params()`
Keep from reference:
- `t_MPC`, `t_delay`, `steering_angle_buf`, `mpc_sd`, `u0`, `fre_s`, `comp_time`
- `previous_frenet_s = 0` (for lap counting)
- `prev_acc = 0` (for acceleration estimation)
- `speed = 0` (initial speed state)

Remove:
- `overtake_d`, `loop_rate`, `gap*`, `v_diff`, `i_gap` (trailing/overtake state)

Replace `rospy.loginfo` with `print`.

### 4. `mpc_initialize_solver(self, xs, ys, vx_ref, kappa_ref, s_ref, d_left, d_right)`
Same signature as KMPC. Build `FrenetConverter` and `SplineTrack` from arrays.
- Call `acados_settings(s_ref, kappa_ref, d_left, d_right, stmpc_config, car_config, tire_config)` (note: STMPC acados_settings doesn't take `vx_ref` — it uses online params for target speed)
- Store `self.kappa`, `self.prev_acc = 0`
- Do NOT call `apply_warm_start` here (unlike reference) — the bridge will seed state via `set(0, "lbx/ubx", x0)` on first main_loop call, and warm start is only needed on solver failure.

### 5. `main_loop(self, position_in_map, waypoint_array_in_map, position_in_map_frenet, single_track_state, compute_time)`

**Parameters:**
- `position_in_map`: shape (1, 3) — [x, y, theta]
- `waypoint_array_in_map`: shape (N, 8) — columns [x, y, vx, d_m, s_m, kappa, yaw, ax]
- `position_in_map_frenet`: shape (3,) — [s, d, alpha]
- `single_track_state`: [v_y, yaw_rate, measured_acc, measured_steer] — bridge fills these from obs
- `compute_time`: float

**Logic (mostly from reference, with removals):**

1. Store instance vars from params. Extract `vel_y`, `yaw_rate`, `measured_steer`, `measured_acc` from `single_track_state`.

2. Compute `center_car_s` and `alpha_center` (same as KMPC/reference).

3. Lap counting using `previous_frenet_s` comparison (reference pattern, NOT KMPC's `fre_s // track_length`):
   ```python
   if self.position_in_map_frenet[0] < self.previous_frenet_s - 1:
       self.nr_laps += 1
   self.previous_frenet_s = self.position_in_map_frenet[0]
   ```

4. Build 8-state x0:
   ```python
   x0 = np.array([fre_s, fre_d, fre_alpha, speed, vel_y, measured_steer, yaw_rate, measured_acc])
   ```
   **IMPORTANT:** `speed` here is `self.speed` (vehicle speed injected by bridge), NOT `speed_now` param. Follow KMPC pattern where bridge sets `controller.speed = linear_vel_x`.

5. Set initial state constraints: `solver.set(0, "lbx", x0)` / `solver.set(0, "ubx", x0)`.

6. Set online parameters for each stage `i in range(N+1)`:
   - Look up `target_v_speed` from `waypoint_array_in_map` (same as KMPC)
   - Hardcode `overtake_d = 0` (time-trial)
   - **Steps delay mechanism**: For `i < steps_delay`, multiply `qddelta` by 1e6 to freeze steering during delay compensation period
   - 12 online params: `[target_v_speed, qadv, qv, qn, qalpha, qjerk, qddelta, alat_max, a_min, a_max, track_safety_margin, overtake_d]`

7. Set input bounds: `lbu = [-50, ddelta_min]`, `ubu = [50, ddelta_max]` (jerk bounds hardcoded to ±50 matching reference — these are dynamically settable but reference uses constants)
   - **Note:** Actually use `jerk_min`/`jerk_max` from config since YAML has them and acados_settings also uses them. The reference hardcodes -50/50 which happens to match config values.

8. Set state bounds at `i+1`: `[v_min, delta_min, a_min]` to `[v_max, delta_max, a_max]` (3 bounds vs KMPC's 2).

9. Solve OCP. On failure (status != 0), call `apply_warm_start`.

10. Extract solution:
    - `u0 = solver.get(0, "u")`
    - `prev_acc = solver.get(0, "x")[StateIndex.ACCEL.value]`
    - `delayed_index = steps_delay + 1`
    - `pred_x = solver.get(delayed_index, "x")`
    - `steering_angle = pred_x[5]` (StateIndex.STEERING_ANGLE_DELTA)
    - `speed = pred_x[3]` (StateIndex.VELOCITY_V_X)
    - Update `prev_acc` with Euler integration: `prev_acc + u0[0] / MPC_freq`

11. Update steering buffer (same pattern as KMPC).

12. Update `mpc_sd` for next step's reference speed lookup.

13. Return `(speed, steering_angle, status)`.
    - On solver failure: return `(0, measured_steer, status)` (safe fallback).

### 6. `apply_warm_start(self, pose_frenet)`
Same pattern as reference. Call `get_warm_start(pose_frenet, const_acc=1, const_steer_vel=0.0)`.

### 7. `get_warm_start(self, pose_frenet, const_acc, const_steer_vel)`
- Warm start array: `(N+1, 10)` — 8 states + 2 inputs
- Initial: `[s, d, alpha, 1, 0, 0, 0, const_acc, 0, const_steer_vel]`
- Propagate with `_dynamics_of_car` + Euler integration at `1/MPC_freq`

### 8. `_dynamics_of_car(self, t, x0)`
- Unpack 10 values: s, n, theta, v_x, v_y, delta, yaw_rate, accel, derDelta, jerk
- Call `self.model.f_expl_func(s, n, theta, v_x, v_y, delta, yaw_rate, accel, jerk, derDelta, p)`
- **CAUTION:** The `f_expl_func` signature in bicycle_model.py orders args as `(..., jerk, derDelta, p)` but the state+input order is `[..., accel, jerk, derDelta]` where jerk and derDelta are controls. The x0 array packs `[...states..., derDelta, jerk]` but the function expects `jerk` before `derDelta`. Match reference exactly: `x0[8]` = derDelta, `x0[9]` = jerk, but call `f_expl_func(..., derDelta, jerk, p)`.
  - **Wait — let me re-check.** In the reference `_dynamics_of_car`: `derDelta = x0[8]`, `jerk = x0[9]`, and it calls `f_expl_func(s, n, theta, v_x, v_y, delta, yaw_rate, accel, derDelta, jerk, p)`. But in `bicycle_model.py` the function is defined as `Function("f_expl_func", [s, n, theta, v_x, v_y, delta, yaw_rate, accel, jerk, derDelta, p], [f_expl])` — **jerk BEFORE derDelta**. This looks like a bug in the reference's `_dynamics_of_car` that swaps them! However, since controls u = `vertcat(jerk, derDelta)`, and the warm start puts `[..., derDelta_val, jerk_val]` at indices 8,9... Actually the warm_start init is `[s, d, alpha, 1, 0, 0, 0, const_acc, 0, const_steer_vel]` — index 8 is 0 (could be either), index 9 is `const_steer_vel`. In `propagate_time_delay`, `x0` is built as `np.concatenate((states, [inputs[0]], [0]))` where `inputs[0]` is jerk (Input.JERK=0). So x0[8]=jerk, x0[9]=0, but `_dynamics_of_car` reads `derDelta = x0[8]`, `jerk = x0[9]`. That IS swapped from propagate_time_delay's convention. BUT propagate_time_delay is disabled. For warm_start, both are 0 so it doesn't matter.
  - **Decision:** Follow the `f_expl_func` definition order (jerk, derDelta) to be correct. Pack warm start as `[..., accel, jerk, derDelta]` where index 8=jerk (Input.JERK), index 9=derDelta (Input.D_STEERING_ANGLE). Call `f_expl_func(..., jerk, derDelta, p)`.

### 9. Methods to DELETE (not port)
- `_transform_waypoints_to_cartesian()` — ROS waypoint conversion
- `_transform_waypoints_to_coords()` — ROS waypoint conversion
- `propagate_time_delay()` — disabled in reference (`t_delay = 0.00`)
- `trailing_controller()` — trailing/overtake logic
- `update_warm_start()` — commented out in reference

## Edge Cases and Potential Issues

1. **`f_expl_func` argument order mismatch**: The reference `_dynamics_of_car` swaps jerk/derDelta relative to the CasADi function definition. Since both are 0 in warm start, this is benign in practice but wrong in principle. We fix this by matching the `bicycle_model.py` definition.

2. **`speed` state initialization**: Bridge must set `controller.speed = linear_vel_x` before calling `main_loop`, just like KMPC. If `speed` starts at 0 and the car is already moving, the first MPC step will have wrong initial state.

3. **`prev_acc` feedback loop**: Reference updates `prev_acc` twice — once from solver state, once from Euler integration. We simplify: use `prev_acc = self.prev_acc + self.u0[0] / self.stmpc_config.MPC_freq` after solve (Euler integration of jerk). This is what the bridge will feed back as `single_track_state[2]`.

4. **`measured_steer` from buffer vs obs**: Reference uses `steering_angle_buf[-1]` for f110 car. Bridge should pass `delta` from obs as `single_track_state[3]`, but controller uses `steering_angle_buf[-1]` internally. For consistency with KMPC pattern, we let bridge set `controller.steering_angle_buf[:] = delta` AND pass it in `single_track_state[3]`. Inside main_loop, use `single_track_state[3]` directly for `measured_steer`.

5. **Solver failure fallback**: Reference returns `(0, 0, 0, measured_steer, states, status)` on failure — speed=0 causes hard braking. For gym, returning `(0, measured_steer, status)` is acceptable; bridge can check status.

6. **Warm start on first call**: Unlike reference (which warm-starts in `mpc_initialize_solver`), we skip it. The solver will get `x0` from `lbx/ubx` constraint. If first solve fails, warm start will be applied via the failure path. This matches KMPC behavior.

7. **`steps_delay` index bounds**: `delayed_index = steps_delay + 1`. With `steps_delay=3` and `N=40`, `delayed_index=4` which is safely within bounds.

8. **`nr_laps` and lap crossing**: The `previous_frenet_s` comparison (`< self.previous_frenet_s - 1`) handles the wrap-around. The `-1` margin prevents noise from triggering false lap increments.

## Verification

1. Run `python examples/stmpc_example.py` (created in Step 8) — solver should initialize and generate C code
2. Check solver returns status 0 on first successful solve
3. Verify car follows centerline visually with `render_mode="human"`
4. Complete a full lap without crashing
5. Compare with KMPC on same map
