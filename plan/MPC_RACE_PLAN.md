# MPC Controller Implementation Plan

## Confirmed Design Decisions
- **Solver**: CasADi + IPOPT
- **Approach**: Staged - Phase 1 kinematic bicycle, Phase 2 upgrade to full STD
- **Control output**: `[accl, steering_angle]` (same as RL controller)
- **Reference path**: Centerline (contouring MPC)
- **Work split**: Claude writes mechanical parts (dynamics translation, NLP setup), guides user through conceptual parts (cost tuning, constraint formulation)

## Context

We have a trained PPO policy that races a 1/10th scale F1TENTH car around a track using the nonlinear STD model. The goal is to implement an equivalent MPC controller for comparison. Since this is the user's first MPC, we take a staged approach: get a working MPC with the simple kinematic bicycle model first, then upgrade the prediction model to the full STD dynamics.

---

## 1. Solver: CasADi + IPOPT

- **CasADi**: Symbolic framework for nonlinear optimization. Handles automatic differentiation, builds NLP problems. `pip install casadi`.
- **IPOPT**: Interior-point NLP solver, ships with CasADi. Warm-starting from previous solution makes MPC fast.
- New dependency: add `casadi` to `pyproject.toml`

---

## 2. Staged Dynamics Approach

### Phase 1: Kinematic Bicycle Model (5 states)

Use the existing model from `kinematic.py:8` (`vehicle_dynamics_ks`):

```
States:   [X, Y, delta, V, psi]     (5 states)
Controls: [steering_velocity, acceleration]  (2 inputs)
ODEs:
  dX/dt    = V * cos(psi)
  dY/dt    = V * sin(psi)
  ddelta/dt = steering_velocity
  dV/dt    = acceleration
  dpsi/dt  = (V / lwb) * tan(delta)
```

This is a direct 1:1 translation to CasADi (`np.cos` -> `ca.cos`, etc.). Only 5 lines of dynamics. No tire model needed.

**Why start here:**
- Trivially simple dynamics - easy to debug MPC formulation issues
- Exercises all the MPC infrastructure: NLP setup, cost function, constraints, Frenet projection, warm-starting
- Works well at moderate speeds (up to ~5 m/s) which is enough to validate the approach
- If MPC doesn't work with kinematic model, the issue is in the MPC formulation, not the dynamics

**Limitations:** No lateral dynamics, no slip - won't capture drift behavior. That's fine for Phase 1.

### Phase 2: Full 9-State STD Model (future)

Once kinematic MPC works, replace the prediction model with a line-by-line CasADi translation of `vehicle_dynamics_std()`. The NLP structure, cost function, and constraints stay the same - only the dynamics function changes.

---

## 3. MPC Formulation: Contouring NMPC

### Objective: Maximize Forward Progress Along Centerline

This mirrors the RL reward structure (progress along centerline × gain, penalty for boundary violation).

### Cost Function

```
J = Sum_{k=0}^{N-1} [
    -w_progress * (s_{k+1} - s_k)             # maximize arc-length progress
    + w_contour * ey_k^2                        # minimize lateral deviation from centerline
    + w_heading * ephi_k^2                      # minimize heading error vs centerline
    + w_steer   * u_steer_k^2                   # penalize large steering inputs
    + w_accel   * u_accel_k^2                   # penalize large throttle inputs
]
+ Terminal: w_contour_N * ey_N^2 + w_heading_N * ephi_N^2
```

Where at each prediction step k:
- `s_k` = arc length along centerline (computed from X_k, Y_k)
- `ey_k` = lateral deviation from centerline
- `ephi_k` = heading error

### Computing Frenet Coordinates in the NLP

The challenge: converting Cartesian `(X, Y, psi)` to Frenet `(s, ey, ephi)` must be differentiable for CasADi.

**Approach:** At each MPC solve, extract a local window of centerline waypoints around the current position. Within the NLP, approximate the Frenet projection using these waypoints:
1. At solve time: find current arc length `s_current` using `track.cartesian_to_frenet()` (NumPy, outside NLP)
2. Extract centerline waypoints in range `[s_current, s_current + horizon_distance]`
3. In the NLP: for each predicted state `(X_k, Y_k, psi_k)`, compute `ey_k` and `ephi_k` relative to the nearest extracted waypoint using CasADi-compatible vector math (cross products, dot products)

This avoids putting the full spline inside CasADi - instead we use a piecewise-linear centerline approximation over the prediction window.

### Constraints

**Control input constraints (box):**
- `-3.2 <= u_steer_vel <= 3.2` rad/s
- `-5.0 <= u_accel <= 5.0` m/s² (with speed-dependent limit: `u_accel <= a_max * min(1, v_switch/V)`)

**State constraints (box):**
- `-0.5 <= delta <= 0.5` rad (steering angle)
- `-5.0 <= V <= 20.0` m/s (velocity)
- **Track boundaries:** `-w_right(s_k) <= ey_k <= w_left(s_k)` (vary along track, interpolated from centerline data)

**Soft boundary constraints:** Add slack variables with large penalty to prevent solver infeasibility when car is near boundary.

---

## 4. MPC Horizon & Timing

- **Prediction horizon:** N = 20 steps × dt_mpc = 0.05s = **1.0 second lookahead**
- **MPC control rate:** Solve every 5 env steps (every 50 ms = 20 Hz)
  - Apply first control action, hold for 5 steps
  - ~50 ms computation budget per solve (should be ample for kinematic model)
- **Warm-starting:** Shift previous solution forward by 1 step as initial guess for next solve

---

## 5. Integration with Existing Simulation

### File Structure
```
examples/controllers/
    mpc_controller.py          # MPCController class (inherits Controller)
    mpc_dynamics.py            # CasADi dynamics model (kinematic, later STD)
    mpc_utils.py               # Frenet projection, centerline extraction helpers
examples/
    mpc_example.py             # Example script to run MPC around a track
```

### Controller Interface

```python
class MPCController(Controller):
    def __init__(self, track, params, N=20, dt_mpc=0.05, ...):
        # Extract centerline data from track
        # Build CasADi NLP once

    def get_action(self, obs) -> np.ndarray:
        # Get full state from env (need X, Y, psi - not in standard obs)
        # Every Kth call: solve NLP, cache solution
        # Other calls: return cached action
        # Output: [[steering_angle, acceleration]]

    def get_env_config(self) -> dict:
        # control_input: ["accl", "steering_angle"]
        # model: "ks" for Phase 1, "std" for Phase 2
        # observation_config: {"type": "drift"} or {"type": "kinematic_state"}
```

### State Access

The MPC needs the full Cartesian state `[X, Y, delta, V, psi]`. Two options:
- Access `env.sim.agents[0].state` directly (array)
- Access `env.sim.agents[0].standard_state` (dict with named keys: x, y, delta, v_x, yaw, etc.)

For Phase 1 (kinematic env with `model: "ks"`), the state vector IS `[X, Y, delta, V, psi]` - a direct match.

### Track Data (reuse existing infrastructure)

From `env.track.centerline` (Raceline object in `f1tenth_gym/envs/track/raceline.py`):
- `ss` (arc lengths), `xs`, `ys`, `yaws`, `ks` (curvatures)
- `w_lefts`, `w_rights` (track boundary widths)
- `spline` (CubicSpline2D for precise interpolation)

Use `track.cartesian_to_frenet()` for initial Frenet state (outside the NLP).

---

## 6. Implementation Steps

### Step 1: Install CasADi & Validate
- `pip install casadi` in the virtual environment
- Write a minimal test: create CasADi variable, solve trivial NLP
- **Claude writes this**

### Step 2: Kinematic Dynamics in CasADi
- Translate `vehicle_dynamics_ks()` (5 ODEs) to CasADi symbolic function
- Implement RK4 discretization in CasADi
- Validate: compare CasADi prediction vs NumPy for same initial state + controls
- **Claude writes this** (mechanical translation)

### Step 3: Centerline Data Extraction
- Write helper to extract local centerline window: waypoints, widths, arc lengths
- Write CasADi-compatible Frenet projection (piecewise-linear approximation)
- **Claude writes extraction, guides user through Frenet projection concept**

### Step 4: NLP Construction
- Build CasADi `Opti` problem with:
  - Decision variables: states[0..N], controls[0..N-1]
  - Dynamics constraints (RK4 shooting)
  - Input/state box constraints
  - Track boundary constraints (soft)
  - Cost function (progress + contour + control smoothness)
- **Claude writes NLP structure, user understands cost weight choices**

### Step 5: Controller Class
- Wrap NLP in `MPCController(Controller)`
- Handle warm-starting, solve timing, fallback on solver failure
- **Claude writes this**

### Step 6: Test & Tune
- Run on a simple track with rendering
- Tune cost weights: start with `w_progress=1, w_contour=10, w_heading=5, w_steer=0.1, w_accel=0.01`
- Profile solve times
- **User drives tuning, Claude helps debug**

### Step 7 (Phase 2): Upgrade to STD Dynamics
- Replace kinematic dynamics with full STD CasADi translation
- Switch env to `model: "std"`, params to `f1tenth_std_vehicle_params()`
- Re-tune cost weights
- Compare against RL controller

---

## 7. Key Files to Modify/Create

| File | Action | Description |
|------|--------|-------------|
| `examples/controllers/mpc_controller.py` | Create | MPCController class |
| `examples/controllers/mpc_dynamics.py` | Create | CasADi dynamics (kinematic + later STD) |
| `examples/controllers/mpc_utils.py` | Create | Centerline extraction, Frenet helpers |
| `examples/mpc_example.py` | Create | Run script |
| `examples/controllers/__init__.py` | Modify | Register new controller |
| `pyproject.toml` | Modify | Add casadi dependency |

### Existing Code to Reuse
- `examples/controllers/base.py` — Controller base class
- `f1tenth_gym/envs/dynamic_models/kinematic.py:8` — Reference kinematic model equations
- `f1tenth_gym/envs/track/raceline.py` — Centerline data (ss, xs, ys, yaws, w_lefts, w_rights)
- `f1tenth_gym/envs/track/track.py:382` — `cartesian_to_frenet()` for initial state conversion
- `train/config/env_config.py` — `get_drift_test_config()` for env configuration template

---

## 8. Verification

1. **CasADi dynamics test**: Forward-simulate kinematic model in CasADi vs NumPy for 100 steps with same inputs — states should match within 1e-10
2. **Straight-line test**: Place car on straight section, verify MPC accelerates forward smoothly
3. **Corner test**: Place car before a corner, verify MPC steers and slows down
4. **Full lap**: Run MPC around complete track with `render_mode="human"`, verify stays within boundaries
5. **Timing**: Print solve times per MPC call — should be <50ms for kinematic model
