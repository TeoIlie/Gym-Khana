# STMPC Recovery Tuning Plan

## Current Baseline
- **14.6% recovery rate** (294 episodes, 7x7 grid x 6 inner combos)
- Recovery only succeeds in a narrow band near (beta~0, r~0)
- Complete failure at extreme beta (+/-80 deg) and extreme r (+/-745 deg/s)

## Understanding the Problem

The MPC state is `[s, n, theta, v_x, v_y, delta, yaw_rate, accel]`, with inputs `[jerk, derDelta]`. The cost function is:

```
qn*(n)^2 + qalpha*(theta)^2 + qv*(vx - V_target)^2 + qjerk*jerk^2 + qddelta*derDelta^2 + 100*vy^2
```

At extreme states (e.g. beta=80 deg, v=4 m/s -> v_y ~ 22.7 m/s), the hardcoded `100 * v_y^2` cost ~ **51,500** per stage, completely dominating all other terms. Combined with tight constraints, the solver likely goes **infeasible** (status != 0), triggering the warm-start fallback which commands `(0 steering, measured_steer)` -- effectively doing nothing useful.

## Phase 1: Diagnostic Run (Do First)

Before changing params, add temporary prints to confirm the failure mode:

1. In `single_track_mpc.py:216-218`, count how often `status != 0` per episode
2. In `gym_bridge.py:191-193`, check how often the startup guard (`vx < v_min`) fires
3. Log the initial v_y magnitude to see how extreme the states really are

This takes 10 minutes and tells you whether the problem is **solver infeasibility** vs **bad cost tuning** vs **the startup guard**.

## Phase 2: Code Fixes (Biggest Impact)

These are not YAML param changes but are critical blockers:

### 2a: Fix startup guard for recovery (`gym_bridge.py:191-193`)
```python
# Current: bypasses MPC when v_x < v_min
if vx < v_min:
    startup_speed = v_min + 3.0
    return np.array([[0.0, startup_speed]])  # zero steering!
```
For recovery at large beta, v_x could drop below v_min during the episode even if it starts above. Sending **zero steering** during a spin is catastrophic. Fix: either skip this guard in recovery mode, or at minimum use the current steering angle instead of 0.

### 2b: Fix init_from_obs reference velocity (`gym_bridge.py:227`)
```python
self.waypoint_array[:, 2] = float(agent_obs["linear_vel_x"])
```
This sets `V_target` to the car's current v_x for all future waypoints. For recovery, V_target should be a fixed moderate value (e.g., 3-4 m/s), not the perturbed state's v_x.

## Phase 3: YAML Parameter Tuning (Ordered by Expected Impact)

### 3a: Relax constraints -- allow maximum control authority

The car config (`car_model.yaml`) allows `max_steering_angle: 0.5` and `max_steering_velocity: 3.2`, but the MPC config is more restrictive. For recovery, give the MPC full authority:

```yaml
# State bounds -- match car_config limits
delta_min: !!float -0.5   # was -0.4 -- full steering range
delta_max: !!float 0.5    # was 0.4 -- full steering range
a_min: !!float -7         # was -5 -- match car_config, allow hard braking
a_max: !!float 7          # was 5 -- match car_config, allow hard accel
v_min: !!float 0.1        # was 0.5 -- allow planning at near-zero forward speed

# Input bounds -- allow faster corrections
ddelta_min: !!float -5.0  # was -3.2 -- faster emergency counter-steer
ddelta_max: !!float 5.0   # was 3.2

# Nonlinear constraint -- more lateral acceleration headroom
alat_max: !!float 15      # was 6 -- extreme states need extreme authority
```

**Rationale**: The ellipse constraint `(a_lat/alat_max)^2 + (v_x_dot/a_max)^2 <= 1` is a key feasibility bottleneck. At a_lat_max=6 and a_max=5, the solver simply can't plan a trajectory from extreme states. Widening these limits dramatically improves feasibility.

### 3b: Relax track constraints -- use full track width

```yaml
track_safety_margin: 0.0  # was 0.25 -- recovery needs every centimeter
```

### 3c: Reprioritize cost function for stabilization

```yaml
qalpha: 100    # was 50  -- heading alignment is priority #1
qn: 10         # was 40  -- lateral position less important than stability
qv: 1          # was 10  -- don't fight speed during recovery
qddelta: 0.1  # was 0.5 -- allow faster steering corrections
qjerk: 0.005  # was 0.01 -- allow more aggressive acceleration
```

**Key insight**: For recovery, the priority order should be:
1. Kill lateral velocity (v_y -> 0) -- already has weight 100 (hardcoded)
2. Align heading (theta -> 0) -- increase qalpha
3. Damp yaw rate (via heading dynamics) -- indirect through qalpha
4. Stay on track (n -> 0) -- lower priority
5. Match speed (vx -> V_target) -- lowest priority

### 3d: Consider shorter horizon

```yaml
N: 20  # was 40 -- 1 second horizon at 20 Hz
```

**Trade-off**: Shorter horizon means faster solve times and better short-term control, but less lookahead. For recovery, short-term aggressive control is more valuable than long-term planning. Also halves computation time, reducing risk of solver timeouts.

### 3e: Consider disabling the combined acceleration constraint

```yaml
combined_constraints: "none"  # was "ellipse"
```

**Rationale**: The friction ellipse is a physical validity constraint for normal driving. During recovery from extreme states, the car is already far beyond the friction circle -- constraining the MPC to stay within it prevents it from commanding the aggressive corrections needed. Removing it lets the optimizer find the best trajectory even if it temporarily violates the friction limit. The underlying physics simulator will enforce actual limits anyway.

**Risk**: The MPC plan may be physically unrealizable, but a "best effort" plan is better than an infeasible solver returning nothing.

### 3f: Soften slack penalties for feasibility

```yaml
Zl: !!float 100   # was 1000 -- softer penalty helps solver find feasible solutions
Zu: !!float 100   # was 1000
zl: !!float 10    # was 100
zu: !!float 10    # was 100
```

Lower slack penalties let the solver violate soft constraints (track boundaries) more easily, which helps it find *any* feasible solution in extreme states. Better to have a plan that slightly exceeds track bounds than no plan at all.

## Phase 4: Investigate Adding Yaw Rate Damping (Code Change)

The cost function has no explicit yaw rate (`omega`) term. Adding one directly penalizes spinning:

In `bicycle_model.py:210-230`, add `weight_yr * yaw_rate^2` to the cost expression. This requires:
1. Adding a `qyr` parameter to the YAML and `STMPCConfig`
2. Adding it as an online parameter
3. Adding the cost term

This would be the most impactful single change for recovery -- directly telling the MPC "stop spinning."

## Phase 5: Iterative Testing Protocol

Run the benchmark after each group of changes:
```bash
python examples/analysis/beta_r_avg_plot.py --controller_type stmpc
```

**Suggested order:**
1. Phase 1 (diagnostics) -> understand failure mode
2. Phase 3a + 3b (constraint relaxation) -> test if feasibility is the bottleneck
3. Phase 3c (cost rebalancing) -> test if cost priorities matter
4. Phase 2a + 2b (code fixes) -> fix startup guard and ref velocity
5. Phase 3d + 3e (horizon + friction ellipse) -> further improvements
6. Phase 4 (yaw rate cost) -> if still insufficient

**Expected outcome per phase:**
- Constraint relaxation alone should improve from ~15% to ~25-35%
- Cost rebalancing should add another ~5-10%
- Code fixes + yaw rate damping could push to ~40-60%
- The outer ring of the grid (beta=+/-80 deg, r=+/-745 deg/s) will likely remain unrecoverable -- those states have v_y > 20 m/s with the car nearly sideways, beyond what any controller can reasonably recover from

## Quick-Win Changes

Single YAML to try immediately that combines the highest-impact changes:

```yaml
N: 20
track_safety_margin: 0.0
qjerk: !!float 5e-3
qddelta: !!float 0.1
qn: 10
qalpha: 100
qv: !!float 1
delta_min: !!float -0.5
delta_max: !!float 0.5
v_min: !!float 0.1
a_min: !!float -7
a_max: !!float 7
ddelta_min: !!float -5.0
ddelta_max: !!float 5.0
alat_max: !!float 15
combined_constraints: "none"
Zl: !!float 100
Zu: !!float 100
zl: !!float 10
zu: !!float 10
```

## Key Files

- MPC recover params: `examples/controllers/mpc/config/st_mpc_recover_params.yaml`
- MPC race params: `examples/controllers/mpc/config/st_mpc_race_params.yaml`
- Car config: `examples/controllers/mpc/config/car_model.yaml`
- Tire config: `examples/controllers/mpc/config/pacejka_tire_params.yaml`
- MPC cost function / dynamics: `examples/controllers/mpc/stmpc/bicycle_model.py`
- Acados solver setup: `examples/controllers/mpc/stmpc/acados_settings.py`
- MPC controller loop: `examples/controllers/mpc/stmpc/single_track_mpc.py`
- Gym bridge (startup guard, ref velocity): `examples/controllers/mpc/gym_bridge.py`
- Controller wrapper: `examples/controllers/mpc/stmpc_controller.py`
- Benchmark script: `examples/analysis/beta_r_avg_plot.py`
- Gym env (recovery task definition): `f1tenth_gym/envs/f110_env.py`
