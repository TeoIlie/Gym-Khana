# Drift Reward Implementation

## Current Implementation

### Reward (`f110_env.py:780-804`)
- **Formula**: `sum(progress_i) - sum(collision_penalty_i)` across all agents
- **Progress**: Arc length `s` via `track.centerline.spline.calc_arclength_inaccurate(x, y)`
- **Collision**: Fixed -1.0 penalty (TTC-based, predictive)
- **Collaborative**: Sums all agent rewards

### Reset (`f110_env.py:859-919`)
- Gets poses from `options["poses"]` or `reset_fn.sample()`
- Default: `"rl_grid_static"` (raceline grid, no shuffle)
- Format: `<refline>_<resetfn>_<shuffle>` (e.g., `"rl_random_random"`)
- Resets counters, sim state, takes zero-action step for initial obs

### Collision Detection (`base_classes.py:291-309`)
- **Method**: Time-To-Collision (TTC) with 0.005s threshold
- **Sources**: Track boundaries (LiDAR) + vehicle-vehicle
- **Behavior**: Predictive (triggers before contact)

## Target Implementation

```
r_t = -1                if track boundary exceeded
      s_t - s_{t-1}     otherwise
```

## Available Infrastructure

### Frenet Coordinates (`track.py:349-410`)
```python
s, ey, ephi = track.cartesian_to_frenet(x, y, phi, use_raceline=False)
# s: arc length [m]
# ey: lateral deviation [m] (this is 'n' from DRIFT_PLAN.md)
# ephi: heading error [rad] (this is 'u' from DRIFT_PLAN.md)
```

### Track Boundaries
```python
track.centerline.w_lefts[i]   # Left width at waypoint i [m]
track.centerline.w_rights[i]  # Right width at waypoint i [m]
track.centerline.spline.s[i]  # Arc length at waypoint i [m]

# Out of bounds check: ey > w_left OR ey < -w_right
```

## Implementation Requirements

| Status | Aspect | Current | Drift |
|--------|--------|---------|-------|
| [X] | Wraparound | Potential bug | Fix wraparound aclength reward bug on line 796 of `f110_env.py`: Detect by `progress < -0.5 * track_length`
| [X] | Boundary | TTC (predictive) | Explicit crossing with Frenet coordinates, via new boolean env config param `predictive_collision`, where True is the current TTC, and False is explicit via Frenet coords. Default to True, set to False if observatin type is `"drift"`, and allow user override regardless |
| [X] | Structure | `progress - penalties` | `-1` OR `progress` (exclusive) |
| [X] | Wraparound fix | N/A | Validate why the current wraparound logic uses `0.5` on line 896 of `f110_env.py`
| [ ] | Reset condition | First of: (a) Ego-agent has a collision or (b) Completed 2 laps of the track | Reset when ego-agent has a collision
| [ ] | Reset function | Configurable with `reset_config` param | Set `reset_config` to `cl_random_random` if observation type is `drift`
| [ ] | Multi-agent | Collaborative | Collaborative (unchanged)
| [ ] | Max progress validation | N/A | Consider adding validation that the reward for progress cannot exceed `v_max` * `dt`, possibly by tracking min/max values in `self.obs_min_max_tracker`