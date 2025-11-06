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

| Aspect | Current | Drift |
|--------|---------|-------|
| Wraparound | Potential bug | Fix wraparound aclength reward bug on line 796 of `f110_env.py`: Detect by `progress < -0.5 * track_length`
| Boundary | TTC (predictive) | Explicit crossing with Frenet coordinates |
| Collision result | Car deflected | No behavioural effect 
| Structure | `progress - penalties` | `-1` OR `progress` (exclusive) |
| Reset condition | TODO | Reset when agent exceeds track rollout
| Reset function | Configurable with `reset_config` param | Set `reset_config` to `cl_random_random` if observation type is `drift`
| Multi-agent | Collaborative | Collaborative (unchanged)

### Configuration options
The following changes should be made to the `gym` configuration options:
1. Collision checking can be configured with new parameter `predictive_collision` to `True` (current TTC) or `False` (Frenet coords out-of-bounds)
2. Collision behaviour can be configured with new parameter `collision_deflection` to `True` (current wall deflection) or `False` (no behavioural effect - car passes through wall)