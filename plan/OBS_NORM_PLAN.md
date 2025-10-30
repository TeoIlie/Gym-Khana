
# Implementation Plan for Drift Observation Normalization

  1. Determining Maximum Values for Each State

  Here's how to find the bounds for each observation feature:

  From Vehicle Parameters (accessible via env.params):

  - linear_vel_x: Max velocity from v_max or reasonable physical limit (~10 m/s for F1TENTH)
  - linear_vel_y: Typically much smaller than vx, use friction-based lateral velocity limit (~3-5 m/s)
  - ang_vel_z: From maximum yaw rate = v_max / (wheelbase/2) or use typical limit (~3-5 rad/s)
  - delta: From s_max (maximum steering angle, typically ~0.4 rad = 23°)
  - prev_steering_cmd: Same as delta
  - prev_accl_cmd: From a_max parameter or motor limits (typical: ±9.51 m/s²)
  - curr_accl_cmd: Same as above
  - prev_avg_wheel_omega: Max wheel speed = v_max / wheel_radius
  - curr_vel_cmd: Same as linear_vel_x max

  From Track Data (requires track analysis):

  - lookahead_curvatures: Compute max/min curvature from all tracks or specific track
  - lookahead_widths: Max track width from track data (typically 1-3 meters)

  Physical Constants:

  - frenet_u: Heading error angle, range [-π, π] radians
  - frenet_n: Lateral distance from centerline, bounded by ±max_track_width/2

  2. Implementation Architecture

  VectorObservation
  ├── __init__(normalize=True, norm_bounds=None)
  ├── _compute_normalization_bounds()  # Extract from env params/track
  ├── space()  # Return normalized [-1, 1] or raw bounds
  └── observe()  # Apply normalization if enabled

  3. Where Normalization Should Occur

  Recommended approach: Inside VectorObservation class
  - Pro: Keeps normalization logic with observation generation
  - Pro: Easy to toggle via constructor parameter
  - Pro: Observation space bounds automatically match
  - Con: None significant

  Steps:
  1. Add normalize parameter to VectorObservation.__init__()
  2. Create _compute_normalization_bounds() method to extract bounds
  3. Store bounds as instance variables
  4. Apply normalization in observe() before returning vector
  5. Update space() to return [-1, 1] bounds when normalized

  4. Configuration Design

### Option 1: Via observation_factory parameter
  env = gym.make('f1tenth_gym:f1tenth-v0',
      config={'observation_config': {
          'type': 'drift',
          'normalize': True,  # or False for raw values
          'norm_bounds': 'auto'  # or custom dict
      }}
  )

### Option 2: Separate observation types
  config={'observation_config': {'type': 'drift_normalized'}}  # normalized
  config={'observation_config': {'type': 'drift'}}  # raw (default)

  5. Normalization Strategy

  For each feature, normalize to [-1, 1] range:
  normalized_value = (raw_value - min_bound) / (max_bound - min_bound) * 2 - 1
  # or for symmetric bounds: normalized_value = raw_value / max_abs_value

  Proposed bounds dictionary:
  {
      'linear_vel_x': (0, 10.0),          # m/s
      'linear_vel_y': (-5.0, 5.0),        # m/s  
      'frenet_u': (-np.pi, np.pi),        # rad
      'frenet_n': (-2.0, 2.0),            # m (track-dependent)
      'ang_vel_z': (-5.0, 5.0),           # rad/s
      'delta': (-0.4, 0.4),               # rad
      'prev_steering_cmd': (-0.4, 0.4),   # rad
      'prev_accl_cmd': (-10.0, 10.0),     # m/s²
      'prev_avg_wheel_omega': (0, 300),   # rad/s
      'curr_vel_cmd': (0, 10.0),          # m/s
      'lookahead_curvatures': (-5.0, 5.0), # 1/m (track-dependent)
      'lookahead_widths': (0.5, 3.0),     # m (track-dependent)
  }

  6. Implementation Steps

  1. Add normalization bounds computation:
    - Create method to extract vehicle parameters (v_max, s_max, a_max, etc.)
    - Add track analysis for curvature/width ranges (or use conservative defaults)
    - Store as self.norm_bounds dict
  2. Modify VectorObservation class:
    - Add normalize parameter to __init__
    - Implement _compute_normalization_bounds()
    - Implement _normalize_value(value, feature_name) helper
    - Update observe() to apply normalization
    - Update space() to return appropriate bounds
  3. Update observation_factory:
    - Parse normalize parameter from config
    - Pass to VectorObservation constructor
    - Consider adding "drift_normalized" as shorthand
  4. Debugging support:
    - Add get_raw_observation() method that bypasses normalization
    - Log normalization bounds on first observation
    - Add assertion checks for out-of-bounds values

  7. Testing Strategy

  - Test with normalize=False to verify raw values unchanged
  - Test with normalize=True to verify values in [-1, 1]
  - Test edge cases (zero velocity, maximum steering, etc.)
  - Verify observation space shape matches
  - Test with multiple tracks to ensure bounds are appropriate

  8. Recommended File Changes

  Files to modify:
  - f1tenth_gym/envs/observation.py: Main implementation
  - f1tenth_gym/envs/f110_env.py: Pass normalize config to observation_factory
  - examples/ (optional): Add example using normalized observations

  New test file:
  - tests/test_drift_observation_normalization.py: Unit tests for normalization logic

  ## Note on normalizing observations for DRL
  - The range [-1, 1] is best, because it is zero-centrered, symmetric, and no bias shift

  ---
  ## Summary

  Key decisions:
  1. ✅ Normalize inside VectorObservation.observe() method
  2. ✅ Add normalize=True/False constructor parameter
  3. ✅ Auto-compute bounds from vehicle params + track data
  4. ✅ Support manual bounds override for custom ranges
  5. ✅ Update observation space to reflect normalized [-1, 1] range
  6. ✅ Provide get_raw_observation() method for debugging

  Benefits:
  - Clean separation of concerns
  - Easy to toggle for debugging
  - Observation space accurately reflects actual values
  - Works with RL algorithms expecting normalized inputs