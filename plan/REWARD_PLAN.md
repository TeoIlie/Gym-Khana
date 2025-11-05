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

## Target: Drift Reward

### Formula (from DRIFT_PLAN.md)
```
r_t = -1                if track boundary exceeded
      s_t - s_{t-1}     otherwise
```

### Key Differences

| Aspect | Current | Drift |
|--------|---------|-------|
| Structure | `progress - penalties` | `-1` OR `progress` (exclusive) |
| Multi-agent | Sum all agents | Ego only |
| Boundary | TTC (predictive) | Explicit crossing |
| Penalty | Added to reward | Replaces reward |
| Collision result | Car deflected | No behavioural effect 

## Implementation

### DriftEnv Class (`examples/drift_env.py`)

```python
import numpy as np
from f1tenth_gym.envs import F110Env

class DriftEnv(F110Env):
    """Drift reward: -1 if boundary exceeded, else s_t - s_{t-1}"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.last_s = None
        self.track_length = self.track.centerline.spline.s[-1]
        self.boundary_s = self.track.centerline.spline.s
        self.boundary_left = self.track.centerline.w_lefts
        self.boundary_right = self.track.centerline.w_rights

    def _get_track_width_at_s(self, s):
        """Linear interpolation of track width at arc length s."""
        s = s % self.track_length
        idx = np.searchsorted(self.boundary_s, s)

        if idx == 0:
            return self.boundary_left[0], self.boundary_right[0]
        if idx >= len(self.boundary_s):
            return self.boundary_left[-1], self.boundary_right[-1]

        s0, s1 = self.boundary_s[idx-1], self.boundary_s[idx]
        alpha = (s - s0) / (s1 - s0) if s1 > s0 else 0.0

        w_left = (1-alpha) * self.boundary_left[idx-1] + alpha * self.boundary_left[idx]
        w_right = (1-alpha) * self.boundary_right[idx-1] + alpha * self.boundary_right[idx]

        return w_left, w_right

    def _check_boundary_violation(self, agent_idx):
        """Check if agent exceeded track boundaries via Frenet coords."""
        x, y, theta = self.poses_x[agent_idx], self.poses_y[agent_idx], self.poses_theta[agent_idx]
        s, ey, _ = self.track.cartesian_to_frenet(x, y, theta, use_raceline=False)
        w_left, w_right = self._get_track_width_at_s(s)
        return (ey > w_left) or (ey < -w_right)

    def _get_reward(self):
        """Return -1 if boundary violated, else progress."""
        i = self.ego_idx

        # Initialize on first call
        if self.last_s is None:
            self.last_s = np.zeros(self.num_agents)
            for j in range(self.num_agents):
                self.last_s[j], _ = self.track.centerline.spline.calc_arclength_inaccurate(
                    self.poses_x[j], self.poses_y[j]
                )

        # Check boundary
        if self._check_boundary_violation(i):
            return -1.0

        # Calculate progress
        current_s, _ = self.track.centerline.spline.calc_arclength_inaccurate(
            self.poses_x[i], self.poses_y[i]
        )
        progress = current_s - self.last_s[i]

        # Handle wrap-around
        if progress < -0.5 * self.track_length:
            progress = (self.track_length - self.last_s[i]) + current_s

        self.last_s[i] = current_s
        return progress

    def reset(self, seed=None, options=None):
        """Reinitialize progress tracking."""
        obs, info = super().reset(seed=seed, options=options)
        self.last_s = None
        return obs, info
```

### Training Script (`examples/drift_training.py`)

```python
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from drift_env import DriftEnv

def make_env():
    return DriftEnv(config={
        "map": "Spielberg",
        "num_agents": 1,
        "model": "std",
        "observation_config": {"type": "drift"},
        "params": gym.make("f1tenth_gym:f1tenth-v0").unwrapped.f1tenth_std_vehicle_params(),
        "reset_config": {"type": "rl_random_random"},
    })

if __name__ == "__main__":
    env = VecMonitor(DummyVecEnv([make_env]))
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./drift_tb/")
    model.learn(total_timesteps=2_000_000)
    model.save("drift_ppo")
```

## Key Considerations

1. **Boundary Detection**: Frenet-based (actual crossing) vs TTC (predictive). Frenet recommended for drift.
2. **Interpolation**: Linear interpolation of track width between waypoints.
3. **Wrap-around**: Detect by `progress < -0.5 * track_length`.
4. **Reset Strategy**: Use `"rl_random_random"` for diverse training.
5. **Compatibility**: Requires `model="std"` and `observation_config={"type": "drift"}`.

## Next Steps

1. Create `examples/drift_env.py` and `examples/drift_training.py`
2. Test boundary detection accuracy
3. Validate reward behavior (should see -1 and small positive values)
4. Run short training to verify setup
5. Tune hyperparameters
