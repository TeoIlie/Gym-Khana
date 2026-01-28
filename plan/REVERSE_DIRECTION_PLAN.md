# Implementation Plan: Reverse Direction Driving (Simplified)

## Overview

Add `track_direction` config with values `"normal"`, `"reverse"`, `"random"`. Compute reversed tracks once at Track init, then swap active references at reset time.

## Key Design

- `centerline_regular` / `raceline_regular`: Original direction (computed at init)
- `centerline_reversed` / `raceline_reversed`: Reversed direction (computed at init)
- `centerline` / `raceline`: **Active references** pointing to either regular or reversed version

Existing code continues using `self.track.centerline` unchanged - the reference just points to the appropriate version.

## Files to Modify

1. `f1tenth_gym/envs/track/raceline.py` - Add `reversed()` method
2. `f1tenth_gym/envs/track/track.py` - Store all 4 versions, manage active references
3. `f1tenth_gym/envs/f110_env.py` - Add config, swap active references at reset

## Implementation

### Step 1: Raceline.reversed() Method

**File:** `f1tenth_gym/envs/track/raceline.py`

This single method works for both centerline and raceline files because both produce Raceline objects with the same attributes. The method handles None values for w_lefts/w_rights (which raceline files don't have).

```python
def reversed(self) -> "Raceline":
    """
    Create reversed copy for reverse-direction driving.

    Works for both centerline (has w_lefts/w_rights) and raceline (w_lefts/w_rights are None).
    """
    # Reverse coordinate arrays
    xs_rev = self.xs[::-1].copy()
    ys_rev = self.ys[::-1].copy()

    # Flip yaw by pi, wrap to [-pi, pi]
    yaws_rev = None
    if self.yaws is not None:
        yaws_rev = (self.yaws[::-1] + np.pi).copy()
        yaws_rev = np.arctan2(np.sin(yaws_rev), np.cos(yaws_rev))

    # Negate curvatures (left turns become right turns)
    ks_rev = -self.ks[::-1].copy() if self.ks is not None else None

    # Swap track widths (left boundary becomes right) - handles None for raceline files
    w_lefts_rev = self.w_rights[::-1].copy() if self.w_rights is not None else None
    w_rights_rev = self.w_lefts[::-1].copy() if self.w_lefts is not None else None

    # Reverse velocities/accelerations (keep magnitudes)
    vxs_rev = self.vxs[::-1].copy() if self.vxs is not None else None
    axs_rev = self.axs[::-1].copy() if self.axs is not None else None

    return Raceline(
        xs=xs_rev, ys=ys_rev, velxs=vxs_rev,
        psis=yaws_rev, kappas=ks_rev, accxs=axs_rev,
        w_lefts=w_lefts_rev, w_rights=w_rights_rev,
    )
```

### Step 2: Track Manages Active References

**File:** `f1tenth_gym/envs/track/track.py`

**2a. Update `__init__`** to store all 4 versions:

```python
def __init__(
    self,
    spec: TrackSpec,
    occupancy_map: np.ndarray,
    filepath: Optional[str] = None,
    ext: Optional[str] = None,
    centerline: Optional[Raceline] = None,
    raceline: Optional[Raceline] = None,
):
    # ... existing initialization ...

    # Store regular versions
    self.centerline_regular = centerline
    self.raceline_regular = raceline

    # Compute reversed versions once
    self.centerline_reversed = centerline.reversed() if centerline else None
    self.raceline_reversed = raceline.reversed() if raceline else None

    # Active references (default to regular direction)
    self.centerline = self.centerline_regular
    self.raceline = self.raceline_regular
```

**2b. Add method to swap active references:**

```python
def set_direction(self, reversed: bool) -> None:
    """
    Set track direction by swapping active centerline/raceline references.

    Args:
        reversed: If True, use reversed versions. If False, use regular versions.
    """
    if reversed:
        self.centerline = self.centerline_reversed
        self.raceline = self.raceline_reversed
    else:
        self.centerline = self.centerline_regular
        self.raceline = self.raceline_regular
```

### Step 3: F110Env Swaps Direction at Reset

**File:** `f1tenth_gym/envs/f110_env.py`

**3a. Add config** in `default_config()`:

```python
"track_direction": "normal",  # "normal", "reverse", or "random"
```

**3b. Add instance variable** in `__init__`:

```python
self.track_direction_config = self.config["track_direction"]
self.direction_reversed = False
```

**3c. Update `reset()`** - resolve direction and swap Track references:

```python
def reset(self, seed=None, options=None):
    # ... existing seed/super code ...

    # Resolve direction for this episode
    if self.track_direction_config == "normal":
        self.direction_reversed = False
    elif self.track_direction_config == "reverse":
        self.direction_reversed = True
    elif self.track_direction_config == "random":
        self.direction_reversed = np.random.random() < 0.5
    else:
        raise ValueError(f"Invalid track_direction: {self.track_direction_config}")

    # Swap Track's active centerline/raceline references
    self.track.set_direction(self.direction_reversed)

    # ... existing reset code continues ...

    # When sampling poses, flip yaw if reversed
    if poses is None:
        poses = self.reset_fn.sample()
        if self.direction_reversed:
            poses[:, 2] += np.pi
            poses[:, 2] = np.arctan2(np.sin(poses[:, 2]), np.cos(poses[:, 2]))

    # ... rest of existing reset code ...
```

## What Changes vs What Stays the Same

**Changes:**
- raceline.py: Add `reversed()` method (~25 lines)
- track.py: Store 4 versions, add `set_direction()` method (~15 lines)
- f110_env.py: Add config option, resolve direction at reset (~15 lines)

**Stays the Same:**
- All code using `self.track.centerline` or `self.track.raceline` - unchanged
- `_get_reward()` - unchanged
- `_check_boundary_frenet()` - unchanged
- Observation code - unchanged
- Rendering callbacks - unchanged
- Lap counting - unchanged
- Reset functions - unchanged

## Verification

```python
import gymnasium as gym
import numpy as np

# Test reverse mode
env = gym.make('f1tenth_gym:f1tenth-v0', config={
    'track_direction': 'reverse',
    'render_mode': 'human',
})
obs, _ = env.reset()

# Vehicle should face opposite direction
# Driving forward (positive action) should yield positive reward
for _ in range(500):
    action = np.array([[1.0, 0.0]])  # speed=1, steering=0
    obs, reward, done, truncated, info = env.step(action)
    env.render()
    print(f"Reward: {reward:.3f}")  # Should be positive
    if done or truncated:
        break
```

## Computation Cost

- **One-time (Track init):** `Raceline.reversed()` called twice - fast O(n) array operations
- **Per-reset:** Reference swap only - O(1)
- **Per-step:** Zero additional overhead (same code paths as before)
