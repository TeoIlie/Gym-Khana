# Multi-Map Training Implementation Plan

## Overview
Enable the F1TENTH Gym to train reinforcement learning agents across multiple tracks simultaneously, improving policy generalization beyond single-map training.

## Background Context

### Current Limitation
- Track loaded once at environment initialization via `Track.from_track_name()` in `f110_env.py:189`
- All training happens on single map specified in `gym_config.yaml`
- Agents may overfit to specific track geometry and fail to generalize

### Available Resources
- **11 tracks total**: 7 large racing circuits (Austin, Catalunya, IMS, Monza, MoscowRaceway, Spielberg, Spielberg_blank) + 3 drift tracks (Drift, Drift2, Drift_large)
- **Parallel training**: `SubprocVecEnv` runs 32 parallel environments (2 × CPU cores)
- **Existing randomization**: `track_direction="random"` successfully randomizes driving direction per episode

---

## Approach Comparison

### Approach 1: Different Maps Per Parallel Environment ✅
**Description**: Each of n_envs parallel environments trains on a different map. PPO averages gradients across all environments.

**Pros**:
- **Zero runtime overhead**: Maps loaded once at initialization
- **Normalization solvable**: Requires global bounds across track pool (see implementation details)
- **Natural averaging**: PPO already averages gradients across parallel envs
- **Simple implementation**: ~60 lines of code changes
- **Proven pattern**: Similar to domain randomization in robotics
- **Every batch is diverse**: Each training batch contains experiences from multiple tracks

**Cons**:
- Limited by n_envs: With 32 envs and 3 tracks, each track gets ~10-11 environments
- All difficulties trained simultaneously (no curriculum)
- May slow initial learning if hard tracks dilute easy track signal

**Verdict**: **RECOMMENDED** - Simplest correct implementation with no downsides.

---

### Alternative Approach 2: Randomize Track at Each Reset ❌
**Description**: Every episode trains on a randomly selected map.

**Pros**:
- Maximum diversity - agent sees different tracks every episode
- No curriculum design needed
- Similar to existing `track_direction` randomization

**Cons**:
- **Critical issue**: Observation normalization bounds are track-specific (curvature, width) and become invalid when switching
- **Performance**: Expensive map loading (50-100ms) at every reset vs normal 1ms
- **Learning**: Sample inefficient - no time to learn track-specific patterns before switching
- **Reset system**: Must recreate `reset_fn` each time since it binds to specific track

**Verdict**: NOT RECOMMENDED due to normalization complexity and overhead.

---

### Alternative Approach 3: Curriculum Learning 🔄
**Description**: Train fixed number of steps on each map in sequence (easy → medium → hard).

**Pros**:
- Pedagogically sound - learn simple behaviors first
- Explicit control over progression
- Can combine with Approach 2 for curriculum across parallel envs
- Good for debugging - isolate problematic tracks

**Cons**:
- Manual curriculum design required (define "easy" vs "hard")
- Risk of catastrophic forgetting on earlier tracks
- Rigid schedule - can't adapt to agent's learning
- Arbitrary phase lengths (hyperparameter tuning needed)
- More complex implementation (requires callback system)

**Verdict**: NOT RECOMMENDED due to rigis schedule, and poor adaption

---


## Recommended Implementation: Different Maps Per Parallel Environment
Implement the different map per parallel environment solution (Approach 1). This is most robust.


## Implementation Details

### Implementation Strategy
Modify environment creation to distribute a list of maps across parallel environments:
- Env 0, 3, 6, ... → Map 0
- Env 1, 4, 7, ... → Map 1
- Env 2, 5, 8, ... → Map 2

For example, with 32 parallel envs and 3 drift tracks, each track gets ~10-11 environments.

### Critical Issue: Global Normalization Required

**Problem**: Current implementation normalizes observations per-track, which causes inconsistent observation meanings across environments:
- Track A (max_curv=0.5): Tight turn → normalized to 1.0
- Track B (max_curv=1.0): Tight turn → normalized to 1.0
- Policy learns "1.0" has different real-world meanings, hurting generalization

**Solution**: Use **global normalization bounds** computed across ALL tracks in the pool:
- Track A tight turn (κ=0.5) → normalized to 0.5
- Track B tight turn (κ=1.0) → normalized to 1.0
- Observation value 1.0 consistently means "sharpest possible turn across any track"
- Policy learns consistent meaning, improving transfer between tracks

**Implementation**:
1. In `make_subprocvecenv()`: Compute `track_max_curv`, `track_min_width` and `track_max_width` once across all tracks
2. Pass these as config parameters to each environment
3. In `calculate_norm_bounds()`: Use provided bounds if available, else compute from track
4. Ensure the global bounds are also used in the evalutation env

**Which features need global bounds?**
- **Lookahead curvatures**: YES - curvature varies significantly between tracks (drift tracks have tighter turns than racing circuits)
- **Track width (frenet_n, lookahead_widths)**: YES - track widths may vary, should also use global bounds
- **Other features**: NO - other features (velocity, steering, etc.) are vehicle-dependent, not track-dependent

### Code Changes Required

#### File 1: `train/training_utils.py`
**Location**: Lines 21-37 (function `make_subprocvecenv`)

**Change**: Add `track_pool` parameter, compute global bounds using helper function, and distribute maps across environments.

**Add helper function** (before `make_subprocvecenv`):
```python
def compute_global_track_bounds(track_pool: list[str], track_scale: float) -> dict:
    """
    Compute global normalization bounds across all tracks in a pool.

    Args:
        track_pool: List of track names to compute bounds across
        track_scale: Scale factor for track loading

    Returns:
        Dictionary with keys: track_max_curv, track_min_width, track_max_width
    """
    from f1tenth_gym.envs.track import Track
    from f1tenth_gym.envs.track.track_utils import get_min_max_curvature, get_min_max_track_width

    max_curvatures = []
    min_widths = []
    max_widths = []

    for track_name in track_pool:
        try:
            track = Track.from_track_name(track_name, track_scale=track_scale)
        except FileNotFoundError as e:
            raise ValueError(
                f"Invalid track name '{track_name}' in track_pool. "
                f"Please check available tracks in the 'maps/' directory."
            ) from e

        # get_min_max_curvature returns symmetric bounds (-max, +max)
        _, max_curv = get_min_max_curvature(track)
        max_curvatures.append(max_curv)

        min_width, max_width = get_min_max_track_width(track)
        min_widths.append(min_width)
        max_widths.append(max_width)

    return {
        "track_max_curv": max(max_curvatures),
        "track_min_width": min(min_widths),
        "track_max_width": max(max_widths),
        "_per_track_curvatures": dict(zip(track_pool, max_curvatures)),  # For logging only
    }
```

**Modify `make_subprocvecenv`**:
```python
def make_subprocvecenv(
    seed: int, config: dict, n_envs: int, track_pool: list[str] | None = None
) -> SubprocVecEnv:
    """
    Create a SubprocVecEnv parallelized environment.

    Args:
        seed: Random seed
        config: Base gym configuration dict
        n_envs: Number of parallel environments
        track_pool: Optional list of maps to distribute across envs.
                   If provided, envs will cycle through these maps and use global normalization.
                   Example: ["Drift", "Drift2", "Drift_large"]

    Returns:
        SubprocVecEnv with environments distributed across track pool (if provided)
    """
    if track_pool is not None:
        # Validate track pool
        if not isinstance(track_pool, list) or len(track_pool) == 0:
            raise ValueError("track_pool must be a non-empty list")

        print(f"[Multi-map] Computing global normalization bounds across {len(track_pool)} tracks...")

        # Compute global normalization bounds across all tracks in pool
        track_scale = config.get("scale", 1.0)
        global_bounds = compute_global_track_bounds(track_pool, track_scale)

        print(f"[Multi-map] Global bounds: curvature=+/-{global_bounds['track_max_curv']:.4f}, "
              f"width=[{global_bounds['track_min_width']:.4f}, {global_bounds['track_max_width']:.4f}]")
        print(f"[Multi-map] Per-track curvatures: {global_bounds['_per_track_curvatures']}")

        # Create environments with different maps but shared global bounds
        env_fns = []
        for i in range(n_envs):
            # Cycle through track pool
            map_name = track_pool[i % len(track_pool)]
            # Create per-env config with specific map and global bounds
            env_config = config.copy()
            env_config["map"] = map_name
            env_config["track_max_curv"] = global_bounds["track_max_curv"]
            env_config["track_min_width"] = global_bounds["track_min_width"]
            env_config["track_max_width"] = global_bounds["track_max_width"]
            env_fns.append(make_env(seed=seed, rank=i, config=env_config))

        env = SubprocVecEnv(env_fns)

        # Print distribution summary
        from collections import Counter
        distribution = Counter(track_pool[i % len(track_pool)] for i in range(n_envs))
        print(f"[Multi-map] Track distribution: {dict(distribution)}")

        return env, global_bounds  # Return bounds for eval env
    else:
        # Original single-map behavior
        env = SubprocVecEnv([make_env(seed=seed, rank=i, config=config) for i in range(n_envs)])
        print(f"Successfully created {n_envs} parallel environments as SubProcVecEnv with seed {seed}")
        return env, None  # No global bounds for single-map
```

**Why**:
- Helper function `compute_global_track_bounds` is testable in isolation
- Fixed missing `min_widths = []` initialization
- Removed emojis for terminal compatibility
- Returns `global_bounds` dict so eval env can use the same normalization
- Added return type hint
- Computes global bounds ONCE (not 32x per environment)
- Uses `track_scale` from config for consistent track scaling
- Validates track names early with helpful error messages

**Breaking change note**:
The return type changes from `SubprocVecEnv` to `tuple[SubprocVecEnv, dict | None]`.
Existing callers must update:
```python
# Before:
env = make_subprocvecenv(seed, config, n_envs)

# After:
env, _ = make_subprocvecenv(seed, config, n_envs)
# Or with track_pool:
env, global_bounds = make_subprocvecenv(seed, config, n_envs, track_pool=track_pool)
```

---

#### File 2: `train/config/gym_config.yaml`
**Location**: Anywhere in the file (suggested: after `map` parameter)

**Change**: Add optional `track_pool` parameter.

```yaml
# Multi-map training (optional)
# If specified, parallel environments will train on different maps from this list
# If null/commented, uses single 'map' parameter above
track_pool:
  - "Drift"
  - "Drift2"
  - "Drift_large"
# For single-map training, set to null or comment out:
# track_pool: null
```

**Why**: Allows easy switching between single-map and multi-map training via config.

---

#### File 3: `train/config/env_config.py`
**Location**: Line ~38 (after other constant definitions)

**Change**: Load `TRACK_POOL` from YAML.

```python
# Add after existing constant definitions
TRACK_POOL = _gym_config.get("track_pool", None)
```

**Location**: Line ~113 (inside `get_drift_train_config()` return dict)

**Change**: Add `track_pool` to config dict.

```python
def get_drift_train_config() -> dict:
    return {
        "seed": SEED,
        "map": MAP,
        # ... other config parameters ...
        "track_pool": TRACK_POOL,  # ADD THIS LINE
    }
```

**Why**: Makes track pool accessible to training scripts.

---

#### File 4: `train/ppo_race.py`
**Location**: Line ~73 (where `make_subprocvecenv` is called)

**Change**: Pass `TRACK_POOL` constant to environment creation and pass global bounds to eval env.

```python
from train.config.env_config import (
    # ... existing imports ...
    TRACK_POOL,  # ADD THIS IMPORT
)

def train_ppo_race():
    TRAIN_CONFIG = get_drift_train_config()

    # Create vectorized environment with optional multi-map support
    # Returns global_bounds dict when track_pool is used, None otherwise
    env, global_bounds = make_subprocvecenv(SEED, TRAIN_CONFIG, N_ENVS, track_pool=TRACK_POOL)

    # Create eval env with same normalization bounds as training envs
    eval_env = make_eval_env(EVAL_SEED, TRAIN_CONFIG, global_bounds=global_bounds)

    # ... rest of training code unchanged ...
```

**Why**:
- Captures `global_bounds` returned by `make_subprocvecenv`
- Passes bounds to eval env to ensure consistent observation normalization
- When `track_pool=None`, `global_bounds=None` preserves original single-map behavior

---

#### File 4b: `train/training_utils.py` - Update `make_eval_env`
**Location**: Lines 145-152 (function `make_eval_env`)

**Change**: Add `global_bounds` parameter to apply same normalization as training.

```python
def make_eval_env(seed: int, config: dict, global_bounds: dict | None = None):
    """
    Create a single evaluation environment for EvalCallback.

    Args:
        seed: Random seed for evaluation
        config: Gym env config
        global_bounds: Optional global normalization bounds from multi-map training.
                      If provided, these bounds are injected into config for consistent
                      normalization between training and evaluation.
    """
    eval_config = config.copy()

    # Apply global bounds if provided (multi-map training)
    if global_bounds is not None:
        eval_config["track_max_curv"] = global_bounds["track_max_curv"]
        eval_config["track_min_width"] = global_bounds["track_min_width"]
        eval_config["track_max_width"] = global_bounds["track_max_width"]
        print(f"[Eval env] Using global normalization bounds from multi-map training")

    env = gym.make(get_env_id(), config=eval_config)
    env = Monitor(env)
    env.reset(seed=seed)
    return env
```

**Why**:
- Ensures eval env uses identical normalization bounds as training envs
- Without this, observations would have different meanings during eval vs training
- Backward compatible: `global_bounds=None` preserves original behavior

---

#### File 5: `f1tenth_gym/envs/utils.py`
**Location**: Before `calculate_norm_bounds` function

**Change**: Add helper function to get track bounds from config or compute from track.

**Add helper function**:
```python
def _get_track_width_bounds(env, track) -> tuple[float, float]:
    """Get track width bounds from global config or compute from track."""
    global_min = env.config.get("track_min_width")
    global_max = env.config.get("track_max_width")
    if global_min is not None and global_max is not None:
        return (global_min, global_max)
    return get_min_max_track_width(track)


def _get_track_curvature_bounds(env, track) -> tuple[float, float]:
    """Get track curvature bounds from global config or compute from track."""
    global_max = env.config.get("track_max_curv")
    if global_max is not None:
        return (-global_max, global_max)
    return get_min_max_curvature(track)
```

**Location**: Lines 188-201 (track-dependent bounds calculation)

**Change**: Use helper functions for cleaner code.

**Find these lines**:
```python
# Frenet lateral distance and track widths
if "frenet_n" in features_set or "lookahead_widths" in features_set:
    min_width, max_width = get_min_max_track_width(track)

    if "frenet_n" in features_set:
        half_max_width = 0.5 * max_width
        bounds["frenet_n"] = (-half_max_width, half_max_width)

    if "lookahead_widths" in features_set:
        bounds["lookahead_widths"] = (min_width, max_width)

# Lookahead curvatures
if "lookahead_curvatures" in features_set:
    min_curv, max_curv = get_min_max_curvature(track)
    bounds["lookahead_curvatures"] = (min_curv, max_curv)
```

**Replace with**:
```python
# Frenet lateral distance and track widths
if "frenet_n" in features_set or "lookahead_widths" in features_set:
    min_width, max_width = _get_track_width_bounds(env, track)

    if "frenet_n" in features_set:
        half_max_width = 0.5 * max_width
        bounds["frenet_n"] = (-half_max_width, half_max_width)

    if "lookahead_widths" in features_set:
        bounds["lookahead_widths"] = (min_width, max_width)

# Lookahead curvatures
if "lookahead_curvatures" in features_set:
    min_curv, max_curv = _get_track_curvature_bounds(env, track)
    bounds["lookahead_curvatures"] = (min_curv, max_curv)
```

**Why**:
- Helper functions encapsulate the "global or local bounds" logic
- Main function stays clean and readable
- Logic change is minimal (just swap function calls)
- Backward compatible: Falls back to track-computed bounds if config not provided

---

#### File 6: `f1tenth_gym/envs/f110_env.py`
**Location**: In `__init__` method, after track loading (around line 189-200)

**Change**: Add validation that all three global bounds must be provided together, then log when using global bounds.

**Add these lines after track is loaded**:
```python
# Validate global normalization bounds (multi-map training)
# If any bound is provided, all three must be provided for consistency
track_bound_keys = ["track_max_curv", "track_min_width", "track_max_width"]
provided_bounds = [key for key in track_bound_keys if self.config.get(key) is not None]

if provided_bounds and len(provided_bounds) != 3:
    missing_bounds = set(track_bound_keys) - set(provided_bounds)
    raise ValueError(
        f"Incomplete global track bounds configuration. "
        f"When using multi-map training, all three bounds must be provided. "
        f"Provided: {provided_bounds}, Missing: {list(missing_bounds)}"
    )

# Log if using global normalization bounds
if len(provided_bounds) == 3:
    print(f"[Global bounds] curvature=+/-{self.config['track_max_curv']:.4f}, "
          f"width=[{self.config['track_min_width']:.4f}, {self.config['track_max_width']:.4f}]")
```

**Why**:
- Validates all-or-none constraint: prevents partial configuration that would cause inconsistent normalization
- Fails fast with clear error message if misconfigured
- Removed emojis for terminal compatibility
- Logs configuration for debugging

---

### Summary of Changes
- **6 files modified**:
  1. `train/training_utils.py` - Add `compute_global_track_bounds()` helper, modify `make_subprocvecenv()` to return bounds, update `make_eval_env()` to accept bounds
  2. `train/config/gym_config.yaml` - Add track_pool config parameter
  3. `train/config/env_config.py` - Load and export track_pool constant
  4. `train/ppo_race.py` - Capture global_bounds and pass to eval env
  5. `f1tenth_gym/envs/utils.py` - Use global bounds (`track_max_curv`, `track_min_width`, `track_max_width`) when provided
  6. `f1tenth_gym/envs/f110_env.py` - Validate all-or-none bounds constraint, log when using global bounds
- **~85 lines of code added** (including helper function, validation, bounds computation, and comments)
- **Key improvements**:
  - Helper function `compute_global_track_bounds()` is testable in isolation
  - Eval env uses same normalization as training (critical for correct evaluation)
  - All-or-none validation prevents partial/invalid configuration
  - No emojis in logs for terminal compatibility
- **Key efficiency**: Global bounds computed once (not 32× per environment)
- **Respects track_scale**: Uses config's `scale` parameter when loading tracks
- **Backward compatible**: Setting `track_pool: null` preserves original single-map behavior with per-track normalization

---

## Testing & Verification
- To-do after implementation
### Evaluation Protocol
After training with multi-map support:

1. **Per-track evaluation**: Test trained policy on each individual track
   - Success rate (% episodes without collision)
   - Average lap time
   - Boundary violation frequency

2. **Cross-track generalization**: Train on 2 tracks, evaluate on held-out 3rd track
   - Example: Train on `["Drift", "Drift2"]`, test on `"Drift_large"`

3. **Compare to baseline**: Train identical policy on single map, compare generalization

4. **Metrics to track in wandb**:
   - Episode reward per track
   - Success rate per track
   - Steps to convergence
   - Policy entropy (measure of exploration)

---

## Configuration Examples

### Example 1: Multi-Map Drift Training
```yaml
# gym_config.yaml
map: "Drift"  # Fallback/default
track_pool:
  - "Drift"
  - "Drift2"
  - "Drift_large"
```

### Example 2: Single-Map Training (Original Behavior)
```yaml
# gym_config.yaml
map: "Drift_large"
track_pool: null  # Disable multi-map
```

### Example 3: Multi-Map Racing Training
```yaml
# gym_config.yaml
map: "Spielberg"
track_pool:
  - "Spielberg"
  - "Monza"
  - "Catalunya"
```

---

## Critical Files Reference

### Files Modified
- `train/training_utils.py`:
  - Add `compute_global_track_bounds()` helper function
  - Modify `make_subprocvecenv()` to return `(env, global_bounds)` tuple
  - Modify `make_eval_env()` to accept `global_bounds` parameter
- `train/config/gym_config.yaml` - Add track_pool parameter
- `train/config/env_config.py` (lines ~38, ~113) - Load and export TRACK_POOL constant
- `train/ppo_race.py` (line ~73) - Capture global_bounds and pass to eval env
- `f1tenth_gym/envs/utils.py` (lines ~188-201) - Use `track_max_curv`, `track_min_width`, `track_max_width` when provided
- `f1tenth_gym/envs/f110_env.py` (line ~189-200) - Validate all-or-none bounds, log global bounds usage

### Files Referenced (Context)
- `f1tenth_gym/envs/track/track.py` (lines 138-205) - `Track.from_track_name()`
- `f1tenth_gym/envs/track/track_utils.py` - `get_min_max_curvature()`, `get_min_max_track_width()`
- `f1tenth_gym/envs/reset/__init__.py` (lines 8-43) - Reset function factory
- `maps/` directory - All available track definitions

---

## Implementation Tasks (Step-by-Step Validation)

### [X] Task 1: Add Configuration Support
**Files**: `train/config/gym_config.yaml`, `train/config/env_config.py`

**Changes**:
- Add `track_pool` list parameter to `gym_config.yaml`
- Load and export `TRACK_POOL` constant in `env_config.py`

**Validation**:
```python
from train.config.env_config import TRACK_POOL
print(TRACK_POOL)  # Should print ["Drift", "Drift2", "Drift_large"] or None
```

---

### [X] Task 2: Add Global Bounds Computation Helper
**Files**: `train/training_utils.py`

**Changes**:
- Add `compute_global_track_bounds(track_pool, track_scale)` function

**Validation**:
```python
from train.training_utils import compute_global_track_bounds
bounds = compute_global_track_bounds(["Drift", "Drift2"], track_scale=1.0)
print(bounds)  # Should show track_max_curv, track_min_width, track_max_width
```

---

### [X] Task 3: Add Global Bounds Usage in Environment
**Files**: `f1tenth_gym/envs/utils.py`, `f1tenth_gym/envs/f110_env.py`

**Changes**:
- Add `_get_track_width_bounds()` and `_get_track_curvature_bounds()` helpers in `utils.py`
- Update `calculate_norm_bounds()` to use helpers
- Add bounds validation and logging in `f110_env.py`

**Validation**:
```python
import gymnasium as gym
config = {
    "map": "Drift",
    "track_max_curv": 1.5,
    "track_min_width": 2.0,
    "track_max_width": 4.0,
    # ... other required config ...
}
env = gym.make("f1tenth_gym:f1tenth-v0", config=config)
# Should log: "[Global bounds] curvature=+/-1.5000, width=[2.0000, 4.0000]"
```

---

### [ ] Task 4: Update `make_subprocvecenv()` for Multi-Map
**Files**: `train/training_utils.py`

**Changes**:
- Modify `make_subprocvecenv()` to accept `track_pool` parameter
- Return `(env, global_bounds)` tuple instead of just `env`
- Compute global bounds and distribute maps across envs

**Validation**:
```python
from train.training_utils import make_subprocvecenv
from train.config.env_config import get_drift_train_config

config = get_drift_train_config()
env, global_bounds = make_subprocvecenv(seed=42, config=config, n_envs=6, track_pool=["Drift", "Drift2"])
print(f"Global bounds: {global_bounds}")
# Should show track distribution: {"Drift": 3, "Drift2": 3}
env.close()
```

---

### [ ] Task 5: Update `make_eval_env()` and Training Script
**Files**: `train/training_utils.py`, `train/ppo_race.py`

**Changes**:
- Add `global_bounds` parameter to `make_eval_env()`
- Update `ppo_race.py` to capture bounds and pass to eval env

**Validation**:
- Run `python train/ppo_race.py` with `track_pool` configured
- Verify logs show multi-map distribution and eval env using global bounds
- Verify training starts without errors

---
### [ ] Task 6: Add documentation in `README.md`
- Document new feature config use

---

### Validation Order
1. **Task 1** → Config loads correctly
2. **Task 2** → Bounds computation works in isolation
3. **Task 3** → Single env respects injected global bounds
4. **Task 4** → Multi-env creation distributes maps correctly
5. **Task 5** → Full integration works end-to-end
6. **Task 6** → Documentation
