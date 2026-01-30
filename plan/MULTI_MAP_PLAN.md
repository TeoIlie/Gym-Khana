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

### Approach 1: Randomize Track at Each Reset ❌
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

### Approach 2: Different Maps Per Parallel Environment ✅
**Description**: Each of n_envs parallel environments trains on a different map. PPO averages gradients across all environments.

**Pros**:
- **Zero runtime overhead**: Maps loaded once at initialization
- **Normalization solved**: Each env has correct track-specific bounds
- **Natural averaging**: PPO already averages gradients across parallel envs
- **Simple implementation**: ~20 lines of code changes
- **Proven pattern**: Similar to domain randomization in robotics
- **Every batch is diverse**: Each training batch contains experiences from multiple tracks

**Cons**:
- Limited by n_envs: With 32 envs and 3 tracks, each track gets ~10-11 environments
- All difficulties trained simultaneously (no curriculum)
- May slow initial learning if hard tracks dilute easy track signal

**Verdict**: **RECOMMENDED** - Simplest correct implementation with no downsides.

---

### Approach 3: Curriculum Learning 🔄
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

**Verdict**: Good for later iteration, but start simpler first.

---

### Approach 4: Hybrid (Map Pools + Randomization) 🔄
**Description**: Each parallel env gets subset of tracks and randomly switches between them.

**Pros**: Best of both worlds - parallel diversity + episode diversity
**Cons**: Most complex, combines all challenges of Approaches 1 and 2

**Verdict**: NOT RECOMMENDED initially - too complex.

---

## Recommended Implementation: Phased Approach

### Phase 1: Approach 2 (Different Maps Per Parallel Environment)
Start with simplest correct implementation to validate if multi-map training helps at all.

### Phase 2: Add Episode Randomization (If Needed)
If Phase 1 shows benefit but agent still overfits, add lazy switching (every N resets).

### Phase 3: Add Curriculum (If Manual Control Needed)
If certain tracks consistently fail, implement progression from easy to hard.

---

## Phase 1 Implementation Details

### Implementation Strategy
Modify environment creation to distribute a list of maps across parallel environments:
- Env 0, 3, 6, ... → Map 0
- Env 1, 4, 7, ... → Map 1
- Env 2, 5, 8, ... → Map 2

With 32 parallel envs and 3 drift tracks, each track gets ~10-11 environments.

### Code Changes Required

#### File 1: `train/training_utils.py`
**Location**: Lines 21-37 (function `make_subprocvecenv`)

**Change**: Add `track_pool` parameter and distribute maps across environments.

```python
def make_subprocvecenv(seed: int, config: dict, n_envs: int, track_pool: list[str] | None = None):
    """
    Create a SubprocVecEnv parallelized environment.

    Args:
        seed: Random seed
        config: Base gym configuration dict
        n_envs: Number of parallel environments
        track_pool: Optional list of maps to distribute across envs.
                   If provided, envs will cycle through these maps.
                   Example: ["Drift", "Drift2", "Drift_large"]
    """
    if track_pool is not None:
        print(f"🗺️  Distributing {len(track_pool)} tracks across {n_envs} environments...")
        env_fns = []
        for i in range(n_envs):
            # Cycle through track pool
            map_name = track_pool[i % len(track_pool)]
            # Create per-env config with specific map
            env_config = config.copy()
            env_config["map"] = map_name
            env_fns.append(make_env(seed=seed, rank=i, config=env_config))

        env = SubprocVecEnv(env_fns)

        # Print distribution summary
        from collections import Counter
        distribution = Counter(track_pool[i % len(track_pool)] for i in range(n_envs))
        print(f"✅ Map distribution: {dict(distribution)}")
    else:
        # Original single-map behavior
        env = SubprocVecEnv([make_env(seed=seed, rank=i, config=config) for i in range(n_envs)])
        print(f"✅ Successfully created {n_envs} parallel environments")

    return env
```

**Why**: This distributes different maps to different environments while keeping all other config identical.

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

**Change**: Extract and pass `track_pool` to environment creation.

```python
def train_ppo_race():
    TRAIN_CONFIG = get_drift_train_config()

    # Extract track_pool from config
    track_pool = TRAIN_CONFIG.get("track_pool", None)

    # Create vectorized environment with optional multi-map support
    env = make_subprocvecenv(SEED, TRAIN_CONFIG, N_ENVS, track_pool=track_pool)
    eval_env = make_eval_env(EVAL_SEED, TRAIN_CONFIG)

    # ... rest of training code unchanged ...
```

**Why**: Passes track pool information to environment factory.

---

### Summary of Changes
- **4 files modified**
- **~30 lines of code added** (including comments)
- **Zero changes to core gym environment** (`f110_env.py`)
- **Backward compatible**: Setting `track_pool: null` preserves original single-map behavior

---

## Testing & Verification

### Unit Testing
1. **Test track pool distribution**:
   ```python
   # Create env with track_pool
   env = make_subprocvecenv(seed=12345, config=base_config, n_envs=9,
                           track_pool=["Drift", "Drift2", "Drift_large"])

   # Verify each sub-env has correct map assigned
   for i in range(9):
       expected_map = ["Drift", "Drift2", "Drift_large"][i % 3]
       # Check via env.env_method or logging
   ```

2. **Test backward compatibility**:
   ```python
   # Create env without track_pool (original behavior)
   env = make_subprocvecenv(seed=12345, config=base_config, n_envs=9,
                           track_pool=None)
   # Should work exactly as before
   ```

### Integration Testing
1. **Run short training with multi-map**:
   ```bash
   # Set track_pool in gym_config.yaml
   python train/ppo_race.py
   # Verify logs show correct map distribution
   # Train for 100k steps, check no crashes
   ```

2. **Verify normalization**: Check that observations stay within [-1, 1] bounds across all tracks

3. **Check wandb logging**: Ensure metrics are logged correctly with multi-map setup

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

## Future Extensions (Not Implemented in Phase 1)

### Phase 2: Episode-Level Randomization
If agent still overfits to assigned track:
- Modify `f110_env.py:reset()` to switch maps every N resets
- Implement global normalization bounds across track pool
- Add track caching to reduce loading overhead

### Phase 3: Curriculum Learning
If manual control over progression needed:
- Implement `CurriculumCallback` in `training_utils.py`
- Define track difficulty metrics (length, max curvature, width variance)
- Schedule progression: easy → medium → hard tracks

### Phase 4: Adaptive Curriculum
If sophisticated scheduling needed:
- Track per-map success rates during training
- Dynamically adjust time spent on each track
- Focus on tracks where agent struggles most

---

## Critical Files Reference

### Files Modified (Phase 1)
- `train/training_utils.py` (lines 21-37) - Distribute track pool
- `train/config/gym_config.yaml` - Add track_pool parameter
- `train/config/env_config.py` (lines ~38, ~113) - Load and pass track_pool
- `train/ppo_race.py` (line ~73) - Extract and pass track_pool

### Files Referenced (Context)
- `f1tenth_gym/envs/f110_env.py` (line 189) - Track loading in `__init__`
- `f1tenth_gym/envs/track/track.py` (lines 138-205) - `Track.from_track_name()`
- `f1tenth_gym/envs/reset/__init__.py` (lines 8-43) - Reset function factory
- `maps/` directory - All available track definitions

---

## Decision Point

**Question for user**: Which approach would you like to implement?

**Recommendation**: Start with **Approach 2 (Phase 1)** because:
- Simplest correct implementation (~30 lines)
- Zero runtime overhead
- Normalization works correctly out of the box
- Easy to test if multi-map training helps
- Can always add randomization or curriculum later

If you'd like to start with a different approach or modify this plan, let me know before implementation begins.
