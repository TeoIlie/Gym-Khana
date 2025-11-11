# Action Normalization Implementation Plan

## Overview

This document describes the implementation plan for adding normalized action spaces to the F1TENTH Gym environment. The feature enables RL networks to output actions in the standardized range `[-1, 1]`, which are then automatically scaled to the appropriate physical values (steering angles in radians, acceleration in m/s²).

## Motivation

### Benefits for RL Training

1. **Scale Invariance**: Neural networks don't need to learn that steering is in `[-0.4189, 0.4189]` rad while acceleration is in `[-9.51, 9.51]` m/s²
2. **Balanced Learning**: Both action dimensions have equal importance initially, preventing one dimension from dominating gradients
3. **Better Initialization**: Tanh output naturally centers around 0, which is often safe for vehicles
4. **Transfer Learning**: Easier to transfer policies between different vehicle configurations with different physical limits
5. **Numerical Stability**: Gradients are more stable when actions are in similar ranges
6. **Standard Practice**: Aligns with common RL implementation patterns used in research papers

### Physical Interpretation

When `normalize_act=True`:
- Action `[0.0, 0.0]` → No steering, no acceleration (neutral/safe state)
- Action `[1.0, 0.0]` → Maximum right steering, no acceleration
- Action `[-1.0, 0.0]` → Maximum left steering, no acceleration
- Action `[0.0, 1.0]` → No steering, maximum acceleration
- Action `[0.0, -1.0]` → No steering, maximum braking (deceleration)

## Scope

### Supported Action Types (Phase 1)

This initial implementation focuses on the most commonly used action types for RL-based autonomous racing:

- ✅ **SteeringAngleAction**: Direct steering angle control
- ✅ **AcclAction**: Direct acceleration control

### Not Included (Future Work)

- ❌ **SpeedAction**: Speed-based longitudinal control (with internal P controller)
- ❌ **SteeringSpeedAction**: Steering velocity control

**Rationale**: The excluded action types are less commonly used for RL training. Direct control (steering angle + acceleration) provides the lowest-level interface suitable for learning. Support for these can be added in future iterations if needed.

## Architecture

### Current Action System Structure

```
CarAction (main composite action class)
├── LongitudinalAction (abstract base)
│   ├── AcclAction ← MODIFY
│   └── SpeedAction (future work)
└── SteerAction (abstract base)
    ├── SteeringAngleAction ← MODIFY
    └── SteeringSpeedAction (future work)
```

### Action Processing Flow

```
RL Policy Network
    ↓
Outputs normalized action: [steering, accl] ∈ [-1, 1]²
    ↓
CarAction.act(action, state, params)
    ↓
    ├─ SteeringAngleAction.act(action[0], ...)
    │    └─ Scale: action[0] * scale_factor → steering_angle [rad]
    │         └─ bang_bang_steer(steering_angle, ...) → steering_velocity
    │
    └─ AcclAction.act(action[1], ...)
         └─ Scale: action[1] * scale_factor → acceleration [m/s²]
    ↓
Returns (acceleration, steering_velocity)
    ↓
Vehicle dynamics integration (RK4, Euler, etc.)
```

## Implementation Details

### File Modifications

#### 1. `f1tenth_gym/envs/f110_env.py`

**A) Add to `default_config()` method (~line 654):**

```python
"normalize_act": None,  # Normalize actions to [-1, 1] range for RL
```

**B) Add initialization logic in `__init__()` method (after line 237):**

Use the same initialization logic as for `normalize_obs`

**C) Modify action type initialization (~line 103):**

```python
self.action_type = CarAction(
    self.config["control_input"],
    params=self.params,
    normalize=self.normalize_act  # ← NEW PARAMETER
)
```

**D) Update `configure()` method (~line 666):**

```python
def configure(self, config: dict) -> None:
    if config:
        self.config = deep_update(self.config, config)
        self.params = self.config["params"]

        if hasattr(self, "sim"):
            self.sim.update_params(self.config["params"])

        if hasattr(self, "action_space"):
            # if some parameters changed, recompute action space
            self.action_type = CarAction(
                self.config["control_input"],
                params=self.params,
                normalize=self.config.get("normalize_act", False)  # ← ADD
            )
            self.action_space = from_single_to_multi_action_space(
                self.action_type.space, self.num_agents
            )
```

#### 2. `f1tenth_gym/envs/action.py`

**A) Modify `LongitudinalAction` base class (~line 26):**

```python
class LongitudinalAction:
    def __init__(self, normalize: bool = False) -> None:
        self._type = None
        self.normalize = normalize  # ← NEW
        self.lower_limit = None
        self.upper_limit = None
        self.scale_factor = 1.0  # ← NEW
```

**B) Modify `AcclAction` class (~line 46):**

```python
class AcclAction(LongitudinalAction):
    def __init__(self, params: Dict, normalize: bool = False) -> None:
        super().__init__(normalize=normalize)
        self._type = "accl"

        if normalize:
            # When normalized: action space is [-1, 1]
            self.lower_limit = -1.0
            self.upper_limit = 1.0
            # Scale factor: maximum acceleration magnitude
            self.scale_factor = params["a_max"]
        else:
            # Original behavior: action space is [-a_max, a_max]
            self.lower_limit = -params["a_max"]
            self.upper_limit = params["a_max"]
            self.scale_factor = 1.0

    def act(self, action: float, state, params) -> float:
        # Apply scaling when normalized
        # When normalize=True: maps [-1, 1] → [-a_max, a_max]
        # When normalize=False: pass-through (scale_factor=1.0)
        return action * self.scale_factor
```

**C) Modify `SteerAction` base class (~line 74):**

```python
class SteerAction:
    def __init__(self, normalize: bool = False) -> None:
        self._type = None
        self.normalize = normalize  # ← NEW
        self.lower_limit = None
        self.upper_limit = None
        self.scale_factor = 1.0  # ← NEW
```

**D) Modify `SteeringAngleAction` class (~line 94):**

```python
class SteeringAngleAction(SteerAction):
    def __init__(self, params: Dict, normalize: bool = False) -> None:
        super().__init__(normalize=normalize)
        self._type = "steering_angle"

        if normalize:
            # When normalized: action space is [-1, 1]
            self.lower_limit = -1.0
            self.upper_limit = 1.0
            # Scale factor: use maximum absolute steering angle
            # Assumes symmetric limits: s_min = -s_max
            self.scale_factor = max(abs(params["s_min"]), abs(params["s_max"]))
        else:
            # Original behavior: action space is [s_min, s_max]
            self.lower_limit = params["s_min"]
            self.upper_limit = params["s_max"]
            self.scale_factor = 1.0

    def act(self, action: float, state: np.ndarray, params: Dict) -> float:
        # Scale normalized action to actual steering angle
        # When normalize=True: maps [-1, 1] → [s_min, s_max]
        # When normalize=False: pass-through (scale_factor=1.0)
        desired_angle = action * self.scale_factor

        # Apply existing bang-bang controller to convert angle → velocity
        sv = bang_bang_steer(
            desired_angle,
            state[2],
            params["sv_max"],
        )
        return sv
```

**E) Modify `CarAction` class constructor (~line 134):**

```python
class CarAction:
    def __init__(
        self,
        control_mode: list[str, str],
        params: Dict,
        normalize: bool = False  # ← NEW PARAMETER
    ) -> None:
        # ... [existing control_mode parsing logic remains unchanged] ...

        # Pass normalize parameter to action instances
        self._longitudinal_action: LongitudinalAction = long_act_type_fn(
            params, normalize=normalize
        )
        self._steer_action: SteerAction = steer_act_type_fn(
            params, normalize=normalize
        )
        self.normalize = normalize  # Store for reference
```

**Note**: The `act()` method in `CarAction` remains unchanged - it already correctly delegates to the sub-actions.

### Scaling Logic

#### SteeringAngleAction Scaling

```python
# Parameters (typical F1TENTH values)
s_min = -0.4189  # rad (~-24 degrees)
s_max =  0.4189  # rad (~+24 degrees)
scale_factor = max(abs(s_min), abs(s_max)) = 0.4189  # rad

# Transformation
normalized_action ∈ [-1, 1]
actual_steering_angle = normalized_action * scale_factor
                     = normalized_action * 0.4189  [rad]

# Examples
normalized = -1.0 → actual = -0.4189 rad (full left)
normalized =  0.0 → actual =  0.0000 rad (straight)
normalized = +1.0 → actual = +0.4189 rad (full right)
```

#### AcclAction Scaling

```python
# Parameters (typical F1TENTH values)
a_max = 9.51  # m/s²
scale_factor = a_max = 9.51  # m/s²

# Transformation
normalized_action ∈ [-1, 1]
actual_acceleration = normalized_action * scale_factor
                   = normalized_action * 9.51  [m/s²]

# Examples
normalized = -1.0 → actual = -9.51 m/s² (full braking)
normalized =  0.0 → actual =  0.00 m/s² (coasting)
normalized = +1.0 → actual = +9.51 m/s² (full acceleration)
```

### Action Space Definition

When `normalize_act=True`, the Gymnasium action space becomes:

```python
# Single agent
gym.spaces.Box(
    low=[-1.0, -1.0],
    high=[1.0, 1.0],
    shape=(2,),
    dtype=np.float32
)

# Multi-agent (e.g., 2 agents)
gym.spaces.Box(
    low=[[-1.0, -1.0],
         [-1.0, -1.0]],
    high=[[1.0, 1.0],
          [1.0, 1.0]],
    shape=(2, 2),
    dtype=np.float32
)
```

When `normalize_act=False` (default), the original behavior is preserved:

```python
# Single agent
gym.spaces.Box(
    low=[s_min, -a_max],     # [-0.4189, -9.51]
    high=[s_max, a_max],     # [0.4189, 9.51]
    shape=(2,),
    dtype=np.float32
)
```

## Validation Criteria

### Implementation Checklist

| Status | File | Location | Task | Notes |
|---|------|----------|------|--------|
| [ ] | `f1tenth_gym/envs/f110_env.py` | ~line 654 | Add `"normalize_act": None` to `default_config()` |
| [ ] | `f1tenth_gym/envs/f110_env.py` | ~line 211-237 | Add initialization logic for `normalize_act` (similar to `normalize_obs`) |
| [ ] | `f1tenth_gym/envs/f110_env.py` | ~line 103 | Pass `normalize=self.normalize_act` to `CarAction` constructor |
| [ ] | `f1tenth_gym/envs/f110_env.py` | ~line 666 | Update `configure()` method to handle `normalize_act` parameter |
| [ ] | `f1tenth_gym/envs/action.py` | ~line 26-32 | Add `normalize` and `scale_factor` attributes to `LongitudinalAction` base class |
| [ ] | `f1tenth_gym/envs/action.py` | ~line 46-53 | Modify `AcclAction.__init__()` to accept `normalize` and set scale factor |
| [ ] | `f1tenth_gym/envs/action.py` | ~line 52-53 | Modify `AcclAction.act()` to apply scaling: `return action * self.scale_factor` |
| [ ] | `f1tenth_gym/envs/action.py` | ~line 74-80 | Add `normalize` and `scale_factor` attributes to `SteerAction` base class |
| [ ] | `f1tenth_gym/envs/action.py` | ~line 94-98 | Modify `SteeringAngleAction.__init__()` to accept `normalize` and set scale factor |
| [ ] | `f1tenth_gym/envs/action.py` | ~line 100-106 | Modify `SteeringAngleAction.act()` to apply scaling before `bang_bang_steer()` |
| [ ] | `f1tenth_gym/envs/action.py` | ~line 134 | Add `normalize: bool = False` parameter to `CarAction.__init__()` |
| [ ] | `f1tenth_gym/envs/action.py` | ~line 174-175 | Pass `normalize` parameter when creating action instances in `CarAction` |
| [ ] | `f1tenth_gym/envs/action.py` | N/A | Add validation: raise error if `normalize=True` with `SpeedAction` or `SteeringSpeedAction` |
| 14 | `tests/test_normalized_actions.py` | New file | Create comprehensive unit tests for normalized actions |
| 15 | All files | N/A | Verify backward compatibility: all existing tests pass with `normalize_act=False` (default) |

### Validation Criteria

After implementation, verify:
- ✅ Action space is `Box([-1, 1], [-1, 1])` when `normalize_act=True`
- ✅ Actions are correctly scaled using vehicle parameters (`s_min/s_max` for steering, `a_max` for acceleration)
- ✅ Backward compatibility: `normalize_act=False` (or `None`) preserves original behavior
- ✅ Multi-agent environments work correctly with normalized actions
- ✅ Appropriate error messages when using unsupported action types (SpeedAction, SteeringSpeedAction)
- ✅ All existing tests continue to pass
