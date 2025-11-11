# Action Normalization Implementation Plan

## Overview

This document describes the implementation plan for adding normalized action spaces to the F1TENTH Gym environment. The feature enables RL networks to output actions in the standardized range `[-1, 1]`, which are then automatically scaled to the appropriate physical values based on the selected action type (e.g., steering angles in radians, acceleration in m/s², speed in m/s, steering velocity in rad/s).

**Default Behavior**: Action normalization is **enabled by default** (`normalize_act=True`). Users can explicitly set `normalize_act=False` in the environment configuration to use physical units directly.

## Motivation

### Benefits for RL Training

When `normalize_act=True` (default), normalized actions in `[-1, 1]` are mapped to physical values.

1. **Scale Invariance**: Neural networks don't need to learn that steering is in `[-0.4189, 0.4189]` rad while acceleration is in `[-9.51, 9.51]` m/s²
2. **Balanced Learning**: Both action dimensions have equal importance initially, preventing one dimension from dominating gradients
3. **Better Initialization**: Tanh output naturally centers around 0, which is often safe for vehicles
4. **Transfer Learning**: Easier to transfer policies between different vehicle configurations with different physical limits
5. **Numerical Stability**: Gradients are more stable when actions are in similar ranges
6. **Standard Practice**: Aligns with common RL implementation patterns used in research papers

## Scope

### Supported Action Types

This implementation provides normalized action spaces for **all** F1TENTH Gym action types:

- ✅ **SteeringAngleAction**: Direct steering angle control
- ✅ **AcclAction**: Direct acceleration control
- ✅ **SpeedAction**: Speed-based longitudinal control (with internal P controller)
- ✅ **SteeringSpeedAction**: Steering velocity control

## Architecture

### Current Action System Structure

```
CarAction (main composite action class)
├── LongitudinalAction (abstract base) ← MODIFY
│   ├── AcclAction ← MODIFY
│   └── SpeedAction ← MODIFY
└── SteerAction (abstract base) ← MODIFY
    ├── SteeringAngleAction ← MODIFY
    └── SteeringSpeedAction ← MODIFY
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
"normalize_act": True,  # Normalize actions to [-1, 1] range for RL (default: True)
```

**B) Initialization logic in `__init__()` method (after line 237):**

```python
# Action normalization - always taken from config (default is True in default_config)
self.normalize_act = self.config["normalize_act"]
```

**Note**: Unlike `normalize_obs` which has complex default logic based on observation type, action normalization always uses the value from config. The default is set to `True` in `default_config()`, and users can override it to `False` if they want raw physical units.

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
            # Update normalize_act from config (no default - must be in config)
            self.normalize_act = self.config["normalize_act"]

            self.action_type = CarAction(
                self.config["control_input"],
                params=self.params,
                normalize=self.normalize_act  # ← ADD
            )
            self.action_space = from_single_to_multi_action_space(
                self.action_type.space, self.num_agents
            )
```

#### 2. `f1tenth_gym/envs/action.py`

**A) Modify `LongitudinalAction` base class (~line 26):**

```python
class LongitudinalAction:
    def __init__(self, normalize: bool) -> None:
        self._type = None
        self.normalize = normalize  # ← NEW
        self.lower_limit = None
        self.upper_limit = None
        self.scale_factor = 1.0  # ← NEW
```

**B) Modify `AcclAction` class (~line 46):**

```python
class AcclAction(LongitudinalAction):
    def __init__(self, params: Dict, normalize: bool) -> None:
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

**C) Modify `SpeedAction` class (~line 58):**

```python
class SpeedAction(LongitudinalAction):
    def __init__(self, params: Dict, normalize: bool) -> None:
        super().__init__(normalize=normalize)
        self._type = "speed"

        if normalize:
            # When normalized: action space is [-1, 1]
            self.lower_limit = -1.0
            self.upper_limit = 1.0
            # Scale factor: compute from v_min and v_max
            # Center point: (v_max + v_min) / 2
            # Range: (v_max - v_min) / 2
            # Mapping: normalized * range + center
            self.v_center = (params["v_max"] + params["v_min"]) / 2.0
            self.v_range = (params["v_max"] - params["v_min"]) / 2.0
        else:
            # Original behavior: action space is [v_min, v_max]
            self.lower_limit = params["v_min"]
            self.upper_limit = params["v_max"]
            self.v_center = 0.0
            self.v_range = 1.0

    def act(self, action: float, state, params) -> float:
        # Scale normalized action to actual speed
        # When normalize=True: maps [-1, 1] → [v_min, v_max]
        # When normalize=False: pass-through
        if self.normalize:
            desired_speed = action * self.v_range + self.v_center
        else:
            desired_speed = action

        # Apply existing P controller logic
        # (rest of the original act() method remains unchanged)
        current_speed = state[3]
        error = desired_speed - current_speed
        accl = params["kp"] * error
        # Clip acceleration
        accl = np.clip(accl, -params["a_max"], params["a_max"])
        return accl
```

**D) Modify `SteerAction` base class (~line 74):**

```python
class SteerAction:
    def __init__(self, normalize: bool) -> None:
        self._type = None
        self.normalize = normalize  # ← NEW
        self.lower_limit = None
        self.upper_limit = None
        self.scale_factor = 1.0  # ← NEW
```

**E) Modify `SteeringAngleAction` class (~line 94):**

```python
class SteeringAngleAction(SteerAction):
    def __init__(self, params: Dict, normalize: bool) -> None:
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

**F) Modify `SteeringSpeedAction` class (~line 111):**

```python
class SteeringSpeedAction(SteerAction):
    def __init__(self, params: Dict, normalize: bool) -> None:
        super().__init__(normalize=normalize)
        self._type = "steering_speed"

        if normalize:
            # When normalized: action space is [-1, 1]
            self.lower_limit = -1.0
            self.upper_limit = 1.0
            # Scale factor: compute from sv_min and sv_max
            # Center point: (sv_max + sv_min) / 2
            # Range: (sv_max - sv_min) / 2
            # Mapping: normalized * range + center
            self.sv_center = (params["sv_max"] + params["sv_min"]) / 2.0
            self.sv_range = (params["sv_max"] - params["sv_min"]) / 2.0
        else:
            # Original behavior: action space is [sv_min, sv_max]
            self.lower_limit = params["sv_min"]
            self.upper_limit = params["sv_max"]
            self.sv_center = 0.0
            self.sv_range = 1.0

    def act(self, action: float, state: np.ndarray, params: Dict) -> float:
        # Scale normalized action to actual steering velocity
        # When normalize=True: maps [-1, 1] → [sv_min, sv_max]
        # When normalize=False: pass-through
        if self.normalize:
            steering_velocity = action * self.sv_range + self.sv_center
        else:
            steering_velocity = action

        return steering_velocity
```

**G) Modify `CarAction` class constructor (~line 134):**

```python
class CarAction:
    def __init__(
        self,
        control_mode: list[str, str],
        params: Dict,
        normalize: bool  # ← NEW PARAMETER
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

### Action Space Definition

When `normalize_act=True` (default), the Gymnasium action space becomes:

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

When `normalize_act=False` (user override), the original behavior with physical units is preserved:

```python
# Single agent
gym.spaces.Box(
    low=[s_min, -a_max],     # [-0.4189, -9.51]
    high=[s_max, a_max],     # [0.4189, 9.51]
    shape=(2,),
    dtype=np.float32
)
```

## Implementation Checklist

#### Environment Configuration (`f1tenth_gym/envs/f110_env.py`)

| Status | Location | Task | Notes |
|---|----------|------|--------|
| [X] | ~line 654 | Add `"normalize_act": True` to `default_config()` | Default is True for normalized actions |
| [X] | ~line 237 | Add initialization logic: `self.normalize_act = self.config["normalize_act"]` | Get from config (no default - config must have it) |
| [X] | ~line 103 | Pass `normalize=self.normalize_act` to `CarAction` constructor | |
| [X] | ~line 666 | Update `configure()` method to use `self.normalize_act = self.config["normalize_act"]` (no default) | |

#### Base Action Classes (`f1tenth_gym/envs/action.py`)

| Status | Location | Task | Notes |
|---|----------|------|--------|
| [X] | ~line 26-32 | Add `normalize` and `scale_factor` attributes to `LongitudinalAction` base class | |
| [X] | ~line 74-80 | Add `normalize` and `scale_factor` attributes to `SteerAction` base class | |

#### AcclAction (`f1tenth_gym/envs/action.py`)

| Status | Location | Task | Notes |
|---|----------|------|--------|
| [X] | ~line 46-53 | Modify `AcclAction.__init__()` to accept `normalize` parameter | Set limits and scale_factor based on `a_max` |
| [X] | ~line 52-53 | Modify `AcclAction.act()` to apply scaling: `return action * self.scale_factor` | |

#### SpeedAction (`f1tenth_gym/envs/action.py`)

| Status | Location | Task | Notes |
|---|----------|------|--------|
| [X] | ~line 58-72 | Modify `SpeedAction.__init__()` to accept `normalize` parameter | Compute v_center and v_range from v_min/v_max |
| [X] | ~line 65-70 | Modify `SpeedAction.act()` to apply scaling before P controller | Map [-1, 1] to [v_min, v_max] |

#### SteeringAngleAction (`f1tenth_gym/envs/action.py`)

| Status | Location | Task | Notes |
|---|----------|------|--------|
| [X] | ~line 94-98 | Modify `SteeringAngleAction.__init__()` to accept `normalize` parameter | Set scale_factor from s_min/s_max |
| [X] | ~line 100-106 | Modify `SteeringAngleAction.act()` to apply scaling before `bang_bang_steer()` | |

#### SteeringSpeedAction (`f1tenth_gym/envs/action.py`)

| Status | Location | Task | Notes |
|---|----------|------|--------|
| [X] | ~line 111-125 | Modify `SteeringSpeedAction.__init__()` to accept `normalize` parameter | Compute sv_center and sv_range from sv_min/sv_max |
| [X] | ~line 118-120 | Modify `SteeringSpeedAction.act()` to apply scaling | Map [-1, 1] to [sv_min, sv_max] |

#### CarAction Integration (`f1tenth_gym/envs/action.py`)

| Status | Location | Task | Notes |
|---|----------|------|--------|
| [X] | ~line 134 | Add `normalize: bool` parameter to `CarAction.__init__()` | |
| [X] | ~line 174-175 | Pass `normalize` parameter when creating action instances in `CarAction` | |
| [X] | ~line 176 | Store `self.normalize` for reference | |

#### Testing

| Status | File | Task | Notes |
|---|------|------|--------|
| [ ] | `tests/test_normalized_actions.py` | Create comprehensive unit tests for all action types with normalization | Test AcclAction, SpeedAction, SteeringAngleAction, SteeringSpeedAction |
| [ ] | All test files | Verify backward compatibility with `normalize_act=False` | All existing tests should pass |
| [ ] | `tests/test_normalized_actions.py` | Test default behavior with `normalize_act=True` | Verify new default works correctly |

### Validation Criteria

After implementation, verify:

#### Functional Requirements
- ✅ Action space is `Box([-1, 1], [-1, 1])` when `normalize_act=True` (default)
- ✅ Action space uses physical units when `normalize_act=False` (user override)
- ✅ **AcclAction**: Correctly scales using `a_max` parameter (symmetric: [-a_max, a_max])
- ✅ **SpeedAction**: Correctly scales using `v_min` and `v_max` parameters (asymmetric mapping)
- ✅ **SteeringAngleAction**: Correctly scales using `s_min` and `s_max` parameters (symmetric)
- ✅ **SteeringSpeedAction**: Correctly scales using `sv_min` and `sv_max` parameters (symmetric)
- ✅ Multi-agent environments work correctly with normalized actions

