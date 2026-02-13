# Recovery Training Implementation Plan

## Context

The project currently supports training racing policies via `train/ppo_race.py`. The goal is to add a parallel capability to train recovery policies — agents that learn to regain control of a vehicle from perturbed states (high sideslip, yaw rate) on a straight track segment. This is described in `plan/RECOVERY_PLAN.md`.

## Critical Design Consideration: SB3 Auto-Reset

With Stable-Baselines3's `SubprocVecEnv`, environments auto-reset when `done=True` or `truncated=True`. We **cannot** pass custom `options` to these auto-resets from the training script. Therefore, the random initialization logic (sampling v, beta, r at S=96) **must live inside the environment's `reset()` method**, not in the training script. When `training_mode == "recover"` and no explicit options are provided, `reset()` will generate perturbed initial states internally.

---

## Step 1: Recovery Gym Config YAML

**Create** `train/config/gym_config_recover.yaml`

Contents (only recovery-specific values — shared defaults come from `env_config.py` constants):

```yaml
project_name: "f1tenth-ppo-recover"
map: "IMS"
track_pool: null              # single map for now
track_direction: "normal"     # CCW on IMS
reset_config: "cl_random_static"  # fallback, overridden by training_mode logic

# Recovery-specific
training_mode: "recover"
recovery_s_init: 96       # starting S on IMS straight
recovery_s_max: 140           # truncation arc-length
recovery_v_range: [2, 20]     # velocity perturbation [m/s]
recovery_beta_range: [-1.047, 1.047]   # [-pi/3, pi/3] rad
recovery_r_range: [-1.571, 1.571]      # [-pi/2, pi/2] rad/s
recovery_yaw_range: [-1.047, 1.047]    # [-pi/3, pi/3] rad (perturbation from track heading)

# Recovery reward gains
recovery_euclid_gain: 1.0     # K_e
recovery_timestep_penalty: 1.0  # K_c
recovery_success_reward: 100
recovery_collision_penalty: -50

# Recovery success thresholds
recovery_delta_thresh: 0.05
recovery_beta_thresh: 0.05
recovery_r_thresh: 0.1
recovery_d_beta_thresh: 0.1
recovery_d_r_thresh: 0.2
recovery_frenet_u_thresh: 0.05

# Override shared params for recovery
max_episode_steps: 2048
obs_type: "drift"
normalize_obs: true
normalize_act: true
```

## Step 2: Config Factory Functions

**Modify** `train/config/env_config.py`

Add a second YAML loader block for recovery config, and two new factory functions:

```python
# Load recovery gym config
_recover_config_path = os.path.join(os.path.dirname(__file__), "gym_config_recover.yaml")
with open(_recover_config_path, "r") as f:
    _recover_config = yaml.safe_load(f)
```

Extract recovery-specific constants (similar pattern to existing):
```python
RECOVER_PROJECT_NAME = _recover_config["project_name"]
RECOVER_MAP = _recover_config["map"]
# ... etc
```

Add two factory functions `get_recovery_train_config()` and `get_recovery_test_config()` that follow the same pattern as the existing drift functions but include recovery-specific keys. These use shared constants (`MODEL`, `TIMESTEP`, `INTEGRATOR`, `PARAMS`, `ACTION_INPUT`, `NUM_BEAMS`) from the existing config to avoid duplication.

## Step 3: Environment Modifications

**Modify** `f1tenth_gym/envs/f110_env.py`

### 3a. Add recovery config to `default_config()` (line ~670)

Add these keys with safe defaults that don't affect existing "race" behavior:

```python
"training_mode": "race",           # "race" or "recover"
"recovery_s_init": 96,
"recovery_s_max": 140,
"recovery_v_range": [2, 20],
"recovery_beta_range": [-1.047, 1.047],
"recovery_r_range": [-1.571, 1.571],
"recovery_yaw_range": [-1.047, 1.047],
"recovery_euclid_gain": 1.0,
"recovery_timestep_penalty": 1.0,
"recovery_success_reward": 100,
"recovery_collision_penalty": -50,
"recovery_delta_thresh": 0.05,
"recovery_beta_thresh": 0.05,
"recovery_r_thresh": 0.1,
"recovery_d_beta_thresh": 0.1,
"recovery_d_r_thresh": 0.2,
"recovery_frenet_u_thresh": 0.05,
```

### 3b. Extract config in `__init__()` (after line ~130)

```python
self.training_mode = self.config["training_mode"]
if self.training_mode == "recover":
    self.recovery_s_init = self.config["recovery_s_init"]
    self.recovery_s_max = self.config["recovery_s_max"]
    self.recovery_v_range = self.config["recovery_v_range"]
    self.recovery_beta_range = self.config["recovery_beta_range"]
    self.recovery_r_range = self.config["recovery_r_range"]
    self.recovery_yaw_range = self.config["recovery_yaw_range"]
    self.recovery_euclid_gain = self.config["recovery_euclid_gain"]
    self.recovery_timestep_penalty = self.config["recovery_timestep_penalty"]
    self.recovery_success_reward = self.config["recovery_success_reward"]
    self.recovery_collision_penalty = self.config["recovery_collision_penalty"]
    self.recovery_delta_thresh = self.config["recovery_delta_thresh"]
    self.recovery_beta_thresh = self.config["recovery_beta_thresh"]
    self.recovery_r_thresh = self.config["recovery_r_thresh"]
    self.recovery_d_beta_thresh = self.config["recovery_d_beta_thresh"]
    self.recovery_d_r_thresh = self.config["recovery_d_r_thresh"]
    self.recovery_frenet_u_thresh = self.config["recovery_frenet_u_thresh"]
    # Derivative tracking
    self.prev_beta = 0.0
    self.prev_r = 0.0
```

Also remove or gate the assertion `assert self.progress_gain >= 1.0` since recovery mode may not use progress_gain.

### 3c. Modify `reset()` (line ~1074)

**How this interacts with `reset_fn`:** The existing `reset()` flow is:

1. Parse `options` for explicit `poses`/`states` (lines 1117-1144)
2. If neither provided: `poses = self.reset_fn.sample()` (line 1148)
3. Call `self.sim.reset(poses, states=states)` (line 1172)

The `reset_fn` (e.g., `AllTrackResetFn`) is still created in `__init__()` at line 287 via `make_reset_fn()`. In recovery mode it exists but is **never called** — the new recovery block (inserted between steps 1 and 2) always sets `states` and `poses` before the fallback is reached. `make_reset_fn` is unchanged, no new reset class is needed.

For SB3 auto-resets: when an episode ends, SB3 calls `env.reset()` with no arguments. The recovery block fires, generates a new random perturbed state, and the episode starts fresh. For manual evaluation, you can still pass `options={"states": ...}` to test specific initial conditions — the recovery block only fires when `states is None and poses is None`.

**Insert after line ~1146** (after options parsing, before the `if poses is None` fallback):

```python
# Recovery mode: generate perturbed initial state if no options given
if self.training_mode == "recover" and states is None and poses is None:
    v = np.random.uniform(*self.recovery_v_range)
    beta = np.random.uniform(*self.recovery_beta_range)
    r = np.random.uniform(*self.recovery_r_range)
    yaw_perturbation = np.random.uniform(*self.recovery_yaw_range)
    x, y, base_yaw = self.track.frenet_to_cartesian(
        self.recovery_s_init, ey=0, ephi=0
    )
    yaw = base_yaw + yaw_perturbation
    states = np.array([[x, y, 0.0, v, yaw, r, beta]])
    poses = np.column_stack([states[:, 0], states[:, 1], states[:, 4]])
```

At end of `reset()` (before the return), initialize derivative tracking:

```python
if self.training_mode == "recover":
    agent = self.sim.agents[0]
    self.prev_beta = agent.standard_state["slip"]
    self.prev_r = agent.standard_state["yaw_rate"]
```

### 3d. Add `_check_recovery_success()` method

New method that checks all 6 recovery conditions:

```python
def _check_recovery_success(self) -> bool:
    agent = self.sim.agents[self.ego_idx]
    std_state = agent.standard_state

    # Current state values
    delta = std_state["delta"]
    beta = std_state["slip"]
    r = std_state["yaw_rate"]

    # Derivatives via finite difference
    d_beta = (beta - self.prev_beta) / self.timestep
    d_r = (r - self.prev_r) / self.timestep

    # Frenet heading error
    x, y, theta = std_state["x"], std_state["y"], std_state["yaw"]
    _, _, frenet_u = self.track.cartesian_to_frenet(x, y, theta, use_raceline=False)

    # Check all 6 conditions
    return (
        abs(delta) < self.recovery_delta_thresh
        and abs(beta) < self.recovery_beta_thresh
        and abs(r) < self.recovery_r_thresh
        and abs(d_beta) < self.recovery_d_beta_thresh
        and abs(d_r) < self.recovery_d_r_thresh
        and abs(frenet_u) < self.recovery_frenet_u_thresh
    )
```

### 3e. Modify `_check_done()` (line ~737)

Add recovery mode branch:

```python
if self.training_mode == "recover":
    # Terminated: crash (boundary exceeded) or successful recovery
    self.recovery_succeeded = self._check_recovery_success()
    terminated = self.boundary_exceeded[self.ego_idx] or self.recovery_succeeded

    # Truncated: arc-length exceeded OR timestep limit
    current_s, _ = self.track.centerline.spline.calc_arclength_inaccurate(
        self.poses_x[self.ego_idx], self.poses_y[self.ego_idx]
    )
    truncated = (
        current_s > self.recovery_s_max
        or self.current_step > self.max_episode_steps
    )
    return bool(terminated), bool(truncated), self.toggle_list >= 4
```

The existing race logic goes into an `else` branch.

### 3f. Add `_get_recovery_reward()` and modify `_get_reward()`

New method for recovery reward:

```python
def _get_recovery_reward(self) -> float:
    agent = self.sim.agents[self.ego_idx]
    std_state = agent.standard_state
    beta = std_state["slip"]
    r = std_state["yaw_rate"]

    # R_Euclid: dense signal toward (beta=0, r=0)
    r_euclid = -self.recovery_euclid_gain * np.sqrt(beta**2 + r**2)

    # R_col: collision penalty
    r_col = self.recovery_collision_penalty if self.boundary_exceeded[self.ego_idx] else 0.0

    # R_rec: recovery success reward
    r_rec = self.recovery_success_reward if self.recovery_succeeded else 0.0

    # R_const: constant timestep penalty
    r_const = self.recovery_timestep_penalty * self.timestep

    reward = r_euclid + r_col + r_rec - r_const

    # Update derivative tracking
    self.prev_beta = beta
    self.prev_r = r

    return reward
```

Modify `_get_reward()` to dispatch:

```python
def _get_reward(self):
    if self.training_mode == "recover":
        return self._get_recovery_reward()
    # ... existing race reward logic unchanged ...
```

## Step 4: Training Script

**Create** `train/ppo_recover.py`

Structure mirrors `train/ppo_race.py` exactly. Key differences:
- Imports `get_recovery_train_config`, `get_recovery_test_config` instead of drift variants
- Uses `RECOVER_PROJECT_NAME` for wandb
- `MODEL_PREFIX = "ppo_recover"`
- All 4 modes (train, eval, download, continue) work identically
- No changes to `training_utils.py` needed — all existing utilities are reused

## Files Modified/Created

| File | Action |
|------|--------|
| `train/config/gym_config_recover.yaml` | **Create** |
| `train/config/env_config.py` | **Modify** — add recovery config loader + factory functions |
| `f1tenth_gym/envs/f110_env.py` | **Modify** — add training_mode, recovery reward/done/reset logic |
| `train/ppo_recover.py` | **Create** — recovery training script |

## Existing Code Reused

| What | Where |
|------|-------|
| State init via `reset(options={"states": ...})` | `f110_env.py:1125-1141` |
| `frenet_to_cartesian(s, ey, ephi)` | `track.py:356-380` |
| `cartesian_to_frenet(x, y, theta)` | `track.py:382-459` |
| `agent.standard_state` (beta, r, delta, v_x) | `base_classes.py:421-428` |
| Frenet boundary checking | `f110_env.py:903-965` |
| `make_subprocvecenv`, `make_eval_env`, callbacks | `training_utils.py` |
| `linear_schedule`, `save_config`, `extract_rl_config` | `training_utils.py` |
| Init state pattern from `beta_r_traj_IMS_plot.py:40-75` | `examples/analysis/` |

## Implementation Tasks (Sequential)

### [X] Task 1: Add `training_mode` config key to the environment

**What:** Add `"training_mode": "race"` to `default_config()` in `f110_env.py`. Extract it in `__init__()`. Gate the `assert self.progress_gain >= 1.0` so it only applies when `training_mode == "race"`.

**Files:** `f1tenth_gym/envs/f110_env.py`

**Validate:** `python3 -m pytest` passes. Create a quick env with `training_mode: "race"` (default) and confirm it works identically to before.

---

### [X] Task 2: Add recovery config keys to `default_config()`

**What:** Add all recovery-specific keys to `default_config()` with safe defaults (see Step 3a in plan above). Extract them in `__init__()` behind `if self.training_mode == "recover"` guard. Initialize derivative tracking state (`prev_beta`, `prev_r`).

**Files:** `f1tenth_gym/envs/f110_env.py`

**Validate:** `python3 -m pytest` passes. Create env with `training_mode: "recover"` and default recovery keys — no crash on construction.

---

### [ ] Task 3: Implement recovery initialization in `reset()`

**What:** In `reset()`, add the recovery initialization block between options parsing and the `reset_fn.sample()` fallback. When `training_mode == "recover"` and no explicit options given: sample random `v`, `beta`, `r`, and `yaw_perturbation` from uniform distributions, compute Cartesian position and base heading at `recovery_s_init` via `frenet_to_cartesian`, add yaw perturbation to base heading, build full state array, set `states` and `poses`. At end of `reset()`, initialize `prev_beta` and `prev_r` from the actual agent state.

**Files:** `f1tenth_gym/envs/f110_env.py`

**Validate:** Create recovery env, call `reset()` with no options multiple times. Print `agent.standard_state` after each reset — confirm `slip`, `yaw_rate`, `v_x`, and `yaw` values vary within configured ranges. Confirm position is at S=96 on IMS. Confirm yaw is base track heading ± perturbation. Confirm passing explicit `options={"states": ...}` still overrides the random perturbation.

---

### [ ] Task 4: Implement recovery success check

**What:** Add `_check_recovery_success()` method that checks all 6 conditions: `|delta|`, `|beta|`, `|r|`, `|d_beta|`, `|d_r|`, `|frenet_u|` all below their respective thresholds. Computes derivatives via finite difference using `prev_beta`/`prev_r` and `self.timestep`.

**Files:** `f1tenth_gym/envs/f110_env.py`

**Validate:** Create recovery env, reset with small perturbation (beta=0.01, r=0.01, v=5), step with zero action. Manually check `_check_recovery_success()` returns `True` (near-zero state should satisfy all conditions). Reset with large perturbation — should return `False`.

---

### [ ] Task 5: Implement recovery done conditions in `_check_done()`

**What:** Add recovery branch at the top of `_check_done()`. When `training_mode == "recover"`: terminated if boundary exceeded OR recovery succeeded. Truncated if current arc-length > `recovery_s_max` OR `current_step > max_episode_steps`. Store `self.recovery_succeeded` for use by reward. Existing race logic goes in `else` branch.

**Files:** `f1tenth_gym/envs/f110_env.py`

**Validate:** Test 3 termination scenarios:
(a) Reset with large beta → car crashes into boundary → `terminated=True`
(b) Reset with tiny perturbation, step with a reasonable controller → eventually `terminated=True` via recovery success
(c) Reset with moderate perturbation, step many times → `truncated=True` at step 2048

---

### [ ] Task 6: Implement recovery reward function

**What:** Add `_get_recovery_reward()` method computing `R = -K_e * sqrt(beta^2 + r^2) + R_col + R_rec - K_c * dt`. Update `prev_beta`/`prev_r` at end of method. Modify `_get_reward()` to dispatch to `_get_recovery_reward()` when `training_mode == "recover"`.

**Files:** `f1tenth_gym/envs/f110_env.py`

**Validate:** Step through several episodes manually. Check: reward is negative when beta/r are large, reward includes +100 on recovery success, reward includes -50 on boundary crash, reward includes small constant penalty each step. Print reward components to verify.

---

### [ ] Task 7: Create recovery config YAML and factory functions

**What:** Create `train/config/gym_config_recover.yaml` with recovery-specific values (see Step 1). Add recovery config loader and constants to `train/config/env_config.py`. Add `get_recovery_train_config()` and `get_recovery_test_config()` factory functions that use shared constants (MODEL, TIMESTEP, INTEGRATOR, PARAMS, etc.) plus recovery-specific overrides.

**Files:** `train/config/gym_config_recover.yaml` (create), `train/config/env_config.py` (modify)

**Validate:** Import and call `get_recovery_train_config()` — returns a valid config dict. Create env with it — no errors. Confirm `training_mode` is `"recover"`, map is `"IMS"`, and all recovery keys are present.

---

### [ ] Task 8: Create `ppo_recover.py` training script

**What:** Create `train/ppo_recover.py` mirroring `train/ppo_race.py` structure. Import recovery config functions. All 4 modes (train, eval, download, continue). Uses existing `training_utils.py` functions unchanged.

**Files:** `train/ppo_recover.py` (create)

**Validate:** Run `python train/ppo_recover.py --m t` with a small `total_timesteps` (override in YAML to ~100k for testing). Confirm: training starts, episodes reset correctly with perturbed states, rewards are computed, checkpoints saved, wandb logs created. Run `python train/ppo_recover.py --m e --path <model>` with `render_mode="human"` — visually confirm vehicle starts perturbed at S=96 on IMS.

---

### [ ] Task 9: End-to-end validation

**What:** Run existing test suite + manual validation of the full recovery pipeline.

**Validate:**
1. `python3 -m pytest` — all existing tests pass (no regressions)
2. Short training run completes without errors
3. Evaluation shows correct behavior: perturbed start, boundary termination, recovery detection, arc-length truncation
4. Test with a known controller (e.g., stanley from `beta_r_traj_IMS_plot.py`) in recovery env to verify recovery detection triggers correctly
