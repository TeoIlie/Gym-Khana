# Curriculum Learning for Recovery Training

## Context

Recovery training (`ppo_recover.py`) samples initial states from fixed ranges for v, beta, r, and yaw. This makes early training inefficient — the agent faces states it cannot yet handle. Curriculum learning starts with narrow ranges and expands them as the agent demonstrates competence, measured by a rolling success rate from `info["recovered"]`.

## Approach

Custom SB3 `BaseCallback` that tracks rolling episode success rate. When success rate >= 80% over a 500-episode window, all four ranges expand simultaneously by fixed increments. Uses `env_method()` to push updated ranges to **all** SubprocVecEnv subprocess environments at once — every subprocess moves to the next stage together.

### Curriculum stages

All four ranges expand in lockstep on each expansion. The number of expansions is determined by whichever range needs the most steps to reach its max:

| Range | Initial | Max | Increment | Steps to max |
|-------|---------|-----|-----------|-------------|
| v_lo | 5 | 2 | 0.5 | 6 |
| v_hi | 9 | 12 | 0.5 | 6 |
| beta_half | 0.10 | 0.349 | 0.05 | ~5 |
| r_half | 0.20 | 0.785 | 0.10 | ~6 |
| yaw_half | 0.20 | 0.785 | 0.10 | ~6 |

This gives **6 expansions = 7 total stages** (initial + 6 expansions). Ranges that reach their max earlier are clamped and stop growing while the others continue.

### Stopping condition

Expansion stops when **all** ranges have reached their max values (`is_at_max()` returns True for all four). After that, the callback continues to log metrics but no longer modifies ranges. Training itself continues until `total_timesteps` is reached — the curriculum only controls range expansion, not training termination.

### `num_timesteps` vs `n_calls`

SB3's `_on_step()` fires once per `env.step()` call, but with `n_envs` parallel environments each call advances `n_envs` timesteps. Two counters exist:

- **`self.n_calls`**: number of times `_on_step()` was called (= number of `env.step()` calls)
- **`self.num_timesteps`**: total environment steps = `n_calls * n_envs`

The callback uses **`self.num_timesteps`** for both `log_freq` and `max_curriculum_timestep` comparisons, so these thresholds behave consistently regardless of `n_envs`. The `min_episodes_between_expansions` hysteresis uses an episode counter (not timesteps) and is unaffected.

## Files to Modify

| File | Change |
|------|--------|
| `train/callbacks.py` | **New file** — `CurriculumRange` dataclass, `CurriculumLearningCallback`, `make_curriculum_callback()` factory |
| `f1tenth_gym/envs/f110_env.py` | Add `set_recovery_ranges()` method (~8 lines, after `_get_recovery_reward` at ~line 1122) |
| `train/config/gym_config.yaml` | Add `curriculum:` section with all tuning knobs |
| `train/config/env_config.py` | Load `CURRICULUM_CONFIG`, add `get_curriculum_config()` |
| `train/train_common.py` | Wire curriculum callback into `train()` and `continue_training()` callback lists |

## Step 1: Add setter method to F110Env

**File:** `f1tenth_gym/envs/f110_env.py` (after `_get_recovery_reward`, ~line 1122)

Add `set_recovery_ranges(self, v_range, beta_range, r_range, yaw_range)` that sets the four `self.recovery_*_range` attributes. This is called via `SubprocVecEnv.env_method()` which uses `get_wrapper_attr` (subproc_vec_env.py:57) to traverse the Monitor wrapper and find the method on F110Env.

## Step 2: Create `train/callbacks.py`

### `CurriculumRange` dataclass
- Handles both **symmetric** ranges (beta, r, yaw: `[-half, +half]`) and **asymmetric** ranges (v: `[lo, hi]`)
- Fields: `initial_*`, `max_*`, `increment`, `current_*` (runtime state)
- Methods: `expand()` → returns bool if changed, `get_range()` → `[lo, hi]`, `is_at_max()` → bool

### `CurriculumLearningCallback(BaseCallback)`

Constructor params:
- 4x `CurriculumRange` configs (v, beta, r, yaw)
- `window_size=500` — rolling success window size
- `success_threshold=0.8` — expansion trigger
- `min_episodes_between_expansions=200` — hysteresis guard
- `max_curriculum_timestep=None` — stop expanding after N `num_timesteps` (None = no limit)
- `log_freq=10000` — wandb logging frequency (compared against `num_timesteps`, consistent across `n_envs`)

Key methods:
- **`_on_training_start()`**: Push initial (narrow) ranges to all envs via `env_method` (preferred over `_init_callback` — semantically correct as it runs "before the first rollout starts" when the training env is fully ready)
- **`_on_step()`**: Check `self.locals["dones"]` and `self.locals["infos"]` for completed episodes. When `dones[i]` is True, read `infos[i]["recovered"]` and append to rolling window. Check expansion conditions. Log periodically.
- **`_should_expand()`**: Returns True when: window is full AND success_rate >= threshold AND hysteresis satisfied AND not past max_timestep AND not all ranges at max
- **`_expand_ranges()`**: Call `expand()` on each range, clear window (agent must re-prove competence), reset episode counter, push new ranges to envs, log to wandb
- **`_push_ranges_to_envs()`**: `self.training_env.env_method("set_recovery_ranges", v, beta, r, yaw)`
- **`_log_metrics()`**: Log `curriculum/success_rate`, `curriculum/expansion_count`, `curriculum/v_range_lo`, `curriculum/v_range_hi`, `curriculum/beta_half`, `curriculum/r_half`, `curriculum/yaw_half` via `wandb.log()`

### `make_curriculum_callback(config: dict)` factory
- Returns `None` if `config.get("enabled")` is False
- Builds `CurriculumRange` objects from config dict
- Returns configured `CurriculumLearningCallback`

## Step 3: Add curriculum config to `gym_config.yaml`

Append to recovery section:

```yaml
curriculum:
  enabled: true
  window_size: 500
  success_threshold: 0.8
  min_episodes_between_expansions: 200
  max_curriculum_timestep: null
  log_freq: 10000
  # Initial (narrow) ranges
  initial_v_range: [5, 9]
  initial_beta_half: 0.10
  initial_r_half: 0.20
  initial_yaw_half: 0.20
  # Maximum (target) ranges
  max_v_range: [2, 12]
  max_beta_half: 0.349
  max_r_half: 0.785
  max_yaw_half: 0.785
  # Per-expansion increments
  v_increment_lo: 0.5
  v_increment_hi: 0.5
  beta_increment: 0.05
  r_increment: 0.10
  yaw_increment: 0.10
```

## Step 4: Load config in `env_config.py`

After `RECOVERY_TRACK_POOL` (line 77):
- Add `CURRICULUM_CONFIG = _config.get("curriculum", {})`
- Add `get_curriculum_config()` function that returns `CURRICULUM_CONFIG`

## Step 5: Wire into `train_common.py`

In both `train()` (~line 84) and `continue_training()` (~line 198):
- Import `make_curriculum_callback` from `train.callbacks` and `get_curriculum_config` from `train.config.env_config`
- Build callbacks list, conditionally append curriculum callback if `make_curriculum_callback()` returns non-None
- Save curriculum config to YAML alongside gym config

## Verification

1. **Smoke test**: Run `python train/ppo_recover.py --m t` with `total_timesteps` temporarily set to ~500k. Verify console prints show expansion events with updated ranges.
2. **Wandb check**: Confirm `curriculum/*` metrics appear on the wandb dashboard.
3. **Disabled mode**: Set `curriculum.enabled: false` in YAML, run training, verify no curriculum output and training proceeds normally.
4. **Max cap**: Set very small max ranges (equal to initial), verify no expansions trigger.
5. **Existing tests**: Run `python3 -m pytest` to confirm no regressions (the env change is additive-only).
