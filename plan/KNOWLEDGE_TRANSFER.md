# Knowledge Transfer: Racing Policy to Recovery Task

## Context

You've trained a PPO racing policy (`ppo_race`) that has learned the underlying dynamics of the STD vehicle model (tire limits, steering response, velocity management). You want to transfer this knowledge to bootstrap recovery training (`ppo_recover`), which needs the same dynamics understanding but for a different objective (stabilizing from perturbed states rather than maximizing track progress).

**Key compatibility finding**: Both tasks share identical observation spaces (18D "drift" type), action spaces (2D normalized), and network architecture (MlpPolicy, 2x64 FC). This means the racing model can be loaded directly into a recovery environment with no dimension mismatches.

**The challenge**: The reward functions are completely different (progress-based vs. equilibrium-seeking), reset distributions differ (random track position vs. fixed position with perturbed dynamics), and the optimizer's Adam state is tuned for racing gradients.

---

## PPO implementation in SB3

`PPO.load()` loads everything — both actor and critic weights, plus log_std. In SB3's default MlpPolicy, the architecture is:

```
mlp_extractor.policy_net  (actor features)     ← encodes dynamics knowledge
mlp_extractor.value_net   (critic features)    ← calibrated for racing rewards
action_net                (actor head)         ← racing actions
value_net                 (critic head)        ← racing value estimates
log_std                                        ← racing exploration level
```

The actor parameters in `mlp_extractor.policy_net` encode useful dynamics knowledge from the racing task. The `action_net` head maps these features to action-space means — tuned for racing-optimal actions, not recovery-optimal, but still grounded in the same vehicle dynamics. As for the critic, `mlp_extractor.value_net` and `value_net` are calibrated for the racing reward function and will produce misleading value estimates for recovery, causing noisy advantage estimates (`A = R - V`) until readapted. The `log_std` value is also stale, as this new task requires fresh exploration.

---

## Approach Comparison

### Approach 1: Direct Fine-Tuning
Load racing model, continue training in recovery env. Essentially `--m c` cross-task.

- **Pro**: Simplest possible implementation
- **Con**: Stale optimizer momentum fights new gradients; `log_std` likely too low for recovery exploration; critic miscalibrated for recovery reward scale
- **Verdict**: Dominated by Approach 2

### Approach 2: Fine-Tuning with Fresh Optimizer + LR Reset ✅ Implemented
Load racing weights, reset Adam optimizer state, apply fresh LR schedule, reset `log_std` for exploration.

- **Pro**: Preserves all learned weight knowledge while removing stale optimizer state. Fresh LR schedule gives proper warm-up. `log_std` reset enables exploration in new task.
- **Con**: No explicit protection against catastrophic forgetting (but for 2x64 network this is less of a concern -- adapts quickly)
- **Verdict**: Best bang-for-buck. Expected to outperform both from-scratch and Approach 1.

### Approach 3: Fine-Tuning with Fresh Optimizer + Critic Reinitialization ✅ Implemented
Same as Approach 2 (fresh optimizer, fresh LR schedule, `log_std` reset), but also reinitialize `mlp_extractor.value_net` and `value_net` to random weights.

- **Pro**: Critic starts clean — no misleading value estimates from racing reward scale. Actor retains dynamics knowledge while critic learns recovery values from scratch.
- **Con**: Asymmetry between confident pre-trained actor and untrained critic can cause high-variance advantage estimates in early training. This is transient and should resolve within a few hundred updates for a 2x64 network.
- **Verdict**: Toggled via `transfer_reset_critic` in `rl_config.yaml`. Compares against Approach 2: Approach 2 bets that racing critic features partially transfer; Approach 3 bets that a clean slate is better than stale estimates.

### Approach 4: Selective Layer Freezing
Freeze `mlp_extractor` (feature extraction layers), only train output heads (`action_net`, `value_net`, `log_std`).

- **Pro**: Strongest forgetting protection; fewer trainable params = faster per-step; tests whether racing features are task-agnostic
- **Con**: Only ~130 trainable parameters in heads -- may lack capacity. Critic features frozen for wrong reward scale. High variance outcome depending on feature quality.
- **Verdict**: Worth having as a `--freeze` flag to test the hypothesis. If it works, it's a strong result. If not, fall back to Approach 2.

### Approach 5: Progressive Unfreezing
Start frozen, unfreeze after N steps with lower LR.

- **Pro**: Best of both worlds theoretically
- **Con**: Most complex; adds 2 hyperparameters to tune; marginal benefit at 2x64 network scale
- **Verdict**: Defer. Implement later if scaling to larger networks.

---

## Implementation Details

### Approach 2 — `transfer_train()` in `train/train_common.py`

Mode `--m f` routes to `transfer_train()`. After `PPO.load(model_path, env=env, device="auto")`:

1. **Fresh LR schedule** — sets both `model.learning_rate` and `model.lr_schedule` to a new `linear_schedule(START_LR, END_LR)`.
2. **Fresh Adam optimizer** — reconstructs via `model.policy.optimizer_class` with `optimizer_kwargs` to preserve SB3 defaults (eps=1e-5). Clears all momentum/variance buffers.
3. **Reset `_n_updates`** — cosmetic, for clean tensorboard logging.
4. **Reset `log_std`** — `model.policy.log_std.data.fill_(reset_log_std)` when `transfer_reset_log_std` is not `null` in `rl_config.yaml`.
5. **`reset_num_timesteps=True`** — new task = fresh step counter and LR schedule progression.
6. **Save `transfer_config.yaml`** alongside gym/rl configs with source model path, timesteps, and reset settings.

### Approach 3 — Critic Reset (additive to Approach 2)

When `transfer_reset_critic: true` in `rl_config.yaml`, after the log_std reset:

```python
model.policy.mlp_extractor.value_net.apply(partial(BasePolicy.init_weights, gain=np.sqrt(2)))
model.policy.value_net.apply(partial(BasePolicy.init_weights, gain=1.0))
```

This replicates SB3's original initialization: orthogonal weights with the same gains used in `ActorCriticPolicy._build()`. The `.apply()` call recurses into the `nn.Sequential`, hitting each `nn.Linear` (and skipping activation layers via the `isinstance` check in `init_weights`).

**What gets reset**:
- `mlp_extractor.value_net` — critic hidden layers (gain=√2)
- `value_net` — critic output head (gain=1.0)

**What is preserved**:
- `mlp_extractor.policy_net` — actor hidden layers
- `action_net` — actor output head
- `log_std` — handled separately by reset_log_std

### Configuration

All transfer settings live in `rl_config.yaml` (no CLI flags beyond `--m f --path`):

```yaml
# Transfer/continue training defaults (mode 'f')
additional_timesteps: 100000000
transfer_reset_log_std: -0.5    # reset log_std to this value; use null to skip
transfer_reset_critic: false    # set to true for Approach 3
```

Constants loaded in `env_config.py`: `TRANSFER_RESET_LOG_STD`, `TRANSFER_RESET_CRITIC`.

### Files Modified

- `train/config/rl_config.yaml` — `transfer_reset_critic` key
- `train/config/env_config.py` — `TRANSFER_RESET_CRITIC` constant
- `train/train_common.py` — `transfer_train()` with `reset_critic` parameter, imports for `partial`, `BasePolicy`
- `train/ppo_recover.py` — docstring with transfer usage
- `train/ppo_race.py` — docstring with transfer usage
- `tests/test_train_common.py` — `TestCriticReset` test class (9 tests)

---

## Usage

```bash
# Approach 2: Transfer racing model to recovery (fresh optimizer + LR + log_std reset)
python train/ppo_recover.py --m f --path /path/to/racing_model.zip

# Approach 3: Same as above + critic reset (set transfer_reset_critic: true in rl_config.yaml)
python train/ppo_recover.py --m f --path /path/to/racing_model.zip
```

---

## Verification

1. Run: `python train/ppo_recover.py --m f --path <racing_model>.zip`
2. Confirm console prints reset status for optimizer, LR, log_std, and critic (when enabled)
3. Check `transfer_config.yaml` in run output dir for `reset_critic: true/false`
4. Check tensorboard: LR curve starts at `START_LEARNING_RATE` (not from where racing left off)
5. Run tests: `python3 -m pytest tests/test_train_common.py -v` (22 tests)
6. Compare early training curves (first 2-5M steps) between Approach 2 and 3

---

## Future: Approach 4, 5 (Layer Freezing)

Add `--freeze` flag that sets `requires_grad=False` on `model.policy.mlp_extractor.parameters()` and reconstructs the optimizer with only trainable params. This tests whether the racing feature extractor alone is sufficient for recovery.
