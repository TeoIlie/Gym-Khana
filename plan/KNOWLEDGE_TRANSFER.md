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

### Approach 2: Fine-Tuning with Fresh Optimizer + LR Reset (Recommended Default)
Load racing weights, reset Adam optimizer state, apply fresh LR schedule, reset `log_std` for exploration.

- **Pro**: Preserves all learned weight knowledge while removing stale optimizer state. Fresh LR schedule gives proper warm-up. `log_std` reset enables exploration in new task.
- **Con**: No explicit protection against catastrophic forgetting (but for 2x64 network this is less of a concern -- adapts quickly)
- **Verdict**: Best bang-for-buck. Expected to outperform both from-scratch and Approach 1.

### Approach 3: Fine-Tuning with Fresh Optimizer + Critic Reinitialization
Same as Approach 2 (fresh optimizer, fresh LR schedule, `log_std` reset), but also reinitialize `mlp_extractor.value_net` and `value_net` to random weights.

- **Pro**: Critic starts clean — no misleading value estimates from racing reward scale. Actor retains dynamics knowledge while critic learns recovery values from scratch.
- **Con**: Asymmetry between confident pre-trained actor and untrained critic can cause high-variance advantage estimates in early training. This is transient and should resolve within a few hundred updates for a 2x64 network.
- **Verdict**: Worth implementing as a `--reset_critic` flag to compare against Approach 2. The two approaches test different hypotheses: Approach 2 bets that racing critic features partially transfer; Approach 3 bets that a clean slate is better than stale estimates.

### Approach 4: Selective Layer Freezing (Recommended Option)
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

## Implementation Plan (Approach 2 Only)

Approaches 4, 5 (layer freezing) are deferred to a future iteration.

### 1. Add `transfer_train()` to `train/train_common.py`

New function modeled on `continue_training()` (line 157). Key differences from `continue_training()`:

```python
def transfer_train(profile, model_path, additional_timesteps, reset_log_std=-0.5):
```

After `PPO.load(model_path, env=env, device="auto")`:

1. **Reset Adam optimizer** -- reconstruct with fresh momentum/variance buffers:
   ```python
   model.policy.optimizer = model.policy.optimizer.__class__(
       model.policy.parameters(),
       lr=linear_schedule(START_LEARNING_RATE, END_LEARNING_RATE)(1.0),
   )
   ```

2. **Fresh LR schedule**:
   ```python
   model.learning_rate = linear_schedule(START_LEARNING_RATE, END_LEARNING_RATE)
   ```

3. **Reset log_std** for exploration (when `reset_log_std` is not None):
   ```python
   model.policy.log_std.data.fill_(reset_log_std)
   ```

4. Use `reset_num_timesteps=True` in `model.learn()` (new task = fresh step counter and LR schedule).

5. Log transfer metadata to wandb:
   ```python
   wandb.config.update({"transfer_source": model_path, "transfer_reset_log_std": reset_log_std})
   ```

6. Save `transfer_config.yaml` alongside gym/rl configs.

### 2. Add CLI mode `--m f` to `main()` in `train/train_common.py`

- Add `"f"` to `--m` choices (for "fine-tune"/"transfer")
- Add `--reset_log_std` arg (float, default -0.5; string `"none"` to skip reset)
- Require `--path` when mode is `"f"`
- Route to `transfer_train()`

### 3. Update docstrings

- `train/ppo_recover.py`: Add transfer usage example
- `train/ppo_race.py`: Add transfer usage example (works in either direction)

---

## Files to Modify

- `train/train_common.py` -- Add `transfer_train()`, modify `main()` for `--m f`
- `train/ppo_recover.py` -- Update docstring
- `train/ppo_race.py` -- Update docstring

No changes to: `env_config.py`, `train_utils.py`, `callbacks.py`, gym configs.

---

## Usage

```bash
# Transfer racing model to recovery (fresh optimizer + LR + log_std reset)
python train/ppo_recover.py --m f --path /path/to/racing_model.zip

# Transfer without log_std reset (keep racing exploration level)
python train/ppo_recover.py --m f --path /path/to/racing_model.zip --reset_log_std none

# Transfer with custom timesteps
python train/ppo_recover.py --m f --path /path/to/racing_model.zip --additional_timesteps 20000000
```

---

## Verification

1. Run: `python train/ppo_recover.py --m f --path <racing_model>.zip`
2. Confirm wandb run logs transfer metadata (source model path, log_std reset value)
3. Check tensorboard: LR curve starts at `START_LEARNING_RATE` (not from where racing left off)
4. Compare early training curves (first 2-5M steps) vs. from-scratch baseline

---

## Future: Approach 4,5 (Layer Freezing)

Add `--freeze` flag that sets `requires_grad=False` on `model.policy.mlp_extractor.parameters()` and reconstructs the optimizer with only trainable params. This tests whether the racing feature extractor alone is sufficient for recovery.
