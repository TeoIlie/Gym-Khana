# System ID with Optuna — Overview & Implementation Plan

## Context

This repo's STD (Single-Track Drift) dynamics model uses PAC2002 tire parameters that cannot be physically measured on a 1/10 F1TENTH car. To close the sim2real gap for RL drift policy transfer, we identify these parameters by replaying real Vicon-recorded command sequences through the simulator under candidate parameter sets and minimizing a sim-vs-real trajectory residual via Optuna.

- Model: `gymkhana/envs/dynamic_models/single_track_drift/single_track_drift.py`
- Parameters: `gymkhana/envs/params/f1tenth_std.yaml`
- Real bags (NPZ, 100 Hz, Vicon): `examples/analysis/bags/`
- Existing sim/real comparison reference: `examples/analysis/traj_compare.py`

## Strategy

Per-trial loop:
1. Optuna proposes a parameter dict.
2. Configure a single env (`GKEnv.configure({"params": ...})`) — env is constructed once per trial, not once per window (`gymkhana_env.py:626-648`).
3. For each pre-built window of real data: reset to the Vicon state with `env.reset(options={"states": ...})` (`gymkhana_env.py:1075-1192`), step through recorded commands, score the rollout against real signals.
4. Aggregate weighted NMSE across channels and windows → trial loss.

We target **STD** directly — the model used by drift RL training. STD's user state is 7 elements (`[x, y, delta, v, yaw, yaw_rate, slip_angle]`); the two extra internal states (`omega_front`, `omega_rear`) are seeded from `v` at reset (`single_track_drift.py:42-43, 244-245`), so the rollout/loss interface is the same dimensionality the env exposes for any model. STD is over-parameterized relative to what the bags can identify (no measured wheel speeds; no full-friction-circle excitation), so the param subset is small even though the model is the full PAC2002.

## Resolved design decisions

These were the open questions in the original draft.

### How many parameters does Optuna optimize?

**Not all 20.** Identifiability — not Optuna capacity — is the binding constraint. PAC2002 is over-parameterized for the F1TENTH operating envelope, and many coefficients are unidentifiable from indoor Vicon data.

**Freeze permanently:**
- All camber-coupled coefficients (`tire_p_dx3`, `tire_p_dy3`, `tire_p_hy3`, `tire_p_vy3`) — 1/10 cars run ~zero camber.
- Pure-shift terms with negligible effect (`tire_p_hx1`, `tire_p_vx1`, `tire_p_hy1`) — they trade off against many other params.

**Multi-stage fit** (each stage opens a small, identifiable subset):
- **Stage 1 — linear regime (4 params):** `tire_p_ky1`, `tire_p_kx1`, `tire_p_cy1`, `tire_p_cx1`. Restrict to low-slip windows.
- **Stage 2 — saturation (+4 params):** `tire_p_dy1`, `tire_p_dx1`, `tire_p_ey1`, `tire_p_ex1`. Use full dataset; initialize from Stage 1 best.
- **Stage 3 — combined slip (optional):** unfreeze a small subset of `r_*` coefficients only if Stage 2 residuals show systematic combined-slip error.

A **sensitivity sweep** (one-at-a-time perturbation around YAML defaults) runs *before* any Optuna study and finalizes the ranked subset. Tight, physically motivated bounds (e.g., `D ∈ [0.5, 1.5]`, `C ∈ [1.0, 2.0]`) — wide bounds waste trials.

**Sampler:** TPE for Stage 1 (≤4 dims, fast). Switch to `CmaEsSampler` for Stages 2–3 (better behavior in 8+ continuous dims).

### How to perform rollouts?

**Use the gym env, not raw dynamics calls.** Reasons:
- `GKEnv.configure({"params": ...})` hot-swaps params on a *single* env instance — no per-trial reconstruction overhead.
- Reset accepts a full 7-element user state directly (`gymkhana_env.py:1132-1144`), so we can warm-start from Vicon at every window without integrating from `t=0`.
- Bypassing the env (calling `p_accl`, `steering_constraint`, `vehicle_dynamics_std` directly) duplicates control-logic code that drifts out of sync the next time the env is touched.
- Overhead is acceptable: 30 windows × 150 steps × 2000 trials ≈ 9M steps. At ~1 ms/step that's ~2.5 hours per stage on a single core; parallelism via Optuna RDB storage gives a 4–8× speedup.

If profiling shows the env is the bottleneck, the escalation is to disable rendering/collision/observation hooks first, *then* drop to direct dynamics calls — not before.

## Loss function (locked)

Detailed in [`OPTUNA_SYS_ID_LOSS.md`](OPTUNA_SYS_ID_LOSS.md). Summary: per-channel NMSE on `{yaw_rate, v_y, a_x, v_x}` with weights `{3, 2, 1, 0.5}`, 1.5 s windows at 0.5 s stride, 0.2 s warmup discard, low-speed mask, optional left/right mirroring. Real `a_x` smoothed once at dataset load (Savitzky–Golay, w=21, p=2); sim `a_x` finite-differenced raw.

## Implementation phases

Each phase below is independently mergeable and produces a usable artifact. Detailed sub-plans live in `docs/plan/`; this document only sets phase scope and dependencies.

### Phase 0 — Scaffold

Create `examples/analysis/sysid/` and `tests/sysid/`. No code yet.

### Phase 1 — Loss, dataset, rollout (STD)

Locked plan: [`OPTUNA_SYS_ID_LOSS.md`](OPTUNA_SYS_ID_LOSS.md).

Produces:
- ✅ `examples/analysis/sysid/dataset.py` — `Window`, `Dataset`, `load_dataset`, `mirror_window`, `CHANNELS`.
- ✅ `examples/analysis/sysid/loss.py` — `channel_nmse`, `window_loss`, `dataset_loss`, `DEFAULT_WEIGHTS`. Sim signals are passed as `dict[str, np.ndarray]` keyed by `CHANNELS` (no dedicated type).
- ⏳ `examples/analysis/sysid/rollout.py` — `make_rollout_fn` closure (constructs env once, reuses across windows). Must return a `dict[str, np.ndarray]` with all four `CHANNELS` keys, each shape `(N+1,)` — enforced by an assertion in `window_loss`.
- ✅ `tests/sysid/test_dataset.py`, ✅ `tests/sysid/test_loss.py` (identity, warmup discard, weighting, aggregation, mirror symmetry, zero-variance safety), ⏳ `tests/sysid/test_rollout.py`.

**Exit criterion:** `dataset_loss` runs end-to-end on `examples/analysis/bags/circle_Apr6_100Hz.npz` with YAML defaults; baseline per-channel NMSE recorded; identity self-test = 0.

### Phase 2 — Sensitivity analysis

Sub-plan: `OPTUNA_SYS_ID_SENSITIVITY.md` (to be written).

Produces:
- `examples/analysis/sysid/sensitivity.py` — for each candidate param, perturb ±10%/±25%/±50% around YAML default, compute `dataset_loss`, plot per-channel response.
- A markdown report or CSV ranking params by `dLoss/dParam` magnitude.
- A **maneuver coverage audit**: histograms of `α_front`, `α_rear`, `κ_front`, `κ_rear` across the dataset. If `|α|` never exceeds ~5°, saturation params are unidentifiable regardless of trial budget — must surface this before Phase 3.

**Exit criterion:** ranked param list + coverage histograms reviewed; final Stage 1/2/3 search-space membership confirmed.

### Phase 3 — Stage 1 Optuna study (STD, lateral-linear)

Sub-plan: `OPTUNA_SYS_ID_STUDY.md` (to be written).

Produces:
- `examples/analysis/sysid/search_spaces.py` — bounded distributions per param per stage.
- `examples/analysis/sysid/study.py` — Optuna objective, study creation, SQLite RDB storage, `MedianPruner` with per-window intermediate `trial.report`, seeded sampler, parallel-worker entry point.
- A CLI in the same module to run a study by name + stage + bag.
- Output YAML: `gymkhana/envs/params/f1tenth_std_optuna.yaml` (per-stage suffix while iterating, e.g. `_stage1`, `_stage2`).

**Exit criterion:** Stage 1 best loss < baseline; reproducible re-run within 1% on same seed.

### Phase 4 — Stage 2 Optuna study (STD, saturation)

Reuses Phase 3 infrastructure. Initial point = Stage 1 best. Sampler switches to CMA-ES. No new sub-plan unless the search-space management diverges materially.

**Exit criterion:** Stage 2 best loss < Stage 1 best; per-channel residual breakdown shows balanced reduction (not just one channel); identified params beat the hand-tuned `f1tenth_std_drift_bias.yaml` on a held-out bag.

### Phase 5 — Stage 3 Optuna study (STD, combined slip — conditional)

Run only if Phase 4 residuals show systematic combined-slip error (e.g. `a_x` and `v_y` errors correlated under braking-while-cornering). Unfreezes a small subset of `r_*` coefficients (lateral and/or longitudinal) chosen from the sensitivity ranking.

**Exit criterion:** Stage 3 reduces the dominant Phase 4 residual without inflating others; otherwise Phase 4's params remain canonical.

### Phase 6 — Validation, plotting, wandb

Sub-plan: `OPTUNA_SYS_ID_CLI.md` (to be written).

Produces:
- Held-out bag selection convention; one bag is excluded from every study and used only for final reporting.
- Plot helpers (sim-vs-real overlays per channel per window) reusing `traj_compare.py` patterns.
- Optional wandb logging of trial loss, per-channel breakdown, best-params YAML as artifact.

**Exit criterion:** final identified params used in an RL training run; sim2real gap qualitatively reduced versus `f1tenth_std.yaml` and `f1tenth_std_drift_bias.yaml` baselines.

## File / directory layout

```
examples/analysis/
  traj_compare.py                    # existing — reused for env config + rollout patterns
  bags/                              # existing — real NPZ data, 100 Hz Vicon + commands
  sysid/                             # NEW
    dataset.py                       # Phase 1
    loss.py                          # Phase 1
    rollout.py                       # Phase 1
    sensitivity.py                   # Phase 2
    search_spaces.py                 # Phase 3
    study.py                         # Phase 3 (CLI lives here)

tests/sysid/                         # NEW
  test_loss.py                       # Phase 1
  test_dataset.py                    # Phase 1
  test_rollout.py                    # Phase 1
  test_sensitivity.py                # Phase 2

gymkhana/envs/params/                # existing
  f1tenth_std_optuna.yaml            # NEW — Phase 3/4/5 output (per-stage suffixes while iterating)

docs/plan/
  OPTUNA_SYS_ID_OVERVIEW.md          # this document
  OPTUNA_SYS_ID_LOSS.md              # locked
  OPTUNA_SYS_ID_SENSITIVITY.md       # to be written (Phase 2)
  OPTUNA_SYS_ID_STUDY.md             # to be written (Phase 3, reused for 4 and 5)
  OPTUNA_SYS_ID_CLI.md               # to be written (Phase 6)
```

## Cross-cutting concerns

These apply across phases and are called out here so individual sub-plans don't have to re-derive them.

- **`delta_init` is a hidden parameter, not just a warmup choice.** `delta_init = cmd_steer[t0]` is a guess; the servo's actual transient depends on slew rate (`sv_max`) and prior cmd history. Two cleaner options if the 0.2 s discard proves insufficient: (a) warm up with 0.3–0.5 s of pre-`t0` commands instead of cold-starting `delta`; (b) treat `delta_init` as a per-window nuisance variable. Defer until Phase 1 baseline exposes the residual.
- **Mirroring vs. real asymmetry.** The existing `f1tenth_std_drift_bias.yaml` implies LR asymmetry on the real car. Default mirroring (locked in the LOSS plan) averages this out — fine for a symmetric RL target policy, wrong if asymmetry is what we're trying to capture. Be explicit in each study's README which assumption holds.
- **Held-out validation set.** With ~tens of windows, overfitting to bag-specific noise is real. Reserve at least one bag from every study; report on it post-hoc.
- **Pruning.** `MedianPruner` with per-window `trial.report(loss, window_idx)` kills bad trials in <10% of full cost. Treat as required, not optional, from Phase 3 onward.
- **Parallelism.** SQLite RDB storage (`optuna.create_study(storage="sqlite:///...")`) enables N processes on one machine. Rollouts are embarrassingly parallel.
- **Reproducibility.** Seed the sampler. Confirm `env.reset(options={"states": ...})` is fully deterministic — no internal RNG firing — otherwise trial-to-trial noise masquerades as a bad parameter.
- **Loss-landscape robustness.** Tire-saturation regions can produce near-discontinuous `dLoss/dParam`. If a study stalls, escalate to `log(1 + loss)` aggregation and/or winsorize the worst window per trial before reporting up.

## Deferred upgrades

### Wheel-speed-aware reset + `omega_rear` loss channel

**Trigger:** end of Phase 4. If `a_x` NMSE plateaus high after exhausting `tire_p_dx1` / `tire_p_ex1`, longitudinal-tire identifiability is the bottleneck and this upgrade is the right escalation. If `a_x` reduces cleanly through Stage 2 alone, skip it.

**Why it matters:** the bags include VESC wheel-speed feedback (rear wheel — F1TENTH is RWD; front wheel free-rolls). Without this feature: (a) longitudinal tire params are constrained only through `a_x = (Fx_f + Fx_r)/m` — one equation, under-determined for two wheels; (b) every window's reset seeds `omega = v/R_w` (`single_track_drift.py:42-43`), so windows starting mid-drift or under hard braking begin with the wrong κ until the 0.2 s warmup absorbs the transient — which it won't, under aggressive maneuvers.

**Scope:**
- `gymkhana_env.py` reset path (`gymkhana_env.py:1132`): relax `expected_state_len` to accept 7 or 9 for STD.
- `RaceCar.reset` / `init_std` in `single_track_drift.py`: branch on input length; if 9 elements, use the provided `omega_front`, `omega_rear` directly instead of recomputing from `v`.
- Loss: add `omega_rear` channel with default weight ~0.5; exclude `omega_front` from loss but still seed it as `v/R_w` (free-rolling).
- Tests: 9-element reset path, omega-channel mirror invariant, identity invariant unchanged.

**Pre-flight check before implementing:** plot VESC `omega_rear` vs Vicon-derived `v/R_w` on a free-rolling segment. If steady-state values disagree, VESC feedback is filtered/biased and the measurement problem must be solved (or characterized) before adding the channel — otherwise you'll fit tire params to firmware lag.

## Verification (cross-phase)

Each phase has its own exit criterion above. Three cross-phase invariants must hold throughout:

1. **Identity invariant:** for any phase's pipeline, feeding real signals as `sim` produces total loss = 0 (within 1e-9). Catches dataset/loss bugs.
2. **Mirror invariant:** for symmetric params, mirrored windows produce identical loss to originals (within 1e-6). Catches sign bugs in the mirror transform.
3. **Sampler determinism:** running the same study twice with the same seed produces the same trial sequence and same best loss (within numerical noise). Catches hidden RNG dependencies.

## Out of scope for this overview

- Multi-NPZ aggregation policy (mixing bags from different surfaces / driving styles) — defer to Phase 6 sub-plan.
- Per-window weighting by maneuver type (drift vs. straight vs. corner entry) — only revisit if Phase 4 residuals are dominated by one regime.
- Online / on-vehicle adaptation — out of scope; this is offline batch identification.
