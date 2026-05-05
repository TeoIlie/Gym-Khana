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

We target **STD** directly — the model used by drift RL training. STD's user state is 9 elements (`[x, y, delta, v, yaw, yaw_rate, slip_angle, omega_front, omega_rear]`); the wheel speeds are seeded from VESC `rs_core_speed/R_w` (single AWD scalar to both wheels) at every window reset. STD is over-parameterized relative to what the bags can identify (no full-friction-circle excitation; longitudinal tire still effectively constrained only via `a_x` until `omega_rear` becomes a loss channel), so the param subset is small even though the model is the full PAC2002.

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
- Reset accepts a full 7- or 9-element user state directly (`gymkhana_env.py:1132-1144` + `model.user_state_lens()`), so we can warm-start from Vicon (and VESC wheel speed) at every window without integrating from `t=0`.
- Bypassing the env (calling `p_accl`, `steering_constraint`, `vehicle_dynamics_std` directly) duplicates control-logic code that drifts out of sync the next time the env is touched.
- Overhead is acceptable: 30 windows × 150 steps × 2000 trials ≈ 9M steps. At ~1 ms/step that's ~2.5 hours per stage on a single core; parallelism via Optuna RDB storage gives a 4–8× speedup.

If profiling shows the env is the bottleneck, the escalation is to disable rendering/collision/observation hooks first, *then* drop to direct dynamics calls — not before.

## Loss function (locked)

Detailed in [`OPTUNA_SYS_ID_LOSS.md`](OPTUNA_SYS_ID_LOSS.md). Summary: per-channel NMSE on `{yaw_rate, v_y, a_x, v_x}` with weights `{3, 2, 1, 0.5}`, 1.5 s windows at 0.5 s stride, 0.2 s warmup discard, low-speed mask, optional left/right mirroring. Real `a_x` smoothed once at dataset load (Savitzky–Golay, w=21, p=2); sim `a_x` finite-differenced raw.

## Implementation phases

Each phase below is independently mergeable and produces a usable artifact. Detailed sub-plans live in `docs/plan/`; this document only sets phase scope and dependencies.

### Phase 0 — Scaffold

Create `examples/analysis/sysid/` and `tests/sysid/`. No code yet.

### Phase 1 — Loss, dataset, rollout (STD) ✅ **complete**

Locked plan: [`OPTUNA_SYS_ID_LOSS.md`](OPTUNA_SYS_ID_LOSS.md).

Produces:
- ✅ `examples/analysis/sysid/dataset.py` — `Window`, `Dataset`, `load_dataset`, `mirror_window`, `CHANNELS`. CLI plot helper for dataset overview.
- ✅ `examples/analysis/sysid/loss.py` — `channel_nmse`, `window_loss`, `dataset_loss`, `DEFAULT_WEIGHTS`. Sim signals are passed as `dict[str, np.ndarray]` keyed by `CHANNELS` (no dedicated type).
- ✅ `examples/analysis/sysid/rollout.py` — `Rollout` class (env constructed once per worker, `set_params` hot-swaps PAC2002 coefficients, context-manager support, NaN/inf guard) + `make_rollout_fn` convenience wrapper. Reads body-frame `v_x`/`v_y`/`yaw_rate` from the `dynamic_state` obs dict; finite-diffs `a_x` from sim `v_x`. CLI: `python -m examples.analysis.sysid.rollout --path <bag>` produces a sim-vs-real overlay plot for visual validation.
- ✅ `tests/sysid/test_dataset.py`, `tests/sysid/test_loss.py`, `tests/sysid/test_rollout.py` — full coverage of the three Phase-1 invariants (identity, mirror, sampler-determinism) plus warmup/weighting/aggregation, NaN guard, hot-swap, and end-to-end loss smoke test.

**Exit criterion (met):** `dataset_loss` runs end-to-end on `examples/analysis/bags/circle_Apr6_100Hz.npz` with YAML defaults in ~4.5 s. Baseline (no mirror, 11 windows, **VESC-seeded omegas**): **total = 5.0224**, per-channel NMSE = `{v_y: 1.85, a_x: 0.33, yaw_rate: 0.30, v_x: 0.15}` — lateral-dominant residual as expected. Identity self-test = 0 within numerical noise. Any future Optuna study must beat this baseline.

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

### `omega` as a loss channel

**Status:** the *reset* half of this upgrade is done — `dataset.py` reads VESC `rs_core_speed`, `init_state` is 9-wide, and the env's 9-branch (`init_std(..., compute_wheel_speeds=False)`) consumes the seeded omegas. AWD assumption is in effect (single VESC scalar to both `omega_front` and `omega_rear`). What remains is adding wheel-speed *as a scored channel*, which is what actually constrains longitudinal-tire identifiability.

**Trigger:** end of Phase 4. If `a_x` NMSE plateaus high after exhausting `tire_p_dx1` / `tire_p_ex1`, longitudinal-tire identifiability is the bottleneck and this is the right escalation. If `a_x` reduces cleanly through Stage 2 alone, skip it.

**Why it matters:** without an omega channel, longitudinal tire params are constrained only through `a_x = (Fx_f + Fx_r)/m` — one equation, under-determined for two wheels.

**Scope:**
- Add `omega` channel(s) to `CHANNELS` and `Window` (need to record `omega_full` slices on each window, not just the t0 sample).
- `Rollout.run` extracts sim wheel speed(s) from `agent.state[7:9]` (no obs key today; either expose via observation or read raw state).
- Loss: weight ~0.5 by default; mirror invariance is straightforward (omega is sign-symmetric under L/R).
- If the AWD assumption ever needs revisiting (i.e. the rig is actually RWD), seed front from `v*cos(β)*cos(δ)/R_w` instead and only add `omega_rear` to the loss.

**Pre-flight (already done):** VESC ↔ Vicon agreement was confirmed on `circle_Apr6_100Hz.npz` before pulling the reset half forward.

## Verification (cross-phase)

Each phase has its own exit criterion above. Three cross-phase invariants must hold throughout:

1. **Identity invariant:** for any phase's pipeline, feeding real signals as `sim` produces total loss = 0 (within 1e-9). Catches dataset/loss bugs.
2. **Mirror invariant:** for symmetric params, mirrored windows produce identical loss to originals (within 1e-6). Catches sign bugs in the mirror transform.
3. **Sampler determinism:** running the same study twice with the same seed produces the same trial sequence and same best loss (within numerical noise). Catches hidden RNG dependencies.

## Out of scope for this overview

- Multi-NPZ aggregation policy (mixing bags from different surfaces / driving styles) — defer to Phase 6 sub-plan.
- Per-window weighting by maneuver type (drift vs. straight vs. corner entry) — only revisit if Phase 4 residuals are dominated by one regime.
- Online / on-vehicle adaptation — out of scope; this is offline batch identification.
