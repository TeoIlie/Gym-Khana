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
- Pure-shift terms (`tire_p_hx1`, `tire_p_vx1`, `tire_p_hy1`, `tire_p_vy1`) — `_vy1` added to the original 3 because for a symmetric tire it should be ≈0 and the bag can't constrain it. Phase 2's frozen-audit pass empirically verifies the freeze decision rather than assuming it.

**Multi-stage fit** (locked by Phase 2 — see [`OPTUNA_SYS_ID_SENSITIVITY_REPORT.md`](OPTUNA_SYS_ID_SENSITIVITY_REPORT.md)):
- **Stage 1 — chassis + linear regime (6 params):** `I_z`, `a_max`, `I_y_w` (chassis, promoted), plus `tire_p_cy1`, `tire_p_cx1`, `tire_p_ky1` (tire). Chassis params dominate sensitivity on the drift bag (`I_z` Δ=155, `a_max` Δ=144) and must be locked before tire-saturation params, otherwise the latter sponge chassis residuals into non-physical values.
- **Stage 2 — saturation (4 params):** `tire_p_dy1`, `tire_p_dx1`, `tire_p_ex1`, `tire_p_ey1`. Initialize from Stage 1 best. `dy1` / `dx1` bounds from physical priors (μ ∈ [0.7, 1.5]); the wide-δ pass placed their unconstrained minima at non-physical μ values, indicating they were absorbing chassis-inertia / longitudinal-cap error that Stage 1 should remove.
- **Stage 3 — combined slip (optional):** unfreeze a small subset of `r_*` coefficients only if Stage 2 residuals show systematic combined-slip error.

**Demoted from search after Phase 2:** `tire_p_kx1` (Δ=1.87, flat), `sv_min` / `sv_max` (slew limits not exercised by the bag). `tire_p_ey1` is retained in Stage 2 only to close the lateral curve shape and is the first drop candidate if convergence is otherwise clean.

A **sensitivity sweep** (one-at-a-time perturbation around YAML defaults) runs *before* any Optuna study and finalizes the ranked subset. Tight, physically motivated bounds (e.g., `D ∈ [0.5, 1.5]`, `C ∈ [1.0, 2.0]`) — wide bounds waste trials. A **wide-δ orientation pass** (`--wide-deltas`, `±90% to +200%`) precedes the standard sweep when YAML defaults come from a different vehicle scale (current case: PAC2002 dict from a full-scale-vehicle reference) — confirms the loss surface is basin-shaped near defaults rather than a cliff. On the drift bag this pass revealed `dy1`/`dx1`/`ky1` minima past the sampled range at non-physical values; the vehicle-dyn pass then showed `I_z`/`a_max` were the actual culprits, justifying the chassis-first Stage-1.

**Sampler:** `CmaEsSampler` from Stage 1 onward (Stage 1 is 6-dimensional after chassis promotion, past TPE's comfort zone).

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

### Phase 2 — Sensitivity analysis ✅ **complete**

Locked plan: [`OPTUNA_SYS_ID_SENSITIVITY.md`](OPTUNA_SYS_ID_SENSITIVITY.md).

Produces:
- ✅ `examples/analysis/sysid/sensitivity.py` — multi-mode sweep runner. Three sweep groups: (a) Stage-1/2 tire candidates (multiplicative, default ±50% ladder), (b) frozen-param audit (absolute mode, per-param ladders, review-only ranking), (c) vehicle-dyn sweep (`I_z`, `I_y_w`, `sv_min`, `sv_max`, `a_max`, multiplicative −0.7 .. +2.0 ladder). Plus a wide-δ orientation preset (`--wide-deltas`) and CLI flags `--frozen-audit` / `--vehicle-dyn` / `--include-combined`.
- ✅ Coverage audit: histograms of `α_front`, `α_rear`, `κ_front`, `κ_rear` across the dataset with default-Pacejka 95%-of-peak saturation-knee annotations and warning flags when coverage falls short. If `|α|` never exceeds ~5°, saturation params are unidentifiable regardless of trial budget — surfaced loudly in `ranking.md` before Phase 3 launches.
- ✅ Auto-generated proposed Optuna bounds (`p₀ · (1 ± 1.5·δ_safe)`, threshold = 2× baseline). Marked review-only; for high-sensitivity params (where the ±δ_safe interval collapses to the smallest sampled δ), the report flags this and bounds come from physical priors instead.
- ✅ `tests/sysid/test_sensitivity.py` — 6 invariants (frozen-list disjointness, frozen-audit-ladder completeness, vehicle-dyn disjointness, frozen opt-in gate, coverage geometry signs, baseline-anchor δ=0 reproducibility).
- ✅ Full Phase-2 run on `rosbag2_2026_05_04-17_54_17_100Hz.npz` and [`OPTUNA_SYS_ID_SENSITIVITY_REPORT.md`](OPTUNA_SYS_ID_SENSITIVITY_REPORT.md) — locks Stage-1/2 membership above.

**Headline findings from the run** (full detail in the report):
- **Coverage:** |α| and |κ| both exceed default-Pacejka 95%-of-peak knees by orders of magnitude. No identifiability veto on saturation params.
- **Chassis dominance:** `I_z` (Δ=155, truth past +200% of default) and `a_max` (Δ=144, truth below −70%) are the most impactful parameters on this bag, exceeding every tire coefficient. Both are promoted into Stage 1.
- **Wide-δ tire minima are spurious:** `dy1`/`dx1`/`ky1` wide-δ optima land at non-physical μ values — they were sponging chassis-inertia / longitudinal-cap residuals. Hence chassis-first Stage 1, then re-evaluate tire bounds in Stage 2.
- **Frozen audit:** 6/8 frozen params show dLoss = 0 across their ladders (camber and lateral pure-shift confirmed). `tire_p_vx1` shows non-zero sensitivity (Δ=19.7) and acts as a longitudinal-bias absorber — **stays frozen** for physical reasons; the implied bias is investigated outside the tire model in Phase 4.

**Recommended run sequence** (single drift bag, `rosbag2_2026_05_04-17_54_17_100Hz.npz`):

1. **Orientation pass** — *check that defaults aren't on a cliff before trusting local sensitivity*

   ```bash
   python -m examples.analysis.sysid.sensitivity --wide-deltas --no-mirror --progress
   ```

   The PAC2002 defaults in `f1tenth_std.yaml` are lifted from a full-scale-vehicle reference; some coefficients (esp. `tire_p_ky1`, `tire_p_kx1`) could plausibly be 2–3× off the true 1/10 values. The wide ladder (`±90% .. +200%`) tells us *where* the loss minimum is, not just the local gradient. Outputs go to `figures/analysis/sysid/sensitivity/<bag>_wide/` so they don't clobber the main run.

   **What to look for in `<bag>_wide/sweep_total.png`:**
   - Clear U with min at δ=0 → defaults are in the basin, the standard sweep is meaningful.
   - U with min at δ≠0 → defaults are off-center; the truth is at `p₀ · (1 + δ_min)`. Re-anchor before the main sweep, or lock Optuna bounds around the new min.
   - Strictly monotone → defaults are on a cliff; truth is past the sampled range. Don't trust `propose_bounds`; set Optuna search from physical priors.

2. **Full Phase-2 deliverable** — *the headline run*

   ```bash
   python -m examples.analysis.sysid.sensitivity --frozen-audit --vehicle-dyn --progress
   ```

   Produces all three rankings + coverage audit in one go (single `Rollout` reused). Outputs land in `figures/analysis/sysid/sensitivity/<bag>/`.

   **What each output tells us:**
   - `ranking.md` — which of the 8 tire candidates dominate `dLoss/dParam`. Drives Stage-1 vs Stage-2 split. Flat-line params get demoted; high-sensitivity ones stay. Top-of-page ⚠️ flags fire if `|α|` or `|κ|` coverage is below the default-Pacejka 95% saturation knee — that's a hard veto on identifying the corresponding saturation params (`tire_p_dy1` / `tire_p_ey1` / `tire_p_dx1` / `tire_p_ex1`) on this bag.
   - `ranking_frozen.md` — empirically confirms the freeze decisions (camber-coupled coefficients should show dLoss ≈ 0 across ±1.0 absolute; pure-shift terms may show some sensitivity but stay frozen for physical reasons — they're indistinguishable from steering bias / sensor offset). If a frozen param shows huge unexpected sensitivity, that's a signal something is off and merits investigation, not promotion.
   - `ranking_vehicle_dyn.md` — decides whether `I_z`, `I_y_w`, `sv_min`, `sv_max`, `a_max` join Stage 1. Smoke testing already showed `I_z` and `I_y_w` are highly sensitive on a stride-5s subset; full-stride is expected to confirm and likely promote both.
   - `coverage.png` — slip-angle / slip-ratio histograms with red knee lines. Confirms (or refutes) that the bag excites the saturation regime.
   - `sweep_total.png` / `sweep_per_channel.png` — visual confirmation of the ranking; reveal which channel each param drives (e.g. `tire_p_ky1` → yaw_rate, `tire_p_dx1` → a_x).

3. **Stage-3 r-coefficient sweep** — *defer*

   ```bash
   python -m examples.analysis.sysid.sensitivity --include-combined --progress
   ```

   Adds 16 combined-slip `tire_r_*` coefficients to the main sweep (~30 min extra). **Don't run this yet** — it's only useful once Phase 4 residuals show *systematic* combined-slip error (e.g. correlated `a_x` and `v_y` residuals under braking-while-cornering). Running it pre-emptively wastes time on rankings whose interpretation depends on later context.

**Exit criterion (met):** all three rankings + coverage histograms reviewed; Stage 1/2 search-space membership locked in `OPTUNA_SYS_ID_SENSITIVITY_REPORT.md`; `I_z`, `a_max`, `I_y_w` promoted into Stage 1; `tire_p_kx1`, `sv_min`, `sv_max` demoted; bounds for `dy1`/`dx1`/`ky1` taken from physical priors (auto-bounds did not contain the true minima).

### Phase 3 — Stage 1 Optuna study (STD, lateral-linear)

Sub-plan: `OPTUNA_SYS_ID_STUDY.md` (to be written).

Produces:
- `examples/analysis/sysid/search_spaces.py` — bounded distributions per param per stage.
- `examples/analysis/sysid/study.py` — Optuna objective, study creation, SQLite RDB storage, `MedianPruner` with per-window intermediate `trial.report`, seeded sampler, parallel-worker entry point.
- A CLI in the same module to run a study by name + stage + bag.
- Output YAML: `gymkhana/envs/params/f1tenth_std_optuna.yaml` (per-stage suffix while iterating, e.g. `_stage1`, `_stage2`).

**Exit criterion:** Stage 1 best loss < baseline; reproducible re-run within 1% on same seed.

### Phase 4 — Stage 2 Optuna study (STD, saturation)

Reuses Phase 3 infrastructure. Initial point = Stage 1 best. Sampler stays CMA-ES. No new sub-plan unless the search-space management diverges materially.

**Pre-Stage-2 investigation:** address the longitudinal-bias source flagged by the frozen audit (`tire_p_vx1` absorbs ~2 NMSE points on `a_x` despite physically belonging at zero). Candidates: VESC current-to-Fx map error, IMU `a_x` offset, drivetrain coastdown drag. If the bias can be calibrated outside the tire model before Stage 2, `tire_p_dx1` won't have to absorb it.

**Stage-2 sentinel:** if `dy1` or `dx1` saturate against their physical-prior bounds (μ_y or μ_x at 1.5), Stage 1 did not fully absorb the chassis residual — investigate (likely `I_z` upper bound, or unmodeled bias) before extending the bound.

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
  test_loss.py                       # Phase 1 ✅
  test_dataset.py                    # Phase 1 ✅
  test_rollout.py                    # Phase 1 ✅
  test_sensitivity.py                # Phase 2 ✅

gymkhana/envs/params/                # existing
  f1tenth_std_optuna.yaml            # NEW — Phase 3/4/5 output (per-stage suffixes while iterating)

docs/plan/
  OPTUNA_SYS_ID_OVERVIEW.md          # this document
  OPTUNA_SYS_ID_LOSS.md              # locked (Phase 1)
  OPTUNA_SYS_ID_SENSITIVITY.md       # locked (Phase 2)
  OPTUNA_SYS_ID_SENSITIVITY_REPORT.md  # to be written after the Phase-2 full-bag run
  OPTUNA_SYS_ID_STUDY.md             # to be written (Phase 3, reused for 4 and 5)
  OPTUNA_SYS_ID_CLI.md               # to be written (Phase 6)
```

## Cross-cutting concerns

These apply across phases and are called out here so individual sub-plans don't have to re-derive them.

- **`delta_init` is a hidden parameter, not just a warmup choice.** `delta_init = cmd_steer[t0]` is a guess; the servo's actual transient depends on slew rate (`sv_max`) and prior cmd history. Two cleaner options if the 0.2 s discard proves insufficient: (a) warm up with 0.3–0.5 s of pre-`t0` commands instead of cold-starting `delta`; (b) treat `delta_init` as a per-window nuisance variable. Note that `sv_max` and `sv_min` are themselves Phase-2 vehicle-dyn candidates — if the slew rate sweep shows non-trivial sensitivity, that's a signal the warmup-discard window may be inadequate.
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
