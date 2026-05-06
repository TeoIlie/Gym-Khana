# Plan: Sensitivity Analysis (Phase 2)

## Context

Phase 1 produced a working `dataset_loss(rollout_fn, dataset)` pipeline. The Phase-1 baseline was recorded on `circle_Apr6_100Hz.npz` (steady ~1 m/s circle, **total = 5.0224**), but that bag is too tame for Phase 2: it never excites high slip-angle / high slip-ratio regimes, so the saturation params we most need to rank would all look indistinguishable from noise.

**Phase 2 uses `examples/analysis/bags/rosbag2_2026_05_04-17_54_17_100Hz.npz` as its primary bag** — this recording captures aggressive cornering / drifting maneuvers with the side-slip and longitudinal-slip variation needed to actually expose `dy1 / ey1 / dx1 / ex1` sensitivity. The Phase-1 circle-bag baseline is retained only as a regression sanity check (the Phase-1 invariants must still pass on it; nothing in Phase 2 changes that).

Before launching any Optuna study (Phase 3+), we need to:

1. Empirically rank PAC2002 parameters by how much each one moves `dataset_loss` when perturbed around the YAML default.
2. Audit whether the bag dataset actually excites the regimes those parameters control (slip-angle / slip-ratio coverage).
3. Lock the Stage-1/2/3 search-space membership and per-parameter bounds for the study plan.

This plan covers **only the sensitivity sweep, the coverage audit, and the resulting param-set lockdown**. Optuna study mechanics, search-space distributions, and parallel-worker plumbing live in `OPTUNA_SYS_ID_STUDY.md` (Phase 3).

## Goals (and explicit non-goals)

**Goals:**
- Produce a ranked CSV/markdown table of `dLoss/dParam` magnitudes per channel and total, around YAML defaults.
- **Empirically audit every permanently-frozen param** with absolute perturbations to verify by data — not assumption — that they are negligible / unidentifiable. Output as a parallel review-only ranking; frozen params are still not promoted into the search space, but the freeze decision becomes defensible.
- Produce slip-angle and slip-ratio coverage histograms across the dataset, computed from real (Vicon + VESC) signals — not sim.
- Decide, in writing, the final membership of Stage 1 (linear) and Stage 2 (saturation) parameter subsets, with bounds.
- Catch identifiability failures *before* burning Optuna trials on dead dimensions.

**Non-goals:**
- Sobol indices / pairwise interaction sweeps. OAT misses interactions; we accept that — Optuna handles in-stage interactions, and the cost of full Sobol (~10× the OAT budget) is not justified at this stage.
- Tuning bounds for Stage-3 combined-slip params. Stage 3 is conditional on Phase 4 residuals, so its sensitivity sweep can be deferred and run as a one-off at that point.
- Multi-bag aggregation. Single-bag (`rosbag2_2026_05_04-17_54_17_100Hz.npz`) is enough to rank; cross-bag stability is a Phase 6 concern.

## Strategy

### Method: one-at-a-time (OAT) perturbation

For each candidate param `p`, evaluate `dataset_loss` with `p ← p_default · (1 + δ)` for `δ ∈ {-0.5, -0.25, -0.10, 0, +0.10, +0.25, +0.50}`. All other params held at YAML default. Re-use the same `Dataset` and the same `Rollout` instance across all evaluations — only `set_params` changes between sweeps.

Rationale: cheap (`P_candidates × 7 ≈ 100–150` evaluations; per-eval cost depends on this bag's window count — likely longer than the circle bag's ~4.5 s; budget 15–30 min single-core and re-confirm after the first run), easy to interpret, and sufficient for the prune-and-rank goal. Optuna will discover interactions inside each stage on its own.

Special-cased params:
- Sign-bearing coefficients (e.g. `tire_p_ky1 = -21.92`) — multiplicative perturbation preserves sign. ✅
- Defaults exactly at zero (e.g. `tire_p_dx3 = 0`, frozen-camber list) — multiplicative `(1+δ)` is a no-op. Handled by the **frozen audit** (below) using absolute perturbation `p ← p_default + δ` with per-param δ scales.
- Defaults very close to zero but non-zero (e.g. `tire_p_hx1 = 0.0012297`) — multiplicative ±50% leaves the absolute change below the bag's noise floor. Also routed to the frozen audit so we get a real signal instead of a flat line.

### Frozen audit (parallel pass, absolute mode)

The 8 permanently-frozen params (4 camber-coupled, 4 pure-shift) are *not* swept multiplicatively — they're routed through a parallel audit pass with absolute perturbation `p ← p₀ + δ`, separate output artifacts (`frozen_audit.csv`, `ranking_frozen.md`, `frozen_audit/sweep_*.png`), and `allow_frozen=True` opt-in. The ranking is **review-only**: a frozen param is never promoted into the Optuna search space, regardless of what `dLoss/dParam` it shows here. The audit exists so the freeze decisions in the OVERVIEW (camber on a zero-camber car; pure-shift unidentifiable) are defensible by data.

Per-param absolute-delta ladders:

| Param group | Defaults | Ladder (absolute δ) | Rationale |
|---|---|---|---|
| `tire_p_dx3`, `tire_p_dy3` | 0 | ±{0.1, 0.5, 1.0} | Friction-multiplier coefficients on γ². ±1.0 is well past any plausible value; if dLoss stays ~0 across this band, freeze is confirmed. |
| `tire_p_hy3`, `tire_p_vy3` | 0.031, -0.33 | ±{0.01, 0.05, 0.1} | Camber-shift terms; ±0.1 ≈ ±6° equivalent slip-angle shift at the upper end. |
| `tire_p_hx1`, `tire_p_hy1` | 0.0012, 0.0027 | ±{0.005, 0.02, 0.05} | Horizontal slip shifts (Shx/Shy); ±0.05 ≈ ±3° α at the upper end. |
| `tire_p_vx1`, `tire_p_vy1` | -8.8e-6, 0.037 | ±{0.005, 0.02, 0.05} | Vertical force shifts (Svx/Fz, Svy/Fz); ±0.05 ≈ ±5% of μ. |

### Candidate parameter list

From `gymkhana/envs/params/f1tenth_std.yaml` PAC2002 block, exclude permanently-frozen params (per OVERVIEW §"Resolved design decisions"):

**Frozen (not in main sweep; audited separately via the frozen-audit pass):**
`tire_p_dx3`, `tire_p_dy3`, `tire_p_hy3`, `tire_p_vy3` (camber-coupled),
`tire_p_hx1`, `tire_p_vx1`, `tire_p_hy1`, `tire_p_vy1` (pure-shift; `_vy1` included because for a symmetric tire it should be ≈0 and the bag can't constrain it).

**Swept (Stage 1/2 candidates) — 8 params:**
`tire_p_cx1`, `tire_p_dx1`, `tire_p_ex1`, `tire_p_kx1`,
`tire_p_cy1`, `tire_p_dy1`, `tire_p_ey1`, `tire_p_ky1`.

**Swept (Stage 3, deferred — combined slip):**
`tire_r_bx1`, `tire_r_bx2`, `tire_r_cx1`, `tire_r_ex1`, `tire_r_hx1`,
`tire_r_by1`, `tire_r_by2`, `tire_r_by3`, `tire_r_cy1`, `tire_r_ey1`, `tire_r_hy1`,
`tire_r_vy1`, `tire_r_vy3`, `tire_r_vy4`, `tire_r_vy5`, `tire_r_vy6`.

Phase 2 sweeps the **8 Stage-1/2 candidates** in the main pass and **all 8 frozen params** in the audit pass. A `--include-combined` CLI flag enables the Stage-3 r-coefficients in the main sweep for the conditional Phase 5 work; not run as part of the standard Phase-2 deliverable. A `--frozen-audit` flag enables the parallel absolute-mode pass over `FROZEN_PARAMS`.

### Coverage audit

Compute, across the dataset (real signals only, all retained windows pre-mirror):
- `α_front[k] = atan2(v_y + lf·yaw_rate, v_x) − delta_cmd[k]`
- `α_rear[k]  = atan2(v_y − lr·yaw_rate, v_x)`
- `κ_front[k] = (R_w·omega_front − v_x_front) / max(|v_x_front|, ε)` where `omega_*` from VESC `rs_core_speed/R_w` (AWD). Equivalent for rear.
- (Effective `v_x_front = v_x·cos(δ) + (v_y + lf·yaw_rate)·sin(δ)`; for rear use `v_x` directly since rear isn't steered.)

Histogram each. Annotate the plots with the saturation knees of the default Pacejka curves (`α` where `Fy / (μ·Fz)` reaches 0.95 — solved analytically from the default `B/C/D/E`). **If `|α|`-coverage never crosses ~5°, flag in the report that `tire_p_dy1` / `tire_p_ey1` are unidentifiable on this dataset and must not be unfrozen for Stage 2 until a richer bag is recorded.** Same logic for `κ` vs `tire_p_dx1` / `tire_p_ex1`.

This is the most important deliverable of Phase 2. The OAT sweep can rank dead params; only the coverage audit can rule out a *category* of failure.

## Files to create

1. **`examples/analysis/sysid/sensitivity.py`** — sweep runner, coverage computation, plotting, CLI.
2. **`tests/sysid/test_sensitivity.py`** — invariants on the sweep plumbing.
3. **`figures/analysis/sysid/sensitivity/<bag_stem>/`** — output directory for plots + CSV (created by CLI; not committed).
4. **`docs/plan/OPTUNA_SYS_ID_SENSITIVITY_REPORT.md`** — short results write-up authored *after* the sweep runs, locking the Stage 1/2 search-space membership. Template lives in this plan; filled in once data is in hand.

## Module design — `sensitivity.py`

```python
DELTAS_DEFAULT = (-0.50, -0.25, -0.10, 0.0, +0.10, +0.25, +0.50)

STAGE12_CANDIDATES = (
    "tire_p_cx1", "tire_p_dx1", "tire_p_ex1", "tire_p_kx1",
    "tire_p_cy1", "tire_p_dy1", "tire_p_ey1", "tire_p_ky1",
)
STAGE3_CANDIDATES = (
    "tire_r_bx1", "tire_r_bx2", "tire_r_cx1", "tire_r_ex1", "tire_r_hx1",
    "tire_r_by1", "tire_r_by2", "tire_r_by3", "tire_r_cy1", "tire_r_ey1",
    "tire_r_hy1", "tire_r_vy1", "tire_r_vy3", "tire_r_vy4", "tire_r_vy5", "tire_r_vy6",
)

@dataclass(frozen=True)
class SweepRow:
    param: str
    delta: float          # multiplicative fraction, 0.0 == baseline
    abs_value: float      # the value actually used in the trial
    total: float
    per_channel: dict[str, float]

def run_sweep(
    rollout: Rollout,
    dataset: Dataset,
    base_params: dict,
    candidates: Sequence[str] = STAGE12_CANDIDATES,
    deltas: Sequence[float] = DELTAS_DEFAULT,
) -> list[SweepRow]: ...

def rank_table(rows: list[SweepRow]) -> pd.DataFrame:
    """Return a DataFrame ranked by max |total - baseline_total| over deltas."""

def compute_coverage(dataset: Dataset, params: dict) -> dict[str, np.ndarray]:
    """Returns {'alpha_front': ..., 'alpha_rear': ..., 'kappa_front': ..., 'kappa_rear': ...}
    flat-concatenated across all non-mirrored retained windows."""

def saturation_knees(params: dict) -> dict[str, float]:
    """Returns {'alpha_sat_deg': ..., 'kappa_sat': ...} — slip at which the
    default Pacejka pure-slip curve hits 0.95 of peak. Annotates coverage plots."""
```

CLI surface:
```
python -m examples.analysis.sysid.sensitivity \
  --path examples/analysis/bags/rosbag2_2026_05_04-17_54_17_100Hz.npz \
  [--include-combined] \
  [--frozen-audit] \
  [--deltas -0.5,-0.25,-0.1,0,0.1,0.25,0.5]
```
Outputs (all under `figures/analysis/sysid/sensitivity/<stem>/`):
- Main sweep: `sweep.csv`, `ranking.md`, `sweep_total.png`, `sweep_per_channel.png`, `coverage.png`.
- Frozen audit (when `--frozen-audit`): `frozen_audit.csv`, `ranking_frozen.md`, `frozen_audit/sweep_total.png`, `frozen_audit/sweep_per_channel.png`. The frozen-audit ranking explicitly states the rankings are review-only and does **not** emit a proposed-bounds table.

## Implementation notes

- **Re-use the Phase-1 stack.** `sensitivity.py` is a thin orchestrator: `load_dataset(...) → Rollout(...) → loop[set_params, dataset_loss] → pandas → matplotlib`. No new dynamics math.
- **Single Rollout instance.** Build one `Rollout` outside the sweep loop. Each iteration calls `rollout.set_params(perturbed)` then `dataset_loss(rollout.run, dataset)`. Mirrors the pattern Phase 3's worker will use.
- **NaN guard.** `Rollout.run` already raises `FloatingPointError` on non-finite sim. Catch it in the sweep loop and record `total = inf`, `per_channel = {ch: inf}`. Plot code must mask infs before plotting (otherwise matplotlib autoscale dies). Surface these rows prominently in the ranking table — divergence at `+50%` perturbation is itself useful information for bound selection.
- **Baseline anchoring.** Always include `δ = 0` as the first row for every param. Sanity check: every param's `δ=0` row must produce *exactly the same* `total` (and per-channel) loss on this bag — the absolute number is bag-dependent (will not match the circle-bag's 5.0224), but it must be identical across all candidate params at `δ=0`. Assert this in the sweep loop — it catches accidental param mutations between iterations. Also log the `δ=0` baseline at the top of `ranking.md` as the reference loss for this bag.
- **Bounds proposal (auto-generated, manually reviewed).** From the sweep, compute the `δ_safe` interval per param: largest symmetric ±δ around 0 where (a) total loss < 2× baseline AND (b) no NaN. Propose Stage 1/2 bounds as `p_default · (1 + 1.5·δ_safe)` (give Optuna some headroom past what we sampled). These are *proposals*; the report locks them after human review.
- **Reproducibility.** No RNG involved on the sweep side, but `Rollout` resets via `env.reset(seed=...)`. The Phase-1 determinism test confirmed bit-identical sim signals across reset seeds; rely on that. Still, log the bag SHA + git commit at the top of `ranking.md` so the report is reproducible.

## `tests/sysid/test_sensitivity.py`

Five focused tests:

1. **Baseline anchoring (`test_delta_zero_reproduces_baseline`):** for every candidate param, the `δ = 0` `SweepRow.total` equals `dataset_loss(rollout.run, dataset)` to within 1e-12. Catches accidental param mutation across iterations.
2. **Frozen params not in candidates (`test_frozen_params_excluded_from_candidates`):** static guarantee — `STAGE12_CANDIDATES ∩ FROZEN_PARAMS == ∅`. Protects against accidentally re-adding a frozen param to the main sweep list.
3. **Coverage geometry (`test_coverage_signs_constant_yaw_rate`):** on a synthetic constant-`yaw_rate` window with `delta=0`, `α_front` and `α_rear` have opposite signs and `α_front − α_rear ≈ atan2(lf·r, v_x) − atan2(−lr·r, v_x)`. Catches sign / lf-lr swap bugs in the slip computation — these would silently invalidate the coverage histograms.
4. **Frozen-audit completeness (`test_frozen_audit_deltas_cover_every_frozen_param`):** every member of `FROZEN_PARAMS` has a `FROZEN_AUDIT_DELTAS` ladder (and vice versa), and every ladder includes `δ=0`. Adding a new frozen param without an audit ladder must fail loudly, not silently skip the audit.
5. **Frozen opt-in gate (`test_run_sweep_rejects_frozen_param_without_opt_in`):** calling `run_sweep` with a frozen param raises `ValueError` unless `allow_frozen=True`. Hard-stops accidental contamination of the main sweep.

No NaN-guard test — that path is exercised by `test_rollout.py::test_run_raises_on_nonfinite` already, and `sensitivity.py` only forwards the exception.

## Exit criteria

Phase 2 is done when:

1. Sweep CSV + ranking + coverage plots committed under `figures/analysis/sysid/sensitivity/rosbag2_2026_05_04-17_54_17_100Hz/`.
2. `OPTUNA_SYS_ID_SENSITIVITY_REPORT.md` written, containing:
   - The ranked table.
   - Coverage plots inline with verdict (e.g. *"max |α_rear| = 3.2°, default Pacejka knee at 6.8° — saturation params not identifiable on this bag"*).
   - **Frozen-audit summary**: confirm dLoss ≈ 0 for camber-coupled params; for any frozen param showing non-trivial sensitivity, restate why it remains frozen (physical reason, not just empirical).
   - **Locked Stage 1 candidate list + bounds.**
   - **Locked Stage 2 candidate list + bounds** (or explicit deferral if coverage fails).
   - Any unfrozen params that should be re-frozen based on the sweep, with reasoning.
3. The three invariants in `test_sensitivity.py` pass.
4. Single-core runtime of the full Phase-2 sweep is under ~45 minutes on this bag (sanity check on the cost model — drift bag has more windows than the circle bag, so per-eval cost is higher; if it blows past this by a wide margin, something is wrong with `set_params` and Phase 3's parallel cost estimates are also wrong). Re-tune this budget after the first end-to-end run.

## Open questions (resolve during implementation, not before)

- Whether to also sweep `lf`, `lr`, `m`, `I_z`, `h_s`, `R_w` as a control. These are "measurable" per the YAML comments but `I_z` in particular was estimated, not measured. Cheap to add; defer the decision until after the Pacejka sweep — if the Pacejka residuals look clean, the geometric params are next on the list.
- Whether `δ = ±0.5` is too aggressive for `tire_p_ky1` (already large-magnitude negative; ±50% may push into a regime the integrator can't handle). If divergence is widespread at ±50% across multiple params, drop the outermost δ and re-run — divergence is information, but only the first time.
- Whether the report needs cross-bag corroboration before locking bounds. Lean: no, single-bag is enough for *ranking*; multi-bag stability is what the held-out validation set in Phase 6 is for. Revisit only if Stage 1 fails to beat baseline.
