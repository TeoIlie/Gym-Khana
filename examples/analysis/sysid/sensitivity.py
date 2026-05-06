"""STD system identification — Phase 2 sensitivity analysis.

One-at-a-time multiplicative perturbation of PAC2002 parameters around
the YAML defaults, plus a real-signal coverage audit (slip-angle and
slip-ratio histograms with default-Pacejka saturation-knee annotations).

See docs/plan/OPTUNA_SYS_ID_SENSITIVITY.md for the locked design.

Usage
-----
Default run (Stage-1/2 sweep + frozen audit + coverage, drift bag):

    python -m examples.analysis.sysid.sensitivity --frozen-audit --progress

Common flags:
  --path <npz>          Different bag (default: rosbag2_2026_05_04-17_54_17_100Hz.npz)
  --include-combined    Also sweep Stage-3 r-coefficients (slow; Phase 5 prep)
  --frozen-audit        Run absolute-mode audit on FROZEN_PARAMS (review-only ranking)
  --vehicle-dyn         Sweep I_z, I_y_w, sv_min, sv_max, a_max — chassis/servo params
                        that were estimated rather than measured precisely. Decides
                        whether to add them to the Optuna search space.
  --wide-deltas         One-time scoping pass with DELTAS_WIDE (±90% to +200%) to check
                        whether the loss surface is basin-shaped near YAML defaults.
                        Use when starting params come from a different vehicle scale
                        (e.g. full-scale PAC2002 dict). Overrides --deltas if both set.
  --stride-s 2.0        Bigger stride → fewer windows → faster smoke test
  --no-mirror           Skip mirrored windows (sensitivity ranking is mirror-invariant)
  --deltas <csv>        Override the multiplicative δ ladder

Outputs land in figures/analysis/sysid/sensitivity/<bag_stem>/:
  ranking.md            Headline: coverage stats, ranked params, proposed Optuna bounds
  sweep.csv             Long-format table (one row per param × δ)
  sweep_total.png       Total loss vs δ, one line per param
  sweep_per_channel.png Per-channel (yaw_rate / v_y / a_x / v_x) breakdown
  coverage.png          α/κ histograms with saturation-knee annotations
  frozen_audit/...      Same outputs for the frozen audit (when --frozen-audit set)
  ranking_frozen.md     Frozen-param ranking (review-only — never promoted to search space)
  vehicle_dyn/...       Sweep plots for vehicle-dynamics params (when --vehicle-dyn set)
  ranking_vehicle_dyn.md  Ranking + proposed bounds for I_z, I_y_w, sv_*, a_max
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from examples.analysis.sysid.dataset import CHANNELS, Dataset
from examples.analysis.sysid.env import SYSID_PARAMS
from examples.analysis.sysid.loss import DEFAULT_WEIGHTS, dataset_loss
from examples.analysis.sysid.rollout import Rollout
from gymkhana.envs.dynamic_models.tire_model import formula_lateral, formula_longitudinal

DELTAS_DEFAULT: tuple[float, ...] = (-0.50, -0.25, -0.10, 0.0, +0.10, +0.25, +0.50)

# Wide-δ orientation ladder: confirms the loss surface is basin-shaped near
# defaults rather than a cliff. Use as a one-time scoping pass when YAML
# defaults are suspected of being well off (e.g. PAC2002 params lifted from
# a full-scale-vehicle reference). Asymmetric on the upside because shape /
# stiffness coefficients can plausibly be 2–3× under-estimated for 1/10
# F1TENTH tires; multiplicative δ=+2.0 means p ← 3·p₀.
DELTAS_WIDE: tuple[float, ...] = (-0.90, -0.70, -0.50, -0.25, 0.0, +0.25, +0.50, +1.00, +2.00)

STAGE12_CANDIDATES: tuple[str, ...] = (
    "tire_p_cx1",
    "tire_p_dx1",
    "tire_p_ex1",
    "tire_p_kx1",
    "tire_p_cy1",
    "tire_p_dy1",
    "tire_p_ey1",
    "tire_p_ky1",
)

# Vehicle dynamics / inertial / actuator-limit parameters. Distinct from the
# PAC2002 tire ladder above — they're physical properties of the chassis and
# servo, not tire coefficients — and their identifiability is mostly via
# transient response, not steady-state cornering. Treated as a separate sweep
# group with its own ladder, ranking, and Optuna search-space proposal.
#
#   - `I_z`: yaw moment of inertia. YAML comment admits it was estimated, not
#     measured (no bifilar pendulum experiment). Couples directly to yaw_rate
#     dynamics: ẏaw_rate = (lf·Fyf − lr·Fyr) / I_z. High identifiability.
#   - `I_y_w`: wheel inertia, approximated by `I = 0.5·m·r²` (uniform-disk
#     assumption — real wheels are not uniform disks). Couples to longitudinal
#     slip-ratio dynamics; identifiability comes through `a_x` transients
#     during accel/brake events. Will sharpen further once `omega` becomes a
#     loss channel (Phase 4 deferred upgrade).
#   - `sv_min`, `sv_max`: steering slew rate limits. Determine how fast δ can
#     change; affect the post-warmup yaw_rate response during fast steer
#     inputs. Treated as separate candidates because the simulator's
#     `steering_constraint` clips them independently.
#   - `a_max`: longitudinal accel cap. Only matters if the real car saturates
#     it during the bag — otherwise irrelevant. Sweep tells us which.
VEHICLE_DYN_CANDIDATES: tuple[str, ...] = (
    "I_z",
    "I_y_w",
    "sv_max",
    "sv_min",
    "a_max",
)

# Wider ladder than the tire sweep because some of these (I_y_w in particular)
# are known to be off by factors, not percentages. δ=+2.0 gives 3·p₀;
# δ=-0.7 gives 0.3·p₀. Spans roughly an order of magnitude.
VEHICLE_DYN_DELTAS: tuple[float, ...] = (
    -0.70,
    -0.50,
    -0.25,
    -0.10,
    0.0,
    +0.10,
    +0.25,
    +0.50,
    +1.00,
    +2.00,
)

STAGE3_CANDIDATES: tuple[str, ...] = (
    "tire_r_bx1",
    "tire_r_bx2",
    "tire_r_cx1",
    "tire_r_ex1",
    "tire_r_hx1",
    "tire_r_by1",
    "tire_r_by2",
    "tire_r_by3",
    "tire_r_cy1",
    "tire_r_ey1",
    "tire_r_hy1",
    "tire_r_vy1",
    "tire_r_vy3",
    "tire_r_vy4",
    "tire_r_vy5",
    "tire_r_vy6",
)

# Permanently-frozen per OVERVIEW §"Resolved design decisions". These are
# never promoted into the Optuna search space, but they ARE empirically
# audited via run_sweep with allow_frozen=True so the freeze decision is
# defensible by data, not assumption. Phase 2's frozen audit runs them with
# *absolute* perturbations (FROZEN_AUDIT_DELTAS below) since several have
# default = 0 (multiplicative would be a no-op) and the rest have tiny
# non-zero defaults where ±50% multiplicative is below the bag's noise floor.
FROZEN_PARAMS: frozenset[str] = frozenset(
    {
        "tire_p_dx3",
        "tire_p_dy3",
        "tire_p_hy3",
        "tire_p_vy3",  # camber-coupled (defaults: 0, -2.88, 0.031, -0.33)
        "tire_p_hx1",
        "tire_p_vx1",
        "tire_p_hy1",
        "tire_p_vy1",  # pure-shift (all tiny: ~1e-3 .. ~4e-2)
    }
)

# Absolute-delta ladders for the frozen audit. Scales chosen per-param to
# probe physically-meaningful regions:
#   - camber-coupled friction multipliers (`p_dx3`, `p_dy3`): dimensionless
#     coefficients on γ². With nominal camber ~0, even ±1.0 would barely
#     register; but the audit's job is to confirm that, so the ladder spans
#     ±{0.1, 0.5, 1.0} (last point is well past any plausible value).
#   - camber-coupled shift terms (`p_hy3`, `p_vy3`): unitless shifts in slip-
#     angle / Fz-normalized lateral force. ±{0.01, 0.05, 0.1} covers a band
#     of ~6° equivalent slip-angle shift at the upper end.
#   - pure-shift Shx/Shy (`p_hx1`, `p_hy1`): horizontal slip shift, ~radians
#     of α / unitless κ. ±{0.005, 0.02, 0.05} ≈ ±3° α at the upper end.
#   - pure-shift Svx/Svy (`p_vx1`, `p_vy1`): vertical force shift normalized
#     by Fz. ±{0.005, 0.02, 0.05} ≈ ±5% of μ at the upper end.
_CAMBER_MULT = (-1.0, -0.5, -0.1, 0.0, +0.1, +0.5, +1.0)
_CAMBER_SHIFT = (-0.10, -0.05, -0.01, 0.0, +0.01, +0.05, +0.10)
_PURE_SHIFT = (-0.05, -0.02, -0.005, 0.0, +0.005, +0.02, +0.05)
FROZEN_AUDIT_DELTAS: dict[str, tuple[float, ...]] = {
    "tire_p_dx3": _CAMBER_MULT,
    "tire_p_dy3": _CAMBER_MULT,
    "tire_p_hy3": _CAMBER_SHIFT,
    "tire_p_vy3": _CAMBER_SHIFT,
    "tire_p_hx1": _PURE_SHIFT,
    "tire_p_hy1": _PURE_SHIFT,
    "tire_p_vx1": _PURE_SHIFT,
    "tire_p_vy1": _PURE_SHIFT,
}


@dataclass(frozen=True)
class SweepRow:
    param: str
    delta: float  # interpretation depends on `mode` (mult fraction OR absolute additive step)
    mode: str  # "multiplicative" or "absolute"
    abs_value: float  # value actually fed to the env
    total: float  # weighted total NMSE (inf if rollout diverged)
    per_channel: dict[str, float]  # per-channel NMSE (each inf if diverged)


def _perturbed_params(base: dict, name: str, delta: float, mode: str) -> dict:
    p = deepcopy(base)
    if mode == "multiplicative":
        p[name] = base[name] * (1.0 + delta)
    elif mode == "absolute":
        p[name] = base[name] + delta
    else:
        raise ValueError(f"Unknown perturbation mode: {mode!r}")
    return p


def run_sweep(
    rollout: Rollout,
    dataset: Dataset,
    base_params: dict,
    candidates: Sequence[str] = STAGE12_CANDIDATES,
    deltas: Sequence[float] | dict[str, Sequence[float]] = DELTAS_DEFAULT,
    weights: dict[str, float] = DEFAULT_WEIGHTS,
    warmup_s: float = 0.2,
    mode: str = "multiplicative",
    allow_frozen: bool = False,
    progress: bool = False,
) -> list[SweepRow]:
    """Sweep each candidate's value across `deltas`.

    Modes:
      - "multiplicative": `p ← p0 · (1 + δ)`. Use for non-frozen Stage 1/2/3
        candidates whose defaults are far enough from 0 for a relative ladder
        to make sense.
      - "absolute": `p ← p0 + δ`. Use for the frozen audit, since several
        frozen params have default = 0 (multiplicative is a no-op) and the
        rest have tiny non-zero defaults where ±50% multiplicative is below
        the bag's noise floor.

    `deltas` may be a single sequence (applied to every candidate) or a
    dict mapping each candidate name to its own ladder. The frozen audit
    uses the dict form (FROZEN_AUDIT_DELTAS — different scales per param).

    Set `allow_frozen=True` to sweep params in `FROZEN_PARAMS` (frozen audit).
    Default raises so the main Stage-1/2 sweep can never accidentally include
    a frozen param.

    Re-uses a single `Rollout` instance (one GKEnv) — only `set_params` is
    called between iterations. Catches FloatingPointError from the rollout's
    NaN guard and records inf rows so divergence is visible in the ranking.
    """
    if mode not in ("multiplicative", "absolute"):
        raise ValueError(f"mode must be 'multiplicative' or 'absolute', got {mode!r}")
    for name in candidates:
        if name in FROZEN_PARAMS and not allow_frozen:
            raise ValueError(
                f"Candidate {name!r} is in FROZEN_PARAMS — pass allow_frozen=True to run a frozen-audit sweep."
            )
        if name not in base_params:
            raise KeyError(f"Candidate {name!r} not in base_params")

    if isinstance(deltas, dict):
        deltas_per_param = {name: tuple(deltas[name]) for name in candidates}
    else:
        deltas_tuple = tuple(deltas)
        deltas_per_param = {name: deltas_tuple for name in candidates}

    rows: list[SweepRow] = []
    baseline_total: float | None = None

    n_total = sum(len(v) for v in deltas_per_param.values())
    i = 0
    try:
        for name in candidates:
            for delta in deltas_per_param[name]:
                i += 1
                params = _perturbed_params(base_params, name, delta, mode)
                rollout.set_params(params)
                try:
                    total, per_channel = dataset_loss(rollout.run, dataset, weights=weights, warmup_s=warmup_s)
                except FloatingPointError:
                    total = math.inf
                    per_channel = {ch: math.inf for ch in CHANNELS}

                if delta == 0.0:
                    # Anchoring: every candidate's δ=0 row must produce identical loss.
                    if baseline_total is None:
                        baseline_total = total
                    else:
                        if not math.isclose(total, baseline_total, rel_tol=1e-12, abs_tol=1e-12):
                            raise AssertionError(
                                f"δ=0 baseline drift while sweeping {name!r}: "
                                f"got total={total!r}, expected {baseline_total!r}. "
                                "Param mutation across iterations — check deepcopy in _perturbed_params."
                            )

                rows.append(
                    SweepRow(
                        param=name,
                        delta=float(delta),
                        mode=mode,
                        abs_value=float(params[name]),
                        total=float(total),
                        per_channel={ch: float(per_channel[ch]) for ch in CHANNELS},
                    )
                )
                if progress:
                    print(f"[{i}/{n_total}] {name} ({mode[:3]}) δ={delta:+.4f} total={total:.4f}")
    finally:
        # Always restore base params so the rollout is reusable, even if an
        # iteration raised mid-sweep (e.g. AssertionError on baseline drift).
        rollout.set_params(base_params)
    return rows


def _baseline_row(rows: list[SweepRow]) -> SweepRow:
    for r in rows:
        if r.delta == 0.0:
            return r
    raise ValueError("No δ=0 row in sweep — cannot anchor ranking.")


def _git_short_commit() -> str:
    import subprocess

    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        )
    except Exception:
        return "unknown"


def rank_table(rows: list[SweepRow]):
    """Return a pandas DataFrame ranked by max |Δtotal| over each param's deltas.

    Columns: param, baseline_total, max_abs_dtotal, max_dtotal, min_dtotal,
             diverged (bool), n_diverged_deltas, plus per-channel max |Δ|.
    """
    import pandas as pd

    base = _baseline_row(rows)
    by_param: dict[str, list[SweepRow]] = {}
    for r in rows:
        by_param.setdefault(r.param, []).append(r)

    records = []
    for name, entries in by_param.items():
        finite = [e for e in entries if math.isfinite(e.total)]
        diverged = [e for e in entries if not math.isfinite(e.total)]
        deltas = [e.total - base.total for e in finite]
        max_dtotal = max(deltas) if deltas else math.nan
        min_dtotal = min(deltas) if deltas else math.nan
        max_abs = max(abs(d) for d in deltas) if deltas else math.inf
        per_channel_max_abs = {}
        for ch in CHANNELS:
            ch_deltas = [e.per_channel[ch] - base.per_channel[ch] for e in finite]
            per_channel_max_abs[f"max_abs_d_{ch}"] = max(abs(d) for d in ch_deltas) if ch_deltas else math.nan
        records.append(
            {
                "param": name,
                "baseline_total": base.total,
                "max_abs_dtotal": max_abs,
                "max_dtotal": max_dtotal,
                "min_dtotal": min_dtotal,
                "diverged": len(diverged) > 0,
                "n_diverged_deltas": len(diverged),
                **per_channel_max_abs,
            }
        )

    df = pd.DataFrame(records).sort_values("max_abs_dtotal", ascending=False).reset_index(drop=True)
    return df


def propose_bounds(
    rows: list[SweepRow],
    base_params: dict,
    headroom: float = 1.5,
    max_loss_ratio: float = 2.0,
) -> dict[str, tuple[float, float]]:
    """For each param, the largest symmetric ±δ where (a) total < ratio·baseline
    and (b) no NaN. Final bounds = `p0 · (1 ± headroom · δ_safe)`.

    These are *proposals* for the report to review, not authoritative.
    """
    base = _baseline_row(rows)
    threshold = base.total * max_loss_ratio
    by_param: dict[str, list[SweepRow]] = {}
    for r in rows:
        by_param.setdefault(r.param, []).append(r)

    out: dict[str, tuple[float, float]] = {}
    for name, entries in by_param.items():
        entries = sorted(entries, key=lambda e: e.delta)
        # Walk outward from δ=0 in each direction; stop at the first row that
        # exceeds threshold or diverges. The last "safe" δ on each side caps
        # the symmetric ±δ_safe window.
        max_safe_pos = 0.0
        for e in (e for e in entries if e.delta > 0):
            if math.isfinite(e.total) and e.total <= threshold:
                max_safe_pos = e.delta
            else:
                break
        max_safe_neg = 0.0
        for e in (e for e in reversed(entries) if e.delta < 0):
            if math.isfinite(e.total) and e.total <= threshold:
                max_safe_neg = e.delta
            else:
                break
        delta_safe = min(abs(max_safe_pos), abs(max_safe_neg))
        p0 = base_params[name]
        if delta_safe == 0.0:
            # Param diverges or blows past threshold even at the smallest sampled δ.
            # Emit a token-narrow bound so the report flags it.
            out[name] = (p0 * 0.95, p0 * 1.05) if p0 != 0 else (-1e-3, 1e-3)
        else:
            scale = headroom * delta_safe
            lo, hi = p0 * (1 - scale), p0 * (1 + scale)
            out[name] = (min(lo, hi), max(lo, hi))
    return out


# ---------------- Coverage audit ----------------


def compute_coverage(dataset: Dataset, params: dict) -> dict[str, np.ndarray]:
    """Real-signal slip-angle and slip-ratio coverage across all non-mirrored
    retained windows. Concatenates per-window arrays into one flat 1-D array per
    quantity so they can be histogrammed directly.

    α_front[k] = atan2(v_y + lf·yaw_rate, v_x) − delta_cmd[k]
    α_rear[k]  = atan2(v_y − lr·yaw_rate, v_x)
    κ_front[k] = (R_w·omega − v_x_front) / max(|v_x_front|, ε)
    κ_rear[k]  = (R_w·omega − v_x)       / max(|v_x|, ε)

    Front-wheel longitudinal velocity: v_x·cos(δ) + (v_y + lf·yaw_rate)·sin(δ).

    **Limitation:** `omega` is taken from `init_state[7]` (the VESC-seeded
    value at t0) and held constant across each window. The bag carries
    full `rs_core_speed[t0:end]` but `Window` does not currently store
    that slice. Within a 1.5s window during accel/braking, true omega can
    drift ~10–20% from the seeded value. This biases the κ histograms
    slightly but does not affect the qualitative "is coverage past the
    saturation knee?" verdict. Tighten by extending `Window` with a
    `real_omega` slice if/when the κ verdict is borderline.
    """
    lf = float(params["lf"])
    lr = float(params["lr"])
    R_w = float(params["R_w"])
    eps = 0.2  # m/s — below this, κ is dominated by v_x noise; mask separately.

    alpha_f_chunks: list[np.ndarray] = []
    alpha_r_chunks: list[np.ndarray] = []
    kappa_f_chunks: list[np.ndarray] = []
    kappa_r_chunks: list[np.ndarray] = []

    for w in dataset.windows:
        if w.is_mirrored:
            continue
        # cmd_steer is N-long, real signals are (N+1)-long. Use the N samples
        # aligned with cmds for slip-angle (steer applies during step k).
        n = len(w.cmd_steer)
        v_x = w.real_v_x[:n]
        v_y = w.real_v_y[:n]
        r = w.real_yaw_rate[:n]
        delta = w.cmd_steer[:n]
        omega = w.init_state[7]  # constant within each window — only seeded value available.

        alpha_f = np.arctan2(v_y + lf * r, v_x) - delta
        alpha_r = np.arctan2(v_y - lr * r, v_x)

        v_x_front = v_x * np.cos(delta) + (v_y + lf * r) * np.sin(delta)
        denom_f = np.maximum(np.abs(v_x_front), eps)
        denom_r = np.maximum(np.abs(v_x), eps)
        kappa_f = (R_w * omega - v_x_front) / denom_f
        kappa_r = (R_w * omega - v_x) / denom_r

        # Mask out samples where the longitudinal speed itself is below eps —
        # κ is meaningless there, and α suffers from atan2 noise too.
        speed_mask = v_x > eps
        alpha_f_chunks.append(alpha_f[speed_mask])
        alpha_r_chunks.append(alpha_r[speed_mask])
        kappa_f_chunks.append(kappa_f[speed_mask])
        kappa_r_chunks.append(kappa_r[speed_mask])

    return {
        "alpha_front": np.concatenate(alpha_f_chunks) if alpha_f_chunks else np.array([]),
        "alpha_rear": np.concatenate(alpha_r_chunks) if alpha_r_chunks else np.array([]),
        "kappa_front": np.concatenate(kappa_f_chunks) if kappa_f_chunks else np.array([]),
        "kappa_rear": np.concatenate(kappa_r_chunks) if kappa_r_chunks else np.array([]),
    }


def saturation_knees(params: dict, threshold: float = 0.95) -> dict[str, float]:
    """Slip values at which the default Pacejka pure-slip curves reach
    `threshold` of their peak (default 95%). Used as annotations on the
    coverage histograms — bag samples that never cross these knees imply
    the corresponding saturation params are unidentifiable.

    Computed numerically against `tire_model.formula_lateral` / `formula_longitudinal`
    so the knees stay consistent with whatever the env actually uses, even if the
    formulas are revised.
    """
    F_z = 1.0  # normalized; only the ratio Fy/(mu·Fz) matters.
    gamma = 0.0

    alpha_grid = np.linspace(0.0, np.deg2rad(40.0), 4001)
    fy = np.array([formula_lateral(a, gamma, F_z, params)[0] for a in alpha_grid])
    mu_y = float(params["tire_p_dy1"])
    fy_norm = np.abs(fy) / (mu_y * F_z)
    peak = float(np.max(fy_norm))
    target_y = threshold * peak
    # First crossing where fy_norm exceeds target_y after the peak's leading edge.
    above = np.where(fy_norm >= target_y)[0]
    alpha_sat = float(alpha_grid[above[0]]) if len(above) else math.nan

    kappa_grid = np.linspace(0.0, 1.0, 4001)
    fx = np.array([formula_longitudinal(k, gamma, F_z, params) for k in kappa_grid])
    mu_x = float(params["tire_p_dx1"])
    fx_norm = np.abs(fx) / (mu_x * F_z)
    peak_x = float(np.max(fx_norm))
    above_x = np.where(fx_norm >= threshold * peak_x)[0]
    kappa_sat = float(kappa_grid[above_x[0]]) if len(above_x) else math.nan

    return {
        "alpha_sat_rad": alpha_sat,
        "alpha_sat_deg": float(np.rad2deg(alpha_sat)) if not math.isnan(alpha_sat) else math.nan,
        "kappa_sat": kappa_sat,
    }


# ---------------- Output: CSV / markdown / plots ----------------


def rows_to_dataframe(rows: list[SweepRow]):
    """Long-format DataFrame: one row per (param, delta) with per-channel cols."""
    import pandas as pd

    records = []
    for r in rows:
        rec = {
            "param": r.param,
            "mode": r.mode,
            "delta": r.delta,
            "abs_value": r.abs_value,
            "total": r.total,
        }
        for ch in CHANNELS:
            rec[f"nmse_{ch}"] = r.per_channel[ch]
        records.append(rec)
    return pd.DataFrame.from_records(records)


def _write_ranking_md(
    out_path: str,
    bag_path: str,
    df_rank,
    bounds: dict[str, tuple[float, float]],
    base_params: dict,
    knees: dict[str, float],
    coverage: dict[str, np.ndarray],
    deltas: Sequence[float],
) -> None:
    import os

    base_total = float(df_rank["baseline_total"].iloc[0]) if len(df_rank) else math.nan

    lines: list[str] = []
    lines.append(f"# Sensitivity sweep — {os.path.basename(bag_path)}\n")
    lines.append(f"- bag: `{bag_path}`")
    lines.append(f"- git commit: `{_git_short_commit()}`")
    lines.append(f"- deltas: `{list(deltas)}`")
    lines.append(f"- baseline total (δ=0, all candidates): **{base_total:.4f}**\n")

    lines.append("## Coverage audit\n")
    a_f = coverage["alpha_front"]
    a_r = coverage["alpha_rear"]
    k_f = coverage["kappa_front"]
    k_r = coverage["kappa_rear"]

    def stats_deg(arr):
        if arr.size == 0:
            return "n/a"
        return f"min={np.rad2deg(arr.min()):.2f}°, max={np.rad2deg(arr.max()):.2f}°, |max|={np.rad2deg(np.abs(arr).max()):.2f}°"

    def stats_kappa(arr):
        if arr.size == 0:
            return "n/a"
        return f"min={arr.min():.3f}, max={arr.max():.3f}, |max|={np.abs(arr).max():.3f}"

    lines.append(f"- α_front: {stats_deg(a_f)}")
    lines.append(f"- α_rear:  {stats_deg(a_r)}")
    lines.append(f"- κ_front: {stats_kappa(k_f)}")
    lines.append(f"- κ_rear:  {stats_kappa(k_r)}")
    lines.append(f"- default-Pacejka α saturation knee (95% of peak Fy): **{knees['alpha_sat_deg']:.2f}°**")
    lines.append(f"- default-Pacejka κ saturation knee (95% of peak Fx): **{knees['kappa_sat']:.3f}**")

    cov_max_alpha_deg = max(
        float(np.rad2deg(np.abs(a_f).max())) if a_f.size else 0.0,
        float(np.rad2deg(np.abs(a_r).max())) if a_r.size else 0.0,
    )
    cov_max_kappa = max(
        float(np.abs(k_f).max()) if k_f.size else 0.0,
        float(np.abs(k_r).max()) if k_r.size else 0.0,
    )
    if cov_max_alpha_deg < knees["alpha_sat_deg"]:
        lines.append(
            f"- ⚠️ **|α| coverage ({cov_max_alpha_deg:.2f}°) below saturation knee "
            f"({knees['alpha_sat_deg']:.2f}°)** — `tire_p_dy1` / `tire_p_ey1` may be unidentifiable."
        )
    if cov_max_kappa < knees["kappa_sat"]:
        lines.append(
            f"- ⚠️ **|κ| coverage ({cov_max_kappa:.3f}) below saturation knee "
            f"({knees['kappa_sat']:.3f})** — `tire_p_dx1` / `tire_p_ex1` may be unidentifiable."
        )

    lines.append("\n## Ranking by max |Δtotal|\n")
    cols = list(df_rank.columns)
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, row in df_rank.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                cells.append(f"{v:.4f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("\n\n## Proposed bounds (auto-generated, review before locking)\n")
    lines.append("| param | default | lo | hi | rel. width |")
    lines.append("|---|---|---|---|---|")
    for name in df_rank["param"].tolist():
        lo, hi = bounds[name]
        p0 = base_params[name]
        rel = (hi - lo) / abs(p0) if p0 != 0 else math.inf
        lines.append(f"| `{name}` | {p0:.4g} | {lo:.4g} | {hi:.4g} | {rel:.2f} |")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _write_frozen_md(
    out_path: str,
    bag_path: str,
    df_rank,
    base_params: dict,
    deltas_per_param: dict[str, Sequence[float]],
) -> None:
    """Frozen-audit report. No proposed-bounds table — these params are not
    promoted into the Optuna search space. The ranking exists so the freeze
    decision is defensible by data, not assumption.
    """
    import os

    base_total = float(df_rank["baseline_total"].iloc[0]) if len(df_rank) else math.nan

    lines: list[str] = []
    lines.append(f"# Frozen-param audit — {os.path.basename(bag_path)}\n")
    lines.append(f"- bag: `{bag_path}`")
    lines.append(f"- git commit: `{_git_short_commit()}`")
    lines.append("- mode: **absolute perturbation** (`p ← p₀ + δ`) — multiplicative is")
    lines.append("  a no-op for default-zero params and below noise for tiny defaults.")
    lines.append(f"- baseline total (δ=0): **{base_total:.4f}**")
    lines.append("")
    lines.append("**These rankings are review-only.** Even params that show non-trivial")
    lines.append("sensitivity here remain frozen — they are excluded for physical reasons")
    lines.append("(camber-coupled on a zero-camber car; pure-shift terms unidentifiable from")
    lines.append("the bag's signals). Use this report to confirm the freeze decision is")
    lines.append("supported by data, *not* to promote params into the Optuna search space.\n")

    lines.append("## Per-param ladders\n")
    lines.append("| param | default | absolute deltas |")
    lines.append("|---|---|---|")
    for name in sorted(deltas_per_param):
        ladder = ", ".join(f"{d:+.3f}" for d in deltas_per_param[name])
        lines.append(f"| `{name}` | {base_params[name]:.4g} | {ladder} |")

    lines.append("\n## Ranking by max |Δtotal|\n")
    cols = list(df_rank.columns)
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, row in df_rank.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                cells.append(f"{v:.4f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _write_vehicle_dyn_md(
    out_path: str,
    bag_path: str,
    df_rank,
    bounds: dict[str, tuple[float, float]],
    base_params: dict,
    deltas: Sequence[float],
) -> None:
    """Vehicle-dynamics sweep report. Like `_write_ranking_md` but without
    the coverage audit (that's bag-level, already in the main report).

    These params *are* candidates for the Optuna search space (unlike frozen
    params), so the proposed-bounds table is included and meaningful.
    """
    import os

    base_total = float(df_rank["baseline_total"].iloc[0]) if len(df_rank) else math.nan

    lines: list[str] = []
    lines.append(f"# Vehicle-dynamics sweep — {os.path.basename(bag_path)}\n")
    lines.append(f"- bag: `{bag_path}`")
    lines.append(f"- git commit: `{_git_short_commit()}`")
    lines.append(f"- deltas: `{list(deltas)}`")
    lines.append(f"- baseline total (δ=0): **{base_total:.4f}**")
    lines.append(f"- candidates: {', '.join(f'`{c}`' for c in VEHICLE_DYN_CANDIDATES)}")
    lines.append("")
    lines.append("These parameters are physical properties of the chassis / servo, not")
    lines.append("PAC2002 tire coefficients. Several were estimated rather than measured")
    lines.append("(`I_z`, `I_y_w`); this sweep determines whether to add them to the")
    lines.append("Optuna search space. Coverage audit is in the main report — refer there")
    lines.append("for slip / κ identifiability checks.\n")

    lines.append("## Ranking by max |Δtotal|\n")
    cols = list(df_rank.columns)
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, row in df_rank.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            cells.append(f"{v:.4f}" if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("\n## Proposed bounds (auto-generated, review before locking)\n")
    lines.append("| param | default | lo | hi | rel. width |")
    lines.append("|---|---|---|---|---|")
    for name in df_rank["param"].tolist():
        lo, hi = bounds[name]
        p0 = base_params[name]
        rel = (hi - lo) / abs(p0) if p0 != 0 else math.inf
        lines.append(f"| `{name}` | {p0:.4g} | {lo:.4g} | {hi:.4g} | {rel:.2f} |")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _plot_sweeps(rows: list[SweepRow], base_total: float, out_dir: str) -> None:
    import os

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_param: dict[str, list[SweepRow]] = {}
    for r in rows:
        by_param.setdefault(r.param, []).append(r)

    # All rows in a single sweep share the same mode (run_sweep enforces this).
    mode = rows[0].mode if rows else "multiplicative"
    delta_label = (
        "δ (multiplicative perturbation, p ← p₀·(1+δ))"
        if mode == "multiplicative"
        else "δ (absolute perturbation, p ← p₀+δ)"
    )

    # --- Total loss vs delta, one line per param ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, entries in by_param.items():
        entries = sorted(entries, key=lambda e: e.delta)
        deltas = [e.delta for e in entries]
        totals = [e.total if math.isfinite(e.total) else np.nan for e in entries]
        ax.plot(deltas, totals, marker="o", label=name)
    ax.axhline(base_total, color="k", linestyle="--", linewidth=0.8, alpha=0.5, label=f"baseline ({base_total:.3f})")
    ax.set_xlabel(delta_label)
    ax.set_ylabel("Total weighted NMSE")
    ax.set_title("Sensitivity: total loss vs δ (per param)")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, "sweep_total.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)

    # --- Per-channel grid ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for ax, ch in zip(axes, CHANNELS):
        for name, entries in by_param.items():
            entries = sorted(entries, key=lambda e: e.delta)
            deltas = [e.delta for e in entries]
            ys = [e.per_channel[ch] if math.isfinite(e.per_channel[ch]) else np.nan for e in entries]
            ax.plot(deltas, ys, marker="o", label=name)
        ax.set_xlabel(delta_label)
        ax.set_ylabel(f"NMSE({ch})")
        ax.set_title(ch)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")
    fig.suptitle("Sensitivity: per-channel NMSE vs δ", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "sweep_per_channel.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_coverage(coverage: dict[str, np.ndarray], knees: dict[str, float], out_dir: str) -> None:
    import os

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    panels = [
        ("alpha_front", axes[0, 0], "α_front (deg)", True, knees["alpha_sat_deg"]),
        ("alpha_rear", axes[0, 1], "α_rear (deg)", True, knees["alpha_sat_deg"]),
        ("kappa_front", axes[1, 0], "κ_front", False, knees["kappa_sat"]),
        ("kappa_rear", axes[1, 1], "κ_rear", False, knees["kappa_sat"]),
    ]
    for key, ax, label, is_angle, knee in panels:
        arr = coverage[key]
        if arr.size == 0:
            ax.set_title(f"{label} — empty")
            continue
        plot_arr = np.rad2deg(arr) if is_angle else arr
        ax.hist(plot_arr, bins=80, color="C0", alpha=0.7)
        for sign in (-1, +1):
            ax.axvline(
                sign * knee,
                color="r",
                linestyle="--",
                linewidth=1.0,
                label=f"±knee = ±{knee:.2f}" if sign == +1 else None,
            )
        ax.set_xlabel(label)
        ax.set_ylabel("count")
        ax.set_title(f"{label} (N={arr.size})")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.suptitle("Coverage audit — saturation knees at 95% of peak default-Pacejka force", fontsize=14)
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, "coverage.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------- CLI ----------------


def _parse_deltas(s: str) -> tuple[float, ...]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    deltas = tuple(float(p) for p in parts)
    if 0.0 not in deltas:
        deltas = tuple(sorted(set(deltas) | {0.0}))
    return deltas


if __name__ == "__main__":
    import argparse
    import os
    import time
    from pathlib import Path

    from examples.analysis.sysid.dataset import load_dataset

    parser = argparse.ArgumentParser(description="Phase 2 sensitivity sweep + coverage audit")
    parser.add_argument(
        "--path",
        default="examples/analysis/bags/rosbag2_2026_05_04-17_54_17_100Hz.npz",
        help="Path to 100 Hz NPZ bag (default: drift bag from sensitivity plan)",
    )
    parser.add_argument(
        "--include-combined", action="store_true", help="Also sweep Stage-3 combined-slip r-coefficients"
    )
    parser.add_argument(
        "--frozen-audit",
        action="store_true",
        help="Also run a parallel sweep on FROZEN_PARAMS using absolute "
        "perturbations (FROZEN_AUDIT_DELTAS). Emits frozen_audit.csv and "
        "ranking_frozen.md alongside the main outputs. These rankings are "
        "for review only — frozen params are not promoted into the search space.",
    )
    parser.add_argument(
        "--vehicle-dyn",
        action="store_true",
        help="Also run a sweep on vehicle-dynamics params (I_z, I_y_w, "
        "sv_min, sv_max, a_max). These are estimated/approximated in the "
        "YAML rather than measured precisely; the sweep determines whether "
        "to add them to the Optuna search space. Emits vehicle_dyn.csv and "
        "ranking_vehicle_dyn.md.",
    )
    parser.add_argument("--deltas", type=str, default=",".join(f"{d}" for d in DELTAS_DEFAULT))
    parser.add_argument(
        "--wide-deltas",
        action="store_true",
        help="Use the DELTAS_WIDE preset (-0.9..+2.0) instead of --deltas. "
        "Orientation pass for confirming the loss surface is basin-shaped "
        "around YAML defaults — useful when defaults come from a different "
        "vehicle scale and may be far from the real tire's true values.",
    )
    parser.add_argument("--window-length-s", type=float, default=1.5)
    parser.add_argument("--stride-s", type=float, default=0.5)
    parser.add_argument("--min-speed", type=float, default=0.3)
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="Skip mirroring (sensitivity ranking is mirror-invariant; off saves time)",
    )
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()

    if args.wide_deltas:
        deltas = DELTAS_WIDE
        print(f"Using DELTAS_WIDE preset: {deltas}")
    else:
        deltas = _parse_deltas(args.deltas)
    candidates = STAGE12_CANDIDATES + (STAGE3_CANDIDATES if args.include_combined else ())

    print(f"Loading dataset: {args.path}")
    ds = load_dataset(
        args.path,
        window_length_s=args.window_length_s,
        stride_s=args.stride_s,
        min_speed=args.min_speed,
        mirror=not args.no_mirror,
    )
    n_orig = sum(not w.is_mirrored for w in ds.windows)
    print(f"windows: {len(ds.windows)} total ({n_orig} originals)")

    base_params = SYSID_PARAMS

    print(f"Sweeping {len(candidates)} params × {len(deltas)} deltas = {len(candidates) * len(deltas)} evals")
    stem = Path(args.path).stem
    # Wide-δ scoping passes go in a sibling folder so they don't clobber the
    # standard ranking artifacts. The outputs *look* identical to a normal
    # sweep but their interpretation is different (orientation, not bounds).
    suffix = "_wide" if args.wide_deltas else ""
    out_dir = os.path.join("figures", "analysis", "sysid", "sensitivity", stem + suffix)
    os.makedirs(out_dir, exist_ok=True)
    knees = saturation_knees(base_params)

    # Single Rollout instance reused across the main sweep and (optionally) the
    # frozen audit — building a GKEnv twice would waste ~5–10s of startup.
    with Rollout(params=base_params) as rollout:
        t0 = time.time()
        rows = run_sweep(
            rollout,
            ds,
            base_params,
            candidates=candidates,
            deltas=deltas,
            progress=args.progress,
        )
        coverage = compute_coverage(ds, base_params)
        print(f"sweep done in {time.time() - t0:.1f}s")

        df_rank = rank_table(rows)
        bounds = propose_bounds(rows, base_params)
        rows_to_dataframe(rows).to_csv(os.path.join(out_dir, "sweep.csv"), index=False)
        _write_ranking_md(
            os.path.join(out_dir, "ranking.md"),
            args.path,
            df_rank,
            bounds,
            base_params,
            knees,
            coverage,
            deltas,
        )
        _plot_sweeps(rows, base_total=df_rank["baseline_total"].iloc[0], out_dir=out_dir)
        _plot_coverage(coverage, knees, out_dir)
        print(f"outputs written to {out_dir}/")
        print(df_rank.to_string(index=False))

        if args.frozen_audit:
            frozen_candidates = tuple(sorted(FROZEN_PARAMS))
            n_frozen_evals = sum(len(FROZEN_AUDIT_DELTAS[p]) for p in frozen_candidates)
            print(f"\nFrozen audit: {len(frozen_candidates)} params × variable ladder = {n_frozen_evals} evals")
            t1 = time.time()
            frozen_rows = run_sweep(
                rollout,
                ds,
                base_params,
                candidates=frozen_candidates,
                deltas=FROZEN_AUDIT_DELTAS,
                mode="absolute",
                allow_frozen=True,
                progress=args.progress,
            )
            print(f"frozen audit done in {time.time() - t1:.1f}s")
            df_frozen = rank_table(frozen_rows)
            rows_to_dataframe(frozen_rows).to_csv(os.path.join(out_dir, "frozen_audit.csv"), index=False)
            _write_frozen_md(
                os.path.join(out_dir, "ranking_frozen.md"),
                args.path,
                df_frozen,
                base_params,
                FROZEN_AUDIT_DELTAS,
            )
            _plot_sweeps(
                frozen_rows,
                base_total=df_frozen["baseline_total"].iloc[0],
                out_dir=os.path.join(out_dir, "frozen_audit"),
            )
            print(df_frozen.to_string(index=False))

        if args.vehicle_dyn:
            n_vd_evals = len(VEHICLE_DYN_CANDIDATES) * len(VEHICLE_DYN_DELTAS)
            print(
                f"\nVehicle-dynamics sweep: {len(VEHICLE_DYN_CANDIDATES)} params × "
                f"{len(VEHICLE_DYN_DELTAS)} deltas = {n_vd_evals} evals"
            )
            t2 = time.time()
            vd_rows = run_sweep(
                rollout,
                ds,
                base_params,
                candidates=VEHICLE_DYN_CANDIDATES,
                deltas=VEHICLE_DYN_DELTAS,
                progress=args.progress,
            )
            print(f"vehicle-dyn sweep done in {time.time() - t2:.1f}s")
            df_vd = rank_table(vd_rows)
            vd_bounds = propose_bounds(vd_rows, base_params)
            rows_to_dataframe(vd_rows).to_csv(os.path.join(out_dir, "vehicle_dyn.csv"), index=False)
            _write_vehicle_dyn_md(
                os.path.join(out_dir, "ranking_vehicle_dyn.md"),
                args.path,
                df_vd,
                vd_bounds,
                base_params,
                VEHICLE_DYN_DELTAS,
            )
            _plot_sweeps(
                vd_rows,
                base_total=df_vd["baseline_total"].iloc[0],
                out_dir=os.path.join(out_dir, "vehicle_dyn"),
            )
            print(df_vd.to_string(index=False))
