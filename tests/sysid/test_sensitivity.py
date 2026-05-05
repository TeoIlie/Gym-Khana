"""Tests for examples.analysis.sysid.sensitivity (Phase 2 sweep plumbing).

Three focused invariants per the plan:
  1. Baseline anchoring — δ=0 row reproduces the unswept dataset_loss for every candidate.
  2. Frozen params not swept — STAGE12_CANDIDATES never overlaps FROZEN_PARAMS.
  3. Coverage geometry — α_front − α_rear ≈ (lf+lr)·yaw_rate / v_x with correct signs.

The sweep test uses the real circle bag (cheap; rollout fixture is module-scoped).
The other two are pure-function tests with no env construction.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from examples.analysis.sysid.dataset import Window
from examples.analysis.sysid.env import SYSID_PARAMS
from examples.analysis.sysid.loss import dataset_loss
from examples.analysis.sysid.sensitivity import (
    FROZEN_AUDIT_DELTAS,
    FROZEN_PARAMS,
    STAGE12_CANDIDATES,
    VEHICLE_DYN_CANDIDATES,
    VEHICLE_DYN_DELTAS,
    compute_coverage,
    run_sweep,
)

BAG_PATH = "examples/analysis/bags/circle_Apr6_100Hz.npz"


# ---------------- pure-function tests ----------------


def test_frozen_params_excluded_from_candidates():
    """Static guarantee — never let a frozen param sneak into the sweep list."""
    overlap = set(STAGE12_CANDIDATES) & set(FROZEN_PARAMS)
    assert overlap == set(), f"Frozen params present in STAGE12_CANDIDATES: {overlap}"


def test_frozen_audit_deltas_cover_every_frozen_param():
    """The frozen audit must define an absolute-delta ladder for every frozen
    param, otherwise a frozen param would be silently skipped from the audit.
    """
    missing = set(FROZEN_PARAMS) - set(FROZEN_AUDIT_DELTAS.keys())
    extra = set(FROZEN_AUDIT_DELTAS.keys()) - set(FROZEN_PARAMS)
    assert missing == set(), f"Frozen params without audit ladder: {missing}"
    assert extra == set(), f"Audit ladders for non-frozen params: {extra}"
    for name, ladder in FROZEN_AUDIT_DELTAS.items():
        assert 0.0 in ladder, f"{name!r} ladder missing δ=0 anchor"


def test_vehicle_dyn_candidates_disjoint_and_present():
    """Vehicle-dynamics candidates must not overlap with tire candidates or
    frozen params, and must all exist in SYSID_PARAMS. Catches typos and
    accidental cross-pollination between the three sweep groups.
    """
    assert set(VEHICLE_DYN_CANDIDATES) & set(STAGE12_CANDIDATES) == set()
    assert set(VEHICLE_DYN_CANDIDATES) & set(FROZEN_PARAMS) == set()
    for name in VEHICLE_DYN_CANDIDATES:
        assert name in SYSID_PARAMS, f"{name!r} not in SYSID_PARAMS — typo?"
    assert 0.0 in VEHICLE_DYN_DELTAS, "VEHICLE_DYN_DELTAS missing δ=0 anchor"


def test_run_sweep_rejects_frozen_param_without_opt_in():
    """Default behavior: passing a frozen param raises. Protects the main
    Stage-1/2 sweep from accidentally including a frozen param."""
    import pytest as _pytest

    from examples.analysis.sysid.sensitivity import run_sweep as _run_sweep

    with _pytest.raises(ValueError, match="FROZEN_PARAMS"):
        _run_sweep(
            rollout=None,  # never reached — guard fires first
            dataset=None,
            base_params=SYSID_PARAMS,
            candidates=("tire_p_dx3",),
            deltas=(0.0,),
        )


def test_coverage_signs_constant_yaw_rate():
    """On a synthetic constant-yaw-rate / zero-steer / forward-velocity window:
        α_front − α_rear  ≈  (lf + lr) · yaw_rate / v_x
    with α_front and α_rear opposite signs (front-of-CG point sees +v_y, rear sees -v_y).

    Catches lf/lr swap or sign flip in compute_coverage.
    """
    lf = float(SYSID_PARAMS["lf"])
    lr = float(SYSID_PARAMS["lr"])
    n = 50
    v_x = 3.0
    yaw_rate = 1.0  # rad/s, positive (left turn)
    v_y = 0.0
    delta = 0.0

    init_state = np.array(
        [0.0, 0.0, delta, v_x, 0.0, yaw_rate, 0.0, v_x / SYSID_PARAMS["R_w"], v_x / SYSID_PARAMS["R_w"]]
    )
    w = Window(
        t0_idx=0,
        init_state=init_state,
        cmd_steer=np.full(n, delta),
        cmd_speed=np.full(n, v_x),
        real_v_x=np.full(n + 1, v_x),
        real_v_y=np.full(n + 1, v_y),
        real_yaw_rate=np.full(n + 1, yaw_rate),
        real_a_x=np.zeros(n + 1),
        is_mirrored=False,
    )

    class _DS:
        windows = [w]

    cov = compute_coverage(_DS(), SYSID_PARAMS)
    a_f = cov["alpha_front"]
    a_r = cov["alpha_rear"]
    assert a_f.size == n and a_r.size == n

    # Front: atan2(0 + lf·r, v_x); rear: atan2(0 - lr·r, v_x). With r > 0:
    # α_front > 0, α_rear < 0, opposite signs.
    assert np.all(a_f > 0), "α_front should be positive for +yaw_rate, +v_x"
    assert np.all(a_r < 0), "α_rear should be negative for +yaw_rate, +v_x"

    # Exact (no small-angle approx): atan2(lf·r, v_x) − atan2(−lr·r, v_x).
    expected_diff = np.arctan2(lf * yaw_rate, v_x) - np.arctan2(-lr * yaw_rate, v_x)
    actual_diff = float(np.mean(a_f - a_r))
    np.testing.assert_allclose(actual_diff, expected_diff, rtol=1e-9)


# ---------------- baseline anchoring (uses rollout fixture) ----------------


pytestmark_bag = pytest.mark.skipif(not os.path.exists(BAG_PATH), reason=f"requires test bag {BAG_PATH}")


@pytest.fixture(scope="module")
def small_dataset():
    """Tiny, fast dataset: stride large enough to keep ≤2 windows on the circle bag."""
    from examples.analysis.sysid.dataset import load_dataset

    return load_dataset(BAG_PATH, mirror=False, window_length_s=1.0, stride_s=5.0)


@pytest.fixture(scope="module")
def rollout_fixture():
    from examples.analysis.sysid.rollout import Rollout

    r = Rollout()
    yield r
    r.close()


@pytestmark_bag
def test_delta_zero_reproduces_baseline(rollout_fixture, small_dataset):
    """For every candidate param, δ=0 must yield the same loss as a clean dataset_loss
    call. Catches accidental param mutation across iterations (would surface as
    drifting baselines for later candidates).
    """
    rollout_fixture.set_params(SYSID_PARAMS)
    expected_total, expected_per_channel = dataset_loss(rollout_fixture.run, small_dataset)

    rows = run_sweep(
        rollout_fixture,
        small_dataset,
        SYSID_PARAMS,
        candidates=STAGE12_CANDIDATES,
        deltas=(0.0,),
    )
    assert len(rows) == len(STAGE12_CANDIDATES)
    for row in rows:
        assert row.delta == 0.0
        assert row.total == pytest.approx(expected_total, rel=1e-12, abs=1e-12), (
            f"baseline drift on {row.param}: {row.total} vs {expected_total}"
        )
        for ch, v in expected_per_channel.items():
            assert row.per_channel[ch] == pytest.approx(v, rel=1e-12, abs=1e-12)
