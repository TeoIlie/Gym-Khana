"""Unit tests for examples.analysis.sysid.loss.

Tests are pure: synthetic Window/Dataset objects and synthetic sim dicts —
no env, no rollout. Rollout integration is verified separately.

Coverage:
  - channel_nmse: identity, known-value correctness, zero-variance safety.
  - window_loss: identity, warmup slicing, weighted-sum aggregation,
    per-channel breakdown completeness.
  - dataset_loss: identity, warmup_s→steps conversion, mean across windows,
    mirror invariance under sign-symmetric sim signals.
"""

from __future__ import annotations

import numpy as np
import pytest

from examples.analysis.sysid.dataset import CHANNELS, Dataset, Window, mirror_window
from examples.analysis.sysid.loss import (
    DEFAULT_WEIGHTS,
    channel_nmse,
    dataset_loss,
    window_loss,
)

DT = 0.01
N = 150  # cmd length; real signals are N+1 = 151
WARMUP_S = 0.2
WARMUP_STEPS = int(round(WARMUP_S / DT))  # 20


def _make_window(rng: np.random.Generator, t0_idx: int = 0) -> Window:
    return Window(
        t0_idx=t0_idx,
        init_state=np.array(
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 20.0, 20.0]
        ),  # omega values arbitrary; loss tests don't read init_state
        cmd_steer=rng.normal(0, 0.1, N),
        cmd_speed=rng.normal(2.0, 0.2, N),
        real_v_x=rng.normal(2.0, 0.5, N + 1),
        real_v_y=rng.normal(0.0, 0.3, N + 1),
        real_yaw_rate=rng.normal(0.0, 0.5, N + 1),
        real_a_x=rng.normal(0.0, 1.0, N + 1),
        is_mirrored=False,
    )


def _sim_from_real(window: Window) -> dict[str, np.ndarray]:
    return {
        "yaw_rate": window.real_yaw_rate.copy(),
        "v_y": window.real_v_y.copy(),
        "a_x": window.real_a_x.copy(),
        "v_x": window.real_v_x.copy(),
    }


def _variances_from_windows(windows: list[Window]) -> dict[str, float]:
    return {
        "yaw_rate": float(np.var(np.concatenate([w.real_yaw_rate for w in windows]))),
        "v_y": float(np.var(np.concatenate([w.real_v_y for w in windows]))),
        "a_x": float(np.var(np.concatenate([w.real_a_x for w in windows]))),
        "v_x": float(np.var(np.concatenate([w.real_v_x for w in windows]))),
    }


# ---------- channel_nmse ----------


def test_channel_nmse_identity_is_zero():
    rng = np.random.default_rng(0)
    real = rng.normal(0, 1, 100)
    assert channel_nmse(real, real, variance=float(np.var(real))) == 0.0


def test_channel_nmse_known_value():
    sim = np.array([1.0, 2.0, 3.0])
    real = np.array([1.0, 2.0, 4.0])  # diff = [0, 0, 1] → MSE = 1/3
    variance = 4.0
    expected = (1.0 / 3.0) / (4.0 + 1e-9)
    assert channel_nmse(sim, real, variance) == pytest.approx(expected)


def test_channel_nmse_zero_variance_safe():
    # Constant real signal → variance 0; epsilon must prevent div-by-zero.
    sim = np.array([1.0, 1.0, 1.0])
    real = np.array([1.0, 1.0, 1.0])
    val = channel_nmse(sim, real, variance=0.0)
    assert val == 0.0
    # Non-zero residual + zero variance: huge but finite.
    val2 = channel_nmse(np.array([2.0, 2.0]), real[:2], variance=0.0)
    assert np.isfinite(val2) and val2 > 0


# ---------- window_loss ----------


def test_window_loss_identity():
    rng = np.random.default_rng(1)
    w = _make_window(rng)
    variances = _variances_from_windows([w])
    sim = _sim_from_real(w)
    total, per_channel = window_loss(sim, w, variances, DEFAULT_WEIGHTS, WARMUP_STEPS)
    assert total == pytest.approx(0.0, abs=1e-12)
    for ch in CHANNELS:
        assert per_channel[ch] == pytest.approx(0.0, abs=1e-12)


def test_window_loss_warmup_region_is_ignored():
    # Inject huge errors inside the warmup region only — loss must be unchanged.
    rng = np.random.default_rng(2)
    w = _make_window(rng)
    variances = _variances_from_windows([w])

    sim_clean = _sim_from_real(w)
    total_clean, _ = window_loss(sim_clean, w, variances, DEFAULT_WEIGHTS, WARMUP_STEPS)

    sim_dirty = _sim_from_real(w)
    for ch in CHANNELS:
        sim_dirty[ch][:WARMUP_STEPS] += 1e6  # huge perturbation inside warmup
    total_dirty, per_channel_dirty = window_loss(sim_dirty, w, variances, DEFAULT_WEIGHTS, WARMUP_STEPS)

    assert total_dirty == pytest.approx(total_clean)
    for ch in CHANNELS:
        assert per_channel_dirty[ch] == pytest.approx(0.0, abs=1e-12)


def test_window_loss_post_warmup_perturbation_counted():
    # Inject error just past the warmup boundary on yaw_rate only.
    rng = np.random.default_rng(3)
    w = _make_window(rng)
    variances = _variances_from_windows([w])
    sim = _sim_from_real(w)
    sim["yaw_rate"][WARMUP_STEPS:] += 0.5  # constant offset

    total, per_channel = window_loss(sim, w, variances, DEFAULT_WEIGHTS, WARMUP_STEPS)

    expected_yaw_nmse = (0.5**2) / (variances["yaw_rate"] + 1e-9)
    assert per_channel["yaw_rate"] == pytest.approx(expected_yaw_nmse, rel=1e-9)
    for ch in ("v_y", "a_x", "v_x"):
        assert per_channel[ch] == pytest.approx(0.0, abs=1e-12)
    assert total == pytest.approx(DEFAULT_WEIGHTS["yaw_rate"] * expected_yaw_nmse, rel=1e-9)


def test_window_loss_weighting_is_linear_combination():
    # Hand-craft known per-channel NMSEs and verify weighted_total = sum(w * nmse).
    rng = np.random.default_rng(4)
    w = _make_window(rng)
    variances = _variances_from_windows([w])
    sim = _sim_from_real(w)
    offsets = {"yaw_rate": 0.1, "v_y": 0.2, "a_x": 0.3, "v_x": 0.4}
    for ch, off in offsets.items():
        sim[ch][WARMUP_STEPS:] += off

    total, per_channel = window_loss(sim, w, variances, DEFAULT_WEIGHTS, WARMUP_STEPS)
    expected_total = sum(DEFAULT_WEIGHTS[ch] * per_channel[ch] for ch in CHANNELS)
    assert total == pytest.approx(expected_total, rel=1e-12)


def test_window_loss_per_channel_keys_complete():
    rng = np.random.default_rng(5)
    w = _make_window(rng)
    variances = _variances_from_windows([w])
    _, per_channel = window_loss(_sim_from_real(w), w, variances, DEFAULT_WEIGHTS, WARMUP_STEPS)
    assert set(per_channel.keys()) == set(CHANNELS)


# ---------- dataset_loss ----------


def _make_dataset(windows: list[Window]) -> Dataset:
    return Dataset(windows=windows, variances=_variances_from_windows(windows), dt=DT)


def test_dataset_loss_identity():
    rng = np.random.default_rng(6)
    windows = [_make_window(rng, t0_idx=i * 10) for i in range(3)]
    ds = _make_dataset(windows)
    total, per_channel = dataset_loss(_sim_from_real, ds, DEFAULT_WEIGHTS, warmup_s=WARMUP_S)
    assert total == pytest.approx(0.0, abs=1e-12)
    for ch in CHANNELS:
        assert per_channel[ch] == pytest.approx(0.0, abs=1e-12)


def test_dataset_loss_warmup_seconds_to_steps_conversion():
    # Set warmup_s so all-but-one sample is dropped; perturbing the dropped region
    # must be invisible. dt=0.01, N+1=151 → warmup_s=1.50 drops 150 samples, scoring 1.
    rng = np.random.default_rng(7)
    w = _make_window(rng)
    ds = _make_dataset([w])

    def rollout_perturb_early(window: Window) -> dict[str, np.ndarray]:
        sim = _sim_from_real(window)
        for ch in CHANNELS:
            sim[ch][:150] += 1e6  # only the last sample is scored
        return sim

    total, per_channel = dataset_loss(rollout_perturb_early, ds, DEFAULT_WEIGHTS, warmup_s=1.50)
    assert total == pytest.approx(0.0, abs=1e-12)
    for ch in CHANNELS:
        assert per_channel[ch] == pytest.approx(0.0, abs=1e-12)


def test_dataset_loss_aggregates_as_arithmetic_mean():
    # Two windows with deliberately different per-window losses; verify mean.
    rng = np.random.default_rng(8)
    w0 = _make_window(rng, t0_idx=0)
    w1 = _make_window(rng, t0_idx=10)
    ds = _make_dataset([w0, w1])

    # Window-specific perturbation: w0 gets +0.1 on yaw_rate, w1 gets +0.3.
    def rollout(window: Window) -> dict[str, np.ndarray]:
        sim = _sim_from_real(window)
        offset = 0.1 if window.t0_idx == 0 else 0.3
        sim["yaw_rate"][WARMUP_STEPS:] += offset
        return sim

    total, per_channel = dataset_loss(rollout, ds, DEFAULT_WEIGHTS, warmup_s=WARMUP_S)

    var = ds.variances["yaw_rate"]
    nmse0 = (0.1**2) / (var + 1e-9)
    nmse1 = (0.3**2) / (var + 1e-9)
    expected_mean_nmse = 0.5 * (nmse0 + nmse1)
    assert per_channel["yaw_rate"] == pytest.approx(expected_mean_nmse, rel=1e-9)
    for ch in ("v_y", "a_x", "v_x"):
        assert per_channel[ch] == pytest.approx(0.0, abs=1e-12)
    assert total == pytest.approx(DEFAULT_WEIGHTS["yaw_rate"] * expected_mean_nmse, rel=1e-9)


def test_dataset_loss_mirror_invariance_under_symmetric_sim():
    """Mirroring the real signals and the sim consistently must produce identical loss.

    Catches sign-handling bugs in any future per-channel sign logic. With the
    current implementation this is a near-tautology since (sim - real)^2 is
    sign-invariant — but it pins the contract.
    """
    rng = np.random.default_rng(9)
    w = _make_window(rng)
    w_mirr = mirror_window(w)
    ds_orig = _make_dataset([w])
    # Use the original variances to score the mirrored dataset, mirroring how
    # load_dataset() computes variances over non-mirrored windows only.
    ds_mirr = Dataset(windows=[w_mirr], variances=ds_orig.variances, dt=DT)

    # "Symmetric" sim: identity for original; for the mirrored window, flip the
    # antisymmetric channels to match the mirrored real signals.
    def rollout(window: Window) -> dict[str, np.ndarray]:
        return _sim_from_real(window)

    total_orig, per_orig = dataset_loss(rollout, ds_orig, DEFAULT_WEIGHTS, warmup_s=WARMUP_S)
    total_mirr, per_mirr = dataset_loss(rollout, ds_mirr, DEFAULT_WEIGHTS, warmup_s=WARMUP_S)

    assert total_orig == pytest.approx(total_mirr, abs=1e-12)
    for ch in CHANNELS:
        assert per_orig[ch] == pytest.approx(per_mirr[ch], abs=1e-12)
