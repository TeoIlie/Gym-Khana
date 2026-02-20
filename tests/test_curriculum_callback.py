"""
Verification tests for CurriculumRange and CurriculumLearningCallback.
"""

import pytest

from train.callbacks import CurriculumLearningCallback, CurriculumRange, make_curriculum_callback

# ── CurriculumRange ──────────────────────────────────────────────────


def make_range(name="v", initial_lo=5.0, initial_hi=9.0, max_lo=2.0, max_hi=12.0, n_stages=6):
    return CurriculumRange(
        name=name, initial_lo=initial_lo, initial_hi=initial_hi, max_lo=max_lo, max_hi=max_hi, n_stages=n_stages
    )


def test_range_initial_state():
    r = make_range()
    assert r.get_range() == [5.0, 9.0]
    assert not r.is_at_max()
    assert r.increment == pytest.approx(0.5)


def test_range_expand_all_stages():
    r = make_range(n_stages=3, initial_lo=5.0, initial_hi=9.0, max_lo=2.0, max_hi=12.0)
    # increment = (12 - 9) / 3 = 1.0
    assert r.increment == pytest.approx(1.0)

    assert r.expand()
    assert r.get_range() == pytest.approx([4.0, 10.0])

    assert r.expand()
    assert r.get_range() == pytest.approx([3.0, 11.0])

    assert r.expand()
    assert r.get_range() == pytest.approx([2.0, 12.0])
    assert r.is_at_max()

    # no-op after max
    assert not r.expand()
    assert r.get_range() == pytest.approx([2.0, 12.0])


def test_range_rejects_asymmetric():
    with pytest.raises(ValueError, match="asymmetric"):
        CurriculumRange(name="bad", initial_lo=5.0, initial_hi=9.0, max_lo=3.0, max_hi=12.0, n_stages=3)


def test_range_rejects_zero_stages():
    with pytest.raises(ValueError, match="n_stages"):
        make_range(n_stages=0)


# ── CurriculumLearningCallback ──────────────────────────────────────


def test_callback_rejects_small_hysteresis():
    """min_episodes_between_expansions must be >= window_size."""
    with pytest.raises(ValueError, match="min_episodes_between_expansions"):
        CurriculumLearningCallback(
            v_range=make_range(),
            beta_range=make_range(name="beta", initial_lo=-0.1, initial_hi=0.1, max_lo=-0.4, max_hi=0.4, n_stages=6),
            r_range=make_range(name="r", initial_lo=-0.2, initial_hi=0.2, max_lo=-0.5, max_hi=0.5, n_stages=6),
            yaw_range=make_range(name="yaw", initial_lo=-0.2, initial_hi=0.2, max_lo=-0.5, max_hi=0.5, n_stages=6),
            n_stages=6,
            window_size=500,
            min_episodes_between_expansions=200,  # < 500 → should fail
        )


# ── make_curriculum_callback factory ─────────────────────────────────


VALID_CONFIG = {
    "enabled": True,
    "n_stages": 6,
    "window_size": 500,
    "success_threshold": 0.8,
    "min_episodes_between_expansions": 1000,
    "max_curriculum_timestep": None,
    "log_freq": 10_000,
    "v_range": [5.0, 9.0, 2.0, 12.0],
    "beta_range": [-0.10, 0.10, -0.349, 0.349],
    "r_range": [-0.20, 0.20, -0.785, 0.785],
    "yaw_range": [-0.20, 0.20, -0.785, 0.785],
}


def test_factory_returns_callback():
    cb = make_curriculum_callback(VALID_CONFIG)
    assert isinstance(cb, CurriculumLearningCallback)
    assert cb.n_stages == 6
    assert cb.ranges["v"].get_range() == [5.0, 9.0]


def test_factory_returns_none_when_disabled():
    assert make_curriculum_callback({"enabled": False}) is None
    assert make_curriculum_callback({}) is None


def test_factory_uses_constructor_defaults_for_missing_keys():
    """Optional keys omitted from config should fall back to __init__ defaults."""
    minimal_config = {
        "enabled": True,
        "n_stages": 6,
        "v_range": [5.0, 9.0, 2.0, 12.0],
        "beta_range": [-0.10, 0.10, -0.349, 0.349],
        "r_range": [-0.20, 0.20, -0.785, 0.785],
        "yaw_range": [-0.20, 0.20, -0.785, 0.785],
    }
    cb = make_curriculum_callback(minimal_config)
    assert cb.window_size == 500
    assert cb.success_threshold == 0.8
    assert cb.min_episodes_between_expansions == 1000
    assert cb.max_curriculum_timestep is None
    assert cb.log_freq == 10_000


def test_factory_ranges_reach_max_after_n_stages():
    cb = make_curriculum_callback(VALID_CONFIG)
    for _ in range(6):
        for r in cb.ranges.values():
            r.expand()

    assert cb.ranges["v"].get_range() == pytest.approx([2.0, 12.0])
    assert cb.ranges["beta"].get_range() == pytest.approx([-0.349, 0.349])
    assert cb.ranges["r"].get_range() == pytest.approx([-0.785, 0.785])
    assert cb.ranges["yaw"].get_range() == pytest.approx([-0.785, 0.785])

    for r in cb.ranges.values():
        assert r.is_at_max()
