"""
Tests for the `record_obs_min_max` observation min/max tracking feature.

Covers all three observation classes (Vector, Original, Features), both
normalization modes, the printout's bounded vs unbounded branches, and the
multi-agent collapse in the FeaturesObservation adapter.
"""

import warnings

import gymnasium as gym
import numpy as np

from gymkhana.envs.gymkhana_env import GKEnv


def _make_drift_env(normalize_obs: bool):
    """Build a drift env (VectorObservation) with min/max tracking enabled."""
    cfg = {
        "map": "Spielberg",
        "num_agents": 1,
        "model": "std",
        "observation_config": {"type": "drift"},
        "params": GKEnv.f1tenth_std_vehicle_params(),
        "normalize_obs": normalize_obs,
        "record_obs_min_max": True,
    }
    # normalize_obs=False with drift fires a "recommended" warning at gym.make time
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return gym.make("gymkhana:gymkhana-v0", config=cfg)


def _run_steps(env, n=10):
    env.reset()
    for _ in range(n):
        action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            env.reset()


def _assert_tracker_finite(unwrapped):
    assert unwrapped.obs_tracker_step_count > 0
    for feature in unwrapped.observation_type.features:
        entry = unwrapped.obs_min_max_tracker[feature]
        assert np.isfinite(entry["min"]), f"{feature} min not updated"
        assert np.isfinite(entry["max"]), f"{feature} max not updated"


def test_tracking_and_print_with_normalization(capsys):
    """normalize_obs=True: tracker fills, header reads ENABLED, table shows Coverage."""
    env = _make_drift_env(normalize_obs=True)
    try:
        unwrapped = env.unwrapped
        assert unwrapped.record_obs_min_max is True
        _run_steps(env, n=10)
        _assert_tracker_finite(unwrapped)

        unwrapped._print_obs_min_max_stats()
        out = capsys.readouterr().out
        assert "Normalization: ENABLED" in out
        assert "Coverage" in out
        # With drift+normalize, every feature should have a finite Theor. column,
        # i.e. no "N/A" rows for the populated features.
        for feature in unwrapped.observation_type.features:
            assert feature in out
    finally:
        env.close()


def test_tracking_and_print_without_normalization(capsys):
    """normalize_obs=False: tracker still fills, header reads DISABLED, no violation block."""
    env = _make_drift_env(normalize_obs=False)
    try:
        unwrapped = env.unwrapped
        assert unwrapped.record_obs_min_max is True
        # bounds remain unset in this mode (Vector.__init__ skips calculate_norm_bounds).
        assert unwrapped.observation_type.bounds == {}

        _run_steps(env, n=10)
        _assert_tracker_finite(unwrapped)

        unwrapped._print_obs_min_max_stats()
        out = capsys.readouterr().out
        assert "Normalization: DISABLED" in out
        assert "N/A" in out
        # Violation block references calculate_norm_bounds — must not appear.
        assert "calculate_norm_bounds" not in out
    finally:
        env.close()


def test_tracking_with_original_obs(capsys):
    """OriginalObservation is supported: tracker fills, printout prints N/A for unbounded features."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        env = gym.make(
            "gymkhana:gymkhana-v0",
            config={
                "map": "Spielberg",
                "num_agents": 1,
                "observation_config": {"type": "original"},
                "params": GKEnv.f1tenth_std_vehicle_params(),
                "normalize_obs": False,
                "record_obs_min_max": True,
            },
        )

    # No tracking-related warning should fire — the gate is gone.
    tracking_warnings = [w for w in caught if "min/max tracking" in str(w.message)]
    assert tracking_warnings == [], f"Unexpected tracking warning(s): {tracking_warnings}"

    try:
        unwrapped = env.unwrapped
        assert unwrapped.record_obs_min_max is True
        _run_steps(env, n=5)
        _assert_tracker_finite(unwrapped)

        unwrapped._print_obs_min_max_stats()
        out = capsys.readouterr().out
        # Original features have no entries in `bounds` → theoretical columns are N/A.
        assert "N/A" in out
        for feature in unwrapped.observation_type.features:
            assert feature in out
    finally:
        env.close()


def test_tracking_with_features_obs_single_agent():
    """FeaturesObservation tracker uses bare feature names (no agent_id prefix)."""
    env = gym.make(
        "gymkhana:gymkhana-v0",
        config={
            "map": "Spielberg",
            "num_agents": 1,
            "observation_config": {"type": "kinematic_state"},
            "params": GKEnv.f1tenth_std_vehicle_params(),
            "normalize_obs": False,
            "record_obs_min_max": True,
        },
    )
    try:
        unwrapped = env.unwrapped
        assert unwrapped.record_obs_min_max is True
        # Tracker keys mirror the obs class .features list — no agent_id prefix.
        assert set(unwrapped.obs_min_max_tracker.keys()) == set(unwrapped.observation_type.features)
        _run_steps(env, n=5)
        _assert_tracker_finite(unwrapped)
    finally:
        env.close()


def test_tracking_with_features_obs_multi_agent():
    """Multi-agent FeaturesObservation: across-agent collapse aggregates into one row per feature.

    Validates the adapter at gymkhana_env.py:_update_obs_min_max that flattens
    nested {agent_id: {feature: value}} into {feature: [values across agents]}.
    With two agents at different starting poses, the tracked range for `pose_x`
    must span both agents, i.e. (max - min) must be >= the distance between
    them after a single observation.
    """
    env = gym.make(
        "gymkhana:gymkhana-v0",
        config={
            "map": "Spielberg",
            "num_agents": 2,
            "observation_config": {"type": "kinematic_state"},
            "params": GKEnv.f1tenth_std_vehicle_params(),
            "normalize_obs": False,
            "record_obs_min_max": True,
        },
    )
    try:
        unwrapped = env.unwrapped
        # Tracker still keyed by bare feature names; agents collapse into the same row.
        assert set(unwrapped.obs_min_max_tracker.keys()) == set(unwrapped.observation_type.features)

        obs, _ = env.reset()
        # After reset, env.step is called once internally with skip_integration=True
        # (see GKEnv.reset) so the tracker has already seen both agents' poses.
        agent_ids = list(obs.keys())
        assert len(agent_ids) == 2
        per_agent_x = [float(obs[a]["pose_x"]) for a in agent_ids]
        spread = max(per_agent_x) - min(per_agent_x)
        assert spread > 0, "Test setup invalid: agents are at the same x"

        entry = unwrapped.obs_min_max_tracker["pose_x"]
        recorded_range = entry["max"] - entry["min"]
        # Recorded range must span both agents — proves across-agent aggregation works.
        assert recorded_range >= spread - 1e-6, (
            f"pose_x recorded range {recorded_range:.4f} smaller than across-agent "
            f"spread {spread:.4f} — adapter is not aggregating across agents"
        )

        _run_steps(env, n=5)
        _assert_tracker_finite(unwrapped)
    finally:
        env.close()
