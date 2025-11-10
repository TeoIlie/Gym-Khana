"""
Test collision detection strategy configuration.

Tests the predictive_collision parameter which controls:
- True: Predictive TTC-based collision detection (default for non-drift modes)
- False: Explicit Frenet-based boundary detection (default for drift mode)
"""

import gymnasium as gym
import pytest

from f1tenth_gym.envs.f110_env import F110Env


def test_predictive_collision_default_drift():
    """Test that predictive_collision=False by default for 'drift' observation type."""
    env = gym.make(
        "f1tenth_gym:f1tenth-v0",
        config={
            "map": "Spielberg",
            "num_agents": 1,
            "model": "std",
            "observation_config": {"type": "drift"},
            "params": F110Env.f1tenth_std_vehicle_params(),
        },
    )
    assert env.unwrapped.predictive_collision is False
    env.close()


def test_predictive_collision_default_non_drift():
    """Test that predictive_collision=True by default for non-drift observation types."""
    env = gym.make(
        "f1tenth_gym:f1tenth-v0",
        config={
            "map": "Spielberg",
            "num_agents": 1,
            "observation_config": {"type": "original"},
        },
    )
    assert env.unwrapped.predictive_collision is True
    env.close()


def test_predictive_collision_default_none_type():
    """Test that predictive_collision=True by default when observation type is None."""
    env = gym.make(
        "f1tenth_gym:f1tenth-v0",
        config={
            "map": "Spielberg",
            "num_agents": 1,
            "observation_config": {"type": None},
        },
    )
    assert env.unwrapped.predictive_collision is True
    env.close()


def test_predictive_collision_override_false_with_non_drift():
    """Test that predictive_collision=False with non-drift type warns but allows override."""
    with pytest.warns(UserWarning, match="Disabling predictive collision"):
        env = gym.make(
            "f1tenth_gym:f1tenth-v0",
            config={
                "map": "Spielberg",
                "num_agents": 1,
                "observation_config": {"type": "original"},
                "predictive_collision": False,
            },
        )
        assert env.unwrapped.predictive_collision is False
        env.close()


def test_predictive_collision_override_true_with_drift():
    """Test that predictive_collision=True with drift type warns but allows override."""
    with pytest.warns(UserWarning, match="Enabling predictive collision"):
        env = gym.make(
            "f1tenth_gym:f1tenth-v0",
            config={
                "map": "Spielberg",
                "num_agents": 1,
                "model": "std",
                "observation_config": {"type": "drift"},
                "params": F110Env.f1tenth_std_vehicle_params(),
                "predictive_collision": True,
            },
        )
        assert env.unwrapped.predictive_collision is True
        env.close()


def test_predictive_collision_explicit_false_with_drift():
    """Test that predictive_collision=False with drift type works without warning (aligns with default)."""
    env = gym.make(
        "f1tenth_gym:f1tenth-v0",
        config={
            "map": "Spielberg",
            "num_agents": 1,
            "model": "std",
            "observation_config": {"type": "drift"},
            "params": F110Env.f1tenth_std_vehicle_params(),
            "predictive_collision": False,
        },
    )
    assert env.unwrapped.predictive_collision is False
    env.close()


def test_predictive_collision_explicit_true_with_non_drift():
    """Test that predictive_collision=True with non-drift type works without warning (aligns with default)."""
    env = gym.make(
        "f1tenth_gym:f1tenth-v0",
        config={
            "map": "Spielberg",
            "num_agents": 1,
            "observation_config": {"type": "original"},
            "predictive_collision": True,
        },
    )
    assert env.unwrapped.predictive_collision is True
    env.close()


def test_predictive_collision_with_kinematic_observation():
    """Test that predictive_collision=True by default for 'kinematic_state' observation type."""
    env = gym.make(
        "f1tenth_gym:f1tenth-v0",
        config={
            "map": "Spielberg",
            "num_agents": 1,
            "observation_config": {"type": "kinematic_state"},
        },
    )
    assert env.unwrapped.predictive_collision is True
    env.close()


def test_predictive_collision_combined_with_normalize():
    """Test that predictive_collision and normalize_obs can be configured independently."""
    env = gym.make(
        "f1tenth_gym:f1tenth-v0",
        config={
            "map": "Spielberg",
            "num_agents": 1,
            "model": "std",
            "observation_config": {"type": "drift"},
            "params": F110Env.f1tenth_std_vehicle_params(),
            "predictive_collision": False,
            "normalize_obs": True,
        },
    )
    assert env.unwrapped.predictive_collision is False
    assert env.unwrapped.normalize_obs is True
    env.close()


def test_predictive_collision_none_explicit():
    """Test that predictive_collision=None triggers auto-detection behavior."""
    env = gym.make(
        "f1tenth_gym:f1tenth-v0",
        config={
            "map": "Spielberg",
            "num_agents": 1,
            "model": "std",
            "observation_config": {"type": "drift"},
            "params": F110Env.f1tenth_std_vehicle_params(),
            "predictive_collision": None,
        },
    )
    # Should auto-set to False for drift
    assert env.unwrapped.predictive_collision is False
    env.close()

    env = gym.make(
        "f1tenth_gym:f1tenth-v0",
        config={
            "map": "Spielberg",
            "num_agents": 1,
            "observation_config": {"type": "original"},
            "predictive_collision": None,
        },
    )
    # Should auto-set to True for non-drift
    assert env.unwrapped.predictive_collision is True
    env.close()


if __name__ == "__main__":
    # Run with pytest
    pytest.main([__file__, "-v", "-s"])
