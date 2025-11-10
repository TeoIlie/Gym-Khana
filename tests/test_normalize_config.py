"""
Test normalization configuration for drift observations.
"""

import gymnasium as gym
import pytest

from f1tenth_gym.envs.f110_env import F110Env


def test_normalize_default_drift():
    """Test that normalize_obs=True by default for 'drift' observation type."""
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
    assert env.unwrapped.normalize_obs is True
    env.close()


def test_normalize_default_other():
    """Test that normalize=False by default for non-drift observation types."""
    env = gym.make(
        "f1tenth_gym:f1tenth-v0",
        config={
            "map": "Spielberg",
            "num_agents": 1,
            "observation_config": {"type": "original"},
            "params": F110Env.f1tenth_std_vehicle_params(),
        },
    )
    assert env.unwrapped.normalize_obs is False
    env.close()


def test_normalize_override_true_with_non_drift():
    """Test that normalize=True with non-drift type sets to False with warning."""
    with pytest.warns(UserWarning, match="Normalization is only supported"):
        env = gym.make(
            "f1tenth_gym:f1tenth-v0",
            config={
                "map": "Spielberg",
                "num_agents": 1,
                "observation_config": {"type": "original"},
                "params": F110Env.f1tenth_std_vehicle_params(),
                "normalize_obs": True,
            },
        )
        assert env.unwrapped.normalize_obs is False
        env.close()


def test_normalize_override_false_with_drift():
    """Test that normalize=False with drift type keeps False but warns."""
    with pytest.warns(UserWarning, match="Normalization is recommended"):
        env = gym.make(
            "f1tenth_gym:f1tenth-v0",
            config={
                "map": "Spielberg",
                "num_agents": 1,
                "model": "std",
                "observation_config": {"type": "drift"},
                "params": F110Env.f1tenth_std_vehicle_params(),
                "normalize_obs": False,
            },
        )
        assert env.unwrapped.normalize_obs is False
        env.close()


def test_normalize_explicit_true_with_drift():
    """Test that normalize=True with drift type works correctly."""
    env = gym.make(
        "f1tenth_gym:f1tenth-v0",
        config={
            "map": "Spielberg",
            "num_agents": 1,
            "model": "std",
            "observation_config": {"type": "drift"},
            "params": F110Env.f1tenth_std_vehicle_params(),
            "normalize_obs": True,
        },
    )
    assert env.unwrapped.normalize_obs is True
    env.close()


def test_normalize_explicit_false_with_non_drift():
    """Test that normalize=False with non-drift type works correctly."""
    env = gym.make(
        "f1tenth_gym:f1tenth-v0",
        config={
            "map": "Spielberg",
            "num_agents": 1,
            "observation_config": {"type": "original"},
            "params": F110Env.f1tenth_std_vehicle_params(),
            "normalize_obs": False,
        },
    )
    assert env.unwrapped.normalize_obs is False
    env.close()


def test_observation_space_bounds_with_normalize_true():
    """Test that observation space has bounds [-1, 1] when normalize=True."""
    env = gym.make(
        "f1tenth_gym:f1tenth-v0",
        config={
            "map": "Spielberg",
            "num_agents": 1,
            "model": "std",
            "observation_config": {"type": "drift"},
            "params": F110Env.f1tenth_std_vehicle_params(),
            "normalize_obs": True,
        },
    )
    obs_space = env.observation_space
    assert obs_space.low[0] == -1.0
    assert obs_space.high[0] == 1.0
    env.close()


def test_observation_space_bounds_with_normalize_false():
    """Test that observation space has bounds [-1e30, 1e30] when normalize=False."""
    env = gym.make(
        "f1tenth_gym:f1tenth-v0",
        config={
            "map": "Spielberg",
            "num_agents": 1,
            "model": "std",
            "observation_config": {"type": "drift"},
            "params": F110Env.f1tenth_std_vehicle_params(),
            "normalize_obs": False,
        },
    )
    obs_space = env.observation_space
    assert obs_space.low[0] == -1e30
    assert obs_space.high[0] == 1e30
    env.close()
