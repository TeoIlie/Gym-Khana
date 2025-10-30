"""
Test normalization configuration for drift observations.
"""

import gymnasium as gym

from f1tenth_gym.envs.f110_env import F110Env


def test_normalize_default_drift():
    """Test that normalize=True by default for 'drift' observation type."""
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
    assert env.unwrapped.normalize is True
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
    assert env.unwrapped.normalize is False
    env.close()


def test_normalize_override_true_with_non_drift(capsys):
    """Test that normalize=True with non-drift type sets to False with warning."""
    env = gym.make(
        "f1tenth_gym:f1tenth-v0",
        config={
            "map": "Spielberg",
            "num_agents": 1,
            "observation_config": {"type": "original"},
            "params": F110Env.f1tenth_std_vehicle_params(),
            "normalize": True,
        },
    )
    assert env.unwrapped.normalize is False

    # Check warning was printed
    captured = capsys.readouterr()
    assert "Warning" in captured.out
    assert "Setting normalize=False" in captured.out
    env.close()


def test_normalize_override_false_with_drift(capsys):
    """Test that normalize=False with drift type keeps False but warns."""
    env = gym.make(
        "f1tenth_gym:f1tenth-v0",
        config={
            "map": "Spielberg",
            "num_agents": 1,
            "model": "std",
            "observation_config": {"type": "drift"},
            "params": F110Env.f1tenth_std_vehicle_params(),
            "normalize": False,
        },
    )
    assert env.unwrapped.normalize is False

    # Check warning was printed
    captured = capsys.readouterr()
    assert "Warning" in captured.out
    assert "intentional" in captured.out
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
            "normalize": True,
        },
    )
    assert env.unwrapped.normalize is True
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
            "normalize": False,
        },
    )
    assert env.unwrapped.normalize is False
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
            "normalize": True,
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
            "normalize": False,
        },
    )
    obs_space = env.observation_space
    assert obs_space.low[0] == -1e30
    assert obs_space.high[0] == 1e30
    env.close()
