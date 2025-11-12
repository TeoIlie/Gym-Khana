"""
Test normalization configuration for drift observations and actions.
"""

import gymnasium as gym
import numpy as np
import pytest

from f1tenth_gym.envs.f110_env import F110Env

# ============================================================================
# Observation Normalization Configuration Tests
# ============================================================================


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


# ============================================================================
# Action Normalization Configuration Tests
# ============================================================================


def test_normalize_act_default_true():
    """Test that normalize_act=True by default."""
    env = gym.make(
        "f1tenth_gym:f1tenth-v0",
        config={
            "map": "Spielberg",
            "num_agents": 1,
            "params": F110Env.f1tenth_std_vehicle_params(),
        },
    )
    assert env.unwrapped.normalize_act is True
    env.close()


def test_normalize_act_explicit_false():
    """Test that normalize_act=False when explicitly set by user."""
    env = gym.make(
        "f1tenth_gym:f1tenth-v0",
        config={
            "map": "Spielberg",
            "num_agents": 1,
            "params": F110Env.f1tenth_std_vehicle_params(),
            "normalize_act": False,
        },
    )
    assert env.unwrapped.normalize_act is False
    env.close()


def test_action_space_bounds_normalized():
    """Test that action space is [-1, 1]² when normalize_act=True."""
    env = gym.make(
        "f1tenth_gym:f1tenth-v0",
        config={
            "map": "Spielberg",
            "num_agents": 1,
            "params": F110Env.f1tenth_std_vehicle_params(),
            "normalize_act": True,
        },
    )
    action_space = env.action_space

    # For single agent, action space shape should be (1, 2) - one agent, two actions
    assert action_space.shape == (1, 2), f"Expected shape (1, 2), got {action_space.shape}"

    # Check bounds for the single agent
    np.testing.assert_array_equal(action_space.low[0], np.array([-1.0, -1.0], dtype=np.float32))
    np.testing.assert_array_equal(action_space.high[0], np.array([1.0, 1.0], dtype=np.float32))

    env.close()


def test_action_space_bounds_unnormalized():
    """Test that action space uses physical bounds when normalize_act=False."""
    params = F110Env.f1tenth_std_vehicle_params()
    env = gym.make(
        "f1tenth_gym:f1tenth-v0",
        config={
            "map": "Spielberg",
            "num_agents": 1,
            "params": params,
            "control_input": ["accl", "steering_angle"],  # Explicitly use accl control
            "normalize_act": False,
        },
    )
    action_space = env.action_space

    # For single agent, action space shape should be (1, 2)
    assert action_space.shape == (1, 2), f"Expected shape (1, 2), got {action_space.shape}"

    # Check bounds match physical parameters
    # control_input is ["accl", "steering_angle"], which means:
    # CarAction composes as [steer, longitudinal], so:
    # action[0] = steering angle, bounds: [s_min, s_max]
    # action[1] = acceleration, bounds: [-a_max, a_max]
    expected_low = np.array([params["s_min"], -params["a_max"]], dtype=np.float32)
    expected_high = np.array([params["s_max"], params["a_max"]], dtype=np.float32)

    np.testing.assert_array_almost_equal(action_space.low[0], expected_low, decimal=4)
    np.testing.assert_array_almost_equal(action_space.high[0], expected_high, decimal=4)

    env.close()
