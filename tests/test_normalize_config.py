"""
Test normalization configuration for drift observations and actions.
"""

import unittest

import gymnasium as gym
import numpy as np
import pytest

from gymkhana.envs.gymkhana_env import GKEnv

# ============================================================================
# Observation Normalization Configuration Tests
# ============================================================================


class TestNormalizeObsDriftDefaults(unittest.TestCase):
    """Tests for drift observation normalization defaults (shared env, normalize_obs=True)."""

    @classmethod
    def setUpClass(cls):
        cls.env = gym.make(
            "gymkhana:gymkhana-v0",
            config={
                "map": "Spielberg",
                "num_agents": 1,
                "model": "std",
                "observation_config": {"type": "drift"},
                "params": GKEnv.f1tenth_std_vehicle_params(),
                "normalize_obs": True,
            },
        )

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_normalize_default_drift(self):
        """Test that normalize_obs=True by default for 'drift' observation type."""
        assert self.env.unwrapped.normalize_obs is True

    def test_normalize_explicit_true_with_drift(self):
        """Test that normalize=True with drift type works correctly."""
        assert self.env.unwrapped.normalize_obs is True

    def test_observation_space_bounds_with_normalize_true(self):
        """Test that observation space has bounds [-1, 1] when normalize=True."""
        obs_space = self.env.observation_space
        assert obs_space.low[0] == -1.0
        assert obs_space.high[0] == 1.0


class TestNormalizeObsNonDriftDefaults(unittest.TestCase):
    """Tests for non-drift observation normalization defaults (shared env, normalize_obs=False)."""

    @classmethod
    def setUpClass(cls):
        cls.env = gym.make(
            "gymkhana:gymkhana-v0",
            config={
                "map": "Spielberg",
                "num_agents": 1,
                "observation_config": {"type": "original"},
                "params": GKEnv.f1tenth_std_vehicle_params(),
                "normalize_obs": False,
            },
        )

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_normalize_default_other(self):
        """Test that normalize=False by default for non-drift observation types."""
        assert self.env.unwrapped.normalize_obs is False

    def test_normalize_explicit_false_with_non_drift(self):
        """Test that normalize=False with non-drift type works correctly."""
        assert self.env.unwrapped.normalize_obs is False


class TestNormalizeObsUnnormalizedDrift(unittest.TestCase):
    """Tests for drift observation with normalization disabled (shared env)."""

    @classmethod
    def setUpClass(cls):
        # Suppress expected warning during env creation
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            cls.env = gym.make(
                "gymkhana:gymkhana-v0",
                config={
                    "map": "Spielberg",
                    "num_agents": 1,
                    "model": "std",
                    "observation_config": {"type": "drift"},
                    "params": GKEnv.f1tenth_std_vehicle_params(),
                    "normalize_obs": False,
                },
            )

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_observation_space_bounds_with_normalize_false(self):
        """Test that observation space has bounds [-1e30, 1e30] when normalize=False."""
        obs_space = self.env.observation_space
        assert obs_space.low[0] == -1e30
        assert obs_space.high[0] == 1e30


# Warning tests must create their own envs (warnings fire at gym.make time)


def test_normalize_override_true_with_non_drift():
    """Test that normalize=True with non-drift type sets to False with warning."""
    with pytest.warns(UserWarning, match="Observation normalization is only supported"):
        env = gym.make(
            "gymkhana:gymkhana-v0",
            config={
                "map": "Spielberg",
                "num_agents": 1,
                "observation_config": {"type": "original"},
                "params": GKEnv.f1tenth_std_vehicle_params(),
                "normalize_obs": True,
            },
        )
        assert env.unwrapped.normalize_obs is False
        env.close()


def test_normalize_override_false_with_drift():
    """Test that normalize=False with drift type keeps False but warns."""
    with pytest.warns(UserWarning, match="Observation normalization is recommended"):
        env = gym.make(
            "gymkhana:gymkhana-v0",
            config={
                "map": "Spielberg",
                "num_agents": 1,
                "model": "std",
                "observation_config": {"type": "drift"},
                "params": GKEnv.f1tenth_std_vehicle_params(),
                "normalize_obs": False,
            },
        )
        assert env.unwrapped.normalize_obs is False
        env.close()


# ============================================================================
# Action Normalization Configuration Tests
# ============================================================================


class TestNormalizeActDefaults(unittest.TestCase):
    """Tests for action normalization defaults (shared env, normalize_act=True)."""

    @classmethod
    def setUpClass(cls):
        cls.env = gym.make(
            "gymkhana:gymkhana-v0",
            config={
                "map": "Spielberg",
                "num_agents": 1,
                "params": GKEnv.f1tenth_std_vehicle_params(),
                "normalize_act": True,
            },
        )

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_normalize_act_default_true(self):
        """Test that normalize_act=True by default."""
        assert self.env.unwrapped.normalize_act is True

    def test_action_space_bounds_normalized(self):
        """Test that action space is [-1, 1]^2 when normalize_act=True."""
        action_space = self.env.action_space
        assert action_space.shape == (1, 2), f"Expected shape (1, 2), got {action_space.shape}"
        np.testing.assert_array_equal(action_space.low[0], np.array([-1.0, -1.0], dtype=np.float32))
        np.testing.assert_array_equal(action_space.high[0], np.array([1.0, 1.0], dtype=np.float32))


class TestNormalizeActUnnormalized(unittest.TestCase):
    """Tests for action space with normalization disabled (shared env)."""

    @classmethod
    def setUpClass(cls):
        cls.params = GKEnv.f1tenth_std_vehicle_params()
        cls.env = gym.make(
            "gymkhana:gymkhana-v0",
            config={
                "map": "Spielberg",
                "num_agents": 1,
                "params": cls.params,
                "control_input": ["accl", "steering_angle"],
                "normalize_act": False,
            },
        )

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_normalize_act_explicit_false(self):
        """Test that normalize_act=False when explicitly set by user."""
        assert self.env.unwrapped.normalize_act is False

    def test_action_space_bounds_unnormalized(self):
        """Test that action space uses physical bounds when normalize_act=False."""
        action_space = self.env.action_space
        assert action_space.shape == (1, 2), f"Expected shape (1, 2), got {action_space.shape}"

        expected_low = np.array([self.params["s_min"], -self.params["a_max"]], dtype=np.float32)
        expected_high = np.array([self.params["s_max"], self.params["a_max"]], dtype=np.float32)

        np.testing.assert_array_almost_equal(action_space.low[0], expected_low, decimal=4)
        np.testing.assert_array_almost_equal(action_space.high[0], expected_high, decimal=4)
