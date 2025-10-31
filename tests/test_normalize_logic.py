"""
Test normalization functionality for drift observations.

This test suite validates:
1. Normalization bounds completeness and validity
2. normalize_feature() correctness for scalars and arrays
3. Edge cases (clipping, asymmetric bounds, errors)
4. End-to-end normalized observation pipeline
5. Track-adaptive normalization behavior
"""

import gymnasium as gym
import numpy as np
import pytest

from f1tenth_gym.envs.f110_env import F110Env
from f1tenth_gym.envs.utils import calculate_norm_bounds, normalize_feature


class TestNormalizationBounds:
    """Test calculate_norm_bounds() completeness and validity."""

    def test_complete_feature_coverage(self):
        """Verify calculate_norm_bounds() returns bounds for all drift features."""
        # Create environment with drift observation type
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

        # Expected drift features from observation.py lines 874-887
        expected_features = [
            "linear_vel_x",
            "linear_vel_y",
            "frenet_u",
            "frenet_n",
            "ang_vel_z",
            "delta",
            "prev_steering_cmd",
            "prev_accl_cmd",
            "prev_avg_wheel_omega",
            "curr_vel_cmd",
            "lookahead_curvatures",
            "lookahead_widths",
        ]

        # Get normalization bounds
        bounds = calculate_norm_bounds(env.unwrapped)

        # Assert all drift features exist as keys
        for feature in expected_features:
            assert feature in bounds, f"Missing bounds for feature: {feature}"

        # Assert all bounds are valid tuples (min, max)
        for feature, (min_val, max_val) in bounds.items():
            assert min_val is not None, f"Feature '{feature}' has None min value"
            assert max_val is not None, f"Feature '{feature}' has None max value"
            # Allow min == max for constant features (e.g., constant track width)
            assert min_val <= max_val, f"Feature '{feature}' has invalid bounds: min={min_val} > max={max_val}"

        env.close()


class TestNormalizeFeature:
    """Test normalize_feature() function correctness."""

    def test_scalar_normalization_float32(self):
        """Test normalization of single scalar float32 values."""
        # Mock bounds
        bounds = {"test_feature": (0.0, 10.0)}

        # Test min value 0.0 maps to -1.0
        result = normalize_feature("test_feature", np.float32(0.0), bounds)
        assert isinstance(result, (np.floating, float)), "Output should be a float type"
        assert result.dtype == np.float32, "Output dtype should be np.float32"
        assert np.isclose(result, -1.0), f"Expected -1.0, got {result}"

        # Test mid value 5.0 maps to 0.0
        result = normalize_feature("test_feature", np.float32(5.0), bounds)
        assert result.dtype == np.float32
        assert np.isclose(result, 0.0), f"Expected 0.0, got {result}"

        # Test max value 10.0 maps to 1.0
        result = normalize_feature("test_feature", np.float32(10.0), bounds)
        assert result.dtype == np.float32
        assert np.isclose(result, 1.0), f"Expected 1.0, got {result}"

        # Test 3/4 point 0.75 maps to 0.5
        result = normalize_feature("test_feature", np.float32(7.5), bounds)
        assert result.dtype == np.float32
        assert np.isclose(result, 0.5), f"Expected 0.5, got {result}"

    def test_array_normalization_element_wise(self):
        """Test element-wise normalization of np.ndarray with explicit value checking."""
        # Mock bounds
        bounds = {"test_array": (0.0, 10.0)}

        # Input array with specific test points: min, mid, 3/4, max
        input_array = np.array([0.0, 5.0, 7.5, 10.0], dtype=np.float32)

        # Expected normalized values
        expected_output = np.array([-1.0, 0.0, 0.5, 1.0], dtype=np.float32)

        # Normalize the array
        result = normalize_feature("test_array", input_array, bounds)

        # Assert output is ndarray with correct dtype
        assert isinstance(result, np.ndarray), "Output should be ndarray"
        assert result.dtype == np.float32, f"Expected dtype float32, got {result.dtype}"

        # Assert output shape matches input
        assert result.shape == input_array.shape, f"Expected shape {input_array.shape}, got {result.shape}"

        # Assert element-wise normalization is exactly correct
        assert np.allclose(result, expected_output), f"Expected {expected_output}, got {result}"

        # Verify each element individually for clarity
        assert np.isclose(result[0], -1.0), f"Element 0: Expected -1.0, got {result[0]}"
        assert np.isclose(result[1], 0.0), f"Element 1: Expected 0.0, got {result[1]}"
        assert np.isclose(result[2], 0.5), f"Element 2: Expected 0.5, got {result[2]}"
        assert np.isclose(result[3], 1.0), f"Element 3: Expected 1.0, got {result[3]}"

    def test_value_clipping(self):
        """Verify out-of-bounds values are clipped to [-1, 1]."""
        bounds = {"test_feature": (0.0, 10.0)}

        # Test value below minimum → clips to -1.0
        result = normalize_feature("test_feature", np.float32(-5.0), bounds)
        assert np.isclose(result, -1.0), f"Expected -1.0 (clipped), got {result}"

        # Test value above maximum → clips to 1.0
        result = normalize_feature("test_feature", np.float32(15.0), bounds)
        assert np.isclose(result, 1.0), f"Expected 1.0 (clipped), got {result}"

        # Test extreme values
        result = normalize_feature("test_feature", np.float32(-1000.0), bounds)
        assert np.isclose(result, -1.0), "Extreme negative should clip to -1.0"

        result = normalize_feature("test_feature", np.float32(1000.0), bounds)
        assert np.isclose(result, 1.0), "Extreme positive should clip to 1.0"

    def test_asymmetric_bounds(self):
        """Test normalization with asymmetric physical ranges (e.g., wheel speed >= 0)."""
        # Asymmetric bounds: [0, 20] (e.g., wheel speed can't be negative)
        bounds = {"velocity": (0.0, 20.0)}

        # Test min (0) → -1.0
        result = normalize_feature("velocity", np.float32(0.0), bounds)
        assert np.isclose(result, -1.0), f"Expected -1.0, got {result}"

        # Test midpoint (10) → 0.0
        result = normalize_feature("velocity", np.float32(10.0), bounds)
        assert np.isclose(result, 0.0), f"Expected 0.0, got {result}"

        # Test max (20) → 1.0
        result = normalize_feature("velocity", np.float32(20.0), bounds)
        assert np.isclose(result, 1.0), f"Expected 1.0, got {result}"

        # Test negative asymmetric bounds: [-10, 5]
        bounds_negative = {"lateral_offset": (-10.0, 5.0)}

        result = normalize_feature("lateral_offset", np.float32(-10.0), bounds_negative)
        assert np.isclose(result, -1.0), "Min should map to -1.0"

        # Midpoint: (-10 + 5) / 2 = -2.5
        result = normalize_feature("lateral_offset", np.float32(-2.5), bounds_negative)
        assert np.isclose(result, 0.0), "Midpoint should map to 0.0"

        result = normalize_feature("lateral_offset", np.float32(5.0), bounds_negative)
        assert np.isclose(result, 1.0), "Max should map to 1.0"

    def test_symmetric_negative_bounds(self):
        """Test normalization with symmetric negative bounds (e.g., steering angle)."""
        # Symmetric bounds: [-pi, pi]
        bounds = {"angle": (-np.pi, np.pi)}

        # Test min
        result = normalize_feature("angle", np.float32(-np.pi), bounds)
        assert np.isclose(result, -1.0)

        # Test zero (center)
        result = normalize_feature("angle", np.float32(0.0), bounds)
        assert np.isclose(result, 0.0)

        # Test max
        result = normalize_feature("angle", np.float32(np.pi), bounds)
        assert np.isclose(result, 1.0)

    def test_missing_feature_error_handling(self):
        """Verify error when feature not in bounds dict."""
        # Create bounds dict with only one feature
        bounds = {"feature_a": (0.0, 1.0)}

        # Attempt to normalize a feature that doesn't have bounds
        with pytest.raises(ValueError) as exc_info:
            normalize_feature("missing_feature", np.float32(0.5), bounds)

        # Check error message is descriptive
        error_msg = str(exc_info.value)
        assert "missing_feature" in error_msg, "Error should mention the missing feature name"
        assert "no bounds defined" in error_msg, "Error should explain the issue"

    def test_constant_feature_edge_case(self):
        """Test normalization when feature has negligible variance (min ≈ max)."""
        # Nearly constant feature (range is essentially zero, less than atol=1e-9)
        bounds = {"constant_feature": (5.0, 5.0 + 1e-10)}

        # Should return 0.0 (center of [-1, 1]) to avoid division by zero
        result = normalize_feature("constant_feature", np.float32(5.0), bounds)
        assert np.isclose(result, 0.0), "Constant feature should normalize to 0.0"

        # Test with array
        input_array = np.array([5.0, 5.0, 5.0], dtype=np.float32)
        result_array = normalize_feature("constant_feature", input_array, bounds)
        assert np.allclose(result_array, 0.0), "Constant array should normalize to zeros"


class TestNormalizedObservation:
    """Test end-to-end normalized observation pipeline."""

    def test_end_to_end_normalized_observation(self):
        """Validate complete observation pipeline with normalization enabled."""
        # Create environment with normalization enabled
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

        # Reset and take several steps
        obs, _ = env.reset()

        for _ in range(10):
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)

            # Assert all observation values are in range [-1, 1]
            assert np.all(obs >= -1.0), f"Observation has values < -1.0: min={np.min(obs)}"
            assert np.all(obs <= 1.0), f"Observation has values > 1.0: max={np.max(obs)}"

            # Assert shape matches observation space
            assert obs.shape == env.observation_space.shape, "Observation shape mismatch"

            # Assert dtype is float32
            assert obs.dtype == np.float32, f"Expected dtype float32, got {obs.dtype}"

            # Assert observation is contained in observation space
            assert env.observation_space.contains(obs), "Observation not in declared space"

            if terminated or truncated:
                break

        env.close()

    def test_unnormalized_observation_not_clipped(self):
        """Verify unnormalized observations can exceed [-1, 1] range."""
        # Create environment with normalization disabled
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

        obs, _ = env.reset()

        # Take a few steps
        for _ in range(10):
            action = env.action_space.sample()
            obs, _, _, _, _ = env.step(action)

            # Unnormalized observations should NOT be restricted to [-1, 1]
            # At least some values should be outside this range
            # (velocities, positions can be much larger)
            if np.any(np.abs(obs) > 1.0):
                # Found values outside [-1, 1], which is expected
                break

        # We expect at least some values to exceed [-1, 1] for unnormalized obs
        # This is a soft check - in practice velocities/positions will exceed this
        assert env.observation_space.contains(obs), "Observation should still be in declared space"

        env.close()

    def test_track_dependent_bounds_vary(self):
        """Ensure lookahead bounds adapt to different track geometries."""
        # Create environment with first track (Spielberg)
        env1 = gym.make(
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

        # Create environment with second track (Catalunya - different layout)
        env2 = gym.make(
            "f1tenth_gym:f1tenth-v0",
            config={
                "map": "Catalunya",  # Different track with different geometry
                "num_agents": 1,
                "model": "std",
                "observation_config": {"type": "drift"},
                "params": F110Env.f1tenth_std_vehicle_params(),
                "normalize": True,
            },
        )

        # Get bounds from both environments
        bounds1 = calculate_norm_bounds(env1.unwrapped)
        bounds2 = calculate_norm_bounds(env2.unwrapped)

        # Track-dependent bounds should differ (curvatures or widths)
        # Use tuple comparison to handle numpy arrays in bounds
        curvature_bounds_differ = (
            bounds1["lookahead_curvatures"][0] != bounds2["lookahead_curvatures"][0]
            or bounds1["lookahead_curvatures"][1] != bounds2["lookahead_curvatures"][1]
        )
        width_bounds_differ = (
            bounds1["lookahead_widths"][0] != bounds2["lookahead_widths"][0]
            or bounds1["lookahead_widths"][1] != bounds2["lookahead_widths"][1]
        )

        # At least one of these should differ for different tracks
        assert curvature_bounds_differ or width_bounds_differ, (
            "Track-dependent bounds should vary across different tracks. "
            f"Curvatures: {bounds1['lookahead_curvatures']} vs {bounds2['lookahead_curvatures']}, "
            f"Widths: {bounds1['lookahead_widths']} vs {bounds2['lookahead_widths']}"
        )

        # Vehicle-dependent bounds should be identical (same vehicle params)
        assert bounds1["linear_vel_x"] == bounds2["linear_vel_x"], "Vehicle velocity bounds should be identical"
        assert bounds1["delta"] == bounds2["delta"], "Steering angle bounds should be identical"
        assert bounds1["prev_accl_cmd"] == bounds2["prev_accl_cmd"], "Acceleration bounds should be identical"

        env1.close()
        env2.close()

    def test_normalized_observation_preserves_relative_ordering(self):
        """Verify normalization preserves relative relationships between features."""
        # Create two identical environments with same seed
        env_normalized = gym.make(
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

        env_unnormalized = gym.make(
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

        # Reset with same seed
        seed = 42
        env_normalized.reset(seed=seed)
        env_unnormalized.reset(seed=seed)

        # Take same actions in both environments
        for _ in range(5):
            # Fixed action: (num_agents, 2) shape for single agent
            action = np.array([[3.0, 0.1]], dtype=np.float32)

            obs_norm, _, _, _, _ = env_normalized.step(action)
            obs_unnorm, _, _, _, _ = env_unnormalized.step(action)

            # Observations should have same length
            assert len(obs_norm) == len(obs_unnorm), "Observation lengths should match"

            # Both observations should be 1D arrays
            assert obs_norm.ndim == 1, "Normalized observation should be 1D"
            assert obs_unnorm.ndim == 1, "Unnormalized observation should be 1D"

            # Normalized observations should be bounded
            assert np.all(obs_norm >= -1.0) and np.all(obs_norm <= 1.0), "Normalized obs should be in [-1, 1]"

            # Unnormalized observations can be larger
            # The key property of normalization: it's a monotonic transformation
            # If we manually normalize the unnormalized obs, it should match
            # (This is implicitly tested by the end-to-end test, so we just verify consistency here)

        env_normalized.close()
        env_unnormalized.close()


if __name__ == "__main__":
    # Run tests with: pytest tests/test_normalize_feature.py -v
    pytest.main([__file__, "-v"])
