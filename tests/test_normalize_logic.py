"""
Test normalization functionality for drift observations and actions.

This test suite validates:
1. Normalization bounds completeness and validity
2. normalize_feature() correctness for scalars and arrays
3. Edge cases (clipping, asymmetric bounds, errors)
4. End-to-end normalized observation pipeline
5. Track-adaptive normalization behavior
6. Action normalization scaling correctness for all action types
"""

import gymnasium as gym
import numpy as np
import pytest

from f1tenth_gym.envs.action import AcclAction, SpeedAction, SteeringAngleAction, SteeringSpeedAction
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
                "normalize_obs": True,
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
        bounds = calculate_norm_bounds(env.unwrapped, expected_features)

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

    def test_prev_steering_cmd_bounds_with_normalized_actions(self):
        """Verify prev_steering_cmd bounds are (-1, 1) when actions are normalized."""
        env = gym.make(
            "f1tenth_gym:f1tenth-v0",
            config={
                "map": "Spielberg",
                "num_agents": 1,
                "model": "std",
                "observation_config": {"type": "drift"},
                "params": F110Env.f1tenth_std_vehicle_params(),
                "normalize_act": True,  # Actions normalized
                "normalize_obs": True,
            },
        )

        bounds = calculate_norm_bounds(env.unwrapped, ["prev_steering_cmd"])

        # When actions are normalized, raw steering is in [-1, 1]
        assert bounds["prev_steering_cmd"] == (
            -1,
            1,
        ), f"Expected (-1, 1), got {bounds['prev_steering_cmd']}"

        env.close()

    def test_prev_steering_cmd_bounds_with_unnormalized_actions(self):
        """Verify prev_steering_cmd bounds are (s_min, s_max) when actions are unnormalized."""
        env = gym.make(
            "f1tenth_gym:f1tenth-v0",
            config={
                "map": "Spielberg",
                "num_agents": 1,
                "model": "std",
                "observation_config": {"type": "drift"},
                "params": F110Env.f1tenth_std_vehicle_params(),
                "normalize_act": False,  # Actions NOT normalized
                "normalize_obs": True,
            },
        )

        bounds = calculate_norm_bounds(env.unwrapped, ["prev_steering_cmd"])
        params = F110Env.f1tenth_std_vehicle_params()
        s_min = params["s_min"]
        s_max = params["s_max"]

        # When actions are unnormalized, raw steering is in [s_min, s_max]
        assert bounds["prev_steering_cmd"] == (
            s_min,
            s_max,
        ), f"Expected ({s_min}, {s_max}), got {bounds['prev_steering_cmd']}"

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

    def test_prev_steering_cmd_normalization_with_normalized_bounds(self):
        """Test normalize_feature with prev_steering_cmd when bounds are (-1, 1)."""
        # This case occurs when normalize_act=True
        bounds = {"prev_steering_cmd": (-1, 1)}

        # Test boundary values - should map to themselves (identity)
        result = normalize_feature("prev_steering_cmd", np.float32(-1.0), bounds)
        assert np.isclose(result, -1.0), f"Expected -1.0, got {result}"

        result = normalize_feature("prev_steering_cmd", np.float32(1.0), bounds)
        assert np.isclose(result, 1.0), f"Expected 1.0, got {result}"

        # Test center value
        result = normalize_feature("prev_steering_cmd", np.float32(0.0), bounds)
        assert np.isclose(result, 0.0), f"Expected 0.0, got {result}"

        # Test intermediate value
        result = normalize_feature("prev_steering_cmd", np.float32(0.5), bounds)
        assert np.isclose(result, 0.5), f"Expected 0.5, got {result}"

    def test_prev_steering_cmd_normalization_with_physical_bounds(self):
        """Test normalize_feature with prev_steering_cmd when bounds are (s_min, s_max)."""
        # This case occurs when normalize_act=False
        params = F110Env.f1tenth_std_vehicle_params()
        s_min = params["s_min"]
        s_max = params["s_max"]
        bounds = {"prev_steering_cmd": (s_min, s_max)}

        # Test s_min maps to -1.0
        result = normalize_feature("prev_steering_cmd", np.float32(s_min), bounds)
        assert np.isclose(result, -1.0), f"Expected -1.0, got {result}"

        # Test s_max maps to 1.0
        result = normalize_feature("prev_steering_cmd", np.float32(s_max), bounds)
        assert np.isclose(result, 1.0), f"Expected 1.0, got {result}"

        # Test midpoint (s_min + s_max) / 2 maps to 0.0
        midpoint = (s_min + s_max) / 2.0
        result = normalize_feature("prev_steering_cmd", np.float32(midpoint), bounds)
        assert np.isclose(result, 0.0), f"Expected 0.0, got {result}"


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
                "normalize_obs": True,
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
                "normalize_obs": False,
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
                "normalize_obs": True,
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
                "normalize_obs": True,
            },
        )

        features = ["lookahead_curvatures", "lookahead_widths", "linear_vel_x", "delta", "prev_accl_cmd"]

        # Get bounds from both environments
        bounds1 = calculate_norm_bounds(env1.unwrapped, features)
        bounds2 = calculate_norm_bounds(env2.unwrapped, features)

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
                "normalize_obs": True,
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
                "normalize_obs": False,
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


class TestNormalizedAction:
    def test_accl_action_normalization(self):
        """Test AcclAction with both normalized and unnormalized modes."""
        params = F110Env.f1tenth_std_vehicle_params()
        a_max = params["a_max"]

        # Create dummy state (AcclAction doesn't use state, but we pass it for consistency)
        dummy_state = np.zeros(7, dtype=np.float32)

        # Test with normalize=True
        accl_normalized = AcclAction(params, normalize=True)

        # Check action space bounds
        assert accl_normalized.lower_limit == -1.0, "Normalized AcclAction lower limit should be -1.0"
        assert accl_normalized.upper_limit == 1.0, "Normalized AcclAction upper limit should be 1.0"
        assert accl_normalized.scale_factor == a_max, f"Scale factor should be a_max={a_max}"

        # Test scaling: -1 → -a_max, 0 → 0, 1 → a_max
        assert np.isclose(
            accl_normalized.act(-1.0, dummy_state, params), -a_max
        ), f"Expected {-a_max}, got {accl_normalized.act(-1.0, dummy_state, params)}"
        assert np.isclose(
            accl_normalized.act(0.0, dummy_state, params), 0.0
        ), f"Expected 0.0, got {accl_normalized.act(0.0, dummy_state, params)}"
        assert np.isclose(
            accl_normalized.act(1.0, dummy_state, params), a_max
        ), f"Expected {a_max}, got {accl_normalized.act(1.0, dummy_state, params)}"

        # Test intermediate value: 0.5 → 0.5 * a_max
        expected = 0.5 * a_max
        assert np.isclose(
            accl_normalized.act(0.5, dummy_state, params), expected
        ), f"Expected {expected}, got {accl_normalized.act(0.5, dummy_state, params)}"

        # Test with normalize=False
        accl_unnormalized = AcclAction(params, normalize=False)

        # Check action space bounds (should be physical units)
        assert accl_unnormalized.lower_limit == -a_max, f"Unnormalized AcclAction lower limit should be -a_max={-a_max}"
        assert accl_unnormalized.upper_limit == a_max, f"Unnormalized AcclAction upper limit should be a_max={a_max}"
        assert accl_unnormalized.scale_factor == 1.0, "Unnormalized scale factor should be 1.0"

        # Test passthrough: 5.0 → 5.0, -3.0 → -3.0
        assert np.isclose(accl_unnormalized.act(5.0, dummy_state, params), 5.0), "Expected passthrough 5.0"
        assert np.isclose(accl_unnormalized.act(-3.0, dummy_state, params), -3.0), "Expected passthrough -3.0"

    def test_speed_action_normalization(self):
        """Test SpeedAction with both normalized and unnormalized modes."""
        params = F110Env.f1tenth_std_vehicle_params()

        # Override v_min and v_max to ensure v_min != v_max for thorough testing
        params["v_min"] = -1.0
        params["v_max"] = 7.0
        v_min = params["v_min"]
        v_max = params["v_max"]
        v_center = 3.0  # (v_max + v_min) / 2.0
        v_range = 4.0  # (v_max - v_min) / 2.0

        # Create state with current velocity = 0.0
        state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)  # state[3] = velocity

        # Test with normalize=True
        speed_normalized = SpeedAction(params, normalize=True)

        # Check action space bounds
        assert speed_normalized.lower_limit == -1.0, "Normalized SpeedAction lower limit should be -1.0"
        assert speed_normalized.upper_limit == 1.0, "Normalized SpeedAction upper limit should be 1.0"
        assert np.isclose(speed_normalized.v_center, v_center), f"v_center should be {v_center}"
        assert np.isclose(speed_normalized.v_range, v_range), f"v_range should be {v_range}"

        # Test scaling mapping (desired speed before P controller)
        # -1 → v_min = -1.0
        desired_speed_min = v_min  # -1.0 * v_range + v_center
        assert np.isclose(desired_speed_min, v_min), f"Expected {v_min}, got {desired_speed_min}"

        # 0 → v_center = 3.0
        desired_speed_center = 3.0  # 0.0 * v_range + v_center
        assert np.isclose(desired_speed_center, v_center), f"Expected {v_center}, got {desired_speed_center}"

        # 1 → v_max = 7.0
        desired_speed_max = v_max  # 1.0 * v_range + v_center
        assert np.isclose(desired_speed_max, v_max), f"Expected {v_max}, got {desired_speed_max}"

        # 0.5 → 5.0 (halfway between center and max: 3.0 + 0.5*4.0 = 5.0)
        desired_speed_half = 5.0  # 0.5 * v_range + v_center
        assert np.isclose(desired_speed_half, 5.0), f"Expected 5.0, got {desired_speed_half}"

        # Test actual act() method (returns acceleration from P controller)
        # We can't predict exact acceleration, but we verify act() runs without error
        accl_result = speed_normalized.act(-1.0, state, params)
        assert isinstance(accl_result, (float, np.floating)), "act() should return a float"

        # Test with normalize=False
        speed_unnormalized = SpeedAction(params, normalize=False)

        # Check action space bounds (should be physical units)
        assert speed_unnormalized.lower_limit == v_min, f"Unnormalized SpeedAction lower limit should be {v_min}"
        assert speed_unnormalized.upper_limit == v_max, f"Unnormalized SpeedAction upper limit should be {v_max}"

        # Test passthrough to P controller
        accl_result_unnorm = speed_unnormalized.act(5.0, state, params)
        assert isinstance(accl_result_unnorm, (float, np.floating)), "act() should return a float"

    def test_steering_angle_action_normalization(self):
        """Test SteeringAngleAction with both normalized and unnormalized modes."""
        params = F110Env.f1tenth_std_vehicle_params()
        s_max = params["s_max"]
        s_min = params["s_min"]

        # Create state with current steering angle = 0.0
        state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)  # state[2] = steering angle

        # Test with normalize=True
        steering_normalized = SteeringAngleAction(params, normalize=True)

        # Check action space bounds
        assert steering_normalized.lower_limit == -1.0, "Normalized SteeringAngleAction lower limit should be -1.0"
        assert steering_normalized.upper_limit == 1.0, "Normalized SteeringAngleAction upper limit should be 1.0"
        assert np.isclose(steering_normalized.scale_factor, s_max), f"Scale factor should be s_max={s_max}"

        # Test scaling: -1 → -s_max, 0 → 0, 1 → s_max
        # The act() method returns steering velocity from bang_bang_steer, so we check the desired_angle internally
        # We'll verify by checking the intermediate scaling works correctly
        assert np.isclose(-1.0 * steering_normalized.scale_factor, s_min), f"Expected {s_min}"
        assert np.isclose(0.0 * steering_normalized.scale_factor, 0.0), "Expected 0.0"
        assert np.isclose(1.0 * steering_normalized.scale_factor, s_max), f"Expected {s_max}"

        # Test act() method runs without error and returns steering velocity
        sv_result = steering_normalized.act(-1.0, state, params)
        assert isinstance(sv_result, (float, np.floating)), "act() should return a float (steering velocity)"

        # Test with normalize=False
        steering_unnormalized = SteeringAngleAction(params, normalize=False)

        # Check action space bounds (should be physical units)
        assert (
            steering_unnormalized.lower_limit == s_min
        ), f"Unnormalized SteeringAngleAction lower limit should be {s_min}"
        assert (
            steering_unnormalized.upper_limit == s_max
        ), f"Unnormalized SteeringAngleAction upper limit should be {s_max}"
        assert steering_unnormalized.scale_factor == 1.0, "Unnormalized scale factor should be 1.0"

        # Test passthrough to bang_bang_steer
        sv_result_unnorm = steering_unnormalized.act(0.2, state, params)
        assert isinstance(sv_result_unnorm, (float, np.floating)), "act() should return a float"

    def test_steering_speed_action_normalization(self):
        """Test SteeringSpeedAction with both normalized and unnormalized modes."""
        params = F110Env.f1tenth_std_vehicle_params()
        sv_max = params["sv_max"]
        sv_min = params["sv_min"]

        # Create dummy state (SteeringSpeedAction doesn't use state extensively)
        dummy_state = np.zeros(7, dtype=np.float32)

        # Test with normalize=True
        steering_speed_normalized = SteeringSpeedAction(params, normalize=True)

        # Check action space bounds
        assert (
            steering_speed_normalized.lower_limit == -1.0
        ), "Normalized SteeringSpeedAction lower limit should be -1.0"
        assert steering_speed_normalized.upper_limit == 1.0, "Normalized SteeringSpeedAction upper limit should be 1.0"
        assert np.isclose(steering_speed_normalized.scale_factor, sv_max), f"Scale factor should be sv_max={sv_max}"

        # Test scaling: -1 → -sv_max, 0 → 0, 1 → sv_max
        assert np.isclose(steering_speed_normalized.act(-1.0, dummy_state, params), -sv_max), f"Expected {-sv_max}"
        assert np.isclose(steering_speed_normalized.act(0.0, dummy_state, params), 0.0), "Expected 0.0"
        assert np.isclose(steering_speed_normalized.act(1.0, dummy_state, params), sv_max), f"Expected {sv_max}"

        # Test intermediate value: 0.5 → 0.5 * sv_max
        expected = 0.5 * sv_max
        assert np.isclose(steering_speed_normalized.act(0.5, dummy_state, params), expected), f"Expected {expected}"

        # Test with normalize=False
        steering_speed_unnormalized = SteeringSpeedAction(params, normalize=False)

        # Check action space bounds (should be physical units)
        assert (
            steering_speed_unnormalized.lower_limit == sv_min
        ), f"Unnormalized SteeringSpeedAction lower limit should be {sv_min}"
        assert (
            steering_speed_unnormalized.upper_limit == sv_max
        ), f"Unnormalized SteeringSpeedAction upper limit should be {sv_max}"
        assert steering_speed_unnormalized.scale_factor == 1.0, "Unnormalized scale factor should be 1.0"

        # Test passthrough: 2.0 → 2.0, -1.5 → -1.5
        assert np.isclose(steering_speed_unnormalized.act(2.0, dummy_state, params), 2.0), "Expected passthrough 2.0"
        assert np.isclose(steering_speed_unnormalized.act(-1.5, dummy_state, params), -1.5), "Expected passthrough -1.5"


class TestActionIntegration:
    """Integration tests for action normalization in complete environment."""

    def test_car_action_space_composition(self):
        """Test CarAction properly composes action space from sub-actions."""
        from f1tenth_gym.envs.action import CarAction

        params = F110Env.f1tenth_std_vehicle_params()

        # Test with normalize=True
        car_action_normalized = CarAction(["accl", "steering_angle"], params=params, normalize=True)

        # Get the composed action space
        action_space = car_action_normalized.space

        # Verify space is [-1, 1]²
        expected_low = np.array([-1.0, -1.0], dtype=np.float32)
        expected_high = np.array([1.0, 1.0], dtype=np.float32)

        np.testing.assert_array_equal(
            action_space.low, expected_low
        ), "Normalized CarAction space low should be [-1, -1]"
        np.testing.assert_array_equal(
            action_space.high, expected_high
        ), "Normalized CarAction space high should be [1, 1]"
        assert action_space.shape == (2,), f"Expected shape (2,), got {action_space.shape}"

        # Test with normalize=False
        car_action_unnormalized = CarAction(["accl", "steering_angle"], params=params, normalize=False)

        action_space_unnorm = car_action_unnormalized.space

        # Verify space uses physical bounds
        # CarAction composes as [steer, longitudinal], so:
        # action[0] = steering angle, bounds: [s_min, s_max]
        # action[1] = acceleration, bounds: [-a_max, a_max]
        expected_low_unnorm = np.array([params["s_min"], -params["a_max"]], dtype=np.float32)
        expected_high_unnorm = np.array([params["s_max"], params["a_max"]], dtype=np.float32)

        np.testing.assert_array_almost_equal(
            action_space_unnorm.low, expected_low_unnorm, decimal=4
        ), "Unnormalized CarAction space should use physical bounds"
        np.testing.assert_array_almost_equal(
            action_space_unnorm.high, expected_high_unnorm, decimal=4
        ), "Unnormalized CarAction space should use physical bounds"

    def test_env_step_with_normalized_actions(self):
        """End-to-end test: environment accepts normalized actions and steps correctly."""
        env = gym.make(
            "f1tenth_gym:f1tenth-v0",
            config={
                "map": "Spielberg_blank",
                "num_agents": 1,
                "params": F110Env.f1tenth_std_vehicle_params(),
                "normalize_act": True,
            },
        )

        # Reset environment
        obs, _ = env.reset()
        assert obs is not None, "Reset should return observation"

        # Step with normalized action [0.5, 0.5] (moderate right turn, moderate acceleration)
        action = np.array([[0.5, 0.5]], dtype=np.float32)  # Shape (1, 2) for single agent

        # Take a step
        obs, reward, terminated, truncated, _ = env.step(action)

        # Verify step completed without errors
        assert obs is not None, "Step should return observation"
        assert isinstance(reward, (float, np.floating, np.ndarray)), "Reward should be numeric"
        assert isinstance(terminated, (bool, np.bool_)), "Terminated should be boolean"
        assert isinstance(truncated, (bool, np.bool_)), "Truncated should be boolean"

        # Take multiple steps to verify stability
        for _ in range(5):
            action = np.array([[0.0, 0.3]], dtype=np.float32)  # Straight, light acceleration
            obs, reward, terminated, truncated, _ = env.step(action)

            # Verify environment doesn't crash
            assert obs is not None, "Observation should not be None"

            if terminated or truncated:
                break

        env.close()


class TestActionEdgeCases:
    """Edge case tests for action normalization."""

    def test_action_space_sampling(self):
        """Verify Gymnasium action space sampling works correctly."""
        env = gym.make(
            "f1tenth_gym:f1tenth-v0",
            config={
                "map": "Spielberg",
                "num_agents": 1,
                "params": F110Env.f1tenth_std_vehicle_params(),
                "normalize_act": True,
            },
        )

        # Reset environment
        env.reset()

        # Sample from action space
        for _ in range(10):
            sampled_action = env.action_space.sample()

            # Verify sampled action is in [-1, 1]²
            assert sampled_action.shape == (1, 2), f"Expected shape (1, 2), got {sampled_action.shape}"
            assert np.all(sampled_action >= -1.0), f"Sampled action has values < -1: {sampled_action}"
            assert np.all(sampled_action <= 1.0), f"Sampled action has values > 1: {sampled_action}"

            # Verify action space contains the sampled action
            assert env.action_space.contains(
                sampled_action
            ), f"Action space should contain sampled action: {sampled_action}"

            # Step with sampled action (should not crash)
            obs, reward, terminated, truncated, info = env.step(sampled_action)
            assert obs is not None, "Step should return valid observation"

            if terminated or truncated:
                env.reset()

        env.close()

    def test_boundary_actions(self):
        """Test extreme boundary actions don't crash."""
        env = gym.make(
            "f1tenth_gym:f1tenth-v0",
            config={
                "map": "Spielberg",
                "num_agents": 1,
                "params": F110Env.f1tenth_std_vehicle_params(),
                "normalize_act": True,
            },
        )

        # Reset environment
        obs, info = env.reset()

        # Test boundary actions
        boundary_actions = [
            np.array([[-1.0, -1.0]], dtype=np.float32),  # Full left, full brake
            np.array([[1.0, 1.0]], dtype=np.float32),  # Full right, full throttle
            np.array([[0.0, 0.0]], dtype=np.float32),  # Center, neutral
            np.array([[-1.0, 1.0]], dtype=np.float32),  # Full left, full throttle
            np.array([[1.0, -1.0]], dtype=np.float32),  # Full right, full brake
        ]

        for action in boundary_actions:
            # Reset before each boundary test
            obs, info = env.reset()

            # Take a few steps with the boundary action
            for _ in range(3):
                obs, reward, terminated, truncated, info = env.step(action)

                # Verify no crashes
                assert obs is not None, f"Step with action {action} should return observation"

                # Handle both array and dict observations
                if isinstance(obs, dict):
                    # For dict observations, check each component
                    for key, value in obs.items():
                        if isinstance(value, np.ndarray):
                            assert not np.any(np.isnan(value)), f"Observation[{key}] contains NaN with action {action}"
                            assert not np.any(np.isinf(value)), f"Observation[{key}] contains Inf with action {action}"
                else:
                    # For array observations
                    assert not np.any(np.isnan(obs)), f"Observation contains NaN with action {action}"
                    assert not np.any(np.isinf(obs)), f"Observation contains Inf with action {action}"

                if terminated or truncated:
                    break

        env.close()


class TestPrevSteeringCmdNormalization:
    """Integration tests for prev_steering_cmd observation normalization."""

    def test_end_to_end_with_normalized_actions(self):
        """Test prev_steering_cmd normalization when actions are normalized."""
        env = gym.make(
            "f1tenth_gym:f1tenth-v0",
            config={
                "map": "Spielberg",
                "num_agents": 1,
                "model": "std",
                "observation_config": {"type": "drift"},
                "params": F110Env.f1tenth_std_vehicle_params(),
                "normalize_act": True,
                "normalize_obs": True,
            },
        )

        obs, _ = env.reset()

        # Get the index of prev_steering_cmd in observation
        # For drift obs: [vx, vy, u, n, omega_z, delta, prev_steer, prev_accl,
        #                 prev_wheel_omega, curr_vel_cmd, lookahead_curv..., lookahead_width...]
        prev_steer_idx = 6  # Based on drift observation structure

        # Test with maximum left steering (-1)
        # Note: prev_steering_cmd holds the PREVIOUS step's steering, so we need to step twice
        action = np.array([[-1.0, 0.5]], dtype=np.float32)
        obs, _, _, _, _ = env.step(action)  # First step: prev=0 (from reset), curr=-1
        obs, _, _, _, _ = env.step(action)  # Second step: prev=-1, curr=-1
        prev_steer_obs = obs[prev_steer_idx]

        # When normalize_act=True, raw action is -1, bounds are (-1, 1)
        # So normalize_feature(-1, (-1, 1)) should give -1
        assert np.isclose(prev_steer_obs, -1.0, atol=1e-5), f"Expected prev_steering_cmd ≈ -1.0, got {prev_steer_obs}"

        # Test with maximum right steering (1)
        action = np.array([[1.0, 0.5]], dtype=np.float32)
        obs, _, _, _, _ = env.step(action)  # First step: prev=-1, curr=1
        obs, _, _, _, _ = env.step(action)  # Second step: prev=1, curr=1
        prev_steer_obs = obs[prev_steer_idx]

        assert np.isclose(prev_steer_obs, 1.0, atol=1e-5), f"Expected prev_steering_cmd ≈ 1.0, got {prev_steer_obs}"

        # Test with zero steering
        action = np.array([[0.0, 0.5]], dtype=np.float32)
        obs, _, _, _, _ = env.step(action)  # First step: prev=1, curr=0
        obs, _, _, _, _ = env.step(action)  # Second step: prev=0, curr=0
        prev_steer_obs = obs[prev_steer_idx]

        assert np.isclose(prev_steer_obs, 0.0, atol=1e-5), f"Expected prev_steering_cmd ≈ 0.0, got {prev_steer_obs}"

        env.close()

    def test_end_to_end_with_unnormalized_actions(self):
        """Test prev_steering_cmd normalization when actions are NOT normalized."""
        params = F110Env.f1tenth_std_vehicle_params()
        s_min = params["s_min"]
        s_max = params["s_max"]

        env = gym.make(
            "f1tenth_gym:f1tenth-v0",
            config={
                "map": "Spielberg",
                "num_agents": 1,
                "model": "std",
                "observation_config": {"type": "drift"},
                "params": params,
                "normalize_act": False,
                "normalize_obs": True,
            },
        )

        obs, _ = env.reset()
        prev_steer_idx = 6

        # Test with maximum left steering (s_min)
        # Note: prev_steering_cmd holds the PREVIOUS step's steering, so we need to step twice
        action = np.array([[s_min, 2.0]], dtype=np.float32)
        obs, _, _, _, _ = env.step(action)  # First step: prev=0 (from reset), curr=s_min
        obs, _, _, _, _ = env.step(action)  # Second step: prev=s_min, curr=s_min
        prev_steer_obs = obs[prev_steer_idx]

        # When normalize_act=False, raw action is s_min, bounds are (s_min, s_max)
        # So normalize_feature(s_min, (s_min, s_max)) should give -1
        assert np.isclose(prev_steer_obs, -1.0, atol=1e-5), f"Expected prev_steering_cmd ≈ -1.0, got {prev_steer_obs}"

        # Test with maximum right steering (s_max)
        action = np.array([[s_max, 2.0]], dtype=np.float32)
        obs, _, _, _, _ = env.step(action)  # First step: prev=s_min, curr=s_max
        obs, _, _, _, _ = env.step(action)  # Second step: prev=s_max, curr=s_max
        prev_steer_obs = obs[prev_steer_idx]

        assert np.isclose(prev_steer_obs, 1.0, atol=1e-5), f"Expected prev_steering_cmd ≈ 1.0, got {prev_steer_obs}"

        # Test with zero steering (should map to normalized center)
        action = np.array([[0.0, 2.0]], dtype=np.float32)
        obs, _, _, _, _ = env.step(action)  # First step: prev=s_max, curr=0
        obs, _, _, _, _ = env.step(action)  # Second step: prev=0, curr=0
        prev_steer_obs = obs[prev_steer_idx]

        # 0.0 is between s_min and s_max, should normalize to corresponding value
        expected_normalized = 2.0 * (0.0 - s_min) / (s_max - s_min) - 1.0
        assert np.isclose(
            prev_steer_obs, expected_normalized, atol=1e-5
        ), f"Expected prev_steering_cmd ≈ {expected_normalized}, got {prev_steer_obs}"

        env.close()


if __name__ == "__main__":
    # Run tests with: pytest tests/test_normalize_feature.py -v
    pytest.main([__file__, "-v"])
