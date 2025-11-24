"""
Unit tests for _get_reward() function in F110Env.

These tests expose the wraparound bug where completing a lap
gives a large negative reward instead of a small positive reward.
"""

import pytest
import numpy as np
import gymnasium as gym


class TestGetRewardWraparound:
    """Test suite for reward calculation wraparound handling"""

    @pytest.fixture
    def env(self):
        """Create a basic F110 environment for testing"""
        env = gym.make(
            "f1tenth_gym:f1tenth-v0",
            config={
                "map": "Spielberg",
                "num_agents": 1,
                "timestep": 0.01,
            },
            render_mode=None,
        )
        env.reset()
        # Unwrap to access private methods like _get_reward()
        return env.unwrapped

    def test_normal_forward_progress(self, env):
        """Test that normal forward progress gives positive reward"""
        track_length = env.track.centerline.spline.s[-1]

        # Set up normal forward motion scenario
        last_s = 50.0
        env.last_s = [last_s]
        env.poses_x = [env.track.centerline.xs[0]]
        env.poses_y = [env.track.centerline.ys[0]]
        env.collisions = np.zeros(1)

        # Mock calc_arclength to return position 55m (5m forward)
        original_calc = env.track.centerline.spline.calc_arclength_inaccurate
        env.track.centerline.spline.calc_arclength_inaccurate = lambda x, y: (50.1, 0)

        reward = env._get_reward()

        # Restore original method
        env.track.centerline.spline.calc_arclength_inaccurate = original_calc

        print(f"\nNormal forward progress test:")
        print(f"  Track length: {track_length:.2f}m")
        print(f"  Last s: {last_s:.2f}m")
        print(f"  Current s: 50.1m")
        print(f"  Expected reward: ~0.1m")
        print(f"  Actual reward: {reward:.2f}m")

        assert np.isclose(reward, 0.1), f"Forward progress should give ~0.1m reward, got {reward}"

    def test_backward_wraparound_lap_completion(self, env):
        """
        Test that completing a lap gives positive reward (capped by max_progress).

        Scenario:
        - Agent at s=338m (5m before finish on ~343m track)
        - Moves to s=5m (crossed finish line)
        - Raw progress: 5 - 338 = -333m (large negative due to wraparound)
        - Wraparound correction: -333 + 343 = 10m ✓
        - Clipping with margin=10: np.clip(10, -2, 2) = 2m (max_progress)

        Note: The actual progress is 10m, but it gets clipped to max_progress (2m)
        because margin=10 limits progress to v_max * timestep * 10 = 20 * 0.01 * 10 = 2m.
        This is expected behavior - the clipping acts as a safety net.
        """
        track_length = env.track.centerline.spline.s[-1]

        # Set up lap completion scenario
        # Agent is 5m before finish line
        last_s = track_length - 5.0
        env.last_s = [last_s]
        env.poses_x = [env.track.centerline.xs[0]]
        env.poses_y = [env.track.centerline.ys[0]]
        env.collisions = np.zeros(1)

        # Mock calc_arclength to return position 5m after start (crossed finish)
        env.track.centerline.spline.calc_arclength_inaccurate = lambda x, y: (5.0, 0)

        reward = env._get_reward()

        # Calculate expected reward (clipped to max_progress)
        max_progress = env.params["v_max"] * env.timestep * 10.0  # margin=10
        expected_reward = max_progress  # 2.0m

        print(f"\n✓ Lap completion with clipping:")
        print(f"  Track length: {track_length:.2f}m")
        print(f"  Last s: {last_s:.2f}m (near finish)")
        print(f"  Current s: 5.00m (crossed finish)")
        print(f"  Raw progress: {5.0 - last_s:.2f}m")
        print(f"  Corrected progress (before clip): ~10.00m")
        print(f"  Max progress (margin=10): {max_progress:.2f}m")
        print(f"  Expected reward (after clip): {expected_reward:.2f}m")
        print(f"  Actual reward: {reward:.2f}m")

        # Verify wraparound was corrected and reward is positive (not large negative)
        assert reward > 0, (
            f"Lap completion should give positive reward! "
            f"Got {reward:.2f}m. This indicates wraparound correction failed."
        )

        # Verify reward matches clipped max_progress
        assert np.isclose(
            reward, expected_reward, atol=0.01
        ), f"Expected clipped reward of {expected_reward:.2f}m, got {reward:.2f}m"

    def test_backward_wraparound_mid_track_to_start(self, env):
        """
        Test wraparound from middle of track back to start (capped by max_progress).

        Scenario:
        - Agent at s=333m (10m before finish on ~343m track)
        - Moves to s=10m (crossed finish line)
        - Raw progress: 10 - 333 = -323m (large negative due to wraparound)
        - Wraparound correction: -323 + 343 = 20m ✓
        - Clipping with margin=10: np.clip(20, -2, 2) = 2m (max_progress)

        Note: Similar to lap completion test, the 20m progress gets clipped to 2m.
        """
        track_length = env.track.centerline.spline.s[-1]

        # Set up scenario: from 10m before finish to 10m after start
        last_s = track_length - 10.0
        env.last_s = [last_s]
        env.collisions = np.zeros(1)

        # Mock position 10m from start
        env.track.centerline.spline.calc_arclength_inaccurate = lambda x, y: (10.0, 0)

        reward = env._get_reward()

        # Calculate expected reward (clipped to max_progress)
        max_progress = env.params["v_max"] * env.timestep * 10.0  # margin=10
        expected_reward = max_progress  # 2.0m (not 20m due to clipping)

        print(f"\n✓ Backward wraparound with clipping (10m before finish to 10m after):")
        print(f"  Track length: {track_length:.2f}m")
        print(f"  Last s: {last_s:.2f}m")
        print(f"  Current s: 10.00m")
        print(f"  Corrected progress (before clip): ~20.00m")
        print(f"  Max progress (margin=10): {max_progress:.2f}m")
        print(f"  Expected reward (after clip): {expected_reward:.2f}m")
        print(f"  Actual reward: {reward:.2f}m")

        # Verify reward is positive (wraparound corrected)
        assert reward > 0, f"Wraparound should give positive reward, got {reward}"

        # Verify reward matches clipped max_progress
        assert np.isclose(
            reward, expected_reward, atol=0.01
        ), f"Expected clipped reward of {expected_reward:.2f}m, got {reward:.2f}m"

    def test_no_wraparound_(self, env):
        """Test that movement near finish line (without crossing) works correctly"""
        track_length = env.track.centerline.spline.s[-1]

        # Both positions before finish line
        last_s = track_length - 10.0
        env.last_s = [last_s]
        env.collisions = np.zeros(1)

        # Move forward 5m, still before finish
        env.track.centerline.spline.calc_arclength_inaccurate = lambda x, y: (track_length - 9.9, 0)

        reward = env._get_reward()

        print(f"\nNo wraparound (near finish):")
        print(f"  Last s: {last_s:.2f}m")
        print(f"  Current s: {track_length - 9.9:.2f}m")
        print(f"  Expected reward: ~0.1m")
        print(f"  Actual reward: {reward:.2f}m")

        assert np.isclose(reward, 0.1), f"Expected reward 0.1, got {reward}"

    def test_collision_penalty(self, env):
        """Test that collision penalty is applied correctly"""
        # Set up forward progress with collision
        last_s = 50.0
        env.last_s = [last_s]
        env.collisions = np.array([1])  # Collision detected

        # Mock 5m forward progress
        env.track.centerline.spline.calc_arclength_inaccurate = lambda x, y: (50.1, 0)

        reward = env._get_reward()

        expected_reward = 50.1 - last_s - 1.0  # progress - collision penalty

        print(f"\nCollision penalty test:")
        print(f"  Last s: {last_s:.2f}m")
        print(f"  Progress: 5.00m")
        print(f"  Collision penalty: 1.00")
        print(f"  Expected reward: {expected_reward:.2f}m")
        print(f"  Actual reward: {reward:.2f}m")

        assert np.isclose(reward, expected_reward), f"Expected reward {expected_reward}, got {reward}"

    def test_multi_agent_collaborative(self, env):
        """Test that multi-agent rewards are summed correctly"""
        env.close()

        # Create 2-agent environment
        wrapped_env = gym.make(
            "f1tenth_gym:f1tenth-v0",
            config={
                "map": "Spielberg",
                "num_agents": 2,
            },
            render_mode=None,
        )
        wrapped_env.reset()
        env = wrapped_env.unwrapped

        # Set up both agents with forward progress
        last_s_agent0 = 50.0
        last_s_agent1 = 100.0
        env.last_s = [last_s_agent0, last_s_agent1]
        env.collisions = np.zeros(2)

        # Mock both agents moving forward
        def mock_calc(x, y):
            # Simple alternating return for two agents
            if not hasattr(mock_calc, "call_count"):
                mock_calc.call_count = 0
            result = (50.1, 0) if mock_calc.call_count == 0 else (100.1, 0)
            mock_calc.call_count += 1
            return result

        env.track.centerline.spline.calc_arclength_inaccurate = mock_calc

        reward = env._get_reward()

        expected_reward = 0.2  # Both agents' progress

        print(f"\nMulti-agent collaborative reward:")
        print(f"  Expected total: {expected_reward:.2f}m")
        print(f"  Actual reward: {reward:.2f}m")

        assert np.isclose(reward, expected_reward), f"Expected reward {expected_reward}, got {reward}"


class TestCorrectWraparoundProg:
    """Test suite for _correct_wraparound_prog() method"""

    @pytest.fixture
    def env(self):
        """Create environment for testing"""
        env = gym.make(
            "f1tenth_gym:f1tenth-v0",
            config={
                "map": "Spielberg",
                "num_agents": 1,
                "timestep": 0.01,
            },
            render_mode=None,
        )
        env.reset()
        return env.unwrapped

    def test_forward_wraparound_correction(self, env):
        """
        Test forward wraparound: car crosses finish line going forward.

        Scenario: last_s=99.9m, current_s=0.1m on 100m track
        Raw progress: 0.1 - 99.9 = -99.8m (large negative)
        Should correct to: -99.8 + 100 = 0.2m (actual forward progress)
        """
        track_length = env.track.centerline.spline.s[-1]

        # Simulate crossing finish line forward
        last_s = track_length - 0.1  # 0.1m before finish
        current_s = 0.1  # 0.1m after start
        prog = current_s - last_s  # Large negative value

        corrected = env._correct_wraparound_prog(prog=prog, track_length=track_length)
        expected = 0.2  # Actual distance traveled

        print(f"\n✓ Forward wraparound correction test:")
        print(f"  Track length: {track_length:.2f}m")
        print(f"  Last s: {last_s:.2f}m (near finish)")
        print(f"  Current s: {current_s:.2f}m (crossed to start)")
        print(f"  Raw progress: {prog:.2f}m")
        print(f"  Corrected progress: {corrected:.2f}m")
        print(f"  Expected: {expected:.2f}m")

        assert np.isclose(corrected, expected, atol=0.01), (
            f"Forward wraparound correction failed: " f"expected {expected:.2f}m, got {corrected:.2f}m"
        )

    def test_backward_wraparound_correction(self, env):
        """
        Test backward wraparound: car crosses finish line going backward.

        Scenario: last_s=0.1m, current_s=99.9m on 100m track
        Raw progress: 99.9 - 0.1 = 99.8m (large positive)
        Should correct to: 99.8 - 100 = -0.2m (actual backward progress)
        """
        track_length = env.track.centerline.spline.s[-1]

        # Simulate crossing finish line backward
        last_s = 0.1  # 0.1m after start
        current_s = track_length - 0.1  # 0.1m before finish (reversed across line)
        prog = current_s - last_s  # Large positive value

        corrected = env._correct_wraparound_prog(prog=prog, track_length=track_length)
        expected = -0.2  # Actual distance traveled backward

        print(f"\n✓ Backward wraparound correction test:")
        print(f"  Track length: {track_length:.2f}m")
        print(f"  Last s: {last_s:.2f}m (just after start)")
        print(f"  Current s: {current_s:.2f}m (reversed to near finish)")
        print(f"  Raw progress: {prog:.2f}m")
        print(f"  Corrected progress: {corrected:.2f}m")
        print(f"  Expected: {expected:.2f}m")

        assert np.isclose(corrected, expected, atol=0.01), (
            f"Backward wraparound correction failed: " f"expected {expected:.2f}m, got {corrected:.2f}m"
        )

    def test_normal_forward_progress_no_correction(self, env):
        """
        Test that normal forward progress is not modified.

        Progress within physics bounds should pass through unchanged.
        """
        track_length = env.track.centerline.spline.s[-1]

        # Normal forward progress: 0.15m at v=15m/s with dt=0.01s
        prog_normal = 0.15
        corrected = env._correct_wraparound_prog(prog=prog_normal, track_length=track_length)

        print(f"\n✓ Normal forward progress (no correction):")
        print(f"  Raw progress: {prog_normal:.2f}m")
        print(f"  Corrected progress: {corrected:.2f}m")
        print(f"  Max allowed: {env.params['v_max'] * env.timestep * 1.05:.4f}m")

        assert corrected == prog_normal, (
            f"Normal progress should not be modified: " f"expected {prog_normal}m, got {corrected}m"
        )

    def test_normal_backward_progress_no_correction(self, env):
        """
        Test that normal backward progress is not modified.

        Small backward motion within physics bounds should pass through unchanged.
        """
        track_length = env.track.centerline.spline.s[-1]

        # Normal backward progress: -0.04m at v=-4m/s with dt=0.01s
        prog_backward = -0.04
        corrected = env._correct_wraparound_prog(prog=prog_backward, track_length=track_length)

        print(f"\n✓ Normal backward progress (no correction):")
        print(f"  Raw progress: {prog_backward:.2f}m")
        print(f"  Corrected progress: {corrected:.2f}m")
        print(f"  Min allowed: {-abs(env.params['v_min']) * env.timestep * 1.05:.4f}m")

        assert corrected == prog_backward, (
            f"Normal backward progress should not be modified: " f"expected {prog_backward}m, got {corrected}m"
        )

    def test_zero_progress_no_correction(self, env):
        """Test that zero progress (stationary vehicle) is not modified."""
        track_length = env.track.centerline.spline.s[-1]

        prog_zero = 0.0
        corrected = env._correct_wraparound_prog(prog=prog_zero, track_length=track_length)

        print(f"\n✓ Zero progress (stationary):")
        print(f"  Raw progress: {prog_zero:.2f}m")
        print(f"  Corrected progress: {corrected:.2f}m")

        assert corrected == prog_zero, "Zero progress should not be modified"

    def test_boundary_threshold_forward(self, env):
        """
        Test progress exactly at the forward wraparound threshold.

        Progress just below the threshold should NOT trigger correction.
        """
        track_length = env.track.centerline.spline.s[-1]

        # Exactly at threshold (should not trigger correction)
        max_backward = abs(env.params["v_min"]) * env.timestep * 1.05
        prog_at_threshold = -max_backward

        corrected = env._correct_wraparound_prog(prog=prog_at_threshold, track_length=track_length)

        print(f"\n✓ Boundary test (at forward threshold):")
        print(f"  Threshold: {-max_backward:.4f}m")
        print(f"  Progress: {prog_at_threshold:.4f}m")
        print(f"  Corrected: {corrected:.4f}m")

        # At exact threshold, should NOT trigger wraparound (< not <=)
        assert corrected == prog_at_threshold, "Progress at threshold should not be corrected"

    def test_boundary_threshold_backward(self, env):
        """
        Test progress exactly at the backward wraparound threshold.

        Progress just below the threshold should NOT trigger correction.
        """
        track_length = env.track.centerline.spline.s[-1]

        # Exactly at threshold (should not trigger correction)
        max_forward = env.params["v_max"] * env.timestep * 1.05
        prog_at_threshold = max_forward

        corrected = env._correct_wraparound_prog(prog=prog_at_threshold, track_length=track_length)

        print(f"\n✓ Boundary test (at backward threshold):")
        print(f"  Threshold: {max_forward:.4f}m")
        print(f"  Progress: {prog_at_threshold:.4f}m")
        print(f"  Corrected: {corrected:.4f}m")

        # At exact threshold, should NOT trigger wraparound (> not >=)
        assert corrected == prog_at_threshold, "Progress at threshold should not be corrected"


class TestGetRewardEdgeCases:
    """Test edge cases in reward calculation"""

    @pytest.fixture
    def env(self):
        """Create environment for testing"""
        env = gym.make("f1tenth_gym:f1tenth-v0", config={"map": "Spielberg", "num_agents": 1}, render_mode=None)
        env.reset()
        return env.unwrapped

    def test_zero_progress(self, env):
        """Test that standing still gives zero reward"""
        last_s = 50.0
        env.last_s = [last_s]
        env.collisions = np.zeros(1)

        # No movement
        env.track.centerline.spline.calc_arclength_inaccurate = lambda x, y: (50.0, 0)

        reward = env._get_reward()

        print(f"\nZero progress test:")
        print(f"  Last s: {last_s:.2f}m")
        print(f"  Current s: 50.00m")
        print(f"  Expected reward: 0.00m")
        print(f"  Actual reward: {reward:.2f}m")

        assert np.isclose(reward, 0.0), f"Zero progress should give 0 reward, got {reward}"

    def test_backward_motion_no_wraparound(self, env):
        """Test that moving backward (without wraparound) gives negative reward"""
        last_s = 100.0
        env.last_s = [last_s]
        env.collisions = np.zeros(1)

        # Moved backward 5m
        env.track.centerline.spline.calc_arclength_inaccurate = lambda x, y: (99.9, 0)

        reward = env._get_reward()

        expected_reward = -0.1

        print(f"\nBackward motion (no wraparound):")
        print(f"  Last s: {last_s:.2f}m")
        print(f"  Expected reward: {expected_reward:.2f}m")
        print(f"  Actual reward: {reward:.2f}m")

        assert np.isclose(reward, expected_reward), f"Expected reward {expected_reward}, got {reward}"


if __name__ == "__main__":
    # Run with pytest
    pytest.main([__file__, "-v", "-s"])
