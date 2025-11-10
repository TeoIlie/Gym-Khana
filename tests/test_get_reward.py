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
        env.track.centerline.spline.calc_arclength_inaccurate = lambda x, y: (55.0, 0)

        reward = env._get_reward()

        # Restore original method
        env.track.centerline.spline.calc_arclength_inaccurate = original_calc

        print(f"\nNormal forward progress test:")
        print(f"  Track length: {track_length:.2f}m")
        print(f"  Last s: {last_s:.2f}m")
        print(f"  Current s: 55.00m")
        print(f"  Expected reward: ~5.00m")
        print(f"  Actual reward: {reward:.2f}m")

        assert np.isclose(reward, 5.0), f"Forward progress should give ~5m reward, got {reward}"

    def test_backward_wraparound_lap_completion(self, env):
        """
        CRITICAL BUG TEST: Test that completing a lap gives positive reward.

        This test WILL FAIL with current buggy implementation because (assuming 200m track length):
        - Agent at s=195m (near finish)
        - Moves to s=5m (crossed finish line)
        - Raw progress: 5 - 195 = -190m
        - Buggy code doesn't detect this, so reward = -190m ❌
        - Should be corrected to: -190 + 200 = 10m ✓
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

        print(f"\n🔴 CRITICAL BUG TEST - Lap completion:")
        print(f"  Track length: {track_length:.2f}m")
        print(f"  Last s: {last_s:.2f}m (near finish)")
        print(f"  Current s: 5.00m (crossed finish)")
        print(f"  Raw progress: {5.0 - last_s:.2f}m")
        print(f"  Expected reward: ~10.00m (5m before + 5m after finish)")
        print(f"  Actual reward: {reward:.2f}m")

        if reward < 0:
            print(f"  ❌ BUG CONFIRMED: Negative reward for lap completion!")

        # This assertion WILL FAIL with buggy code
        assert reward > 0, (
            f"Lap completion should give positive reward! "
            f"Got {reward:.2f}m instead of ~10. "
            f"This indicates a backward wraparound bug."
        )
        assert np.isclose(reward, 10), f"Lap completion reward incorrect, got {reward}"

    def test_backward_wraparound_mid_track_to_start(self, env):
        """Test wraparound from middle of track back to start (edge case)"""
        track_length = env.track.centerline.spline.s[-1]

        # Set up scenario: from 190m to 10m
        last_s = track_length - 10.0
        env.last_s = [last_s]
        env.collisions = np.zeros(1)

        # Mock position 10m from start
        env.track.centerline.spline.calc_arclength_inaccurate = lambda x, y: (10.0, 0)

        reward = env._get_reward()

        expected_reward = 20.0  # 10m to finish + 10m from start

        print(f"\nBackward wraparound (10m before finish to 10m after):")
        print(f"  Track length: {track_length:.2f}m")
        print(f"  Last s: {last_s:.2f}m")
        print(f"  Current s: 10.00m")
        print(f"  Expected reward: ~{expected_reward:.2f}m")
        print(f"  Actual reward: {reward:.2f}m")

        assert np.isclose(reward, expected_reward), f"Wraparound should give {expected_reward} reward, got {reward}"

    def test_no_wraparound_near_finish(self, env):
        """Test that movement near finish line (without crossing) works correctly"""
        track_length = env.track.centerline.spline.s[-1]

        # Both positions before finish line
        last_s = track_length - 10.0
        env.last_s = [last_s]
        env.collisions = np.zeros(1)

        # Move forward 5m, still before finish
        env.track.centerline.spline.calc_arclength_inaccurate = lambda x, y: (track_length - 5.0, 0)

        reward = env._get_reward()

        print(f"\nNo wraparound (near finish):")
        print(f"  Last s: {last_s:.2f}m")
        print(f"  Current s: {track_length - 5.0:.2f}m")
        print(f"  Expected reward: ~5.00m")
        print(f"  Actual reward: {reward:.2f}m")

        assert np.isclose(reward, 5.0), f"Expected reward 5.0, got {reward}"

    def test_collision_penalty(self, env):
        """Test that collision penalty is applied correctly"""
        # Set up forward progress with collision
        last_s = 50.0
        env.last_s = [last_s]
        env.collisions = np.array([1])  # Collision detected

        # Mock 5m forward progress
        env.track.centerline.spline.calc_arclength_inaccurate = lambda x, y: (55.0, 0)

        reward = env._get_reward()

        expected_reward = 5.0 - 1.0  # progress - collision penalty

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
            result = (55.0, 0) if mock_calc.call_count == 0 else (103.0, 0)
            mock_calc.call_count += 1
            return result

        env.track.centerline.spline.calc_arclength_inaccurate = mock_calc

        reward = env._get_reward()

        expected_reward = 5.0 + 3.0  # Both agents' progress

        print(f"\nMulti-agent collaborative reward:")
        print(f"  Agent 0: last_s={last_s_agent0:.2f}m, current_s=55.00m, progress=5.00m")
        print(f"  Agent 1: last_s={last_s_agent1:.2f}m, current_s=103.00m, progress=3.00m")
        print(f"  Expected total: {expected_reward:.2f}m")
        print(f"  Actual reward: {reward:.2f}m")

        assert np.isclose(reward, expected_reward), f"Expected reward {expected_reward}, got {reward}"


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
        env.track.centerline.spline.calc_arclength_inaccurate = lambda x, y: (95.0, 0)

        reward = env._get_reward()

        expected_reward = -5.0

        print(f"\nBackward motion (no wraparound):")
        print(f"  Last s: {last_s:.2f}m")
        print(f"  Current s: 95.00m")
        print(f"  Expected reward: {expected_reward:.2f}m")
        print(f"  Actual reward: {reward:.2f}m")

        assert np.isclose(reward, expected_reward), f"Expected reward {expected_reward}, got {reward}"


if __name__ == "__main__":
    # Run with pytest
    pytest.main([__file__, "-v", "-s"])
