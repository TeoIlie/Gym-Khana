import unittest
from unittest.mock import patch

import gymnasium as gym
import numpy as np


class TestFrenetModeReset(unittest.TestCase):
    """
    Tests for environment reset behavior when predictive_collision=False (Frenet mode).
    In this mode, the environment should reset ONLY when the ego agent exceeds track boundaries,
    NOT on lap completion or non-ego agent collisions.
    """

    def setUp(self):
        """Create test environment with Frenet boundary checking enabled."""
        self.env = gym.make(
            "f1tenth_gym:f1tenth-v0",
            config={
                "map": "Spielberg",
                "num_agents": 1,
                "observation_config": {"type": None},
                "reset_config": {"type": "rl_random_static"},
                "predictive_collision": False,  # Enable Frenet boundary checking
            },
        )
        self.env.reset()

    def tearDown(self):
        """Clean up environment."""
        self.env.close()

    def test_frenet_mode_reset_on_boundary_violation(self):
        """Test that _check_done returns True when ego agent exceeds track boundaries."""
        unwrapped = self.env.unwrapped

        # Mock track with known boundaries
        unwrapped.track.centerline.w_lefts = np.array([2.0] * 100)
        unwrapped.track.centerline.w_rights = np.array([2.0] * 100)
        unwrapped.track.centerline.ss = np.linspace(0, 100, 100)

        # Set ego agent poses
        unwrapped.poses_x = np.array([0.0])
        unwrapped.poses_y = np.array([0.0])
        unwrapped.poses_theta = np.array([0.0])

        # Mock: ego agent exceeds boundary (ey=2.5m > half_width=2.0m)
        with patch.object(unwrapped.track, "cartesian_to_frenet", return_value=(50.0, 2.5, 0.0)):
            # Update state to populate boundary_exceeded array
            unwrapped._update_state()
            # Check if done
            done, _, _ = unwrapped._check_done()

        self.assertTrue(done, "Environment should reset when ego agent exceeds track boundary in Frenet mode")

    def test_frenet_mode_no_reset_within_boundaries(self):
        """Test that _check_done returns False when ego agent is within track boundaries."""
        unwrapped = self.env.unwrapped

        # Mock track with known boundaries
        unwrapped.track.centerline.w_lefts = np.array([2.0] * 100)
        unwrapped.track.centerline.w_rights = np.array([2.0] * 100)
        unwrapped.track.centerline.ss = np.linspace(0, 100, 100)

        # Set ego agent poses
        unwrapped.poses_x = np.array([0.0])
        unwrapped.poses_y = np.array([0.0])
        unwrapped.poses_theta = np.array([0.0])

        # Initialize toggle_list to simulate no lap completion
        unwrapped.toggle_list = np.array([0])

        # Mock: ego agent within boundaries (ey=0.5m < half_width=2.0m)
        with patch.object(unwrapped.track, "cartesian_to_frenet", return_value=(50.0, 0.5, 0.0)):
            # Update state to populate boundary_exceeded array
            unwrapped._update_state()
            # Check if done
            done, _, _ = unwrapped._check_done()

        self.assertFalse(done, "Environment should NOT reset when ego agent is within track boundaries in Frenet mode")


class TestTerminatedTruncatedLogic(unittest.TestCase):
    """
    Tests for the distinction between terminated and truncated in _check_done().

    Terminated: Episode ends due to a terminal event (collision, boundary violation, or lap completion)
    Truncated: Episode ends due to reaching the maximum timestep limit
    """

    def test_predictive_collision_mode_terminated_on_collision(self):
        """Test that terminated=True when predictive_collision=True and ego agent collides."""
        env = gym.make(
            "f1tenth_gym:f1tenth-v0",
            config={
                "map": "Spielberg",
                "num_agents": 1,
                "observation_config": {"type": None},
                "reset_config": {"type": "rl_random_static"},
                "predictive_collision": True,  # TTC-based collision detection
            },
        )
        env.reset()
        unwrapped = env.unwrapped

        # Set up state with collision
        unwrapped.collisions = np.array([1.0])  # Ego agent (index 0) has collision
        unwrapped.toggle_list = np.array([0])  # No lap completion
        unwrapped.current_step = 100  # Within max_episode_steps

        terminated, truncated, _ = unwrapped._check_done()

        env.close()
        self.assertTrue(terminated, "terminated should be True when ego agent collides in predictive_collision mode")
        self.assertFalse(truncated, "truncated should be False when collision occurs before time limit")

    def test_predictive_collision_mode_not_terminated_no_collision(self):
        """Test that terminated=False when predictive_collision=True and no collision."""
        env = gym.make(
            "f1tenth_gym:f1tenth-v0",
            config={
                "map": "Spielberg",
                "num_agents": 1,
                "observation_config": {"type": None},
                "reset_config": {"type": "rl_random_static"},
                "predictive_collision": True,  # TTC-based collision detection
            },
        )
        env.reset()
        unwrapped = env.unwrapped

        # Set up state without collision
        unwrapped.collisions = np.array([0.0])  # No collision
        unwrapped.toggle_list = np.array([0])  # No lap completion
        unwrapped.current_step = 100  # Within max_episode_steps

        terminated, truncated, _ = unwrapped._check_done()

        env.close()
        self.assertFalse(
            terminated,
            "terminated should be False when no collision in predictive_collision mode without lap completion",
        )
        self.assertFalse(truncated, "truncated should be False when within time limit")

    def test_predictive_collision_mode_terminated_on_lap_completion(self):
        """Test that terminated=True when predictive_collision=True and all agents complete 2 laps."""
        env = gym.make(
            "f1tenth_gym:f1tenth-v0",
            config={
                "map": "Spielberg",
                "num_agents": 2,
                "observation_config": {"type": None},
                "reset_config": {"type": "rl_random_static"},
                "predictive_collision": True,  # TTC-based collision detection
            },
        )
        env.reset()
        unwrapped = env.unwrapped

        # Set up state with both agents completing 2 laps (toggle_list >= 4)
        unwrapped.collisions = np.array([0.0, 0.0])  # No collisions
        unwrapped.toggle_list = np.array([4, 4])  # Both agents completed 2 laps
        unwrapped.current_step = 100  # Within max_episode_steps

        terminated, truncated, _ = unwrapped._check_done()

        env.close()
        self.assertTrue(
            terminated, "terminated should be True when all agents complete 2 laps in predictive_collision mode"
        )
        self.assertFalse(truncated, "truncated should be False when episode ends before time limit")

    def test_frenet_mode_terminated_on_boundary_violation(self):
        """Test that terminated=True when predictive_collision=False and boundary is violated."""
        env = gym.make(
            "f1tenth_gym:f1tenth-v0",
            config={
                "map": "Spielberg",
                "num_agents": 1,
                "observation_config": {"type": None},
                "reset_config": {"type": "rl_random_static"},
                "predictive_collision": False,  # Frenet-based boundary checking
            },
        )
        env.reset()
        unwrapped = env.unwrapped

        # Mock track with known boundaries
        unwrapped.track.centerline.w_lefts = np.array([2.0] * 100)
        unwrapped.track.centerline.w_rights = np.array([2.0] * 100)
        unwrapped.track.centerline.ss = np.linspace(0, 100, 100)

        # Set ego agent poses
        unwrapped.poses_x = np.array([0.0])
        unwrapped.poses_y = np.array([0.0])
        unwrapped.poses_theta = np.array([0.0])

        # Mock: ego agent exceeds boundary (ey=2.5m > half_width=2.0m)
        with patch.object(unwrapped.track, "cartesian_to_frenet", return_value=(50.0, 2.5, 0.0)):
            unwrapped._update_state()
            terminated, truncated, _ = unwrapped._check_done()

        env.close()
        self.assertTrue(terminated, "terminated should be True when ego agent exceeds track boundary in Frenet mode")
        self.assertFalse(truncated, "truncated should be False when boundary violation occurs before time limit")

    def test_frenet_mode_not_terminated_within_boundaries(self):
        """Test that terminated=False when predictive_collision=False and within boundaries."""
        env = gym.make(
            "f1tenth_gym:f1tenth-v0",
            config={
                "map": "Spielberg",
                "num_agents": 1,
                "observation_config": {"type": None},
                "reset_config": {"type": "rl_random_static"},
                "predictive_collision": False,  # Frenet-based boundary checking
            },
        )
        env.reset()
        unwrapped = env.unwrapped

        # Mock track with known boundaries
        unwrapped.track.centerline.w_lefts = np.array([2.0] * 100)
        unwrapped.track.centerline.w_rights = np.array([2.0] * 100)
        unwrapped.track.centerline.ss = np.linspace(0, 100, 100)

        # Set ego agent poses
        unwrapped.poses_x = np.array([0.0])
        unwrapped.poses_y = np.array([0.0])
        unwrapped.poses_theta = np.array([0.0])

        # Initialize toggle_list to simulate no lap completion
        unwrapped.toggle_list = np.array([0])

        # Mock: ego agent within boundaries (ey=0.5m < half_width=2.0m)
        with patch.object(unwrapped.track, "cartesian_to_frenet", return_value=(50.0, 0.5, 0.0)):
            unwrapped._update_state()
            terminated, truncated, _ = unwrapped._check_done()

        env.close()
        self.assertFalse(
            terminated, "terminated should be False when ego agent is within track boundaries in Frenet mode"
        )
        self.assertFalse(truncated, "truncated should be False when within time limit")

    def test_truncated_when_max_episode_steps_reached(self):
        """Test that truncated=True when current_step exceeds max_episode_steps."""
        env = gym.make(
            "f1tenth_gym:f1tenth-v0",
            config={
                "map": "Spielberg",
                "num_agents": 1,
                "observation_config": {"type": None},
                "reset_config": {"type": "rl_random_static"},
                "max_episode_steps": 100,
                "predictive_collision": True,
            },
        )
        env.reset()
        unwrapped = env.unwrapped

        # Set up state that would normally not be terminal
        unwrapped.collisions = np.array([0.0])  # No collision
        unwrapped.toggle_list = np.array([0])  # No lap completion
        unwrapped.current_step = 101  # Exceeds max_episode_steps

        terminated, truncated, _ = unwrapped._check_done()

        env.close()
        self.assertFalse(terminated, "terminated should be False when time limit is only reason for episode end")
        self.assertTrue(truncated, "truncated should be True when current_step exceeds max_episode_steps")

    def test_truncated_when_at_max_episode_steps(self):
        """Test that truncated=False when current_step equals max_episode_steps (not exceeds)."""
        env = gym.make(
            "f1tenth_gym:f1tenth-v0",
            config={
                "map": "Spielberg",
                "num_agents": 1,
                "observation_config": {"type": None},
                "reset_config": {"type": "rl_random_static"},
                "max_episode_steps": 100,
                "predictive_collision": True,
            },
        )
        env.reset()
        unwrapped = env.unwrapped

        # Set up state at exactly max_episode_steps
        unwrapped.collisions = np.array([0.0])  # No collision
        unwrapped.toggle_list = np.array([0])  # No lap completion
        unwrapped.current_step = 100  # Equals max_episode_steps

        terminated, truncated, _ = unwrapped._check_done()

        env.close()
        self.assertFalse(
            truncated, "truncated should be False when current_step equals (not exceeds) max_episode_steps"
        )

    def test_truncated_not_set_when_terminal_event_occurs(self):
        """Test that truncated remains False even if max_episode_steps is reached when terminal event occurs."""
        env = gym.make(
            "f1tenth_gym:f1tenth-v0",
            config={
                "map": "Spielberg",
                "num_agents": 1,
                "observation_config": {"type": None},
                "reset_config": {"type": "rl_random_static"},
                "max_episode_steps": 100,
                "predictive_collision": True,
            },
        )
        env.reset()
        unwrapped = env.unwrapped

        # Set up state with collision AND max_episode_steps exceeded
        unwrapped.collisions = np.array([1.0])  # Collision occurs
        unwrapped.toggle_list = np.array([0])  # No lap completion
        unwrapped.current_step = 101  # Exceeds max_episode_steps

        terminated, truncated, _ = unwrapped._check_done()

        env.close()
        self.assertTrue(terminated, "terminated should be True when collision occurs")
        self.assertTrue(truncated, "truncated should be True when max_episode_steps exceeded (both can be true)")

    def test_predictive_collision_multi_agent_only_ego_collision_matters(self):
        """Test that in predictive_collision mode, only ego agent collision triggers termination."""
        env = gym.make(
            "f1tenth_gym:f1tenth-v0",
            config={
                "map": "Spielberg",
                "num_agents": 2,
                "ego_idx": 0,
                "observation_config": {"type": None},
                "reset_config": {"type": "rl_random_static"},
                "predictive_collision": True,
            },
        )
        env.reset()
        unwrapped = env.unwrapped

        # Set up state where non-ego agent collides but ego doesn't
        unwrapped.collisions = np.array([0.0, 1.0])  # Only non-ego agent (index 1) collides
        unwrapped.toggle_list = np.array([0, 0])  # No lap completion
        unwrapped.current_step = 100  # Within max_episode_steps

        terminated, truncated, _ = unwrapped._check_done()

        env.close()
        self.assertFalse(
            terminated,
            "terminated should be False when only non-ego agent collides in predictive_collision mode",
        )

    def test_frenet_collision_multi_agent_only_ego_boundary_matters(self):
        """Test that in Frenet mode, only ego agent boundary violation triggers termination."""
        env = gym.make(
            "f1tenth_gym:f1tenth-v0",
            config={
                "map": "Spielberg",
                "num_agents": 2,
                "ego_idx": 0,
                "observation_config": {"type": None},
                "reset_config": {"type": "rl_random_static"},
                "predictive_collision": False,
            },
        )
        env.reset()
        unwrapped = env.unwrapped

        # Directly set boundary_exceeded: only non-ego agent exceeds boundary
        unwrapped.boundary_exceeded = np.array([False, True])  # Ego agent OK, non-ego exceeds
        unwrapped.toggle_list = np.array([0, 0])  # No lap completion
        unwrapped.current_step = 100  # Within max_episode_steps

        terminated, truncated, _ = unwrapped._check_done()

        env.close()
        self.assertFalse(
            terminated,
            "terminated should be False when only non-ego agent exceeds boundary in Frenet mode",
        )


if __name__ == "__main__":
    unittest.main()
