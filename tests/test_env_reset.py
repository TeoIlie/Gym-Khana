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
            done, _ = unwrapped._check_done()

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
            done, _ = unwrapped._check_done()

        self.assertFalse(done, "Environment should NOT reset when ego agent is within track boundaries in Frenet mode")


if __name__ == "__main__":
    unittest.main()
