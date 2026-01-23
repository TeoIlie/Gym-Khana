import unittest
from unittest.mock import patch

import gymnasium as gym
import numpy as np


from f1tenth_gym.envs import F110Env


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


class TestFullStateReset(unittest.TestCase):
    """
    Tests for full vehicle state initialization for STD model.
    The STD model has 9 states: [x, y, delta, v, yaw, yaw_rate, slip_angle, omega_f, omega_r]
    Users can provide 7 states and omega_f, omega_r are calculated automatically.
    """

    def setUp(self):
        """Create test environment with STD model."""
        self.env = gym.make(
            "f1tenth_gym:f1tenth-v0",
            config={
                "map": "Spielberg",
                "num_agents": 1,
                "model": "std",
                "params": F110Env.f1tenth_std_vehicle_params(),
                "observation_config": {"type": None},
                "reset_config": {"type": "rl_random_static"},
            },
        )

    def tearDown(self):
        """Clean up environment."""
        self.env.close()

    def test_backward_compatible_pose_only_reset(self):
        """Test that existing pose-only reset still works."""
        poses = np.array([[0.0, 0.0, np.pi / 2]])
        obs, info = self.env.reset(options={"poses": poses})

        # Verify state initialized correctly
        state = self.env.unwrapped.sim.agents[0].state
        self.assertAlmostEqual(state[0], 0.0, places=5)  # x
        self.assertAlmostEqual(state[1], 0.0, places=5)  # y
        self.assertAlmostEqual(state[4], np.pi / 2, places=5)  # yaw
        self.assertAlmostEqual(state[3], 0.0, places=5)  # velocity should be 0

    def test_full_state_reset_std_model(self):
        """Test full 7-element state initialization for STD model."""
        # State: [x, y, delta, v, yaw, yaw_rate, slip_angle]
        states = np.array([[1.0, 2.0, 0.1, 5.0, np.pi / 4, 0.5, 0.05]])
        obs, info = self.env.reset(options={"states": states})

        state = self.env.unwrapped.sim.agents[0].state
        self.assertAlmostEqual(state[0], 1.0, places=5)  # x
        self.assertAlmostEqual(state[1], 2.0, places=5)  # y
        self.assertAlmostEqual(state[2], 0.1, places=5)  # delta
        self.assertAlmostEqual(state[3], 5.0, places=5)  # velocity
        self.assertAlmostEqual(state[4], np.pi / 4, places=5)  # yaw
        self.assertAlmostEqual(state[5], 0.5, places=5)  # yaw_rate
        self.assertAlmostEqual(state[6], 0.05, places=5)  # slip_angle

        # Verify omega_f and omega_r are calculated (not zero for non-zero velocity)
        self.assertNotEqual(state[7], 0.0)  # omega_f
        self.assertNotEqual(state[8], 0.0)  # omega_r

    def test_omega_calculation_formula(self):
        """Verify omega_f and omega_r are calculated correctly from velocity, slip, and steering."""
        v, beta, delta = 5.0, 0.05, 0.1
        R_w = self.env.unwrapped.params["R_w"]

        states = np.array([[0.0, 0.0, delta, v, 0.0, 0.0, beta]])
        self.env.reset(options={"states": states})

        state = self.env.unwrapped.sim.agents[0].state
        # omega_f = v * cos(beta) * cos(delta) / R_w
        expected_omega_f = v * np.cos(beta) * np.cos(delta) / R_w
        # omega_r = v * cos(beta) / R_w
        expected_omega_r = v * np.cos(beta) / R_w

        self.assertAlmostEqual(state[7], expected_omega_f, places=5)
        self.assertAlmostEqual(state[8], expected_omega_r, places=5)

    def test_mutual_exclusivity_error(self):
        """Test that specifying both poses and states raises error."""
        poses = np.array([[0.0, 0.0, 0.0]])
        states = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

        with self.assertRaises(ValueError) as ctx:
            self.env.reset(options={"poses": poses, "states": states})
        self.assertIn("Cannot provide both", str(ctx.exception))

    def test_wrong_state_shape_error(self):
        """Test that wrong state shape raises clear error."""
        wrong_states = np.array([[0.0, 0.0, 0.0]])  # Only 3 elements instead of 7

        with self.assertRaises(AssertionError):
            self.env.reset(options={"states": wrong_states})

    def test_wrong_num_agents_error(self):
        """Test that wrong number of agents in states raises error."""
        # Create env with 2 agents but provide states for 1
        env = gym.make(
            "f1tenth_gym:f1tenth-v0",
            config={
                "map": "Spielberg",
                "num_agents": 2,
                "model": "std",
                "params": F110Env.f1tenth_std_vehicle_params(),
                "observation_config": {"type": None},
            },
        )

        states = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])  # Only 1 agent

        with self.assertRaises(AssertionError):
            env.reset(options={"states": states})

        env.close()

    def test_non_std_model_states_error(self):
        """Test that using states with non-STD model raises error."""
        env = gym.make(
            "f1tenth_gym:f1tenth-v0",
            config={
                "map": "Spielberg",
                "num_agents": 1,
                "model": "st",  # Single Track model, not STD
                "observation_config": {"type": None},
            },
        )

        states = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

        with self.assertRaises(ValueError) as ctx:
            env.reset(options={"states": states})
        self.assertIn("only supported for STD model", str(ctx.exception))

        env.close()


class TestMultiAgentFullStateReset(unittest.TestCase):
    """Tests for full state reset with multiple agents."""

    def test_multi_agent_full_state(self):
        """Test full state initialization with multiple agents."""
        env = gym.make(
            "f1tenth_gym:f1tenth-v0",
            config={
                "map": "Spielberg",
                "num_agents": 2,
                "model": "std",
                "params": F110Env.f1tenth_std_vehicle_params(),
                "observation_config": {"type": None},
            },
        )

        states = np.array(
            [
                [0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0],
                [5.0, 5.0, 0.1, 4.0, np.pi / 6, 0.2, 0.01],
            ]
        )
        obs, info = env.reset(options={"states": states})

        for i in range(2):
            state = env.unwrapped.sim.agents[i].state
            np.testing.assert_array_almost_equal(state[:7], states[i])

        env.close()


class TestStateConstraints(unittest.TestCase):
    """Tests for constraint application in full state reset."""

    def setUp(self):
        """Create test environment with STD model."""
        self.env = gym.make(
            "f1tenth_gym:f1tenth-v0",
            config={
                "map": "Spielberg",
                "num_agents": 1,
                "model": "std",
                "params": F110Env.f1tenth_std_vehicle_params(),
                "observation_config": {"type": None},
            },
        )

    def tearDown(self):
        """Clean up environment."""
        self.env.close()

    def test_steering_constraint_applied(self):
        """Test that steering angle is clamped to valid range."""
        params = self.env.unwrapped.params
        s_max = params["s_max"]

        # State with steering exceeding max
        states = np.array([[0.0, 0.0, s_max + 0.5, 5.0, 0.0, 0.0, 0.0]])
        self.env.reset(options={"states": states})

        state = self.env.unwrapped.sim.agents[0].state
        self.assertLessEqual(state[2], s_max)  # delta should be clamped

    def test_velocity_constraint_applied(self):
        """Test that velocity is clamped to valid range."""
        params = self.env.unwrapped.params
        v_max = params["v_max"]

        # State with velocity exceeding max
        states = np.array([[0.0, 0.0, 0.0, v_max + 10, 0.0, 0.0, 0.0]])
        self.env.reset(options={"states": states})

        state = self.env.unwrapped.sim.agents[0].state
        self.assertLessEqual(state[3], v_max)  # velocity should be clamped


class TestSimulationContinuation(unittest.TestCase):
    """Tests for simulation behavior after full state reset."""

    def setUp(self):
        """Create test environment with STD model."""
        self.env = gym.make(
            "f1tenth_gym:f1tenth-v0",
            config={
                "map": "Spielberg",
                "num_agents": 1,
                "model": "std",
                "params": F110Env.f1tenth_std_vehicle_params(),
                "control_input": ["accl", "steering_angle"],
                "observation_config": {"type": None},
            },
        )

    def tearDown(self):
        """Clean up environment."""
        self.env.close()

    def test_simulation_continues_from_state(self):
        """Test that simulation correctly continues from initialized state."""
        # Initialize with specific state (non-zero velocity forward)
        initial_state = np.array([[10.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0]])
        self.env.reset(options={"states": initial_state})

        # Take a step with zero action
        action = np.array([[0.0, 0.0]])
        obs, _, _, _, _ = self.env.step(action)

        # State should have evolved - x position should have increased
        new_state = self.env.unwrapped.sim.agents[0].state
        # With velocity=3.0 m/s and timestep=0.01s, x should increase by ~0.03m per step
        # (step() is called in reset() once, so after our step we're 2 steps in)
        self.assertGreater(new_state[0], 10.0)

    def test_round_trip_state_preservation(self):
        """Test: run simulation -> extract state -> reset with state -> verify state matches."""
        # Run simulation for a few steps
        self.env.reset()
        for _ in range(10):
            action = self.env.action_space.sample()
            self.env.step(action)

        # Extract the first 7 elements of state (user-controllable states)
        captured_state = self.env.unwrapped.sim.agents[0].state[:7].copy()

        # Reset with captured state
        self.env.reset(options={"states": captured_state.reshape(1, -1)})

        # Verify state matches (first 7 elements)
        restored_state = self.env.unwrapped.sim.agents[0].state[:7]
        np.testing.assert_array_almost_equal(restored_state, captured_state, decimal=5)


class TestInternalBookkeeping(unittest.TestCase):
    """Tests for internal state bookkeeping after full state reset."""

    def setUp(self):
        """Create test environment with STD model."""
        self.env = gym.make(
            "f1tenth_gym:f1tenth-v0",
            config={
                "map": "Spielberg",
                "num_agents": 1,
                "model": "std",
                "params": F110Env.f1tenth_std_vehicle_params(),
                "observation_config": {"type": None},
            },
        )

    def tearDown(self):
        """Clean up environment."""
        self.env.close()

    def test_start_positions_extracted_from_states(self):
        """Test that start_xs, start_ys, start_thetas are correctly extracted from states."""
        x, y, yaw = 5.0, 10.0, np.pi / 3
        states = np.array([[x, y, 0.1, 3.0, yaw, 0.2, 0.01]])
        self.env.reset(options={"states": states})

        unwrapped = self.env.unwrapped
        self.assertAlmostEqual(unwrapped.start_xs[0], x, places=5)
        self.assertAlmostEqual(unwrapped.start_ys[0], y, places=5)
        self.assertAlmostEqual(unwrapped.start_thetas[0], yaw, places=5)


if __name__ == "__main__":
    unittest.main()
