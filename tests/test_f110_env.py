import unittest

import gymnasium as gym
import numpy as np
from f1tenth_gym.envs.utils import deep_update
from train.config.env_config import get_drift_test_config, get_env_id


class TestDriftEnv(unittest.TestCase):
    def test_gymnasium_api(self):
        from gymnasium.utils.env_checker import check_env

        env = gym.make(get_env_id(), config=get_drift_test_config())
        check_env(env.unwrapped, skip_render_check=True)


class TestEnvInterface(unittest.TestCase):
    def _make_env(self, config={}):
        conf = {
            "map": "Spielberg",
            "num_agents": 1,
            "timestep": 0.01,
            "integrator": "rk4",
            "control_input": ["speed", "steering_angle"],
            "params": {"mu": 1.0},
            "normalize_act": False,
        }
        conf = deep_update(conf, config)

        env = gym.make(
            "f1tenth_gym:f1tenth-v0",
            config=conf,
        )
        return env

    def test_gymnasium_api(self):
        from gymnasium.utils.env_checker import check_env

        env = self._make_env()
        check_env(env.unwrapped, skip_render_check=True)

    def test_configure_method(self):
        """
        Test that the configure method works as expected, and that the parameters are
        correctly updated in the simulator and agents.
        """

        # create a base environment and use configure() to change the width
        config_ext = {"params": {"width": 15.0}}
        base_env = self._make_env()
        base_env.unwrapped.configure(config=config_ext)

        # create an extended environment, with the width set on initialization
        extended_env = self._make_env(config=config_ext)

        # check consistency parameters in config
        for par in base_env.unwrapped.config["params"]:
            base_val = base_env.unwrapped.config["params"][par]
            extended_val = extended_env.unwrapped.config["params"][par]

            self.assertEqual(base_val, extended_val, f"{par} should be the same")

        # check consistency in simulator parameters
        for par in base_env.unwrapped.sim.params:
            base_val = base_env.unwrapped.sim.params[par]
            extended_val = extended_env.unwrapped.sim.params[par]

            self.assertEqual(base_val, extended_val, f"{par} should be the same")

        # check consistency in agent parameters
        for agent, ext_agent in zip(base_env.unwrapped.sim.agents, extended_env.unwrapped.sim.agents):
            for par in agent.params:
                base_val = agent.params[par]
                extended_val = ext_agent.params[par]

                self.assertEqual(base_val, extended_val, f"{par} should be the same")

        # finally, run a simulation and check that the results are the same
        obs0, _ = base_env.reset(options={"poses": np.array([[0.0, 0.0, np.pi / 2]])})
        obs1, _ = extended_env.reset(options={"poses": np.array([[0.0, 0.0, np.pi / 2]])})
        done0 = done1 = False
        t = 0

        while not done0 and not done1:
            action = base_env.action_space.sample()
            obs0, _, done0, _, _ = base_env.step(action)
            obs1, _, done1, _, _ = extended_env.step(action)
            base_env.render()
            for k in obs0:
                self.assertTrue(
                    np.allclose(obs0[k], obs1[k]),
                    f"Observations {k} should be the same",
                )
            self.assertTrue(done0 == done1, "Done should be the same")
            t += 1

        print(f"Done after {t} steps")

        base_env.close()
        extended_env.close()

    def test_configure_action_space(self):
        """
        Try to change the upper bound of the action space, and check that the
        action space is correctly updated.
        """
        base_env = self._make_env()
        action_space_low = base_env.action_space.low
        action_space_high = base_env.action_space.high

        params = base_env.unwrapped.sim.params.copy()
        new_v_max = 5.0
        params["v_max"] = new_v_max

        base_env.unwrapped.configure(config={"params": params})
        new_action_space_low = base_env.action_space.low
        new_action_space_high = base_env.action_space.high

        self.assertTrue(
            (action_space_low == new_action_space_low).all(),
            "Steering action space should be the same",
        )
        self.assertTrue(
            action_space_high[0][0] == new_action_space_high[0][0],
            "Steering action space should be the same",
        )
        self.assertTrue(
            new_action_space_high[0][1] == new_v_max,
            f"Speed action high should be {new_v_max}",
        )

    def test_acceleration_action_space(self):
        """
        Test that the acceleration action space is correctly configured.
        """
        base_env = self._make_env(config={"control_input": ["accl", "steering_speed"]})
        params = base_env.unwrapped.sim.params
        action_space_low = base_env.action_space.low
        action_space_high = base_env.action_space.high

        self.assertTrue(
            (action_space_low[0][0] - params["sv_min"]) < 1e-6,
            "lower sv does not match min steering velocity",
        )
        self.assertTrue(
            (action_space_high[0][0] - params["sv_max"]) < 1e-6,
            "upper sv does not match max steering velocity",
        )
        self.assertTrue(
            (action_space_low[0][1] + params["a_max"]) < 1e-6,
            "lower acceleration bound does not match a_min",
        )
        self.assertTrue(
            (action_space_high[0][1] - params["a_max"]) < 1e-6,
            "upper acceleration bound does not match a_max",
        )

    def test_manual_reset_options_in_synch_vec_env(self):
        """
        Test that the environment can be used in a vectorized environment.
        """
        num_envs, num_agents = 3, 2
        config = {
            "num_agents": num_agents,
            "observation_config": {"type": "kinematic_state"},
            "normalize_act": False,
        }
        vec_env = gym.make_vec("f1tenth_gym:f1tenth-v0", asynchronous=False, config=config, num_envs=num_envs)

        rnd_poses = np.random.random((2, 3))
        obss, infos = vec_env.reset(options={"poses": rnd_poses})

        for i, agent_id in enumerate(obss):
            for ie in range(num_envs):
                agent_obs = obss[agent_id]
                agent_pose = np.array(
                    [
                        agent_obs["pose_x"][ie],
                        agent_obs["pose_y"][ie],
                        agent_obs["pose_theta"][ie],
                    ]
                )
                self.assertTrue(
                    np.allclose(agent_pose, rnd_poses[i]),
                    f"pose of agent {agent_id} in env {ie} should be {rnd_poses[i]}, got {agent_pose}",
                )

    def test_manual_reset_options_in_asynch_vec_env(self):
        """
        Test that the environment can be used in a vectorized environment.
        """
        num_envs, num_agents = 3, 2
        config = {
            "num_agents": num_agents,
            "observation_config": {"type": "kinematic_state"},
            "normalize_act": False,
        }
        vec_env = gym.make_vec("f1tenth_gym:f1tenth-v0", vectorization_mode="async", config=config, num_envs=num_envs)

        rnd_poses = np.random.random((2, 3))
        obss, infos = vec_env.reset(options={"poses": rnd_poses})

        for i, agent_id in enumerate(obss):
            for ie in range(num_envs):
                agent_obs = obss[agent_id]
                agent_pose = np.array(
                    [
                        agent_obs["pose_x"][ie],
                        agent_obs["pose_y"][ie],
                        agent_obs["pose_theta"][ie],
                    ]
                )
                self.assertTrue(
                    np.allclose(agent_pose, rnd_poses[i]),
                    f"pose of agent {agent_id} in env {ie} should be {rnd_poses[i]}, got {agent_pose}",
                )

    def test_auto_reset_options_in_synch_vec_env(self):
        """
        Test that the environment can be used in a vectorized environment without explicit poses.
        """
        num_envs, num_agents = 3, 2
        config = {
            "num_agents": num_agents,
            "observation_config": {"type": "kinematic_state"},
            "reset_config": {"type": "rl_random_random"},
        }
        vec_env = gym.make_vec(
            "f1tenth_gym:f1tenth-v0",
            vectorization_mode="sync",
            config=config,
            num_envs=num_envs,
        )

        obss, infos = vec_env.reset()

        for i, agent_id in enumerate(obss):
            agent_pose0 = np.array(
                [
                    obss[agent_id]["pose_x"][0],
                    obss[agent_id]["pose_y"][0],
                    obss[agent_id]["pose_theta"][0],
                ]
            )
            for ie in range(1, num_envs):
                agent_obs = obss[agent_id]
                agent_pose = np.array(
                    [
                        agent_obs["pose_x"][ie],
                        agent_obs["pose_y"][ie],
                        agent_obs["pose_theta"][ie],
                    ]
                )
                self.assertFalse(
                    np.allclose(agent_pose, agent_pose0),
                    f"pose of agent {agent_id} in env {ie} should be random, got same {agent_pose} == {agent_pose0}",
                )

        # test auto reset
        all_dones_once = [False] * num_envs
        all_dones_twice = [False] * num_envs

        max_steps = 1000
        while not all(all_dones_twice) and max_steps > 0:
            actions = vec_env.action_space.sample()
            obss, rewards, dones, truncations, infos = vec_env.step(actions)

            all_dones_once = [all_dones_once[i] or dones[i] for i in range(num_envs)]
            all_dones_twice = [all_dones_twice[i] or all_dones_once[i] for i in range(num_envs)]
            max_steps -= 1

        vec_env.close()
        self.assertTrue(
            all(all_dones_twice),
            f"All envs should be done twice, got {all_dones_twice}",
        )

    def test_track_direction_config_valid_values(self):
        """Test that valid track_direction values ('normal', 'reverse', 'random') are accepted."""
        # Test 'normal'
        env_normal = self._make_env(config={"track_direction": "normal"})
        self.assertEqual(env_normal.unwrapped.track_direction_config, "normal")
        self.assertFalse(env_normal.unwrapped.direction_reversed)
        env_normal.close()

        # Test 'reverse'
        env_reverse = self._make_env(config={"track_direction": "reverse"})
        self.assertEqual(env_reverse.unwrapped.track_direction_config, "reverse")
        self.assertTrue(env_reverse.unwrapped.direction_reversed)
        env_reverse.close()

        # Test 'random' (just check it's accepted, not the random value)
        env_random = self._make_env(config={"track_direction": "random"})
        self.assertEqual(env_random.unwrapped.track_direction_config, "random")
        self.assertIsInstance(env_random.unwrapped.direction_reversed, bool)
        env_random.close()

    def test_track_direction_config_invalid_value(self):
        """Test that invalid track_direction value raises ValueError."""
        with self.assertRaises(ValueError) as context:
            self._make_env(config={"track_direction": "invalid_mode"})

        error_msg = str(context.exception)
        self.assertIn("Invalid track_direction", error_msg)
        self.assertIn("invalid_mode", error_msg)
        self.assertIn("normal", error_msg)
        self.assertIn("reverse", error_msg)
        self.assertIn("random", error_msg)

    def test_resolve_direction_method(self):
        """
        Test that _resolve_direction() correctly sets direction_reversed based on
        track_direction_config for 'normal', 'reverse', and 'random' modes.
        """
        # Test 'normal' mode
        env_normal = self._make_env(config={"track_direction": "normal"})
        env_normal.unwrapped._resolve_direction()
        self.assertFalse(env_normal.unwrapped.direction_reversed)
        env_normal.close()

        # Test 'reverse' mode
        env_reverse = self._make_env(config={"track_direction": "reverse"})
        env_reverse.unwrapped._resolve_direction()
        self.assertTrue(env_reverse.unwrapped.direction_reversed)
        env_reverse.close()

        # Test 'random' mode (run multiple times to verify randomness)
        env_random = self._make_env(config={"track_direction": "random"})
        results = []
        for _ in range(20):
            env_random.unwrapped._resolve_direction()
            results.append(env_random.unwrapped.direction_reversed)

        # Check that we get both True and False values (probabilistic test)
        # With 20 trials and 50% probability, getting all same is ~0.0001% chance
        self.assertTrue(any(results), "Random mode should produce at least one True")
        self.assertFalse(all(results), "Random mode should produce at least one False")
        env_random.close()

    def test_reset_maintains_direction_for_normal_and_reverse(self):
        """
        Test that reset() maintains consistent direction for 'normal' and 'reverse' modes.
        """
        # Test 'normal' mode stays normal across resets
        env_normal = self._make_env(config={"track_direction": "normal"})
        for _ in range(5):
            env_normal.reset()
            self.assertFalse(env_normal.unwrapped.direction_reversed)
        env_normal.close()

        # Test 'reverse' mode stays reversed across resets
        env_reverse = self._make_env(config={"track_direction": "reverse"})
        for _ in range(5):
            env_reverse.reset()
            self.assertTrue(env_reverse.unwrapped.direction_reversed)
        env_reverse.close()

    def test_reset_rerandomizes_direction_for_random_mode(self):
        """
        Test that reset() re-randomizes direction for 'random' mode.
        """
        env_random = self._make_env(config={"track_direction": "random"})

        # Collect direction_reversed values across multiple resets
        directions = []
        for _ in range(20):
            env_random.reset()
            directions.append(env_random.unwrapped.direction_reversed)

        # Verify we get both True and False (probabilistic test)
        self.assertTrue(any(directions), "Random mode should produce at least one True across resets")
        self.assertFalse(all(directions), "Random mode should produce at least one False across resets")

        env_random.close()

    def test_track_set_direction_called_on_reset(self):
        """
        Test that reset() calls track.set_direction() with the resolved direction.
        """
        # Test with 'normal' mode
        env = self._make_env(config={"track_direction": "normal"})
        env.reset()

        # After reset, track's active references should match direction_reversed
        if env.unwrapped.direction_reversed:
            self.assertIs(env.unwrapped.track.centerline, env.unwrapped.track.centerline_reversed)
            self.assertIs(env.unwrapped.track.raceline, env.unwrapped.track.raceline_reversed)
        else:
            self.assertIs(env.unwrapped.track.centerline, env.unwrapped.track.centerline_regular)
            self.assertIs(env.unwrapped.track.raceline, env.unwrapped.track.raceline_regular)

        env.close()

    def test_random_direction_yaw_alignment(self):
        """
        Test that with track_direction='random', the vehicle yaw aligns with
        the track direction after reset, regardless of direction changes.

        This test verifies the fix for the bug where reset functions cached
        references to centerline/raceline at initialization, causing yaw
        misalignment when track_direction='random' changed directions.
        """
        from unittest.mock import patch

        env = self._make_env(config={"track_direction": "random", "reset_config": {"type": "cl_random_static"}})

        # Test both directions explicitly by mocking _resolve_direction
        for direction_reversed in [False, True]:
            direction_name = "reverse" if direction_reversed else "normal"

            # Mock _resolve_direction to force specific direction
            # We need to capture the current value in the closure
            def mock_resolve_fn(reversed_val=direction_reversed):
                env.unwrapped.direction_reversed = reversed_val

            with patch.object(env.unwrapped, "_resolve_direction", side_effect=mock_resolve_fn):
                obs, info = env.reset()

                # Get vehicle pose
                pose_x = env.unwrapped.start_xs[0]
                pose_y = env.unwrapped.start_ys[0]
                pose_theta = env.unwrapped.start_thetas[0]

                # Find nearest waypoint and get Frenet coordinates
                s, ey, ephi = env.unwrapped.track.cartesian_to_frenet(pose_x, pose_y, pose_theta, use_raceline=False)

                # Vehicle heading should align with track tangent (ephi ~= 0)
                # Allow some tolerance for lateral offset sampling
                self.assertLess(
                    abs(ephi),
                    0.5,  # ~30 degrees tolerance
                    f"Vehicle yaw misaligned with track direction in {direction_name} mode "
                    f"(ephi={ephi:.3f} rad, ~{np.rad2deg(ephi):.1f} deg) at reset",
                )

        env.close()


class TestMultiMapTraining(unittest.TestCase):
    """Integration tests for multi-map training feature."""

    def test_multi_map_environment_creation_and_execution(self):
        """
        End-to-end integration test for multi-map training.

        Verifies that:
        1. Environments can be created with track_pool
        2. Different envs are assigned different maps correctly
        3. Observation spaces are identical despite different tracks (global normalization)
        4. Environments can be reset successfully
        """
        from train.training_utils import make_subprocvecenv
        from train.config.env_config import get_drift_train_config

        # Create base config
        config = get_drift_train_config()
        config["normalize_obs"] = True
        config["normalize_act"] = True

        # Use minimal envs for speed (just need to test multi-map distribution)
        n_envs = 2
        track_pool = ["Drift", "Drift2"]

        # Create multi-map vectorized environment
        env = make_subprocvecenv(seed=42, config=config, n_envs=n_envs, track_pool=track_pool)

        try:
            # 1. Verify environment created successfully
            self.assertEqual(env.num_envs, n_envs)

            # 2. Verify envs are on different tracks (CORE multi-map test)
            configs = env.get_attr("config")
            track_names = [cfg["map"] for cfg in configs]
            self.assertEqual(
                set(track_names),
                set(track_pool),
                f"Environments should be distributed across track pool. Got: {track_names}",
            )
            print(f"✓ Track distribution verified: {dict(zip(range(n_envs), track_names))}")

            # 3. Verify observation spaces are IDENTICAL (CORE global normalization test)
            obs_spaces = env.get_attr("observation_space")
            for i in range(1, n_envs):
                self.assertEqual(
                    obs_spaces[0].shape,
                    obs_spaces[i].shape,
                    f"Observation space shapes should be identical (env 0 vs env {i})",
                )
                self.assertTrue(
                    np.array_equal(obs_spaces[0].low, obs_spaces[i].low),
                    f"Observation space lower bounds should be identical (env 0 vs env {i})",
                )
                self.assertTrue(
                    np.array_equal(obs_spaces[0].high, obs_spaces[i].high),
                    f"Observation space upper bounds should be identical (env 0 vs env {i})",
                )
            print(f"✓ Observation spaces identical across tracks (global normalization working)")

            # 4. Verify action spaces are identical
            action_spaces = env.get_attr("action_space")
            for i in range(1, n_envs):
                self.assertEqual(
                    action_spaces[0].shape,
                    action_spaces[i].shape,
                    f"Action space shapes should be identical (env 0 vs env {i})",
                )
            print(f"✓ Action spaces identical across tracks")

            # 5. Verify reset works and produces valid normalized observations
            obs = env.reset()
            self.assertIsNotNone(obs)
            self.assertIsInstance(obs, np.ndarray, "Drift observations should be numpy arrays")
            self.assertEqual(obs.shape[0], n_envs)

            # Check normalization bounds
            self.assertTrue(np.all(obs >= -1.0), f"Normalized observations should be >= -1.0, got min={np.min(obs)}")
            self.assertTrue(np.all(obs <= 1.0), f"Normalized observations should be <= 1.0, got max={np.max(obs)}")
            print(f"✓ Reset successful, observations in normalized range [-1, 1]")

        finally:
            # Clean up
            env.close()

        print(
            f"\n✅ Multi-map integration test passed: {n_envs} envs on {len(track_pool)} tracks, "
            f"identical observation/action spaces, successful reset"
        )
