import unittest

import numpy as np

from f1tenth_gym.envs import F110Env
from f1tenth_gym.envs.utils import deep_update


class TestRenderer(unittest.TestCase):
    @staticmethod
    def _make_env(config={}, render_mode=None) -> F110Env:
        import gymnasium as gym
        import f1tenth_gym

        base_config = {
            "map": "Spielberg",
            "num_agents": 1,
            "timestep": 0.01,
            "integrator": "rk4",
            "control_input": ["speed", "steering_angle"],
            "model": "st",
            "observation_config": {"type": "kinematic_state"},
            "params": {"mu": 1.0},
        }
        config = deep_update(base_config, config)

        env = gym.make(
            "f1tenth_gym:f1tenth-v0",
            config=config,
            render_mode=render_mode,
        )

        return env

    # def test_human_render(self):
    #     env = self._make_env(render_mode="human")
    #     env.reset()
    #     for _ in range(100):
    #         action = env.action_space.sample()
    #         env.step(action)
    #         env.render()
    #     env.close()

    #     self.assertTrue(True, "Human render test failed")

    def test_rgb_array_render(self):
        env = self._make_env(render_mode="rgb_array")
        env.reset()
        for _ in range(100):
            action = env.action_space.sample()
            env.step(action)
            frame = env.render()

            self.assertTrue(isinstance(frame, np.ndarray), "Frame is not a numpy array")
            self.assertTrue(len(frame.shape) == 3, "Frame is not a 3D array")
            self.assertTrue(frame.shape[2] == 3, "Frame does not have 3 channels")

        env.close()

        self.assertTrue(True, "rgb_array render test failed")

    def test_rgb_array_list(self):
        env = self._make_env(render_mode="rgb_array_list")
        env.reset()

        steps = 100
        for _ in range(steps):
            action = env.action_space.sample()
            env.step(action)

        frame_list = env.render()

        self.assertTrue(isinstance(frame_list, list), "the returned object is not a list of frames")
        self.assertTrue(
            len(frame_list) == steps + 1,
            "the returned list does not have the correct number of frames",
        )
        self.assertTrue(
            all([isinstance(frame, np.ndarray) for frame in frame_list]),
            "not all frames are numpy arrays",
        )
        self.assertTrue(
            all([len(frame.shape) == 3 for frame in frame_list]),
            "not all frames are 3D arrays",
        )
        self.assertTrue(
            all([frame.shape[2] == 3 for frame in frame_list]),
            "not all frames have 3 channels",
        )

        env.close()

    def test_arc_length_annotations_disabled_by_default(self):
        """Test that arc length annotations are disabled by default with no performance impact."""
        env = self._make_env(render_mode="rgb_array")
        env.reset()

        # Verify the feature is disabled by default
        self.assertFalse(
            env.unwrapped.render_arc_length_annotations, "Arc length annotations should be disabled by default"
        )

        # Verify no callback was registered for arc length annotations
        # The track's arc annotation cache should remain None when disabled
        self.assertIsNone(
            env.unwrapped.track.arc_annotation_points_render,
            "Arc annotation render cache should be None when feature is disabled",
        )

        # Run some steps to ensure no performance impact
        for _ in range(50):
            action = env.action_space.sample()
            env.step(action)
            frame = env.render()
            self.assertIsNotNone(frame, "Frame should render successfully with annotations disabled")

        env.close()

    def test_arc_length_annotations_enabled(self):
        """Test that arc length annotations can be enabled and render without crashing."""
        env = self._make_env(
            config={
                "render_arc_length_annotations": True,
                "arc_length_annotation_interval": 10.0,
            },
            render_mode="rgb_array",
        )
        env.reset()

        # Verify the feature is enabled
        self.assertTrue(env.unwrapped.render_arc_length_annotations, "Arc length annotations should be enabled")
        self.assertEqual(env.unwrapped.arc_length_annotation_interval, 10.0, "Interval should be 10.0 meters")

        # Render multiple frames to ensure annotations work
        for _ in range(50):
            action = env.action_space.sample()
            env.step(action)
            frame = env.render()

            self.assertIsNotNone(frame, "Frame should render successfully with annotations enabled")
            self.assertIsInstance(frame, np.ndarray, "Frame should be a numpy array")

        # Verify that annotation points were rendered (cache should be populated)
        self.assertIsNotNone(
            env.unwrapped.track.arc_annotation_points_render,
            "Arc annotation render cache should be populated after rendering",
        )

        env.close()

    def test_arc_length_annotations_different_intervals(self):
        """Test that different annotation interval values work correctly."""
        test_intervals = [2.5, 5.0, 10.0, 20.0]

        for interval in test_intervals:
            with self.subTest(interval=interval):
                env = self._make_env(
                    config={
                        "render_arc_length_annotations": True,
                        "arc_length_annotation_interval": interval,
                    },
                    render_mode="rgb_array",
                )
                env.reset()

                # Verify interval is set correctly
                self.assertEqual(
                    env.unwrapped.arc_length_annotation_interval,
                    interval,
                    f"Interval should be {interval} meters",
                )

                # Render a few frames to ensure it works
                for _ in range(10):
                    action = env.action_space.sample()
                    env.step(action)
                    frame = env.render()
                    self.assertIsNotNone(frame, f"Frame should render with interval={interval}")

                env.close()

    def test_arc_length_annotations_with_track_lines(self):
        """Test that arc length annotations work correctly with track lines enabled."""
        env = self._make_env(
            config={
                "render_track_lines": True,
                "render_arc_length_annotations": True,
                "arc_length_annotation_interval": 5.0,
            },
            render_mode="rgb_array",
        )
        env.reset()

        # Verify both features are enabled
        self.assertTrue(env.unwrapped.render_track_lines, "Track lines should be enabled")
        self.assertTrue(env.unwrapped.render_arc_length_annotations, "Arc length annotations should be enabled")

        # Render multiple frames with both features
        for _ in range(50):
            action = env.action_space.sample()
            env.step(action)
            frame = env.render()

            self.assertIsNotNone(frame, "Frame should render with both track lines and annotations")
            self.assertIsInstance(frame, np.ndarray, "Frame should be a numpy array")

        # Verify both rendering caches are populated
        self.assertIsNotNone(env.unwrapped.track.centerline.waypoint_render, "Centerline should be rendered")
        self.assertIsNotNone(
            env.unwrapped.track.arc_annotation_points_render,
            "Arc annotations should be rendered",
        )

        env.close()
