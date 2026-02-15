import pathlib
import tempfile
import unittest

import numpy as np

from f1tenth_gym.envs.track import Raceline


class TestRacelineReversed(unittest.TestCase):
    """Test suite for Raceline.reversed() method."""

    def test_reversed_centerline_with_track_widths(self):
        """
        Test that reversed() correctly reverses a centerline with track width data.

        Centerline files have w_lefts and w_rights which should be swapped when reversed.
        """
        # Create a simple centerline with known values
        n = 5
        xs = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        ys = np.array([0.0, 0.0, 1.0, 1.0, 0.0], dtype=np.float32)
        psis = np.array([0.0, 0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        ks = np.array([0.5, 0.4, 0.3, 0.2, 0.1], dtype=np.float32)
        vxs = np.array([1.0, 1.5, 2.0, 1.5, 1.0], dtype=np.float32)
        axs = np.array([0.0, 0.1, 0.0, -0.1, 0.0], dtype=np.float32)
        ss = np.array([0.0, 1.0, 2.5, 4.0, 5.5], dtype=np.float32)
        w_lefts = np.array([1.0, 1.2, 1.4, 1.2, 1.0], dtype=np.float32)
        w_rights = np.array([1.5, 1.7, 1.9, 1.7, 1.5], dtype=np.float32)

        centerline = Raceline(
            xs=xs, ys=ys, velxs=vxs, psis=psis, kappas=ks, accxs=axs, ss=ss, w_lefts=w_lefts, w_rights=w_rights
        )

        # Reverse it
        reversed_line = centerline.reversed()

        # Check coordinate arrays are reversed
        np.testing.assert_array_equal(reversed_line.xs, xs[::-1])
        np.testing.assert_array_equal(reversed_line.ys, ys[::-1])

        # Check velocities/accelerations are reversed (magnitudes preserved)
        np.testing.assert_array_equal(reversed_line.vxs, vxs[::-1])
        np.testing.assert_array_equal(reversed_line.axs, axs[::-1])

        # Check curvatures are negated and reversed
        np.testing.assert_array_almost_equal(reversed_line.ks, -ks[::-1])

        # Check track widths are swapped and reversed
        np.testing.assert_array_equal(reversed_line.w_lefts, w_rights[::-1])
        np.testing.assert_array_equal(reversed_line.w_rights, w_lefts[::-1])

        # Check yaws are flipped by pi and wrapped
        expected_yaws = (psis[::-1] + np.pi) % (2 * np.pi)
        # Wrap to [-pi, pi]
        expected_yaws = np.arctan2(np.sin(expected_yaws), np.cos(expected_yaws))
        np.testing.assert_array_almost_equal(reversed_line.yaws, expected_yaws, decimal=5)

        # Check arc lengths are re-parameterized
        total_length = ss[-1]
        expected_ss = total_length - ss[::-1]
        np.testing.assert_array_almost_equal(reversed_line.ss, expected_ss, decimal=5)

        # Check spline is recreated
        self.assertIsNotNone(reversed_line.spline)
        self.assertIsNot(reversed_line.spline, centerline.spline)

    def test_reversed_raceline_without_track_widths(self):
        """
        Test that reversed() correctly reverses a raceline without track width data.

        Raceline files don't have w_lefts/w_rights, so these should remain None.
        """
        # Create a raceline without track widths
        n = 4
        xs = np.array([0.0, 2.0, 4.0, 6.0], dtype=np.float32)
        ys = np.array([0.0, 1.0, 0.0, -1.0], dtype=np.float32)
        psis = np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float32)
        ks = np.array([0.2, -0.3, 0.1, -0.2], dtype=np.float32)
        vxs = np.array([5.0, 6.0, 5.5, 5.0], dtype=np.float32)
        axs = np.array([0.5, 0.0, -0.5, 0.0], dtype=np.float32)
        ss = np.array([0.0, 2.5, 5.0, 7.5], dtype=np.float32)

        raceline = Raceline(
            xs=xs, ys=ys, velxs=vxs, psis=psis, kappas=ks, accxs=axs, ss=ss, w_lefts=None, w_rights=None
        )

        # Reverse it
        reversed_line = raceline.reversed()

        # Check coordinate arrays are reversed
        np.testing.assert_array_equal(reversed_line.xs, xs[::-1])
        np.testing.assert_array_equal(reversed_line.ys, ys[::-1])

        # Check velocities/accelerations are reversed
        np.testing.assert_array_equal(reversed_line.vxs, vxs[::-1])
        np.testing.assert_array_equal(reversed_line.axs, axs[::-1])

        # Check curvatures are negated and reversed
        np.testing.assert_array_almost_equal(reversed_line.ks, -ks[::-1])

        # Check track widths remain None
        self.assertIsNone(reversed_line.w_lefts)
        self.assertIsNone(reversed_line.w_rights)

        # Check yaws are flipped by pi
        expected_yaws = psis[::-1] + np.pi
        expected_yaws = np.arctan2(np.sin(expected_yaws), np.cos(expected_yaws))
        np.testing.assert_array_almost_equal(reversed_line.yaws, expected_yaws, decimal=5)

        # Check spline exists
        self.assertIsNotNone(reversed_line.spline)

    def test_reversed_spline_recreation(self):
        """Test that reversed() creates a new spline from reversed coordinates."""
        # Create simple raceline with arc lengths
        xs = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        ys = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        vxs = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        ss = np.array([0.0, 1.5, 3.0], dtype=np.float32)

        raceline = Raceline(xs=xs, ys=ys, velxs=vxs, ss=ss)
        original_spline = raceline.spline

        # Reverse it
        reversed_line = raceline.reversed()

        # Check spline is different object
        self.assertIsNot(reversed_line.spline, original_spline)

        # Check spline is created from reversed coordinates
        self.assertIsNotNone(reversed_line.spline)

        # Verify spline has correct arc length
        self.assertIsNotNone(reversed_line.ss)

        # Verify coordinates are reversed
        np.testing.assert_array_equal(reversed_line.xs, xs[::-1])
        np.testing.assert_array_equal(reversed_line.ys, ys[::-1])

    def test_reversed_preserves_array_dimensions(self):
        """Test that reversed() preserves the same array dimensions as original."""
        n = 10
        xs = np.linspace(0, 10, n, dtype=np.float32)
        ys = np.sin(xs).astype(np.float32)
        vxs = np.ones(n, dtype=np.float32)
        psis = np.linspace(0, np.pi, n, dtype=np.float32)
        ks = np.random.randn(n).astype(np.float32)

        raceline = Raceline(xs=xs, ys=ys, velxs=vxs, psis=psis, kappas=ks)
        reversed_line = raceline.reversed()

        # Check all arrays have same length
        self.assertEqual(len(reversed_line.xs), n)
        self.assertEqual(len(reversed_line.ys), n)
        self.assertEqual(len(reversed_line.vxs), n)
        self.assertEqual(len(reversed_line.yaws), n)
        self.assertEqual(len(reversed_line.ks), n)

    def test_reversed_from_centerline_file(self):
        """
        Test that reversed() works correctly on a Raceline loaded from a centerline file.
        """
        # Create temporary centerline file
        # Format: [x, y, w_right, w_left]
        waypoints = np.array(
            [
                [0.0, 0.0, 2.0, 2.0],
                [1.0, 0.0, 2.0, 2.0],
                [2.0, 1.0, 2.0, 2.0],
                [2.0, 2.0, 2.0, 2.0],
                [1.0, 2.0, 2.0, 2.0],
            ]
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            np.savetxt(f, waypoints, delimiter=",")
            filepath = pathlib.Path(f.name)

        try:
            # Load centerline
            centerline = Raceline.from_centerline_file(filepath)

            # Reverse it
            reversed_line = centerline.reversed()

            # Verify dimensions match
            self.assertEqual(len(reversed_line.xs), len(centerline.xs))
            self.assertEqual(len(reversed_line.ys), len(centerline.ys))

            # Verify track widths are swapped
            np.testing.assert_array_equal(reversed_line.w_lefts, centerline.w_rights[::-1])
            np.testing.assert_array_equal(reversed_line.w_rights, centerline.w_lefts[::-1])

            # Verify spline exists
            self.assertIsNotNone(reversed_line.spline)

        finally:
            filepath.unlink()


if __name__ == "__main__":
    unittest.main()
