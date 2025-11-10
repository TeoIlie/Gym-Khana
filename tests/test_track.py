import pathlib
import tempfile
import time
import unittest

import numpy as np

from f1tenth_gym.envs.track import Raceline, Track, find_track_dir
from f1tenth_gym.envs.track.track_utils import get_min_max_curvature, get_min_max_track_width


class TestTrack(unittest.TestCase):
    def test_error_handling(self):
        wrong_track_name = "i_dont_exists"
        self.assertRaises(FileNotFoundError, Track.from_track_name, wrong_track_name)

    def test_raceline(self):
        track_name = "Spielberg"
        track = Track.from_track_name(track_name)

        # check raceline is not None
        self.assertNotEqual(track.raceline, None)

        # check loaded raceline match the one in the csv file
        track_dir = find_track_dir(track_name)
        assert track_dir is not None and track_dir.exists(), "track_dir does not exist"

        raceline = np.loadtxt(track_dir / f"{track_name}_raceline.csv", delimiter=";")
        s_idx, x_idx, y_idx, psi_idx, kappa_idx, vx_idx, ax_idx = range(7)

        self.assertTrue(np.isclose(track.raceline.ss, raceline[:, s_idx]).all())
        self.assertTrue(np.isclose(track.raceline.xs, raceline[:, x_idx]).all())
        self.assertTrue(np.isclose(track.raceline.ys, raceline[:, y_idx]).all())
        self.assertTrue(np.isclose(track.raceline.yaws, raceline[:, psi_idx]).all())
        self.assertTrue(np.isclose(track.raceline.ks, raceline[:, kappa_idx]).all())
        self.assertTrue(np.isclose(track.raceline.vxs, raceline[:, vx_idx]).all())
        self.assertTrue(np.isclose(track.raceline.axs, raceline[:, ax_idx]).all())

    def test_map_dir_structure(self):
        """
        Check that the map dir structure is correct:
        - maps/
            - Trackname/
                - Trackname_map.*               # map image
                - Trackname_map.yaml            # map specification
                - [Trackname_raceline.csv]      # raceline (optional)
                - [Trackname_centerline.csv]    # centerline (optional)
        """
        mapdir = pathlib.Path(__file__).parent.parent / "maps"
        for trackdir in mapdir.iterdir():
            if trackdir.is_file():
                continue

            # check subdir is capitalized (at least first letter is capitalized)
            trackdirname = trackdir.stem
            self.assertTrue(trackdirname[0].isupper(), f"trackdir {trackdirname} is not capitalized")

            # check map spec file exists
            file_spec = trackdir / f"{trackdirname}_map.yaml"
            self.assertTrue(
                file_spec.exists(),
                f"map spec file {file_spec} does not exist in {trackdir}",
            )

            # read map image file from spec
            map_spec = Track.load_spec(track=str(trackdir), filespec=str(file_spec))
            file_image = trackdir / map_spec.image

            # check map image file exists
            self.assertTrue(
                file_image.exists(),
                f"map image file {file_image} does not exist in {trackdir}",
            )

            # check raceline and centerline files
            file_raceline = trackdir / f"{trackdir.stem}_raceline.csv"
            file_centerline = trackdir / f"{trackdir.stem}_centerline.csv"

            if file_raceline.exists():
                # try to load raceline files
                # it will raise an assertion error if the file format are not valid
                Raceline.from_raceline_file(file_raceline)

            if file_centerline.exists():
                # try to load raceline files
                # it will raise an assertion error if the file format are not valid
                Raceline.from_centerline_file(file_centerline)

    @unittest.skip("Skipping download test to avoid creating temporary directories")
    def test_download_racetrack(self):
        import shutil

        track_name = "Spielberg"
        track_backup = Track.from_track_name(track_name)

        # rename the track dir
        track_dir = find_track_dir(track_name)
        tmp_dir = track_dir.parent / f"{track_name}_tmp{int(time.time())}"
        track_dir.rename(tmp_dir)

        # download the track
        track = Track.from_track_name(track_name)

        # check the two tracks' specs are the same
        for spec_attr in [
            "name",
            "image",
            "resolution",
            "origin",
            "negate",
            "occupied_thresh",
            "free_thresh",
        ]:
            self.assertEqual(getattr(track.spec, spec_attr), getattr(track_backup.spec, spec_attr))

        # check the two tracks' racelines are the same
        for raceline_attr in ["ss", "xs", "ys", "yaws", "ks", "vxs", "axs"]:
            self.assertTrue(
                np.isclose(
                    getattr(track.raceline, raceline_attr),
                    getattr(track_backup.raceline, raceline_attr),
                ).all()
            )

        # check the two tracks' centerlines are the same
        for centerline_attr in ["ss", "xs", "ys", "yaws", "ks", "vxs", "axs"]:
            self.assertTrue(
                np.isclose(
                    getattr(track.centerline, centerline_attr),
                    getattr(track_backup.centerline, centerline_attr),
                ).all()
            )

        # remove the newly created track dir
        track_dir = find_track_dir(track_name)
        shutil.rmtree(track_dir, ignore_errors=True)

        # rename the backup track dir to its original name
        track_backup_dir = find_track_dir(tmp_dir.stem)
        track_backup_dir.rename(track_dir)

    def test_frenet_to_cartesian(self):
        track_name = "Spielberg"
        track = Track.from_track_name(track_name)

        # Check frenet to cartesian conversion
        # using the track's xs, ys
        for s, x, y in zip(track.centerline.ss, track.centerline.xs, track.centerline.ys):
            x_, y_, _ = track.frenet_to_cartesian(s, 0, 0)
            self.assertAlmostEqual(x, x_, places=4)
            self.assertAlmostEqual(y, y_, places=4)

    def test_frenet_to_cartesian_to_frenet(self):
        track_name = "Spielberg"
        track = Track.from_track_name(track_name)

        # check frenet to cartesian conversion
        s_ = 0
        for s in np.linspace(0, 1, 10):
            x, y, psi = track.frenet_to_cartesian(s, 0, 0)
            s_, d, _ = track.cartesian_to_frenet(x, y, psi, s_guess=s_)
            self.assertAlmostEqual(s, s_, places=4)
            self.assertAlmostEqual(d, 0, places=4)

        # check frenet to cartesian conversion
        # with non-zero lateral offset
        s_ = 0
        for s in np.linspace(0, 1, 10):
            d = np.random.uniform(-1.0, 1.0)
            x, y, psi = track.frenet_to_cartesian(s, d, 0)
            s_, d_, _ = track.cartesian_to_frenet(x, y, psi, s_guess=s_)
            # Handle edge case where we are checking for s=0 but s_ is the last s (same point, but different s)
            self.assertTrue(
                np.isclose(s, s_, atol=1e-4) or np.isclose(s + track.centerline.spline.s[-1], s_, atol=1e-4)
            )
            self.assertAlmostEqual(d, d_, places=4)

    def test_get_min_max_track_width(self):
        """Test extraction of min/max track width from centerline."""
        # Create mock track with known width values
        w_lefts = np.array([1.0, 2.0, 3.0, 1.5], dtype=np.float32)
        w_rights = np.array([1.5, 2.5, 3.5, 2.0], dtype=np.float32)
        # Total widths: [2.5, 4.5, 6.5, 3.5]
        # Expected min = 2.5, max = 6.5

        centerline = Raceline(
            xs=np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32),
            ys=np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32),
            velxs=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            w_lefts=w_lefts,
            w_rights=w_rights,
        )

        track = Track(
            spec=None,
            occupancy_map=np.zeros((10, 10)),
            centerline=centerline,
            raceline=None,
        )

        # Call the function
        min_width, max_width = get_min_max_track_width(track)

        # Verify exact values
        self.assertEqual(min_width, 2.5, "min_width should be exactly 2.5")
        self.assertEqual(max_width, 6.5, "max_width should be exactly 6.5")

        # Verify return types
        self.assertIsInstance(min_width, float)
        self.assertIsInstance(max_width, float)

    def test_get_min_max_track_width_error_cases(self):
        """Test error handling for get_min_max_track_width."""
        # Create a mock track with no centerline
        track_no_centerline = Track(
            spec=None,
            occupancy_map=np.zeros((10, 10)),
            centerline=None,
            raceline=None,
        )

        # Should raise ValueError when centerline is None
        with self.assertRaises(ValueError) as context:
            get_min_max_track_width(track_no_centerline)
        self.assertIn("centerline not available", str(context.exception).lower())

        # Create a mock track with centerline but no width data
        raceline_no_widths = Raceline(
            xs=np.array([0.0, 1.0, 2.0]),
            ys=np.array([0.0, 1.0, 2.0]),
            velxs=np.array([1.0, 1.0, 1.0]),
            w_lefts=None,  # Missing width data
            w_rights=None,
        )

        track_no_widths = Track(
            spec=None,
            occupancy_map=np.zeros((10, 10)),
            centerline=raceline_no_widths,
            raceline=None,
        )

        # Should raise ValueError when width data is missing
        with self.assertRaises(ValueError) as context:
            get_min_max_track_width(track_no_widths)
        self.assertIn("width data not available", str(context.exception).lower())

    def test_get_min_max_curvature(self):
        """Test extraction of min and max curvature from centerline."""
        # Create mock track with known curvature values
        # Include positive, negative, and zero curvatures
        kappas = np.array([-2.5, 0.0, 3.0, -1.5, 4.2], dtype=np.float32)
        # Expected min = -2.5, max = 4.2

        centerline = Raceline(
            xs=np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32),
            ys=np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32),
            velxs=np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            kappas=kappas,
        )

        track = Track(
            spec=None,
            occupancy_map=np.zeros((10, 10)),
            centerline=centerline,
            raceline=None,
        )

        # Call the function
        min_curv, max_curv = get_min_max_curvature(track)

        # Verify values (use assertAlmostEqual for float32 precision)
        self.assertAlmostEqual(min_curv, -2.5, places=5, msg="min curvature should be -2.5")
        self.assertAlmostEqual(max_curv, 4.2, places=5, msg="max curvature should be 4.2")

        # Verify return types
        self.assertIsInstance(min_curv, float)
        self.assertIsInstance(max_curv, float)

    def test_get_min_max_curvature_error_cases(self):
        """Test error handling for get_min_max_curvature."""
        # Create a mock track with no centerline
        track_no_centerline = Track(
            spec=None,
            occupancy_map=np.zeros((10, 10)),
            centerline=None,
            raceline=None,
        )

        # Should raise ValueError when centerline is None
        with self.assertRaises(ValueError) as context:
            get_min_max_curvature(track_no_centerline)
        self.assertIn("centerline not available", str(context.exception).lower())

        # Create a mock track with centerline but no curvature data
        raceline_no_curvature = Raceline(
            xs=np.array([0.0, 1.0, 2.0]),
            ys=np.array([0.0, 1.0, 2.0]),
            velxs=np.array([1.0, 1.0, 1.0]),
            kappas=None,  # Missing curvature data
        )

        track_no_curvature = Track(
            spec=None,
            occupancy_map=np.zeros((10, 10)),
            centerline=raceline_no_curvature,
            raceline=None,
        )

        # Should raise ValueError when curvature data is missing
        with self.assertRaises(ValueError) as context:
            get_min_max_curvature(track_no_curvature)
        self.assertIn("curvature data", str(context.exception).lower())

    def test_centerline_validation_left_width_exceeds_tolerance(self):
        """Test that centerline with w_left > w_right by >10% raises ValueError."""
        # Create temporary centerline file
        # Format: [x, y, w_right, w_left]
        # w_right = 2.0, w_left = 3.0 (50% difference, exceeds 10% tolerance)
        waypoints = np.array(
            [
                [0.0, 0.0, 2.0, 3.0],
                [1.0, 0.0, 2.0, 3.0],
                [2.0, 0.0, 2.0, 3.0],
                [3.0, 0.0, 2.0, 3.0],
            ]
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            np.savetxt(f, waypoints, delimiter=",")
            filepath = pathlib.Path(f.name)

        try:
            # Should raise ValueError due to asymmetry
            with self.assertRaises(ValueError) as context:
                Raceline.from_centerline_file(filepath)

            error_msg = str(context.exception)
            self.assertIn("Centerline validation failed", error_msg)
            self.assertIn("w_right=2.000m, w_left=3.000m", error_msg)
            self.assertIn("exceeds 10% tolerance", error_msg)
        finally:
            filepath.unlink()

    def test_centerline_validation_right_width_exceeds_tolerance(self):
        """Test that centerline with w_right > w_left by >10% raises ValueError."""
        # Create temporary centerline file
        # Format: [x, y, w_right, w_left]
        # w_right = 3.5, w_left = 2.0 (43% difference, exceeds 10% tolerance)
        waypoints = np.array(
            [
                [0.0, 0.0, 3.5, 2.0],
                [1.0, 0.0, 3.5, 2.0],
                [2.0, 0.0, 3.5, 2.0],
                [3.0, 0.0, 3.5, 2.0],
            ]
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            np.savetxt(f, waypoints, delimiter=",")
            filepath = pathlib.Path(f.name)

        try:
            # Should raise ValueError due to asymmetry
            with self.assertRaises(ValueError) as context:
                Raceline.from_centerline_file(filepath)

            error_msg = str(context.exception)
            self.assertIn("Centerline validation failed", error_msg)
            self.assertIn("w_right=3.500m, w_left=2.000m", error_msg)
            self.assertIn("exceeds 10% tolerance", error_msg)
        finally:
            filepath.unlink()

    def test_centerline_validation_left_width_within_tolerance(self):
        """Test that centerline with w_left slightly > w_right (<10%) loads successfully."""
        # Create temporary centerline file
        # Format: [x, y, w_right, w_left]
        # w_right = 2.0, w_left = 2.18 (9% difference, within 10% tolerance)
        waypoints = np.array(
            [
                [0.0, 0.0, 2.0, 2.18],
                [1.0, 0.0, 2.0, 2.18],
                [2.0, 1.0, 2.0, 2.18],
                [3.0, 1.0, 2.0, 2.18],
            ]
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            np.savetxt(f, waypoints, delimiter=",")
            filepath = pathlib.Path(f.name)

        try:
            # Should load without error (within 10% tolerance)
            raceline = Raceline.from_centerline_file(filepath)
            self.assertIsNotNone(raceline)
            self.assertIsNotNone(raceline.xs)
            self.assertIsNotNone(raceline.ys)
        finally:
            filepath.unlink()

    def test_centerline_validation_right_width_within_tolerance(self):
        """Test that centerline with w_right slightly > w_left (<10%) loads successfully."""
        # Create temporary centerline file
        # Format: [x, y, w_right, w_left]
        # w_right = 2.2, w_left = 2.0 (9% difference, within 10% tolerance)
        waypoints = np.array(
            [
                [0.0, 0.0, 2.2, 2.0],
                [1.0, 0.0, 2.2, 2.0],
                [2.0, 1.0, 2.2, 2.0],
                [3.0, 1.0, 2.2, 2.0],
            ]
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            np.savetxt(f, waypoints, delimiter=",")
            filepath = pathlib.Path(f.name)

        try:
            # Should load without error (within 10% tolerance)
            raceline = Raceline.from_centerline_file(filepath)
            self.assertIsNotNone(raceline)
            self.assertIsNotNone(raceline.xs)
            self.assertIsNotNone(raceline.ys)
        finally:
            filepath.unlink()
