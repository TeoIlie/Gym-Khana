"""
Test suite for lookahead width sampling functionality.

Validates:
1. Numerical equivalence between fast and slow implementations
2. Edge cases and boundary conditions
3. Performance improvements from binary search optimization
4. Correct width values at various track positions
"""
import time
import numpy as np
from f1tenth_gym.envs.track.raceline import Raceline
from f1tenth_gym.envs.track.cubic_spline import CubicSpline2D
from f1tenth_gym.envs.observation import (
    sample_lookahead_widths,
    sample_lookahead_widths_fast,
)


class MockTrack:
    """Mock track object for testing width sampling."""

    def __init__(self, centerline):
        self.centerline = centerline


def create_circular_track_with_varying_widths(radius=10.0, n_points=100):
    """
    Create a circular track with varying widths.

    Width varies sinusoidally around the track to test interpolation.
    """
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    # Varying widths: narrow at 0°, wide at 180°
    w_left = 1.0 + 0.5 * np.sin(theta)  # Range: 0.5 to 1.5
    w_right = 1.5 + 0.5 * np.cos(theta)  # Range: 1.0 to 2.0

    # Close the loop
    x = np.append(x, x[0])
    y = np.append(y, y[0])
    w_left = np.append(w_left, w_left[0])
    w_right = np.append(w_right, w_right[0])

    # Create spline
    spline = CubicSpline2D(x, y)

    # Resample at 0.1m intervals (like from_centerline_file does)
    ds = 0.1
    ss, xs, ys, yaws, ks, w_lefts, w_rights = [], [], [], [], [], [], []

    for i_s in np.arange(0, spline.s[-1], ds):
        x_i, y_i = spline.calc_position(i_s)
        yaw = spline.calc_yaw(i_s)
        k = spline.calc_curvature(i_s)

        # Find closest waypoint for width (nearest-neighbor like current implementation)
        # Using closed-loop arrays
        dists = np.hypot(x - x_i, y - y_i)
        closest_idx = np.argmin(dists)

        xs.append(x_i)
        ys.append(y_i)
        yaws.append(yaw)
        ks.append(k)
        ss.append(i_s)
        w_lefts.append(w_left[closest_idx])
        w_rights.append(w_right[closest_idx])

    # Create Raceline
    centerline = Raceline(
        ss=np.array(ss, dtype=np.float32),
        xs=np.array(xs, dtype=np.float32),
        ys=np.array(ys, dtype=np.float32),
        psis=np.array(yaws, dtype=np.float32),
        kappas=np.array(ks, dtype=np.float32),
        velxs=np.ones(len(ss), dtype=np.float32),
        accxs=np.zeros(len(ss), dtype=np.float32),
        spline=spline,
        w_lefts=np.array(w_lefts, dtype=np.float32),
        w_rights=np.array(w_rights, dtype=np.float32),
    )

    return MockTrack(centerline)


def create_constant_width_track(radius=15.0, width_left=2.0, width_right=2.5, n_points=150):
    """Create a circular track with constant width (for easier validation)."""
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    # Constant widths
    w_left = np.full(n_points, width_left)
    w_right = np.full(n_points, width_right)

    # Close the loop
    x = np.append(x, x[0])
    y = np.append(y, y[0])
    w_left = np.append(w_left, w_left[0])
    w_right = np.append(w_right, w_right[0])

    # Create spline
    spline = CubicSpline2D(x, y)

    # Resample
    ds = 0.1
    ss, xs, ys, yaws, ks, w_lefts, w_rights = [], [], [], [], [], [], []

    for i_s in np.arange(0, spline.s[-1], ds):
        x_i, y_i = spline.calc_position(i_s)
        yaw = spline.calc_yaw(i_s)
        k = spline.calc_curvature(i_s)

        dists = np.hypot(x - x_i, y - y_i)
        closest_idx = np.argmin(dists)

        xs.append(x_i)
        ys.append(y_i)
        yaws.append(yaw)
        ks.append(k)
        ss.append(i_s)
        w_lefts.append(w_left[closest_idx])
        w_rights.append(w_right[closest_idx])

    centerline = Raceline(
        ss=np.array(ss, dtype=np.float32),
        xs=np.array(xs, dtype=np.float32),
        ys=np.array(ys, dtype=np.float32),
        psis=np.array(yaws, dtype=np.float32),
        kappas=np.array(ks, dtype=np.float32),
        velxs=np.ones(len(ss), dtype=np.float32),
        accxs=np.zeros(len(ss), dtype=np.float32),
        spline=spline,
        w_lefts=np.array(w_lefts, dtype=np.float32),
        w_rights=np.array(w_rights, dtype=np.float32),
    )

    return MockTrack(centerline)


def test_numerical_equivalence():
    """Test that fast and slow implementations produce identical results."""
    # Create test tracks
    tracks = {
        "varying_width": create_circular_track_with_varying_widths(),
        "constant_width": create_constant_width_track(),
    }

    # Test configurations
    test_configs = [
        {"n_points": 5, "ds": 0.2},
        {"n_points": 10, "ds": 0.3},
        {"n_points": 20, "ds": 0.5},
        {"n_points": 15, "ds": 0.1},
    ]

    tolerance = 1e-5

    for track_name, track in tracks.items():
        track_length = track.centerline.ss[-1]

        for config in test_configs:
            n_points = config["n_points"]
            ds = config["ds"]

            # Test at multiple positions
            test_positions = [
                0.0,
                track_length * 0.25,
                track_length * 0.5,
                track_length * 0.75,
                track_length * 0.9,  # Near end (tests wrap-around)
            ]

            for current_s in test_positions:
                # Call both implementations
                widths_slow = sample_lookahead_widths(track, current_s, n_points, ds)
                widths_fast = sample_lookahead_widths_fast(track, current_s, n_points, ds)

                # Assert numerical equivalence
                max_diff = np.max(np.abs(widths_slow - widths_fast))
                assert max_diff < tolerance, (
                    f"{track_name} track with n={n_points}, ds={ds}, s={current_s:.2f}: "
                    f"max difference {max_diff:.2e} exceeds tolerance {tolerance}"
                )


def test_constant_width_values():
    """Verify that constant width tracks return expected values."""
    width_left = 2.0
    width_right = 2.5
    expected_total = width_left + width_right  # 4.5

    track = create_constant_width_track(width_left=width_left, width_right=width_right)
    track_length = track.centerline.ss[-1]

    n_points = 10
    ds = 0.3

    test_positions = [0.0, track_length * 0.25, track_length * 0.5, track_length * 0.75]

    for current_s in test_positions:
        widths = sample_lookahead_widths_fast(track, current_s, n_points, ds)

        # All widths should be constant
        width_variance = np.var(widths)
        mean_width = np.mean(widths)

        # Assert all values are close to expected
        assert np.allclose(
            widths, expected_total, atol=1e-4
        ), f"Position s={current_s:.2f}m: widths={mean_width:.3f}m != expected {expected_total}m"

        # Assert variance is negligible (constant width)
        assert (
            width_variance < 1e-6
        ), f"Position s={current_s:.2f}m: width variance {width_variance:.2e} too high for constant width"


def test_edge_cases():
    """Test edge cases and boundary conditions."""
    track = create_circular_track_with_varying_widths()
    track_length = track.centerline.ss[-1]

    test_cases = [
        ("Start of track (s=0)", 0.0, 10, 0.3),
        ("Near end with wrap-around", track_length - 1.0, 10, 0.3),
        ("Exactly at end", track_length, 5, 0.2),
        ("Small spacing (ds=0.05m)", track_length * 0.5, 10, 0.05),
        ("Large spacing (ds=2.0m)", track_length * 0.5, 10, 2.0),
        ("Single lookahead point", track_length * 0.25, 1, 0.3),
        ("Many lookahead points (50)", track_length * 0.25, 50, 0.1),
    ]

    tolerance = 1e-5

    for name, current_s, n_points, ds in test_cases:
        widths_slow = sample_lookahead_widths(track, current_s, n_points, ds)
        widths_fast = sample_lookahead_widths_fast(track, current_s, n_points, ds)

        # Assert shapes match
        assert (
            widths_slow.shape == widths_fast.shape
        ), f"{name}: Shape mismatch: {widths_slow.shape} vs {widths_fast.shape}"
        assert widths_slow.shape == (
            n_points,
        ), f"{name}: Wrong output shape: {widths_slow.shape}, expected ({n_points},)"

        # Assert numerical agreement
        max_diff = np.max(np.abs(widths_slow - widths_fast))
        assert max_diff < tolerance, f"{name}: max difference {max_diff:.2e} exceeds tolerance {tolerance}"

        # Assert widths are positive
        assert np.all(widths_fast > 0), f"{name}: Widths should be positive"

        # Assert widths are reasonable (between 0.5 and 5.0 meters for test tracks)
        assert np.all(widths_fast < 5.0) and np.all(
            widths_fast > 0.5
        ), f"{name}: Widths out of reasonable range [0.5, 5.0]m"


def test_performance_comparison():
    """Benchmark performance improvement from binary search optimization."""
    # Create tracks with different sizes to show scaling
    track_configs = [
        ("Small track", create_circular_track_with_varying_widths(radius=10, n_points=100)),
        ("Medium track", create_circular_track_with_varying_widths(radius=20, n_points=300)),
        ("Large track", create_circular_track_with_varying_widths(radius=50, n_points=1000)),
    ]

    n_points = 10
    ds = 0.3
    n_iterations = 500  # Reduced for faster tests

    for track_name, track in track_configs:
        track_length = track.centerline.ss[-1]

        # Generate random positions
        np.random.seed(42)
        test_positions = np.random.uniform(0, track_length, n_iterations)

        # Benchmark slow version
        start = time.perf_counter()
        for s in test_positions:
            sample_lookahead_widths(track, s, n_points, ds)
        time_slow = time.perf_counter() - start

        # Benchmark fast version (with JIT compilation)
        start = time.perf_counter()
        for s in test_positions:
            sample_lookahead_widths_fast(track, s, n_points, ds)
        time_fast = time.perf_counter() - start

        speedup = time_slow / time_fast

        # Assert that fast version is actually faster
        assert speedup > 1.0, f"{track_name}: Fast version not faster (speedup={speedup:.1f}x)"

        # Assert reasonable speedup (at least 2x for binary search vs linear)
        assert speedup > 2.0, f"{track_name}: Speedup {speedup:.1f}x too low (expected >2x)"


def test_wrap_around_behavior():
    """Test that wrap-around at track boundaries works correctly."""
    track = create_constant_width_track(width_left=2.0, width_right=2.5)
    track_length = track.centerline.ss[-1]
    expected_width = 4.5

    # Test position near end of track where lookahead points wrap around
    current_s = track_length - 0.5  # 0.5m before end
    n_points = 10
    ds = 0.3  # Will sample: -0.2, 0.1, 0.4, 0.7, 1.0, 1.3, 1.6, 1.9, 2.2, 2.5 (wrapped)

    widths = sample_lookahead_widths_fast(track, current_s, n_points, ds)

    # Assert all widths are constant even with wrap-around
    assert np.allclose(
        widths, expected_width, atol=1e-4
    ), f"Wrap-around failed: widths={widths}, expected all={expected_width}"

    # Assert variance is negligible
    width_variance = np.var(widths)
    assert width_variance < 1e-6, f"Wrap-around produced varying widths: variance={width_variance:.2e}"


if __name__ == "__main__":
    # Run tests directly if executed as a script
    import pytest
    import sys

    # Run pytest on this file
    exit_code = pytest.main([__file__, "-v"])
    sys.exit(exit_code)
