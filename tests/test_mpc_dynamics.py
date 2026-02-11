"""
Unit tests for MPC dynamics models in CasADi.

Tests validate that CasADi symbolic dynamics match NumPy reference implementations.
"""

import pytest
import casadi as ca
import numpy as np
import gymnasium as gym

from examples.controllers.mpc_dynamics import kinematic_bicycle_dynamics_casadi, rk4_step_casadi
from examples.controllers.mpc_utils import (
    CenterlineExtractor,
    frenet_projection_piecewise_linear,
    frenet_projection_simple,
)
from f1tenth_gym.envs.dynamic_models.kinematic import vehicle_dynamics_ks
from f1tenth_gym.envs.f110_env import F110Env


@pytest.fixture
def vehicle_params():
    """Fixture providing F1TENTH vehicle parameters."""
    return F110Env.f1tenth_std_vehicle_params()


@pytest.fixture
def test_scenario():
    """
    Fixture providing test scenario with initial state and control inputs.

    Uses small control inputs that stay well within constraint bounds to allow
    fair comparison between constrained NumPy and unconstrained CasADi dynamics.
    """
    return {
        "x0": np.array([0.0, 0.0, 0.0, 3.0, 0.0]),  # [X, Y, delta, V, psi]
        "u0": np.array([0.2, 0.5]),  # [steering_velocity, acceleration] - modest inputs
        "dt": 0.05,  # 50ms time step
        "n_steps": 30,  # Shorter test to keep delta within bounds
    }


@pytest.fixture(scope="module")
def test_track():
    """Fixture providing a test track environment."""
    config = {
        "map": "Spielberg",
        "num_agents": 1,
    }
    env = gym.make("f1tenth_gym:f1tenth-v0", config=config)
    env.reset()
    yield env.unwrapped.track
    env.close()


def test_casadi_kinematic_dynamics_vs_numpy(vehicle_params, test_scenario, capsys):
    """
    Test that CasADi kinematic dynamics match NumPy reference implementation.

    Compares RK4-integrated trajectories from:
    1. CasADi symbolic model (used in MPC optimization)
    2. NumPy reference model (ground truth from kinematic.py)

    Validates that max error is below 2e-3 threshold (accounting for minor
    constraint saturation differences in NumPy version).
    """
    x0 = test_scenario["x0"]
    u0 = test_scenario["u0"]
    dt = test_scenario["dt"]
    n_steps = test_scenario["n_steps"]

    print("\n" + "=" * 70)
    print("VALIDATING CASADI KINEMATIC DYNAMICS")
    print("=" * 70)

    print(f"\nTest scenario:")
    print(
        f"  Initial state: X={x0[0]:.2f}m, Y={x0[1]:.2f}m, delta={x0[2]:.3f}rad, V={x0[3]:.2f}m/s, psi={x0[4]:.3f}rad"
    )
    print(f"  Control input: steer_vel={u0[0]:.2f}rad/s, accel={u0[1]:.2f}m/s²")
    print(f"  Time step: {dt}s")
    print(f"  Steps: {n_steps}")

    # -------------------------------------------------------------------------
    # 1. NumPy simulation (ground truth)
    # -------------------------------------------------------------------------
    print("\nRunning NumPy reference simulation...")
    x_numpy = x0.copy()
    x_numpy_trajectory = [x_numpy.copy()]

    for i in range(n_steps):
        # NumPy dynamics include constraint clamping, but for this test
        # we use raw inputs within valid ranges
        f_numpy = vehicle_dynamics_ks(x_numpy, u0, vehicle_params)
        # Simple Euler integration for NumPy (for fair comparison, use RK4 below)
        x_numpy = x_numpy + dt * f_numpy
        x_numpy_trajectory.append(x_numpy.copy())

    x_numpy_trajectory = np.array(x_numpy_trajectory)

    # -------------------------------------------------------------------------
    # 2. CasADi simulation (what MPC uses)
    # -------------------------------------------------------------------------
    print("Running CasADi simulation...")

    # Create CasADi symbolic variables
    x_sym = ca.SX.sym("x", 5)
    u_sym = ca.SX.sym("u", 2)

    # Build single-step dynamics function
    f_casadi_sym = kinematic_bicycle_dynamics_casadi(x_sym, u_sym, vehicle_params)
    f_casadi_fn = ca.Function("f", [x_sym, u_sym], [f_casadi_sym])

    # Build RK4 step function
    x_next_sym = rk4_step_casadi(
        lambda x, u, p: kinematic_bicycle_dynamics_casadi(x, u, p), x_sym, u_sym, vehicle_params, dt
    )
    rk4_step_fn = ca.Function("rk4_step", [x_sym, u_sym], [x_next_sym])

    # Simulate
    x_casadi = x0.copy()
    x_casadi_trajectory = [x_casadi.copy()]

    for i in range(n_steps):
        x_casadi = rk4_step_fn(x_casadi, u0).full().flatten()
        x_casadi_trajectory.append(x_casadi.copy())

    x_casadi_trajectory = np.array(x_casadi_trajectory)

    # -------------------------------------------------------------------------
    # 3. NumPy with RK4 (apples-to-apples comparison)
    # -------------------------------------------------------------------------
    print("Running NumPy RK4 simulation (for fair comparison)...")

    def numpy_rk4_step(x, u, params, dt):
        """RK4 integration using NumPy dynamics"""
        k1 = vehicle_dynamics_ks(x, u, params)
        k2 = vehicle_dynamics_ks(x + (dt / 2) * k1, u, params)
        k3 = vehicle_dynamics_ks(x + (dt / 2) * k2, u, params)
        k4 = vehicle_dynamics_ks(x + dt * k3, u, params)
        return x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    x_numpy_rk4 = x0.copy()
    x_numpy_rk4_trajectory = [x_numpy_rk4.copy()]

    for i in range(n_steps):
        x_numpy_rk4 = numpy_rk4_step(x_numpy_rk4, u0, vehicle_params, dt)
        x_numpy_rk4_trajectory.append(x_numpy_rk4.copy())

    x_numpy_rk4_trajectory = np.array(x_numpy_rk4_trajectory)

    # -------------------------------------------------------------------------
    # 4. Compare results
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("COMPARISON RESULTS")
    print("-" * 70)

    # Compare CasADi vs NumPy RK4
    error = np.abs(x_casadi_trajectory - x_numpy_rk4_trajectory)
    max_error = np.max(error)
    mean_error = np.mean(error)

    print(f"\nCasADi RK4 vs NumPy RK4:")
    print(f"  Max absolute error:  {max_error:.2e}")
    print(f"  Mean absolute error: {mean_error:.2e}")
    print(f"  Error by state:")
    state_names = ["X [m]", "Y [m]", "delta [rad]", "V [m/s]", "psi [rad]"]
    for i, name in enumerate(state_names):
        print(f"    {name:12s}: max={np.max(error[:, i]):.2e}, mean={np.mean(error[:, i]):.2e}")

    # Final state comparison
    print(f"\nFinal states after {n_steps} steps ({n_steps*dt:.1f}s):")
    print(f"  {'State':12s} {'NumPy RK4':>15s} {'CasADi RK4':>15s} {'Error':>15s}")
    print(f"  {'-'*12} {'-'*15} {'-'*15} {'-'*15}")
    for i, name in enumerate(state_names):
        np_val = x_numpy_rk4_trajectory[-1, i]
        ca_val = x_casadi_trajectory[-1, i]
        err = abs(ca_val - np_val)
        print(f"  {name:12s} {np_val:15.6f} {ca_val:15.6f} {err:15.2e}")

    # Validation assertions
    # Threshold set to 2e-3 to account for minor constraint saturation differences
    # between pure CasADi dynamics and saturated NumPy dynamics.
    # For MPC, this level of accuracy is more than sufficient (< 0.12° error).
    threshold = 2e-3

    print("\n" + "=" * 70)
    if max_error < threshold:
        print(f"✓ VALIDATION PASSED (max error {max_error:.2e} < {threshold:.0e})")
    else:
        print(f"✗ VALIDATION FAILED (max error {max_error:.2e} >= {threshold:.0e})")
    print("=" * 70 + "\n")

    # Pytest assertions
    assert max_error < threshold, (
        f"CasADi dynamics diverged from NumPy reference: " f"max_error={max_error:.2e} >= threshold={threshold:.0e}"
    )

    # Check individual state errors for better diagnostics
    state_names = ["X", "Y", "delta", "V", "psi"]
    for i, name in enumerate(state_names):
        state_error = np.max(error[:, i])
        assert state_error < threshold, f"State {name} error too large: {state_error:.2e} >= {threshold:.0e}"


def test_centerline_extraction(test_track, capsys):
    """
    Test CenterlineExtractor functionality.

    Validates that:
    1. Extractor correctly loads track data
    2. Window extraction produces expected number of waypoints
    3. Extracted waypoints are in the correct arc length range
    4. Track width data is properly extracted
    """
    print("\n" + "=" * 70)
    print("TESTING CENTERLINE EXTRACTION")
    print("=" * 70)

    # Create extractor
    extractor = CenterlineExtractor(test_track, use_raceline=False)

    print(f"\nTrack properties:")
    print(f"  Track length: {extractor.track_length:.2f}m")
    print(f"  Total waypoints: {len(extractor.xs)}")

    # Basic assertions
    assert extractor.track_length > 0, "Track length should be positive"
    assert len(extractor.xs) > 0, "Should have waypoints"
    assert len(extractor.xs) == len(extractor.ys), "X and Y should have same length"
    assert len(extractor.xs) == len(extractor.ss), "Coordinates and arc lengths should match"

    # Test window extraction
    s_current = 0.0
    horizon_distance = 20.0
    n_points = 20
    window = extractor.extract_window(s_current, horizon_distance, n_points=n_points)

    print(f"\nExtracted window:")
    print(f"  s_current: {s_current:.2f}m")
    print(f"  Horizon: {horizon_distance:.2f}m")
    print(f"  Requested waypoints: {n_points}")
    print(f"  Actual waypoints: {len(window['x'])}")
    print(f"  Arc length range: [{window['s'][0]:.2f}, {window['s'][-1]:.2f}]m")
    print(
        f"  Track width range: Left=[{window['w_left'].min():.2f}, {window['w_left'].max():.2f}]m, "
        f"Right=[{window['w_right'].min():.2f}, {window['w_right'].max():.2f}]m"
    )

    # Assertions on window
    assert len(window["x"]) == n_points, f"Should extract {n_points} waypoints"
    assert window["s"][0] >= s_current - 1.0, "Window should start near s_current"
    assert window["s"][-1] <= s_current + horizon_distance + 1.0, "Window should end near horizon"
    assert all(window["w_left"] > 0), "Left track width should be positive"
    assert all(window["w_right"] > 0), "Right track width should be positive"

    # Test wraparound case (near end of track)
    s_near_end = extractor.track_length - 10.0
    window_wrap = extractor.extract_window(s_near_end, horizon_distance=20.0, n_points=20)

    print(f"\nWraparound test:")
    print(f"  s_current: {s_near_end:.2f}m (near track end)")
    print(f"  Extracted waypoints: {len(window_wrap['x'])}")

    assert len(window_wrap["x"]) == 20, "Should handle wraparound correctly"

    print("\n" + "=" * 70)
    print("✓ Centerline extraction tests passed")
    print("=" * 70)


def test_frenet_projection_segment(test_track, capsys):
    """
    Test segment-based Frenet projection against reference implementation.

    Validates that the piecewise-linear segment projection produces
    Frenet coordinates close to the spline-based reference.
    """
    print("\n" + "=" * 70)
    print("TESTING FRENET PROJECTION (SEGMENT METHOD)")
    print("=" * 70)

    # Setup
    extractor = CenterlineExtractor(test_track)
    s_current = 50.0  # Test at arbitrary position on track
    window = extractor.extract_window(s_current, horizon_distance=20.0, n_points=30)

    # Pick test point on centerline
    x_test = window["x"][5]
    y_test = window["y"][5]
    psi_test = window["yaw"][5]

    print(f"\nTest point:")
    print(f"  Position: ({x_test:.2f}, {y_test:.2f})")
    print(f"  Heading: {psi_test:.3f} rad")

    # Reference: Track's Frenet conversion
    s_ref, ey_ref, ephi_ref = test_track.cartesian_to_frenet(x_test, y_test, psi_test)

    # CasADi segment projection
    x_sym = ca.SX.sym("x")
    y_sym = ca.SX.sym("y")
    psi_sym = ca.SX.sym("psi")

    s_sym, ey_sym, ephi_sym = frenet_projection_piecewise_linear(
        x_sym, y_sym, psi_sym, window["x"], window["y"], window["yaw"], window["s"]
    )

    frenet_fn = ca.Function("frenet", [x_sym, y_sym, psi_sym], [s_sym, ey_sym, ephi_sym])
    result = frenet_fn(x_test, y_test, psi_test)

    s_casadi = float(result[0])
    ey_casadi = float(result[1])
    ephi_casadi = float(result[2])

    # Compute errors
    error_s = abs(s_casadi - s_ref)
    error_ey = abs(ey_casadi - ey_ref)
    error_ephi = abs(ephi_casadi - ephi_ref)

    print(f"\nResults:")
    print(f"  Reference (spline):  s={s_ref:.3f}m, ey={ey_ref:.3f}m, ephi={ephi_ref:.3f}rad")
    print(f"  Segment projection:  s={s_casadi:.3f}m, ey={ey_casadi:.3f}m, ephi={ephi_casadi:.3f}rad")
    print(f"  Errors:              Δs={error_s:.3f}m, Δey={error_ey:.3f}m, Δephi={error_ephi:.3f}rad")

    # Assertions (allow reasonable tolerances for piecewise-linear approximation)
    assert error_s < 0.5, f"Arc length error too large: {error_s:.3f}m"
    assert error_ey < 0.1, f"Lateral error too large: {error_ey:.3f}m"
    assert error_ephi < 0.05, f"Heading error too large: {error_ephi:.3f}rad"

    print("\n" + "=" * 70)
    print("✓ Segment Frenet projection tests passed")
    print("=" * 70)


def test_frenet_projection_simple(test_track, capsys):
    """
    Test simple softmax-based Frenet projection.

    Validates that the simple waypoint-based projection produces
    reasonable Frenet coordinates compared to reference.
    """
    print("\n" + "=" * 70)
    print("TESTING FRENET PROJECTION (SIMPLE METHOD)")
    print("=" * 70)

    # Setup
    extractor = CenterlineExtractor(test_track)
    s_current = 50.0
    window = extractor.extract_window(s_current, horizon_distance=20.0, n_points=30)

    # Pick test point on centerline
    x_test = window["x"][5]
    y_test = window["y"][5]
    psi_test = window["yaw"][5]

    print(f"\nTest point:")
    print(f"  Position: ({x_test:.2f}, {y_test:.2f})")
    print(f"  Heading: {psi_test:.3f} rad")

    # Reference
    s_ref, ey_ref, ephi_ref = test_track.cartesian_to_frenet(x_test, y_test, psi_test)

    # CasADi simple projection
    x_sym = ca.SX.sym("x")
    y_sym = ca.SX.sym("y")
    psi_sym = ca.SX.sym("psi")

    s_sym, ey_sym, ephi_sym = frenet_projection_simple(
        x_sym, y_sym, psi_sym, window["x"], window["y"], window["yaw"], window["s"]
    )

    frenet_fn = ca.Function("frenet", [x_sym, y_sym, psi_sym], [s_sym, ey_sym, ephi_sym])
    result = frenet_fn(x_test, y_test, psi_test)

    s_casadi = float(result[0])
    ey_casadi = float(result[1])
    ephi_casadi = float(result[2])

    # Compute errors
    error_s = abs(s_casadi - s_ref)
    error_ey = abs(ey_casadi - ey_ref)
    error_ephi = abs(ephi_casadi - ephi_ref)

    print(f"\nResults:")
    print(f"  Reference (spline):  s={s_ref:.3f}m, ey={ey_ref:.3f}m, ephi={ephi_ref:.3f}rad")
    print(f"  Simple projection:   s={s_casadi:.3f}m, ey={ey_casadi:.3f}m, ephi={ephi_casadi:.3f}rad")
    print(f"  Errors:              Δs={error_s:.3f}m, Δey={error_ey:.3f}m, Δephi={error_ephi:.3f}rad")

    # Assertions (looser tolerances for simple method)
    assert error_s < 1.0, f"Arc length error too large: {error_s:.3f}m"
    assert error_ey < 0.2, f"Lateral error too large: {error_ey:.3f}m"
    assert error_ephi < 0.1, f"Heading error too large: {error_ephi:.3f}rad"

    print("\n" + "=" * 70)
    print("✓ Simple Frenet projection tests passed")
    print("=" * 70)
