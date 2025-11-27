from __future__ import annotations
from abc import abstractmethod
from typing import List

import gymnasium as gym
import numpy as np
from numba import njit

from f1tenth_gym.envs.utils import calculate_norm_bounds, normalize_feature


def sample_lookahead_curvatures(track, current_s: float, n_points: int, ds: float) -> np.ndarray:
    """
    Sample N curvature values ahead of vehicle at uniform intervals along centerline.

    Sampling starts at ds meters ahead (not at current position) and proceeds forward.
    For closed tracks, sampling wraps around using modulo arithmetic.

    Parameters
    ----------
    track : Track
        Track object with centerline (must not be None)
    current_s : float
        Current arc length position on centerline (meters)
    n_points : int
        Number of lookahead points to sample (default 10)
    ds : float
        Spacing between points in meters (default 0.3m = 30cm)

    Returns
    -------
    curvatures : np.ndarray (n_points,)
        Curvature at each lookahead point (1/m)

    Raises
    ------
    ValueError
        If track, centerline, or spline is None/invalid
    """
    # Validate inputs
    if track is None or not hasattr(track, "centerline") or track.centerline is None:
        raise ValueError("Track and centerline must be valid")

    if not hasattr(track.centerline, "spline"):
        raise ValueError("Centerline must have a spline")

    if n_points <= 0:
        raise ValueError("n_points must be positive")

    if ds <= 0:
        raise ValueError("ds must be positive")

    centerline = track.centerline
    track_length = centerline.spline.s[-1]  # Total length for wrap-around

    curvatures = np.zeros(n_points, dtype=np.float32)

    for i in range(n_points):
        # Lookahead arc length (wrap around for closed tracks)
        s_ahead = (current_s + ((i + 1) * ds)) % track_length

        # Query curvature from spline
        try:
            curvatures[i] = centerline.spline.calc_curvature(s_ahead)
        except Exception as e:
            # Log warning but continue with zero curvature
            print(f"Warning: Failed to calculate curvature at s={s_ahead}: {e}")
            curvatures[i] = 0.0

    return curvatures


@njit(cache=True)
def _find_spline_segment(spline_x: np.ndarray, s: float) -> int:
    """
    Find the spline segment index for a given arc length s using binary search.

    Returns the index i such that spline_x[i] <= s < spline_x[i+1]
    """
    n = len(spline_x)

    # Handle edge cases
    if s <= spline_x[0]:
        return 0
    if s >= spline_x[n - 1]:
        return n - 2  # Last valid segment

    # Binary search
    left, right = 0, n - 1

    while left < right - 1:
        mid = (left + right) // 2
        if spline_x[mid] <= s:
            left = mid
        else:
            right = mid

    return left


@njit(cache=True)
def _sample_curvatures_numba(
    current_s: float, n_points: int, ds: float, track_length: float, spline_x: np.ndarray, spline_c: np.ndarray
) -> np.ndarray:
    """
    Numba-optimized curvature sampling using pre-extracted spline coefficients.

    This is a high-performance alternative to sample_lookahead_curvatures() for real-time
    applications. It bypasses the scipy CubicSpline interface and directly computes
    curvature from spline derivatives for accuracy.

    Parameters
    ----------
    current_s : float
        Current arc length position on centerline (meters)
    n_points : int
        Number of lookahead points to sample
    ds : float
        Spacing between points in meters
    track_length : float
        Total track length for wrap-around
    spline_x : np.ndarray
        Spline knot positions (arc lengths from centerline.spline.x)
    spline_c : np.ndarray
        Spline coefficients (4, n_segments, n_states) (from centerline.spline.c)

    Returns
    -------
    curvatures : np.ndarray (n_points,)
        Curvature at each lookahead point (1/m)
    """
    curvatures = np.zeros(n_points, dtype=np.float32)

    for i in range(n_points):
        # Lookahead arc length (wrap around for closed tracks)
        s_ahead = (current_s + (i + 1) * ds) % track_length

        # Find segment using binary search (handles non-uniform knot spacing)
        segment = _find_spline_segment(spline_x, s_ahead)

        # Calculate curvature from spline derivatives for accuracy
        # For cubic spline: f(t) = c3*t³ + c2*t² + c1*t + c0
        # f'(t) = 3*c3*t² + 2*c2*t + c1
        # f''(t) = 6*c3*t + 2*c2
        x_offset = s_ahead - spline_x[segment]

        # Get x and y spline coefficients [c3, c2, c1, c0]
        cx = spline_c[:, segment % spline_c.shape[1], 0]  # x coefficients
        cy = spline_c[:, segment % spline_c.shape[1], 1]  # y coefficients

        # Compute first derivatives: dx/ds, dy/ds
        dx = 3.0 * cx[0] * x_offset**2 + 2.0 * cx[1] * x_offset + cx[2]
        dy = 3.0 * cy[0] * x_offset**2 + 2.0 * cy[1] * x_offset + cy[2]

        # Compute second derivatives: d²x/ds², d²y/ds²
        ddx = 6.0 * cx[0] * x_offset + 2.0 * cx[1]
        ddy = 6.0 * cy[0] * x_offset + 2.0 * cy[1]

        # Curvature formula: κ = (dx·d²y - dy·d²x) / (dx² + dy²)^(3/2)
        numerator = ddy * dx - ddx * dy
        denominator = (dx**2 + dy**2) ** 1.5

        curvatures[i] = numerator / denominator

    return curvatures


def sample_lookahead_curvatures_fast(track, current_s: float, n_points: int = 10, ds: float = 0.3) -> np.ndarray:
    """
    High-performance version of sample_lookahead_curvatures using numba JIT compilation.

    This function provides ~20x speedup for real-time applications by using numba
    to compile the curvature sampling loop. Use this when performance is critical
    (e.g., high-frequency control loops at 100Hz+).

    Parameters
    ----------
    track : Track
        Track object with centerline (must not be None)
    current_s : float
        Current arc length position on centerline (meters)
    n_points : int
        Number of lookahead points to sample (default 10)
    ds : float
        Spacing between points in meters (default 0.3m = 30cm)

    Returns
    -------
    curvatures : np.ndarray (n_points,)
        Curvature at each lookahead point (1/m)

    Raises
    ------
    ValueError
        If track, centerline, or spline is None/invalid
    """
    # Validate inputs (same as non-optimized version)
    if track is None or not hasattr(track, "centerline") or track.centerline is None:
        raise ValueError("Track and centerline must be valid")

    if not hasattr(track.centerline, "spline"):
        raise ValueError("Centerline must have a spline")

    if n_points <= 0:
        raise ValueError("n_points must be positive")

    if ds <= 0:
        raise ValueError("ds must be positive")

    centerline = track.centerline
    track_length = centerline.spline.s[-1]

    # Extract spline data for numba (use cached numpy arrays from CubicSpline2D)
    spline_x = np.asarray(centerline.spline.spline_x, dtype=np.float64)
    spline_c = np.asarray(centerline.spline.spline_c, dtype=np.float64)

    # Call numba-optimized function
    return _sample_curvatures_numba(current_s, n_points, ds, track_length, spline_x, spline_c)


def sample_lookahead_widths(track, current_s: float, n_points: int, ds: float) -> np.ndarray:
    """
    Sample N track width values ahead of vehicle at uniform intervals along centerline.

    Sampling starts at ds meters ahead (not at current position) and proceeds forward.
    For closed tracks, sampling wraps around using modulo arithmetic.

    Parameters
    ----------
    track : Track
        Track object with centerline (must not be None)
    current_s : float
        Current arc length position on centerline (meters)
    n_points : int
        Number of lookahead points to sample (default 10)
    ds : float
        Spacing between points in meters (default 0.3m = 30cm)

    Returns
    -------
    widths : np.ndarray (n_points,)
        Total track width (w_left + w_right) at each lookahead point (meters)

    Raises
    ------
    ValueError
        If track, centerline, or width data is None/invalid
    """
    # Validate inputs
    if track is None or not hasattr(track, "centerline") or track.centerline is None:
        raise ValueError("Track and centerline must be valid")

    if n_points <= 0:
        raise ValueError("n_points must be positive")

    if ds <= 0:
        raise ValueError("ds must be positive")

    centerline = track.centerline

    # Check if width data is available
    if centerline.w_lefts is None or centerline.w_rights is None:
        # Return zeros if width data not available (graceful fallback)
        return np.zeros(n_points, dtype=np.float32)

    # Get track length for wrap-around
    if not hasattr(centerline, "ss") or centerline.ss is None:
        raise ValueError("Centerline must have arc length data")

    track_length = centerline.ss[-1]
    widths = np.zeros(n_points, dtype=np.float32)

    for i in range(n_points):
        # Lookahead arc length (wrap around for closed tracks)
        s_ahead = (current_s + ((i + 1) * ds)) % track_length

        # Find closest arc length in centerline data using linear search
        try:
            # Find index of closest arc length value
            idx = np.argmin(np.abs(centerline.ss - s_ahead))

            # Get width at this point (total width = left + right)
            widths[i] = centerline.w_lefts[idx] + centerline.w_rights[idx]
        except Exception as e:
            # Log warning but continue with zero width
            print(f"Warning: Failed to calculate width at s={s_ahead}: {e}")
            widths[i] = 0.0

    return widths


@njit(cache=True)
def _binary_search_nearest(arr: np.ndarray, target: float) -> int:
    """
    Find index of nearest value in monotonically increasing sorted array using binary search.

    This provides O(log n) complexity instead of O(n) for linear search, resulting in
    significant performance improvements for tracks with many centerline points.

    Parameters
    ----------
    arr : np.ndarray
        Sorted array of arc length values (monotonically increasing)
    target : float
        Target value to find nearest neighbor for

    Returns
    -------
    int
        Index of the nearest value in arr
    """
    n = len(arr)

    # Handle edge cases
    if target <= arr[0]:
        return 0
    if target >= arr[n - 1]:
        return n - 1

    # Binary search
    left, right = 0, n - 1

    while left < right - 1:
        mid = (left + right) // 2
        if arr[mid] <= target:
            left = mid
        else:
            right = mid

    # Return closer of the two candidates
    if abs(arr[left] - target) < abs(arr[right] - target):
        return left
    else:
        return right


@njit(cache=True)
def _sample_widths_numba(
    current_s: float,
    n_points: int,
    ds: float,
    track_length: float,
    ss: np.ndarray,
    w_lefts: np.ndarray,
    w_rights: np.ndarray,
) -> np.ndarray:
    """
    Numba-optimized width sampling using pre-extracted arc length and width arrays.

    This is a high-performance alternative to sample_lookahead_widths() for real-time
    applications. It uses nearest-neighbor interpolation with binary search for O(log n)
    complexity, providing ~5-10x speedup over linear search for typical tracks.

    Parameters
    ----------
    current_s : float
        Current arc length position on centerline (meters)
    n_points : int
        Number of lookahead points to sample
    ds : float
        Spacing between points in meters
    track_length : float
        Total track length for wrap-around
    ss : np.ndarray
        Arc length values from centerline (must be monotonically increasing)
    w_lefts : np.ndarray
        Left track widths from centerline
    w_rights : np.ndarray
        Right track widths from centerline

    Returns
    -------
    widths : np.ndarray (n_points,)
        Total track width (w_left + w_right) at each lookahead point (meters)
    """
    widths = np.zeros(n_points, dtype=np.float32)

    for i in range(n_points):
        # Lookahead arc length (wrap around for closed tracks)
        s_ahead = (current_s + (i + 1) * ds) % track_length

        # Find nearest arc length index using binary search (O(log n))
        min_idx = _binary_search_nearest(ss, s_ahead)

        # Get total width at this point
        widths[i] = w_lefts[min_idx] + w_rights[min_idx]

    return widths


def sample_lookahead_widths_fast(track, current_s: float, n_points: int = 10, ds: float = 0.3) -> np.ndarray:
    """
    High-performance version of sample_lookahead_widths using numba JIT compilation.

    This function provides ~1.2x speedup for real-time applications by using numba
    to compile the width sampling loop. Use this when performance is critical
    (e.g., high-frequency control loops at 100Hz+).

    Parameters
    ----------
    track : Track
        Track object with centerline (must not be None)
    current_s : float
        Current arc length position on centerline (meters)
    n_points : int
        Number of lookahead points to sample (default 10)
    ds : float
        Spacing between points in meters (default 0.3m = 30cm)

    Returns
    -------
    widths : np.ndarray (n_points,)
        Total track width (w_left + w_right) at each lookahead point (meters)

    Raises
    ------
    ValueError
        If track, centerline, or width data is None/invalid
    """
    # Validate inputs (same as non-optimized version)
    if track is None or not hasattr(track, "centerline") or track.centerline is None:
        raise ValueError("Track and centerline must be valid")

    if n_points <= 0:
        raise ValueError("n_points must be positive")

    if ds <= 0:
        raise ValueError("ds must be positive")

    centerline = track.centerline

    # Check if width data is available
    if centerline.w_lefts is None or centerline.w_rights is None:
        # Return zeros if width data not available
        return np.zeros(n_points, dtype=np.float32)

    # Check if arc length data is available
    if not hasattr(centerline, "ss") or centerline.ss is None:
        raise ValueError("Centerline must have arc length data")

    track_length = centerline.ss[-1]

    # Extract data for numba
    ss = np.asarray(centerline.ss, dtype=np.float64)
    w_lefts = np.asarray(centerline.w_lefts, dtype=np.float64)
    w_rights = np.asarray(centerline.w_rights, dtype=np.float64)

    # Call numba-optimized function
    return _sample_widths_numba(current_s, n_points, ds, track_length, ss, w_lefts, w_rights)


class Observation:
    """
    Abstract class for observations. Each observation must implement the space and observe methods.

    :param env: The environment.
    :param vehicle_id: The id of the observer vehicle.
    :param kwargs: Additional arguments.
    """

    def __init__(self, env):
        self.env = env
        self._last_raw_features = None  # Store raw features before normalization/flattening

    @abstractmethod
    def space(self):
        """
        Defines the observation space structure for the gym env
        :return gym.spaces object that desribes shape, data types, and observation bounds
        """
        raise NotImplementedError()

    @abstractmethod
    def observe(self):
        """
        Generated the observation data at each time step
        :return current observation values as np array or dict
        """
        raise NotImplementedError()


class OriginalObservation(Observation):
    def __init__(self, env):
        super().__init__(env)

    def space(self):
        num_agents = self.env.unwrapped.num_agents
        scan_size = self.env.unwrapped.sim.agents[0].scan_simulator.num_beams
        scan_range = self.env.unwrapped.sim.agents[0].scan_simulator.max_range + 0.5  # add 1.0 to avoid small errors
        large_num = 1e30  # large number to avoid unbounded obs space (ie., low=-inf or high=inf)

        obs_space = gym.spaces.Dict(
            {
                "ego_idx": gym.spaces.Discrete(num_agents),
                "scans": gym.spaces.Box(
                    low=0.0,
                    high=scan_range,
                    shape=(num_agents, scan_size),
                    dtype=np.float32,
                ),
                "poses_x": gym.spaces.Box(
                    low=-large_num,
                    high=large_num,
                    shape=(num_agents,),
                    dtype=np.float32,
                ),
                "poses_y": gym.spaces.Box(
                    low=-large_num,
                    high=large_num,
                    shape=(num_agents,),
                    dtype=np.float32,
                ),
                "poses_theta": gym.spaces.Box(
                    low=-large_num,
                    high=large_num,
                    shape=(num_agents,),
                    dtype=np.float32,
                ),
                "linear_vels_x": gym.spaces.Box(
                    low=-large_num,
                    high=large_num,
                    shape=(num_agents,),
                    dtype=np.float32,
                ),
                "linear_vels_y": gym.spaces.Box(
                    low=-large_num,
                    high=large_num,
                    shape=(num_agents,),
                    dtype=np.float32,
                ),
                "ang_vels_z": gym.spaces.Box(
                    low=-large_num,
                    high=large_num,
                    shape=(num_agents,),
                    dtype=np.float32,
                ),
                "collisions": gym.spaces.Box(low=0.0, high=1.0, shape=(num_agents,), dtype=np.float32),
                "lap_times": gym.spaces.Box(low=0.0, high=large_num, shape=(num_agents,), dtype=np.float32),
                "lap_counts": gym.spaces.Box(low=0.0, high=large_num, shape=(num_agents,), dtype=np.float32),
            }
        )

        return obs_space

    def observe(self):
        # state indices
        xi, yi, deltai, vxi, yawi, yaw_ratei, slipi = range(7)  # 7 largest state size (ST Model)

        observations = {
            "ego_idx": self.env.unwrapped.sim.ego_idx,
            "scans": [],
            "poses_x": [],
            "poses_y": [],
            "poses_theta": [],
            "linear_vels_x": [],
            "linear_vels_y": [],
            "ang_vels_z": [],
            "collisions": [],
            "lap_times": [],
            "lap_counts": [],
        }

        for i, agent in enumerate(self.env.unwrapped.sim.agents):
            agent_scan = self.env.unwrapped.sim.agent_scans[i]
            lap_time = self.env.unwrapped.lap_times[i]
            lap_count = self.env.unwrapped.lap_counts[i]
            collision = self.env.unwrapped.sim.collisions[i]

            std_state = agent.standard_state

            x, y, theta = std_state["x"], std_state["y"], std_state["yaw"]

            vx = std_state["v_x"]
            vy = std_state["v_y"]
            angvel = std_state["yaw_rate"]

            observations["scans"].append(agent_scan)
            observations["poses_x"].append(x)
            observations["poses_y"].append(y)
            observations["poses_theta"].append(theta)
            observations["linear_vels_x"].append(vx)
            observations["linear_vels_y"].append(vy)
            observations["ang_vels_z"].append(angvel)
            observations["collisions"].append(collision)
            observations["lap_times"].append(lap_time)
            observations["lap_counts"].append(lap_count)

        # cast to match observation space
        for key in observations.keys():
            if isinstance(observations[key], np.ndarray) or isinstance(observations[key], list):
                observations[key] = np.array(observations[key], dtype=np.float32)

        return observations


class FeaturesObservation(Observation):
    def __init__(self, env, features: List[str]):
        super().__init__(env)
        self.features = features

    def space(self):
        scan_size = self.env.unwrapped.sim.agents[0].scan_simulator.num_beams
        scan_range = self.env.unwrapped.sim.agents[0].scan_simulator.max_range
        large_num = 1e30  # large number to avoid unbounded obs space (ie., low=-inf or high=inf)

        complete_space = {}
        for agent_id in self.env.unwrapped.agent_ids:
            agent_dict = {
                "scan": gym.spaces.Box(low=0.0, high=scan_range, shape=(scan_size,), dtype=np.float32),
                "pose_x": gym.spaces.Box(low=-large_num, high=large_num, shape=(), dtype=np.float32),
                "pose_y": gym.spaces.Box(low=-large_num, high=large_num, shape=(), dtype=np.float32),
                "pose_theta": gym.spaces.Box(low=-large_num, high=large_num, shape=(), dtype=np.float32),
                "linear_vel_x": gym.spaces.Box(low=-large_num, high=large_num, shape=(), dtype=np.float32),
                "linear_vel_y": gym.spaces.Box(low=-large_num, high=large_num, shape=(), dtype=np.float32),
                "ang_vel_z": gym.spaces.Box(low=-large_num, high=large_num, shape=(), dtype=np.float32),
                "delta": gym.spaces.Box(low=-large_num, high=large_num, shape=(), dtype=np.float32),
                "beta": gym.spaces.Box(low=-large_num, high=large_num, shape=(), dtype=np.float32),
                "collision": gym.spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float32),
                "lap_time": gym.spaces.Box(low=0.0, high=large_num, shape=(), dtype=np.float32),
                "lap_count": gym.spaces.Box(low=0.0, high=large_num, shape=(), dtype=np.float32),
            }
            complete_space[agent_id] = gym.spaces.Dict({k: agent_dict[k] for k in self.features})

        obs_space = gym.spaces.Dict(complete_space)
        return obs_space

    def observe(self):
        obs = {}  # dictionary agent_id -> observation dict

        for i, agent_id in enumerate(self.env.unwrapped.agent_ids):
            scan = self.env.unwrapped.sim.agent_scans[i]
            agent = self.env.unwrapped.sim.agents[i]
            lap_time = self.env.unwrapped.lap_times[i]
            lap_count = self.env.unwrapped.lap_counts[i]

            std_state = agent.standard_state

            x, y, theta = std_state["x"], std_state["y"], std_state["yaw"]
            delta = std_state["delta"]
            beta = std_state["slip"]
            vx = std_state["v_x"]
            vy = std_state["v_y"]
            angvel = std_state["yaw_rate"]

            # create agent's observation dict
            agent_obs = {
                "scan": scan,
                "pose_x": x,
                "pose_y": y,
                "pose_theta": theta,
                "linear_vel_x": vx,
                "linear_vel_y": vy,
                "ang_vel_z": angvel,
                "delta": delta,
                "beta": beta,
                "collision": int(agent.in_collision),
                "lap_time": lap_time,
                "lap_count": lap_count,
            }

            # add agent's observation to multi-agent observation
            obs[agent_id] = {k: agent_obs[k] for k in self.features}

            # cast to match observation space
            for key in obs[agent_id].keys():
                if (
                    isinstance(obs[agent_id][key], np.ndarray)
                    or isinstance(obs[agent_id][key], list)
                    or isinstance(obs[agent_id][key], float)
                ):
                    obs[agent_id][key] = np.array(obs[agent_id][key], dtype=np.float32)

        return obs


class VectorObservation(Observation):
    def __init__(self, env, features: List[str]):
        super().__init__(env)
        self.features = features
        self.bounds = {}
        self.normalize_obs = self.env.unwrapped.normalize_obs
        if self.normalize_obs:
            self.bounds = calculate_norm_bounds(self.env.unwrapped, self.features)

    def space(self):
        scan_size = self.env.unwrapped.sim.agents[0].scan_simulator.num_beams
        large_num = 1e30  # large number to avoid unbounded obs space (ie., low=-inf or high=inf)

        num_agents = len(self.env.unwrapped.agent_ids)
        assert num_agents == 1, "Vector observation only supports single agent"

        lookahead_points = self.env.unwrapped.lookahead_n_points

        # map observation features to their sizes, used to calculate obs space shape
        obs_size_dict = {
            "scan": scan_size,
            "pose_x": 1,
            "pose_y": 1,
            "pose_theta": 1,
            "linear_vel_x": 1,
            "linear_vel_y": 1,
            "ang_vel_z": 1,
            "delta": 1,
            "beta": 1,
            "collision": 1,
            "lap_time": 1,
            "lap_count": 1,
            "frenet_u": 1,
            "frenet_n": 1,
            "prev_steering_cmd": 1,
            "prev_accl_cmd": 1,
            "curr_accl_cmd": 1,
            "lookahead_curvatures": lookahead_points,
            "lookahead_widths": lookahead_points,
            "curr_avg_wheel_omega": 1,
            "prev_avg_wheel_omega": 1,
            "curr_vel_cmd": 1,
        }

        complete_space_size = sum([obs_size_dict[k] for k in self.features])

        # For normalized observations, bounds should be [-1, 1], otherwise use large_num
        if self.env.unwrapped.normalize_obs:
            obs_low, obs_high = -1.0, 1.0
        else:
            obs_low, obs_high = -large_num, large_num

        obs_space = gym.spaces.Box(
            low=obs_low,
            high=obs_high,
            shape=(complete_space_size,),
            dtype=np.float32,
        )
        return obs_space

    def observe(self):
        scan = self.env.unwrapped.sim.agent_scans[0]
        agent = self.env.unwrapped.sim.agents[0]
        lap_time = self.env.unwrapped.lap_times[0]
        lap_count = self.env.unwrapped.lap_counts[0]

        std_state = agent.standard_state

        x, y, theta = std_state["x"], std_state["y"], std_state["yaw"]
        delta = std_state["delta"]
        beta = std_state["slip"]
        vx = std_state["v_x"]
        vy = std_state["v_y"]
        angvel = std_state["yaw_rate"]

        # Get current and previous average wheel angular velocity (computed in update_pose)
        curr_avg_wheel_omega = agent.curr_avg_wheel_omega
        prev_avg_wheel_omega = agent.prev_avg_wheel_omega

        # Compute Frenet coordinates if track is available
        frenet_u = 0.0  # heading error (0.0 default when unavailable)
        frenet_n = 0.0  # lateral distance (0.0 default when unavailable)

        # Get config values for lookahead points
        n_lookahead = self.env.unwrapped.lookahead_n_points
        ds_lookahead = self.env.unwrapped.lookahead_ds

        # Get curvatures and widths for lookahead points
        lookahead_curvatures = np.zeros(
            n_lookahead, dtype=np.float32
        )  # Lookahead curvatures (zeros default when unavailable)
        lookahead_widths = np.zeros(n_lookahead, dtype=np.float32)  # Lookahead widths (zeros default when unavailable)

        # Check if track and centerline are available
        track = getattr(self.env.unwrapped, "track", None)
        if track is not None and getattr(track, "centerline", None) is not None:
            try:
                # Convert Cartesion coordinates to Frenet
                s, ey, ephi = track.cartesian_to_frenet(
                    x, y, theta, use_raceline=False, debug=self.env.unwrapped.debug_frenet_projection
                )

                frenet_u = float(ephi)  # heading error (vehicle heading - track heading)
                frenet_n = float(ey)  # lateral distance from centerline (left=-ve, right=+ve)

                # Sample lookahead curvatures using configured parameters n, ds
                lookahead_curvatures = sample_lookahead_curvatures_fast(track, s, n_points=n_lookahead, ds=ds_lookahead)

                # Sample lookahead widths using configured parameters n, ds
                lookahead_widths = sample_lookahead_widths_fast(track, s, n_points=n_lookahead, ds=ds_lookahead)

            except Exception as e:
                print(f"Frenet conversion failed: {e}")
                # Keep NaN values to indicate computation failure

        # create agent's observation dict, with all possible observation values for current time step
        agent_obs = {
            "scan": scan,
            "pose_x": x,
            "pose_y": y,
            "pose_theta": theta,
            "linear_vel_x": vx,
            "linear_vel_y": vy,
            "ang_vel_z": angvel,
            "delta": delta,
            "beta": beta,
            "collision": int(agent.in_collision),
            "lap_time": lap_time,
            "lap_count": lap_count,
            "frenet_u": frenet_u,
            "frenet_n": frenet_n,
            "prev_steering_cmd": agent.prev_steering_cmd,
            "prev_accl_cmd": agent.prev_accl_cmd,
            "curr_accl_cmd": agent.curr_accl_cmd,
            "lookahead_curvatures": lookahead_curvatures,
            "lookahead_widths": lookahead_widths,
            "curr_avg_wheel_omega": curr_avg_wheel_omega,
            "prev_avg_wheel_omega": prev_avg_wheel_omega,
            "curr_vel_cmd": agent.curr_vel_cmd,
        }

        # Store raw features before normalization/flattening (for min/max tracking)
        self._last_raw_features = {k: agent_obs[k] for k in self.features}

        # add agent's observation to multi-agent observation
        vec_obs = []
        for k in self.features:
            curr_feat = agent_obs[k]
            if self.normalize_obs:
                curr_feat = normalize_feature(k, curr_feat, self.bounds)
            if isinstance(curr_feat, (list, np.ndarray)):
                vec_obs.extend(list(curr_feat))
            else:
                # Handle scalar values
                vec_obs.append(curr_feat)

        return np.array(vec_obs, dtype=np.float32)


def observation_factory(env, type: str | None, **kwargs) -> Observation:
    type = type or "original"

    if type == "original":
        return OriginalObservation(env)
    elif type == "features":
        return FeaturesObservation(env, **kwargs)
    elif type == "kinematic_state":
        features = ["pose_x", "pose_y", "delta", "linear_vel_x", "pose_theta"]
        return FeaturesObservation(env, features=features)
    elif type == "dynamic_state":
        features = [
            "pose_x",
            "pose_y",
            "delta",
            "linear_vel_x",
            "pose_theta",
            "ang_vel_z",
            "beta",
        ]
        return FeaturesObservation(env, features=features)
    elif type == "frenet_dynamic_state":
        features = [
            "pose_x",
            "pose_y",
            "delta",
            "linear_vel_x",
            "linear_vel_y",
            "pose_theta",
            "ang_vel_z",
            "beta",
        ]
        return FeaturesObservation(env, features=features)
    elif type == "rl":
        features = [
            "scan",
        ]
        return VectorObservation(env, features=features)
    elif type == "drift":
        features = [
            "linear_vel_x",  # vx - longitudinal velocity, vehicle frame
            "linear_vel_y",  # vy - lateral velocity, vehicle frame
            "frenet_u",  # u - angle between car heading, track heading, in Frenet coords
            "frenet_n",  # n - lateral distance from centerline, in Frenet coords
            "ang_vel_z",  # r - yaw rate
            "delta",  # δ - measured steering angle
            "prev_steering_cmd",  # δ_ref - previous commanded steering angle
            "prev_accl_cmd",  # ω_dot_ref - last control input (acceleration)
            "prev_avg_wheel_omega",  # ω - previous measured wheel speed
            "curr_vel_cmd",  # ω_ref - current commanded velocity (integrated from acceleration)
            "lookahead_curvatures",  # c - track curvatures
            "lookahead_widths",  # w - track widths
        ]
        return VectorObservation(env, features=features)
    elif type == "frenet":
        features = ["frenet_u", "frenet_n"]
        return VectorObservation(env, features=features)
    elif type == "race":
        features = [
            "linear_vel_x",  # vx - longitudinal velocity, vehicle frame
            "linear_vel_y",  # vy - lateral velocity, vehicle frame
            "frenet_u",  # u - angle between car heading, track heading, in Frenet coords
            "frenet_n",  # n - lateral distance from centerline, in Frenet coords
            "ang_vel_z",  # r - yaw rate
            "lookahead_curvatures",  # c - track curvatures
        ]
        return VectorObservation(env, features=features)
    else:
        raise ValueError(f"Invalid observation type {type}.")
