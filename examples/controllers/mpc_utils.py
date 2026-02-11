"""
MPC utility functions for centerline extraction and Frenet coordinate transformations.

This module provides helper functions to work with track centerline data in a way
that's compatible with CasADi symbolic optimization.
"""

import numpy as np
import casadi as ca


class CenterlineExtractor:
    """
    Extracts local centerline windows for MPC optimization.

    The MPC controller needs centerline data (waypoints, track widths, arc lengths)
    over its prediction horizon. This class efficiently extracts the relevant
    window of data around the vehicle's current position.
    """

    def __init__(self, track, use_raceline=False):
        """
        Initialize centerline extractor.

        Args:
            track: Track object with centerline/raceline data
            use_raceline: If True, use raceline; if False, use centerline
        """
        self.track = track
        self.line = track.raceline if use_raceline else track.centerline

        # Cache centerline data for fast access
        self.ss = self.line.ss  # Arc lengths
        self.xs = self.line.xs  # X coordinates
        self.ys = self.line.ys  # Y coordinates
        self.yaws = self.line.yaws  # Heading angles
        self.ks = self.line.ks  # Curvatures
        self.w_lefts = self.line.w_lefts  # Left track width
        self.w_rights = self.line.w_rights  # Right track width
        self.track_length = self.line.length

    def extract_window(self, s_current, horizon_distance, n_points=None):
        """
        Extract a local window of centerline waypoints around current position.

        This provides the data needed for MPC's prediction horizon. The window
        extends from s_current to s_current + horizon_distance along the track.

        Args:
            s_current: Current arc length position on track [m]
            horizon_distance: Look-ahead distance for MPC horizon [m]
            n_points: Number of points to extract (if None, uses all points in range)

        Returns:
            dict with keys:
                's': Arc lengths in window (n_points,)
                'x': X coordinates (n_points,)
                'y': Y coordinates (n_points,)
                'yaw': Heading angles (n_points,)
                'kappa': Curvatures (n_points,)
                'w_left': Left track widths (n_points,)
                'w_right': Right track widths (n_points,)
                'indices': Original waypoint indices (for debugging)
        """
        # Wrap s_current to track length
        s_current = s_current % self.track_length
        s_end = s_current + horizon_distance

        # Handle track wraparound (when horizon crosses finish line)
        if s_end <= self.track_length:
            # Simple case: no wraparound
            mask = (self.ss >= s_current) & (self.ss <= s_end)
            indices = np.where(mask)[0]
        else:
            # Wraparound case: extract from s_current to end, then from start to s_end % length
            s_end_wrapped = s_end % self.track_length
            mask = (self.ss >= s_current) | (self.ss <= s_end_wrapped)
            indices = np.where(mask)[0]

        # Extract waypoints
        if len(indices) == 0:
            # Fallback: if no waypoints in range, find closest waypoint
            closest_idx = np.argmin(np.abs(self.ss - s_current))
            indices = np.array([closest_idx])

        # Optionally resample to fixed number of points
        if n_points is not None and len(indices) > n_points:
            # Uniformly sample n_points from the extracted indices
            indices = indices[np.linspace(0, len(indices) - 1, n_points, dtype=int)]

        # Build output dict
        window = {
            "s": self.ss[indices].copy(),
            "x": self.xs[indices].copy(),
            "y": self.ys[indices].copy(),
            "yaw": self.yaws[indices].copy(),
            "kappa": self.ks[indices].copy(),
            "w_left": self.w_lefts[indices].copy(),
            "w_right": self.w_rights[indices].copy(),
            "indices": indices,
        }

        return window


def frenet_to_cartesian_casadi(s, ey, centerline_x, centerline_y, centerline_yaw):
    """
    Convert Frenet coordinates to Cartesian (symbolic CasADi version).

    Given a position in Frenet frame (s, ey) and centerline data, compute
    Cartesian coordinates (X, Y). This is used to validate Frenet projections.

    Args:
        s: Arc length along centerline (CasADi variable or float)
        ey: Lateral deviation from centerline (CasADi variable or float)
        centerline_x: X coordinates of centerline waypoints (numpy array)
        centerline_y: Y coordinates of centerline waypoints (numpy array)
        centerline_yaw: Heading angles of centerline waypoints (numpy array)

    Returns:
        X, Y: Cartesian coordinates (CasADi variables)

    Note: This uses linear interpolation between waypoints. For symbolic
    compatibility, we can't use splines or conditional logic.
    """
    # For now, this is a placeholder - full implementation would require
    # symbolic interpolation logic
    raise NotImplementedError("This function needs symbolic interpolation")


def frenet_projection_piecewise_linear(x, y, psi, wx, wy, wyaw, ws):
    """
    Project Cartesian state to Frenet coordinates using piecewise-linear centerline.

    This is a CasADi-compatible Frenet projection that works with symbolic variables.
    Uses point-to-segment projection with soft-weighting for differentiability.

    **Algorithm:**
    1. For each line segment (waypoint i to i+1), compute:
       - Projection of vehicle position onto the segment
       - Lateral deviation (signed perpendicular distance)
       - Arc length at projection point
    2. Weight segments by inverse distance squared (soft-minimum)
    3. Return weighted average of Frenet coordinates

    This is more accurate than waypoint-only projection since it captures
    positions between waypoints correctly.

    Args:
        x: Vehicle X position (CasADi variable)
        y: Vehicle Y position (CasADi variable)
        psi: Vehicle heading angle (CasADi variable)
        wx: Waypoint X coordinates (numpy array, n_waypoints)
        wy: Waypoint Y coordinates (numpy array, n_waypoints)
        wyaw: Waypoint heading angles (numpy array, n_waypoints)
        ws: Waypoint arc lengths (numpy array, n_waypoints)

    Returns:
        s: Arc length along centerline (CasADi variable)
        ey: Lateral deviation from centerline (CasADi variable)
        ephi: Heading error w.r.t. centerline (CasADi variable)
    """
    n_waypoints = len(wx)
    n_segments = n_waypoints - 1

    # For each segment, compute projection and weight
    segment_s = []
    segment_ey = []
    segment_yaw = []
    segment_weights = []

    epsilon = 0.01  # Regularization for numerical stability

    for i in range(n_segments):
        # Segment endpoints
        x1, y1 = wx[i], wy[i]
        x2, y2 = wx[i + 1], wy[i + 1]
        s1, s2 = ws[i], ws[i + 1]
        yaw1, yaw2 = wyaw[i], wyaw[i + 1]

        # Segment vector
        seg_dx = x2 - x1
        seg_dy = y2 - y1
        seg_length_sq = seg_dx * seg_dx + seg_dy * seg_dy + epsilon

        # Vector from segment start to vehicle
        veh_dx = x - x1
        veh_dy = y - y1

        # Project vehicle onto infinite line through segment
        # Parameter t: 0 = at waypoint i, 1 = at waypoint i+1
        t = (veh_dx * seg_dx + veh_dy * seg_dy) / seg_length_sq

        # Smooth clamping to [0, 1] using tanh
        # This keeps t in valid range while maintaining differentiability
        # tanh maps: (-inf, inf) -> (-1, 1), so (tanh(x) + 1)/2 -> (0, 1)
        t_clamped = 0.5 * (ca.tanh(5.0 * (t - 0.5)) + 1.0)  # Soft clamp to [0,1]

        # Projection point on segment
        proj_x = x1 + t_clamped * seg_dx
        proj_y = y1 + t_clamped * seg_dy

        # Arc length at projection
        proj_s = s1 + t_clamped * (s2 - s1)

        # Lateral deviation: cross product gives signed distance
        # cross(segment_vec, vehicle_vec) = seg_dx * veh_dy - seg_dy * veh_dx
        # Normalize by segment length
        seg_length = ca.sqrt(seg_length_sq)
        cross = seg_dx * veh_dy - seg_dy * veh_dx
        proj_ey = cross / seg_length

        # Heading at projection (linear interpolation)
        proj_yaw = yaw1 + t_clamped * (yaw2 - yaw1)

        # Distance to projection point (for weighting)
        dist_x = x - proj_x
        dist_y = y - proj_y
        dist_sq = dist_x * dist_x + dist_y * dist_y + epsilon

        # Weight: inverse distance squared (closer segments have higher weight)
        weight = 1.0 / dist_sq

        segment_s.append(proj_s)
        segment_ey.append(proj_ey)
        segment_yaw.append(proj_yaw)
        segment_weights.append(weight)

    # Normalize weights
    total_weight = sum(segment_weights)
    weights_norm = [w / total_weight for w in segment_weights]

    # Weighted average of Frenet coordinates
    s = sum(weights_norm[i] * segment_s[i] for i in range(n_segments))
    ey = sum(weights_norm[i] * segment_ey[i] for i in range(n_segments))
    track_yaw = sum(weights_norm[i] * segment_yaw[i] for i in range(n_segments))

    # Heading error: wrap to [-pi, pi]
    ephi = psi - track_yaw
    ephi = ca.atan2(ca.sin(ephi), ca.cos(ephi))

    return s, ey, ephi


def frenet_projection_simple(x, y, psi, wx, wy, wyaw, ws):
    """
    Simplified Frenet projection using soft-argmin (closest waypoint).

    This is the SIMPLEST CasADi-compatible approach:
    - Compute distance to each waypoint
    - Use softmax to select the closest waypoint (differentiable argmin)
    - Return Frenet coords relative to that waypoint

    **Pros:**
    - Very fast (no segment projections)
    - Simple and easy to understand
    - Works well for densely sampled centerlines (waypoints every 0.3-1.0m)

    **Cons:**
    - Less accurate when vehicle is between waypoints
    - Discontinuous gradients when switching between waypoints

    Args:
        x, y, psi: Vehicle state (CasADi variables)
        wx, wy, wyaw, ws: Waypoint data (numpy arrays)

    Returns:
        s, ey, ephi: Frenet coordinates (CasADi variables)
    """
    n_waypoints = len(wx)
    epsilon = 0.01

    # Compute squared distances
    sq_distances = []
    for i in range(n_waypoints):
        dx = x - wx[i]
        dy = y - wy[i]
        sq_distances.append(dx * dx + dy * dy)

    # Softmax weights: exp(-beta * distance^2)
    # Higher beta = sharper selection (closer to true argmin)
    # Lower beta = smoother blending
    beta = 2.0  # Tunable parameter
    exp_weights = [ca.exp(-beta * sq_dist) for sq_dist in sq_distances]
    total_exp = sum(exp_weights)
    weights = [w / total_exp for w in exp_weights]

    # Weighted Frenet coordinates
    s = sum(weights[i] * ws[i] for i in range(n_waypoints))
    track_yaw = sum(weights[i] * wyaw[i] for i in range(n_waypoints))

    # Lateral deviation: perpendicular distance to each waypoint
    lateral_devs = []
    for i in range(n_waypoints):
        dx = x - wx[i]
        dy = y - wy[i]
        # Normal at waypoint: [-sin(yaw), cos(yaw)]
        ey_i = dx * (-ca.sin(wyaw[i])) + dy * ca.cos(wyaw[i])
        lateral_devs.append(ey_i)

    ey = sum(weights[i] * lateral_devs[i] for i in range(n_waypoints))

    # Heading error
    ephi = psi - track_yaw
    ephi = ca.atan2(ca.sin(ephi), ca.cos(ephi))

    return s, ey, ephi


def compute_track_boundaries(s, centerline_s, w_left, w_right):
    """
    Interpolate track boundary widths at a given arc length position.

    Given arc length s, linearly interpolate the left and right track widths
    from the centerline data. This is used for track boundary constraints in MPC.

    Args:
        s: Arc length position (CasADi variable or float)
        centerline_s: Arc lengths of centerline waypoints (numpy array)
        w_left: Left track widths at waypoints (numpy array)
        w_right: Right track widths at waypoints (numpy array)

    Returns:
        w_left_interp: Interpolated left track width (CasADi variable)
        w_right_interp: Interpolated right track width (CasADi variable)

    Note: Uses linear interpolation between waypoints.
    """
    # For CasADi compatibility, we use weighted average similar to Frenet projection
    n_waypoints = len(centerline_s)

    # Compute distance along s (with wraparound handling)
    # For now, simple approach: weight by inverse distance in s
    epsilon = 0.1  # Small constant to avoid division by zero
    weights = []
    for i in range(n_waypoints):
        ds = ca.fabs(s - centerline_s[i])
        weights.append(1.0 / (ds + epsilon))

    # Normalize weights
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]

    # Weighted average
    w_left_interp = sum(weights[i] * w_left[i] for i in range(n_waypoints))
    w_right_interp = sum(weights[i] * w_right[i] for i in range(n_waypoints))

    return w_left_interp, w_right_interp


# ============================================================================
# Validation and Testing Utilities
# ============================================================================
