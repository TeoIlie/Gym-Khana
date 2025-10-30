from typing import Any, Dict, TypeVar

import numpy as np

from f1tenth_gym.envs.track.track_utils import get_min_max_curvature, get_min_max_track_width

KeyType = TypeVar("KeyType")


def deep_update(mapping: Dict[KeyType, Any], *updating_mappings: Dict[KeyType, Any]) -> Dict[KeyType, Any]:
    """
    Dictionary deep update for nested dictionaries from pydantic:
    https://github.com/pydantic/pydantic/blob/fd2991fe6a73819b48c906e3c3274e8e47d0f761/pydantic/utils.py#L200
    """
    updated_mapping = mapping.copy()
    for updating_mapping in updating_mappings:
        for k, v in updating_mapping.items():
            if k in updated_mapping and isinstance(updated_mapping[k], dict) and isinstance(v, dict):
                updated_mapping[k] = deep_update(updated_mapping[k], v)
            else:
                updated_mapping[k] = v
    return updated_mapping


def calculate_norm_bounds(env):
    """
    Calculate normalization bounds for each feature in the drift observation.

    Extracts bounds from three sources:
    1. Vehicle parameters (velocity limits, control limits, wheel dynamics)
    2. Track geometry (width, curvature) when available
    3. Physical constants (angle wrapping)

    All features are normalized to [-1, 1] for optimal neural network training.

    Parameters
    ----------
    env : F110Env
        The environment to extract parameters from

    Returns
    -------
    dict
        Mapping of feature names to (min, max) tuples for normalization

    Notes
    -----
    - ValueError raised if requires params are missing
    - Track-dependent bounds use defaults if track/centerline not available
    - Asymmetric physical bounds (e.g., wheel speed ≥ 0) still map to [-1, 1] for best DRL training
    """
    params = env.params
    bounds = {}

    # ===========================
    # 1. VELOCITY BOUNDS
    # ===========================

    # Longitudinal velocity: [0, v_max]
    v_max = params.get("v_max", None)  # m/s
    bounds["linear_vel_x"] = (0.0, v_max)
    bounds["curr_vel_cmd"] = (0.0, v_max)  # Same as vx

    # Lateral velocity: symmetric, friction-limited
    # Set to 1/2 of max longitudinal velocity
    linear_vel_bound = 0.5 * v_max if v_max is not None else None
    bounds["linear_vel_y"] = (
        (-linear_vel_bound, linear_vel_bound) if linear_vel_bound is not None else (None, None)
    )  # m/s

    # Yaw rate: can compute from v_max / min(lf, lr) or use conservative bound
    # For F1TENTH: v_max=10 / wheelbase~0.3 ≈ 33 rad/s, but use conservative 5 rad/s

    # TODO current max yaw found experimentally in sim, but this could be off
    experimental_max_yaw = 0.75
    bounds["ang_vel_z"] = (-experimental_max_yaw, experimental_max_yaw)  # rad/s

    # ===========================
    # 2. CONTROL INPUT BOUNDS
    # ===========================

    # Steering angle: from vehicle parameters
    s_min = params.get("s_min", None)
    s_max = params.get("s_max", None)
    bounds["delta"] = (s_min, s_max)
    bounds["prev_steering_cmd"] = (s_min, s_max)

    # Acceleration: symmetric about zero (negative acceleration is braking)
    a_max = params.get("a_max", None)  # m/s²
    bounds["prev_accl_cmd"] = (-a_max, a_max) if a_max is not None else (None, None)
    bounds["curr_accl_cmd"] = (-a_max, a_max) if a_max is not None else (None, None)

    # ===========================
    # 3. WHEEL DYNAMICS BOUNDS
    # ===========================

    # Wheel angular velocity: depends on wheel radius if available
    # omega_max = v_max / R_w
    R_w = params.get("R_w", None)  # Wheel radius (STD model has this)

    if R_w is not None and R_w > 0 and v_max is not None:
        omega_max = v_max / R_w  # rad/s
        bounds["prev_avg_wheel_omega"] = (0.0, omega_max)
    else:
        bounds["prev_avg_wheel_omega"] = (None, None)

    # ===========================
    # 4. PHYSICAL CONSTANTS
    # ===========================

    # Heading error: wraps at +/- pi
    bounds["frenet_u"] = (-np.pi, np.pi)  # rad

    # ===========================
    # 5. TRACK-DEPENDENT BOUNDS
    # ===========================

    # Ensure track exists
    if not hasattr(env, "track") or env.track is None:
        raise ValueError(
            "Track must be set before calculating normalization bounds. "
            "Ensure env.track is initialized before using normalized observations."
        )

    # Extract track width and curvature info
    track = env.track
    min_width, max_width = get_min_max_track_width(track)
    half_max_width = 0.5 * max_width
    min_curv, max_curv = get_min_max_curvature(track)

    # Use info for frenet_n, lookahead obs bounds
    bounds["frenet_n"] = (-half_max_width, half_max_width)
    bounds["lookahead_widths"] = (min_width, max_width)
    bounds["lookahead_curvatures"] = (min_curv, max_curv)

    # ===========================
    # 6. VALIDATION
    # ===========================

    # Check for missing parameters (None values in bounds)
    missing_features = []
    for feature_name, (min_val, max_val) in bounds.items():
        if min_val is None or max_val is None:
            missing_features.append(feature_name)

    if missing_features:
        raise ValueError(
            f"Missing required vehicle parameters. The following features have None bounds: {missing_features}. "
            f"Please ensure all required parameters (v_max, s_min, s_max, a_max, R_w) are configured in env.params."
        )

    # Ensure all bounds have min < max
    for feature_name, (min_val, max_val) in bounds.items():
        if min_val > max_val:
            raise ValueError(
                f"Invalid bounds for feature '{feature_name}': min={min_val} >= max={max_val}. "
                f"Bounds must satisfy min < max."
            )

    return bounds


def normalize_feature(
    feature_name: str, feature_value: np.float32 | np.ndarray, norm_bounds: dict
) -> np.float32 | np.ndarray:
    """
    Normalize a feature value using min-max normalization to [-1, 1] range.

    Parameters
    ----------
    feature_name : str
        Name of the feature to normalize
    feature_value : np.float32 | np.ndarray
        The feature value(s) to normalize (can be scalar or array)
    norm_bounds : dict
        Dictionary mapping feature names to (min, max) tuples

    Returns
    -------
    np.float32 | np.ndarray
        Normalized feature value(s) in [-1, 1] range

    Raises
    ------
    ValueError
        If feature_name is not found in norm_bounds
    """
    # Raise error if feature does not have normalization bounds defined
    if feature_name not in norm_bounds:
        available_features = list(norm_bounds.keys())
        raise ValueError(
            f"Cannot normalize feature '{feature_name}': no bounds defined. "
            f"Available features with bounds: {available_features}"
        )

    min_val, max_val = norm_bounds[feature_name]

    # Check if range is too small (constant feature) - prevents division by zero
    range_val = max_val - min_val
    if np.isclose(range_val, 0.0, atol=1e-9):
        # Feature is essentially constant, return 0.0 (center of [-1, 1])
        if isinstance(feature_value, np.ndarray):
            return np.zeros_like(feature_value, dtype=np.float32)
        else:
            return np.float32(0.0)

    # Min-max normalization to [-1, 1]: 2 * (x - min) / (max - min) - 1
    normalized = 2.0 * (feature_value - min_val) / range_val - 1.0

    # Clip to [-1, 1] to handle any values outside expected bounds
    return np.clip(normalized, -1.0, 1.0)
