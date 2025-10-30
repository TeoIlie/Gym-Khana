from typing import Any, Dict, TypeVar
import numpy as np

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

    Parameters
    ----------
    env : F110Env
        The environment to extract parameters from

    Returns
    -------
    dict
        Mapping of feature names to (min, max) tuples for normalization
    """
    # TODO: Implement proper bounds calculation based on vehicle params, track info, etc.
    # For now, return empty dict (normalization will be identity function)
    return {}


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
    """
    # If bounds not defined for this feature, return unchanged
    if feature_name not in norm_bounds:
        return feature_value

    min_val, max_val = norm_bounds[feature_name]

    # Min-max normalization to [-1, 1]: 2 * (x - min) / (max - min) - 1
    normalized = 2.0 * (feature_value - min_val) / (max_val - min_val) - 1.0

    # Clip to [-1, 1] to handle any values outside expected bounds
    return np.clip(normalized, -1.0, 1.0)
