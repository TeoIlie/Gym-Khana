"""Vehicle parameter loading from YAML files."""

import copy
from pathlib import Path

import yaml

_PARAMS_DIR = Path(__file__).parent


def load_params(name: str) -> dict:
    """Load a vehicle parameter set by name.

    Resolves *name* to ``<params_dir>/<name>.yaml`` and returns the contents
    as a plain ``dict``.  If the YAML contains ``_base`` and ``_overrides``
    keys, the base parameter set is loaded first and the overrides are applied
    on top (single level of inheritance).

    A fresh copy is returned on every call so callers may mutate the result
    without affecting subsequent loads.

    Args:
        name: Parameter set name (without ``.yaml`` extension), e.g.
            ``"f1tenth_std"`` or ``"f1tenth_std_drift_bias"``.

    Returns:
        Vehicle parameter dictionary.
    """
    path = _PARAMS_DIR / f"{name}.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)

    if "_base" in data:
        base = load_params(data["_base"])
        base.update(data.get("_overrides", {}))
        return base

    return copy.deepcopy(data)
