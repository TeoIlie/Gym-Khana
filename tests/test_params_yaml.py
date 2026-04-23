"""Validate vehicle parameter YAML files load correctly and contain expected keys."""

import pytest

from gymkhana.envs.params import load_params

# Minimum required keys for each model type.
# ST/KS models need the base set; STD models add tire + wheel dynamics keys.
_BASE_KEYS = {
    "mu",
    "C_Sf",
    "C_Sr",
    "lf",
    "lr",
    "h",
    "m",
    "s_min",
    "s_max",
    "sv_min",
    "sv_max",
    "v_switch",
    "a_max",
    "v_min",
    "v_max",
    "width",
    "length",
}

_TIRE_KEYS = {
    "tire_p_cx1",
    "tire_p_dx1",
    "tire_p_dx3",
    "tire_p_ex1",
    "tire_p_kx1",
    "tire_p_hx1",
    "tire_p_vx1",
    "tire_r_bx1",
    "tire_r_bx2",
    "tire_r_cx1",
    "tire_r_ex1",
    "tire_r_hx1",
    "tire_p_cy1",
    "tire_p_dy1",
    "tire_p_dy3",
    "tire_p_ey1",
    "tire_p_ky1",
    "tire_p_hy1",
    "tire_p_hy3",
    "tire_p_vy1",
    "tire_p_vy3",
    "tire_r_by1",
    "tire_r_by2",
    "tire_r_by3",
    "tire_r_cy1",
    "tire_r_ey1",
    "tire_r_hy1",
    "tire_r_vy1",
    "tire_r_vy3",
    "tire_r_vy4",
    "tire_r_vy5",
    "tire_r_vy6",
}

_STD_EXTRA_KEYS = {"I_z", "h_s", "R_w", "I_y_w", "T_sb", "T_se"} | _TIRE_KEYS


@pytest.mark.parametrize(
    "yaml_name, required_keys",
    [
        ("fullscale", _BASE_KEYS | {"I"} | _TIRE_KEYS),
        ("f1fifth", _BASE_KEYS | {"I"}),
        ("f1tenth_st", _BASE_KEYS | {"I"}),
        ("f1tenth_std", _BASE_KEYS | _STD_EXTRA_KEYS),
        ("f1tenth_std_drift_bias", _BASE_KEYS | _STD_EXTRA_KEYS),
    ],
)
def test_yaml_has_required_keys(yaml_name, required_keys):
    """Each YAML parameter file must contain all keys required by its model type."""
    params = load_params(yaml_name)
    missing = required_keys - set(params.keys())
    assert not missing, f"{yaml_name} missing keys: {missing}"


def test_yaml_values_are_numeric():
    """All parameter values should be int or float, not strings."""
    for name in ("fullscale", "f1fifth", "f1tenth_st", "f1tenth_std", "f1tenth_std_drift_bias"):
        params = load_params(name)
        for key, val in params.items():
            assert isinstance(val, (int, float)), (
                f"{name}[{key!r}] has type {type(val).__name__}, expected int or float"
            )


def test_load_params_returns_fresh_copy():
    """Each call to load_params should return an independent dict."""
    a = load_params("f1tenth_std")
    b = load_params("f1tenth_std")
    assert a is not b
    a["m"] = 999.0
    assert b["m"] != 999.0


def test_drift_bias_inherits_from_std():
    """Drift bias params should match f1tenth_std except for the known overrides."""
    base = load_params("f1tenth_std")
    bias = load_params("f1tenth_std_drift_bias")
    overridden = {"lf", "lr", "a_max", "tire_p_dy1", "tire_p_ky1", "T_se"}
    assert set(base.keys()) == set(bias.keys())
    for key in base:
        if key in overridden:
            assert base[key] != bias[key], f"{key} should differ but both are {base[key]}"
        else:
            assert base[key] == bias[key], f"{key} should match: base={base[key]}, bias={bias[key]}"
