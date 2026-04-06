"""
Tests for CarAction __init__ parsing and order-independent control_input handling.
"""

import warnings

import numpy as np
import pytest

from gymkhana.envs.action import (
    AcclAction,
    CarAction,
    SpeedAction,
    SteeringAngleAction,
    SteeringSpeedAction,
)


@pytest.fixture
def params():
    """Minimal parameter dict for constructing CarAction."""
    return {
        "a_max": 9.51,
        "v_max": 20.0,
        "v_min": -5.0,
        "s_max": 0.4189,
        "s_min": -0.4189,
        "sv_max": 3.2,
        "sv_min": -3.2,
    }


# ============================================================================
# List input: order-independent parsing
# ============================================================================


class TestListOrderIndependence:
    """control_input as a list of two strings should work in any order."""

    def test_accl_steering_angle(self, params):
        ca = CarAction(["accl", "steering_angle"], params=params, normalize=False)
        assert isinstance(ca._longitudinal_action, AcclAction)
        assert isinstance(ca._steer_action, SteeringAngleAction)

    def test_steering_angle_accl(self, params):
        ca = CarAction(["steering_angle", "accl"], params=params, normalize=False)
        assert isinstance(ca._longitudinal_action, AcclAction)
        assert isinstance(ca._steer_action, SteeringAngleAction)

    def test_speed_steering_angle(self, params):
        ca = CarAction(["speed", "steering_angle"], params=params, normalize=False)
        assert isinstance(ca._longitudinal_action, SpeedAction)
        assert isinstance(ca._steer_action, SteeringAngleAction)

    def test_steering_angle_speed(self, params):
        ca = CarAction(["steering_angle", "speed"], params=params, normalize=False)
        assert isinstance(ca._longitudinal_action, SpeedAction)
        assert isinstance(ca._steer_action, SteeringAngleAction)

    def test_accl_steering_speed(self, params):
        ca = CarAction(["accl", "steering_speed"], params=params, normalize=False)
        assert isinstance(ca._longitudinal_action, AcclAction)
        assert isinstance(ca._steer_action, SteeringSpeedAction)

    def test_steering_speed_accl(self, params):
        ca = CarAction(["steering_speed", "accl"], params=params, normalize=False)
        assert isinstance(ca._longitudinal_action, AcclAction)
        assert isinstance(ca._steer_action, SteeringSpeedAction)

    def test_speed_steering_speed(self, params):
        ca = CarAction(["speed", "steering_speed"], params=params, normalize=False)
        assert isinstance(ca._longitudinal_action, SpeedAction)
        assert isinstance(ca._steer_action, SteeringSpeedAction)

    def test_steering_speed_speed(self, params):
        ca = CarAction(["steering_speed", "speed"], params=params, normalize=False)
        assert isinstance(ca._longitudinal_action, SpeedAction)
        assert isinstance(ca._steer_action, SteeringSpeedAction)

    def test_reversed_order_same_space(self, params):
        """Both orderings should produce identical action spaces."""
        ca1 = CarAction(["accl", "steering_angle"], params=params, normalize=False)
        ca2 = CarAction(["steering_angle", "accl"], params=params, normalize=False)
        np.testing.assert_array_equal(ca1.space.low, ca2.space.low)
        np.testing.assert_array_equal(ca1.space.high, ca2.space.high)

    def test_reversed_order_same_type(self, params):
        """Both orderings should report the same type tuple."""
        ca1 = CarAction(["speed", "steering_angle"], params=params, normalize=False)
        ca2 = CarAction(["steering_angle", "speed"], params=params, normalize=False)
        assert ca1.type == ca2.type


# ============================================================================
# List input: error cases
# ============================================================================


class TestListErrors:
    """Invalid list inputs should raise clear ValueErrors."""

    def test_two_longitudinal_types(self, params):
        with pytest.raises(ValueError, match="two longitudinal types"):
            CarAction(["accl", "speed"], params=params, normalize=False)

    def test_two_steering_types(self, params):
        with pytest.raises(ValueError, match="two steering types"):
            CarAction(["steering_angle", "steering_speed"], params=params, normalize=False)

    def test_unknown_mode(self, params):
        with pytest.raises(ValueError, match="Unknown control mode 'unknown'"):
            CarAction(["accl", "unknown"], params=params, normalize=False)

    def test_both_unknown(self, params):
        with pytest.raises(ValueError, match="Unknown control mode"):
            CarAction(["foo", "bar"], params=params, normalize=False)

    def test_empty_list(self, params):
        with pytest.raises(ValueError, match="exactly 2 elements"):
            CarAction([], params=params, normalize=False)

    def test_single_element_list(self, params):
        with pytest.raises(ValueError, match="exactly 2 elements"):
            CarAction(["accl"], params=params, normalize=False)

    def test_three_element_list(self, params):
        with pytest.raises(ValueError, match="exactly 2 elements"):
            CarAction(["accl", "steering_angle", "speed"], params=params, normalize=False)

    def test_non_string_non_list(self, params):
        with pytest.raises(ValueError, match="Unknown control mode"):
            CarAction(123, params=params, normalize=False)


# ============================================================================
# String input: single control mode with defaults
# ============================================================================


class TestSingleStringInput:
    """A single string should default the missing control type with a warning."""

    def test_accl_defaults_to_steering_speed(self, params):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ca = CarAction("accl", params=params, normalize=False)
            assert len(w) == 1
            assert "defaulting to steering speed" in str(w[0].message)
        assert isinstance(ca._longitudinal_action, AcclAction)
        assert isinstance(ca._steer_action, SteeringSpeedAction)

    def test_speed_defaults_to_steering_angle(self, params):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ca = CarAction("speed", params=params, normalize=False)
            assert len(w) == 1
            assert "defaulting to steering angle" in str(w[0].message)
        assert isinstance(ca._longitudinal_action, SpeedAction)
        assert isinstance(ca._steer_action, SteeringAngleAction)

    def test_steering_angle_defaults_to_speed(self, params):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ca = CarAction("steering_angle", params=params, normalize=False)
            assert len(w) == 1
            assert "defaulting to speed" in str(w[0].message)
        assert isinstance(ca._longitudinal_action, SpeedAction)
        assert isinstance(ca._steer_action, SteeringAngleAction)

    def test_steering_speed_defaults_to_accl(self, params):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ca = CarAction("steering_speed", params=params, normalize=False)
            assert len(w) == 1
            assert "defaulting to acceleration" in str(w[0].message)
        assert isinstance(ca._longitudinal_action, AcclAction)
        assert isinstance(ca._steer_action, SteeringSpeedAction)

    def test_unknown_string_raises(self, params):
        with pytest.raises(ValueError, match="Unknown control mode"):
            CarAction("unknown", params=params, normalize=False)


# ============================================================================
# Action space and type properties
# ============================================================================


class TestSpaceAndType:
    """Verify action space bounds and type tuple are consistent."""

    def test_space_shape(self, params):
        ca = CarAction(["accl", "steering_angle"], params=params, normalize=False)
        assert ca.space.shape == (2,)

    def test_space_ordering_steer_first(self, params):
        """Action space index 0 should be steer bounds, index 1 should be longitudinal bounds."""
        ca = CarAction(["accl", "steering_angle"], params=params, normalize=False)
        # index 0: steering angle bounds
        assert ca.space.low[0] == pytest.approx(params["s_min"])
        assert ca.space.high[0] == pytest.approx(params["s_max"])
        # index 1: acceleration bounds
        assert ca.space.low[1] == pytest.approx(-params["a_max"])
        assert ca.space.high[1] == pytest.approx(params["a_max"])

    def test_space_ordering_speed(self, params):
        ca = CarAction(["speed", "steering_angle"], params=params, normalize=False)
        # index 0: steering angle bounds
        assert ca.space.low[0] == pytest.approx(params["s_min"])
        assert ca.space.high[0] == pytest.approx(params["s_max"])
        # index 1: speed bounds
        assert ca.space.low[1] == pytest.approx(params["v_min"])
        assert ca.space.high[1] == pytest.approx(params["v_max"])

    def test_type_tuple_is_steer_then_long(self, params):
        """type property should return (steer_type, longitudinal_type)."""
        ca = CarAction(["accl", "steering_angle"], params=params, normalize=False)
        assert ca.type == ("steering_angle", "accl")

    def test_normalized_space(self, params):
        """When normalized, both dimensions should be [-1, 1]."""
        ca = CarAction(["accl", "steering_angle"], params=params, normalize=True)
        np.testing.assert_array_almost_equal(ca.space.low, [-1.0, -1.0])
        np.testing.assert_array_almost_equal(ca.space.high, [1.0, 1.0])
