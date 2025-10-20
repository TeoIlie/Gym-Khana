from dataclasses import dataclass
from typing import Optional


@dataclass
class SteeringParameters:
    """
    Class defines all steering related parameters
    """

    # constraints regarding steering
    min: Optional[float] = None  # minimum steering angle [rad]
    max: Optional[float] = None  # maximum steering angle [rad]
    v_min: Optional[float] = None  # minimum steering velocity [rad/s]
    v_max: Optional[float] = None  # maximum steering velocity [rad/s]
    kappa_dot_max: Optional[float] = None  # maximum curvature rate
    kappa_dot_dot_max: Optional[float] = None  # maximum curvature rate rate
