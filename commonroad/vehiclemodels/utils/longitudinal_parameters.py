from dataclasses import dataclass
from typing import Optional


@dataclass
class LongitudinalParameters:
    """
    Class defines all parameters related to the longitudinal dynamics
    """

    # constraints regarding longitudinal dynamics
    v_min: Optional[float] = None  # minimum velocity [m/s]
    v_max: Optional[float] = None  # minimum velocity [m/s]
    v_switch: Optional[float] = None  # switching velocity [m/s]
    a_max: Optional[float] = None  # maximum absolute acceleration [m/s^2]
    j_max: Optional[float] = None  # maximum longitudinal jerk [m/s^3]
    j_dot_max: Optional[float] = None  # maximum change of longitudinal jerk [m/s^4]
