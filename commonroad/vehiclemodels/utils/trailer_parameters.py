from dataclasses import dataclass
from typing import Optional


@dataclass
class TrailerParameters:
    """
    Class defines all trailer parameters (for on-axle trailer-truck models)
    """

    # class for trailer parameters
    l: Optional[float] = None  # trailer length [m]
    w: Optional[float] = None  # trailer width [m]
    l_hitch: Optional[float] = None  # hitch length [m]
    l_total: Optional[float] = None  # total system length [m]
    l_wb: Optional[float] = None  # trailer wheel base [m]
