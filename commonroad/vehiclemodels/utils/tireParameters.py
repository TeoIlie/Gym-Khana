from dataclasses import dataclass
from typing import Optional


@dataclass
class TireParameters:
    """
    Class defines all Tire Parameters
    """

    # tire parameters from ADAMS handbook
    # longitudinal coefficients
    p_cx1: Optional[float] = None  # Shape factor Cfx for longitudinal force
    p_dx1: Optional[float] = None  # Longitudinal friction Mux at Fznom
    p_dx3: Optional[float] = None  # Variation of friction Mux with camber
    p_ex1: Optional[float] = None  # Longitudinal curvature Efx at Fznom
    p_kx1: Optional[float] = None  # Longitudinal slip stiffness Kfx/Fz at Fznom
    p_hx1: Optional[float] = None  # Horizontal shift Shx at Fznom
    p_vx1: Optional[float] = None  # Vertical shift Svx/Fz at Fznom
    r_bx1: Optional[float] = None  # Slope factor for combined slip Fx reduction
    r_bx2: Optional[float] = None  # Variation of slope Fx reduction with kappa
    r_cx1: Optional[float] = None  # Shape factor for combined slip Fx reduction
    r_ex1: Optional[float] = None  # Curvature factor of combined Fx
    r_hx1: Optional[float] = None  # Shift factor for combined slip Fx reduction

    # lateral coefficients
    p_cy1: Optional[float] = None  # Shape factor Cfy for lateral forces
    p_dy1: Optional[float] = None  # Lateral friction Muy
    p_dy3: Optional[float] = None  # Variation of friction Muy with squared camber
    p_ey1: Optional[float] = None  # Lateral curvature Efy at Fznom
    p_ky1: Optional[float] = None  # Maximum value of stiffness Kfy/Fznom
    p_hy1: Optional[float] = None  # Horizontal shift Shy at Fznom
    p_hy3: Optional[float] = None  # Variation of shift Shy with camber
    p_vy1: Optional[float] = None  # Vertical shift in Svy/Fz at Fznom
    p_vy3: Optional[float] = None  # Variation of shift Svy/Fz with camber
    r_by1: Optional[float] = None  # Slope factor for combined Fy reduction
    r_by2: Optional[float] = None  # Variation of slope Fy reduction with alpha
    r_by3: Optional[float] = None  # Shift term for alpha in slope Fy reduction
    r_cy1: Optional[float] = None  # Shape factor for combined Fy reduction
    r_ey1: Optional[float] = None  # Curvature factor of combined Fy
    r_hy1: Optional[float] = None  # Shift factor for combined Fy reduction
    r_vy1: Optional[float] = None  # Kappa induced side force Svyk/Muy*Fz at Fznom
    r_vy3: Optional[float] = None  # Variation of Svyk/Muy*Fz with camber
    r_vy4: Optional[float] = None  # Variation of Svyk/Muy*Fz with alpha
    r_vy5: Optional[float] = None  # Variation of Svyk/Muy*Fz with kappa
    r_vy6: Optional[float] = None  # Variation of Svyk/Muy*Fz with atan(kappa)
