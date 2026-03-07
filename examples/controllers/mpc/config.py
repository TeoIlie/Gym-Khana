from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict


class KMPCConfig(BaseModel):
    """Kinematic MPC configuration class"""

    model_config = ConfigDict(extra="forbid", use_attribute_docstrings=True)

    # general parameters
    N: int
    """Prediction horizon"""
    t_delay: float
    """Delay in seconds accounted for propagation before the MPC is applied"""
    steps_delay: int
    """Delays for the steering angle"""
    MPC_freq: int
    """Frequency of the MPC controller"""
    track_safety_margin: float
    """Safety margin for the track"""
    track_max_width: float
    """Maximum width of the track"""
    overtake_d: float
    """Distance to overtake"""

    # Cost function settings #
    qac: float
    """acceleration cost weight"""
    qddelta: float
    """steering angle rate cost weight"""
    qadv: float
    """advancement cost weight"""
    qn: float
    """lateral deviation cost weight"""
    qalpha: float
    """heading deviation cost weight"""
    qv: float
    """velocity tracking cost weight"""

    Zl: float
    """quadratic coefficient state slack variable"""
    Zu: float
    """quadratic coefficient input slack variable"""
    zl: float
    """linear coefficient state slack variable"""
    zu: float
    """linear coefficient input slack variable"""

    # Model constraints
    # state bounds
    delta_min: float
    """Minimum steering angle"""
    delta_max: float
    """Maximum steering angle"""
    v_min: float
    """Minimum velocity"""
    v_max: float
    """Maximum velocity"""

    # input bounds
    ddelta_min: float
    """Minimum steering angle rate"""
    ddelta_max: float
    """Maximum steering angle rate"""
    a_min: float
    """Minimum acceleration (Maximum Breaking)"""
    a_max: float
    """Maximum acceleration"""

    # nonlinear constraint
    alat_max: float
    """Maximum lateral acceleration"""

    @classmethod
    def from_yaml(cls, path: str | Path) -> "KMPCConfig":
        with open(path, "r") as f:
            return cls(**yaml.safe_load(f))


class CarConfig(BaseModel):
    """Car Parameters configuration class"""

    model_config = ConfigDict(extra="forbid", use_attribute_docstrings=True)

    m: float
    """Mass of the car"""
    Iz: float
    """Moment of inertia of the car"""
    lf: float
    """Distance from the center of mass to the front axle"""
    lr: float
    """Distance from the center of mass to the rear axle"""
    wheelbase: float
    """Wheelbase of the car"""
    h_cg: float
    """Height of the center of gravity"""
    a_max: float
    """Maximum acceleration"""
    a_min: float
    """Minimum acceleration (maximum deceleration)"""
    v_max: float
    """Maximum velocity"""
    v_min: float
    """Minimum velocity"""

    C_0d: float
    """Steering Angle-to-Servo Offset"""
    C_d: float
    """Steering Angle-to-Servo Gain"""
    C_acc: float
    """Acceleration to Current Gain"""
    C_dec: float
    """Deceleration to Current Gain"""
    C_R: float
    """Velocity to Current Gain"""
    C_0v: float
    """Velocity-to-erpm Offset"""
    C_v: float
    """Velocity-to-erpm Gain"""

    tau_steer: float
    """Steering Angle Time Constant"""
    max_steering_angle: float
    """Maximum Steering Angle in radians"""
    max_steering_velocity: float
    """Maximum Steering Velocity"""

    @classmethod
    def from_yaml(cls, path: str | Path) -> "CarConfig":
        with open(path, "r") as f:
            return cls(**yaml.safe_load(f))


class STMPCConfig(BaseModel):
    """Single track MPC configuration class"""

    model_config = ConfigDict(extra="forbid", use_attribute_docstrings=True)

    # general parameters
    N: int
    """Prediction horizon"""
    t_delay: float
    """Delay in seconds accounted for propagation before the MPC is applied"""
    steps_delay: int
    """Delays for the steering angle"""
    MPC_freq: int
    """Frequency of the MPC controller"""
    track_safety_margin: float
    """Safety margin for the track"""
    track_max_width: float
    """Maximum width of the track"""
    overtake_d: float
    """MPC Overtake path lateral tracking distance"""

    # Cost function settings #
    qjerk: float
    """jerk cost weight"""
    qddelta: float
    """steering angle rate cost weight"""
    qadv: float
    """advancement cost weight"""
    qn: float
    """lateral deviation cost weight"""
    qalpha: float
    """heading deviation cost weight"""
    qv: float
    """velocity tracking cost weight"""

    Zl: float
    """quadratic coefficient state slack variable"""
    Zu: float
    """quadratic coefficient input slack variable"""
    zl: float
    """linear coefficient state slack variable"""
    zu: float
    """linear coefficient input slack variable"""

    # Model constraints
    # state bounds
    delta_min: float
    """Minimum steering angle"""
    delta_max: float
    """Maximum steering angle"""
    v_min: float
    """Minimum velocity"""
    v_max: float
    """Maximum velocity"""
    a_min: float
    """Minimum acceleration (Maximum Breaking)"""
    a_max: float
    """Maximum acceleration"""

    # input bounds
    ddelta_min: float
    """Minimum steering angle rate"""
    ddelta_max: float
    """Maximum steering angle rate"""
    jerk_min: float
    """Minimum jerk"""
    jerk_max: float
    """Maximum jerk"""

    # nonlinear constraint
    alat_max: float
    """Maximum lateral acceleration"""

    # Cost Flags
    vy_minimization: bool
    """If true, minimize the lateral velocity"""
    adv_maximization: bool
    """If true, maximize the advancement"""

    # constraints flags
    combined_constraints: str
    """ellipse/diamond/anything else for None"""

    # model flags
    load_transfer: bool
    """If true, load transfer is considered"""
    correct_v_y_dot: bool
    """If true, the lateral velocity derivative is from the model and not approximated"""

    @classmethod
    def from_yaml(cls, path: str | Path) -> "STMPCConfig":
        with open(path, "r") as f:
            return cls(**yaml.safe_load(f))


class PacejkaTireConfig(BaseModel):
    """Pacejka Tire Parameters configuration class"""

    model_config = ConfigDict(extra="forbid", use_attribute_docstrings=True)

    friction_coeff: float
    """Friction coefficient of the tire"""
    Bf: float
    """Longitudinal stiffness coefficient (B) of the FRONT tire"""
    Cf: float
    """Cornering stiffness coefficient (C) of the FRONT tire"""
    Df: float
    """Peak value of the lateral force (D) of the FRONT tire"""
    Ef: float
    """Lateral stiffness coefficient (E) of the FRONT tire"""

    Br: float
    """Longitudinal stiffness coefficient (B) of the REAR tire"""
    Cr: float
    """Cornering stiffness coefficient (C) of the REAR tire"""
    Dr: float
    """Peak value of the lateral force (D) of the REAR tire"""
    Er: float
    """Lateral stiffness coefficient (E) of the REAR tire"""

    floor: str
    """Floor indicating location of the track"""

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PacejkaTireConfig":
        with open(path, "r") as f:
            return cls(**yaml.safe_load(f))
