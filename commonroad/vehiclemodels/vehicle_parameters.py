from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
from omegaconf import OmegaConf

from commonroad.vehiclemodels.utils.longitudinal_parameters import LongitudinalParameters
from commonroad.vehiclemodels.utils.steering_parameters import SteeringParameters
from commonroad.vehiclemodels.utils.tireParameters import TireParameters
from commonroad.vehiclemodels.utils.trailer_parameters import TrailerParameters


@dataclass
class VehicleParameters:
    """
    VehicleParameters base class: defines all parameters used by the vehicle models described in
    Althoff, M. and Würsching, G. "CommonRoad: Vehicle Models", 2020
    """

    # vehicle body dimensions
    l: Optional[float] = None
    w: Optional[float] = None

    # steering parameters
    steering: SteeringParameters = field(default_factory=SteeringParameters)

    # longitudinal parameters
    longitudinal: LongitudinalParameters = field(default_factory=LongitudinalParameters)

    # masses
    m: Optional[float] = None
    m_s: Optional[float] = None
    m_uf: Optional[float] = None
    m_ur: Optional[float] = None

    # axes distances
    a: Optional[float] = None  # distance from spring mass center of gravity to front axle [m]  LENA
    b: Optional[float] = None  # distance from spring mass center of gravity to rear axle [m]  LENB

    # moments of inertia of sprung mass
    I_Phi_s: Optional[float] = None  # moment of inertia for sprung mass in roll [kg m^2]  IXS
    I_y_s: Optional[float] = None  # moment of inertia for sprung mass in pitch [kg m^2]  IYS
    I_z: Optional[float] = None  # moment of inertia for sprung mass in yaw [kg m^2]  IZZ
    I_xz_s: Optional[float] = None  # moment of inertia cross product [kg m^2]  IXZ

    # suspension parameters
    K_sf: Optional[float] = None  # suspension spring rate (front) [N/m]  KSF
    K_sdf: Optional[float] = None  # suspension damping rate (front) [N s/m]  KSDF
    K_sr: Optional[float] = None  # suspension spring rate (rear) [N/m]  KSR
    K_sdr: Optional[float] = None  # suspension damping rate (rear) [N s/m]  KSDR

    # geometric parameters
    T_f: Optional[float] = None  # track width front [m]  TRWF
    T_r: Optional[float] = None  # track width rear [m]  TRWB
    K_ras: Optional[
        float
    ] = None  # lateral spring rate at compliant compliant pin joint between M_s and M_u [N/m]  KRAS

    K_tsf: Optional[
        float
    ] = None  # auxiliary torsion roll stiffness per axle (normally negative) (front) [N m/rad]  KTSF
    K_tsr: Optional[
        float
    ] = None  # auxiliary torsion roll stiffness per axle (normally negative) (rear) [N m/rad]  KTSR
    K_rad: Optional[float] = None  # damping rate at compliant compliant pin joint between M_s and M_u [N s/m]  KRADP
    K_zt: Optional[float] = None  # vertical spring rate of tire [N/m]  TSPRINGR

    h_cg: Optional[
        float
    ] = None  # center of gravity height of total mass [m]  HCG (mainly required for conversion to other vehicle models)
    h_raf: Optional[float] = None  # height of roll axis above ground (front) [m]  HRAF
    h_rar: Optional[float] = None  # height of roll axis above ground (rear) [m]  HRAR

    h_s: Optional[float] = None  # M_s center of gravity above ground [m]  HS

    I_uf: Optional[float] = None  # moment of inertia for unsprung mass about x-axis (front) [kg m^2]  IXUF
    I_ur: Optional[float] = None  # moment of inertia for unsprung mass about x-axis (rear) [kg m^2]  IXUR
    I_y_w: Optional[float] = None  # wheel inertia, from internet forum for 235/65 R 17 [kg m^2]

    K_lt: Optional[float] = None  # lateral compliance rate of tire, wheel, and suspension, per tire [m/N]  KLT
    R_w: Optional[
        float
    ] = None  # effective wheel/tire radius  chosen as tire rolling radius RR  taken from ADAMS documentation [m]

    # split of brake and engine torque
    T_sb: Optional[float] = None
    T_se: Optional[float] = None

    # suspension parameters
    D_f: Optional[float] = None  # [rad/m]  DF
    D_r: Optional[float] = None  # [rad/m]  DR
    E_f: Optional[float] = None  # [needs conversion if nonzero]  EF
    E_r: Optional[float] = None  # [needs conversion if nonzero]  ER

    # tire parameters
    tire: TireParameters = field(default_factory=TireParameters)

    # trailer parameters
    trailer: TrailerParameters = field(default_factory=TrailerParameters)


def setup_vehicle_parameters(vehicle_id: int, dir_params: str = None) -> VehicleParameters:
    """
    Creates a VehicleParameters object holding all vehicle parameters for a given vehicle type ID
    The parameters are read from the YAML-files in vehiclemodels/parameters or from the folder specified by dir_params
    :param vehicle_id: CommonRoad vehicle ID (see Althoff, M. and Würsching, G. "CommonRoad: Vehicle Models", 2020)
    :param dir_params: [optional] path to folder containing the parameter yaml files. If None, the default location
    in /vehiclemodels/parameters is used
    :return VehicleParameters object with vehicle parameters
    """
    # create structured config from dataclass
    structured_conf = OmegaConf.structured(VehicleParameters)

    # set path
    if dir_params:
        path_root = dir_params
    else:
        path_root = Path(__file__).parent / "parameters"

    # load configurations from yaml files
    conf_vehicle = OmegaConf.load(path_root / f'{"parameters_vehicle"}{vehicle_id}.yaml')
    conf_tires = OmegaConf.load(path_root / "parameters_tire.yaml")

    # create merged configuration and set as Read-only
    p = OmegaConf.merge(structured_conf, conf_vehicle, conf_tires)
    OmegaConf.set_readonly(p, True)

    return OmegaConf.to_object(p)


# *******************************
# Test parameter setup
# *******************************
if __name__ == "__main__":
    params1 = setup_vehicle_parameters(vehicle_id=1)
    params2 = setup_vehicle_parameters(vehicle_id=2)
    params3 = setup_vehicle_parameters(vehicle_id=3)
    params4 = setup_vehicle_parameters(vehicle_id=4)
