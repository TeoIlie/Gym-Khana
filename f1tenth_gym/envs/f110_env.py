# MIT License

# Copyright (c) 2020 Joseph Auckley, Matthew O'Kelly, Aman Sinha, Hongrui Zheng

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Author: Hongrui Zheng
"""

# gym imports
import gymnasium as gym

# others
import numpy as np
import warnings

from .action import CarAction, from_single_to_multi_action_space

# base classes
from .base_classes import DynamicModel, Simulator
from .integrator import IntegratorType
from .observation import observation_factory
from .rendering import make_renderer
from .reset import make_reset_fn
from .track import Track
from .utils import deep_update


class F110Env(gym.Env):
    """
    Gymnasium environment for F1TENTH. For API specs, see https://gymnasium.farama.org/api/env/#

    Args:
        kwargs:
            seed (int, default=12345): seed for random state and reproducibility
            map (str, default='vegas'): name of the map used for the environment.

            params (dict, default={'mu': 1.0489, 'C_Sf':, 'C_Sr':, 'lf': 0.15875, 'lr': 0.17145, 'h': 0.074, 'm': 3.74, 'I': 0.04712, 's_min': -0.4189, 's_max': 0.4189, 'sv_min': -3.2, 'sv_max': 3.2, 'v_switch':7.319, 'a_max': 9.51, 'v_min':-5.0, 'v_max': 20.0, 'width': 0.31, 'length': 0.58}): dictionary of vehicle parameters.
            mu: surface friction coefficient
            C_Sf: Cornering stiffness coefficient, front
            C_Sr: Cornering stiffness coefficient, rear
            lf: Distance from center of gravity to front axle
            lr: Distance from center of gravity to rear axle
            h: Height of center of gravity
            m: Total mass of the vehicle
            I: Moment of inertial of the entire vehicle about the z axis
            s_min: Minimum steering angle constraint
            s_max: Maximum steering angle constraint
            sv_min: Minimum steering velocity constraint
            sv_max: Maximum steering velocity constraint
            v_switch: Switching velocity (velocity at which the acceleration is no longer able to create wheel spin)
            a_max: Maximum longitudinal acceleration
            v_min: Minimum longitudinal velocity
            v_max: Maximum longitudinal velocity
            width: width of the vehicle in meters
            length: length of the vehicle in meters

            num_agents (int, default=2): number of agents in the environment

            timestep (float, default=0.01): physics timestep

            ego_idx (int, default=0): ego's index in list of agents
    """

    # NOTE: change matadata with default rendering-modes, add definition of render_fps
    metadata = {"render_modes": ["human", "human_fast", "rgb_array"], "render_fps": 100}

    def __init__(self, config: dict = None, render_mode=None, **kwargs):
        super().__init__()

        # Configuration
        self.config = self.default_config()
        self.configure(config)

        self.seed = self.config["seed"]
        self.params = self.config["params"]
        self.num_agents = self.config["num_agents"]
        self.timestep = self.config["timestep"]
        self.ego_idx = self.config["ego_idx"]
        self.integrator = IntegratorType.from_string(self.config["integrator"])
        self.model = DynamicModel.from_string(self.config["model"])
        self.observation_config = self.config["observation_config"]
        self.normalize_act = self.config["normalize_act"]
        self.action_type = CarAction(self.config["control_input"], params=self.params, normalize=self.normalize_act)
        self.num_beams = self.config["num_beams"]

        # training mode
        self.training_mode = self.config["training_mode"]

        # conditionally set parameters based on training_mode
        match self.training_mode:
            case "race":
                self.map = self.config["map"]

            case "recover":
                # set recovery map
                self.map = self.config["recovery_map"]

                # set initial and final arc-lengths
                self.recovery_s_init = self.config["recovery_s_init"]
                self.recovery_s_max = self.config["recovery_s_max"]

                # set ranges for initial state perturbation
                self.recovery_v_range = self.config["recovery_v_range"]
                self.recovery_beta_range = self.config["recovery_beta_range"]
                self.recovery_yaw_range = self.config["recovery_yaw_range"]
                self.recovery_r_range = self.config["recovery_r_range"]

                # set reward parameters
                self.recovery_euclid_gain = self.config["recovery_euclid_gain"]
                self.recovery_timestep_penalty = self.config["recovery_timestep_penalty"]
                self.recovery_success_reward = self.config["recovery_success_reward"]
                self.recovery_collision_penalty = self.config["recovery_collision_penalty"]

                # set recovery condition threshold values
                self.recovery_delta_thresh = self.config["recovery_delta_thresh"]
                self.recovery_beta_thresh = self.config["recovery_beta_thresh"]
                self.recovery_r_thresh = self.config["recovery_r_thresh"]
                self.recovery_d_beta_thresh = self.config["recovery_d_beta_thresh"]
                self.recovery_d_r_thresh = self.config["recovery_d_r_thresh"]
                self.recovery_frenet_u_thresh = self.config["recovery_frenet_u_thresh"]

                # initialize values for beta, r derivative tracking
                self.prev_beta = 0.0
                self.prev_r = 0.0

            case _:
                raise ValueError(f"Invalid training_mode: '{self.training_mode}'")

        # rendering and debug configuration
        self.render_track_lines = self.config["render_track_lines"]
        self.render_arc_length_annotations = self.config["render_arc_length_annotations"]
        self.arc_length_annotation_interval = self.config["arc_length_annotation_interval"]
        self.render_lookahead_curvatures = self.config["render_lookahead_curvatures"]
        self.lookahead_n_points = self.config["lookahead_n_points"]
        self.lookahead_ds = self.config["lookahead_ds"]
        self.sparse_width_obs = self.config["sparse_width_obs"]
        self.debug_frenet_projection = self.config["debug_frenet_projection"]
        self.record_obs_min_max = self.config["record_obs_min_max"]

        # reward params
        self.out_of_bounds_penalty = self.config["out_of_bounds_penalty"]
        self.progress_gain = self.config["progress_gain"]
        self.negative_vel_penalty = self.config["negative_vel_penalty"]
        self.max_episode_steps = self.config["max_episode_steps"]
        self.current_step = 0

        # collision detection strategy
        self.predictive_collision = self.config["predictive_collision"]

        # wall deflection behavior
        self.wall_deflection = self.config["wall_deflection"]

        # track direction configuration
        self.track_direction_config = self.config["track_direction"]

        # Validate track direction config
        if self.track_direction_config not in ["normal", "reverse", "random"]:
            raise ValueError(
                f"Invalid track_direction: '{self.track_direction_config}'. "
                f"Must be one of: 'normal', 'reverse', 'random'"
            )

        # Set initial direction
        self._resolve_direction()

        assert self.progress_gain >= 1.0, "Progress gain must be >= 1."

        assert self.lookahead_n_points >= 2, "Minimum of 2 lookahead track observation points required"

        # radius to consider done
        self.start_thresh = 0.5  # 10cm

        # env states
        self.poses_x = []
        self.poses_y = []
        self.poses_theta = []
        self.collisions = np.zeros((self.num_agents,))
        self.boundary_exceeded = np.zeros((self.num_agents,), dtype=bool)

        # loop completion
        self.near_start = True
        self.num_toggles = 0

        # race info
        self.lap_times = np.zeros((self.num_agents,))
        self.lap_counts = np.zeros((self.num_agents,))
        self.current_time = 0.0

        # finish line info
        self.num_toggles = 0
        self.near_start = True
        self.near_starts = np.array([True] * self.num_agents)
        self.toggle_list = np.zeros((self.num_agents,))
        self.start_xs = np.zeros((self.num_agents,))
        self.start_ys = np.zeros((self.num_agents,))
        self.start_thetas = np.zeros((self.num_agents,))
        self.start_rot = np.eye(2)

        # initiate stuff
        self.sim = Simulator(
            self.params,
            self.num_agents,
            self.seed,
            wall_deflection=self.wall_deflection,
            time_step=self.timestep,
            integrator=self.integrator,
            model=self.model,
            action_type=self.action_type,
            num_beams=self.num_beams,
        )
        self.sim.set_map(self.map, self.config["scale"])

        if isinstance(self.map, Track):
            self.track = self.map
        else:
            self.track = Track.from_track_name(
                self.map,
                track_scale=self.config["scale"],
            )  # load track in gym env for convenience

        # Set initial track direction
        self.track.set_direction(self.direction_reversed)

        # observations
        self.agent_ids = [f"agent_{i}" for i in range(self.num_agents)]

        assert "type" in self.observation_config, "observation_config must contain 'type' key"

        # steering min and steering velocity min should be symmetrical to max values
        assert self.params["s_min"] == -self.params["s_max"], "s_min must be equal to -s_max"
        assert self.params["sv_min"] == -self.params["sv_max"], "sv_min must be equal to -sv_max"

        # Store observation type for validation
        obs_type = self.observation_config["type"]

        # Validate drift observation requires STD model
        if obs_type == "drift" and self.model != DynamicModel.STD:
            raise ValueError(
                "The 'drift' observation type requires the single_track_drift (STD) model. "
                f"Current model: {self.model}. "
                "Please set model='std' (or model=DynamicModel.STD) when creating the environment."
            )

        # Handle normalization configuration
        normalize_obs = self.config["normalize_obs"]

        # Identify whether the chosen observation type is supported for normalization
        supported_obs_types = ["drift", "race", "frenet"]
        obs_norm_supported = obs_type in supported_obs_types

        if normalize_obs is None:
            # User did not set normalize - auto-set based on observation type
            # Default to True for observation types that support normalization (drift, race, frenet, etc.)
            self.normalize_obs = obs_norm_supported
        else:
            # User explicitly set normalize
            if normalize_obs and not obs_norm_supported:
                # If user wants normalization, but obs_type is incompatible, warn and overwrite normalize to False to prevent failures
                warnings.warn(
                    f"Observation normalization is only supported for {supported_obs_types} observation types, not '{obs_type}'. "
                    "Setting normalize_obs=False.",
                    UserWarning,
                )
                self.normalize_obs = False
            elif not normalize_obs and obs_norm_supported:
                # If user chose supported obs_type but set normalize to False, allow but warn
                warnings.warn(
                    f"Observation normalization is recommended for {obs_type} observation type but was disabled. "
                    "Verify this is intentional.",
                    UserWarning,
                )
                self.normalize_obs = False
            else:
                # In all other cases, accept user input
                self.normalize_obs = normalize_obs

        self.observation_type = observation_factory(env=self, **self.observation_config)
        self.observation_space = self.observation_type.space()

        # Initialize observation min/max tracking if requested

        # If user requests observation min/max tracking, check it is allowed
        if self.record_obs_min_max:
            if not obs_norm_supported:
                warnings.warn(
                    f"Observation min/max tracking only supported for {supported_obs_types} observation types, not '{obs_type}'. "
                    "Setting record_obs_min_max=False.",
                    UserWarning,
                )
                self.record_obs_min_max = False
            if not self.normalize_obs:
                warnings.warn(
                    f"Observation min/max tracking only supported if 'normalize_obs' is True. "
                    "Setting record_obs_min_max=False.",
                    UserWarning,
                )
                self.record_obs_min_max = False

        # Set up obs tracking if requested by user and allowed
        if self.record_obs_min_max:
            self.obs_min_max_tracker = {}
            for feature in self.observation_type.features:
                self.obs_min_max_tracker[feature] = {"min": float("inf"), "max": float("-inf")}
            self.record_obs_min_max = True
            self.obs_tracker_step_count = 0

        # action space
        self.action_space = from_single_to_multi_action_space(self.action_type.space, self.num_agents)

        # reset modes
        self.reset_fn = make_reset_fn(**self.config["reset_config"], track=self.track, num_agents=self.num_agents)

        # stateful observations for rendering
        # add choice of colors (same, random, ...)
        self.render_obs = None
        self.render_mode = render_mode

        # match render_fps to integration timestep
        self.metadata["render_fps"] = int(1.0 / self.timestep)
        if self.render_mode == "human_fast":
            self.metadata["render_fps"] *= 10  # boost fps by 10x
        self.renderer, self.render_spec = make_renderer(
            params=self.params,
            track=self.track,
            agent_ids=self.agent_ids,
            render_mode=render_mode,
            render_fps=self.metadata["render_fps"],
        )

        # automatically add track line rendering callback if configured
        if self.render_track_lines:
            self.add_render_callback(self.track.render_both_lines)

        # automatically add arc length annotation rendering callback if configured
        if self.render_arc_length_annotations:
            self.add_render_callback(
                lambda e: self.track.render_arc_length_annotations(e, interval=self.arc_length_annotation_interval)
            )

        # automatically add lookahead curvature rendering callback if configured
        if self.render_lookahead_curvatures:
            # Initialize cache for rendering callback's own s tracking
            self.add_render_callback(self._render_lookahead_curvatures_callback)

        # automatically add frenet projection debug visualization if configured
        if self.debug_frenet_projection:
            self.add_render_callback(self.track.render_frenet_projection)

    def _render_lookahead_curvatures_callback(self, e) -> None:
        """
        Render callback for lookahead curvature visualization.

        This method is called during rendering to display the lookahead
        curvature sampling points ahead of the ego vehicle on the centerline.
        Visualizes the points where curvature is sampled for drift control.

        Parameters
        ----------
        e : EnvRenderer
            Environment renderer object

        Notes
        -----
        - Only executed when env.render() is called, not during env.step() - confirmed this!
        - Silently skips on errors to avoid breaking rendering
        - Uses config parameters: lookahead_n_points, lookahead_ds
        - Maintains its own s_guess cache for accurate Frenet conversion
        """
        try:
            # Get ego agent state
            agent = self.sim.agents[self.ego_idx]
            x, y, theta = agent.state[0], agent.state[1], agent.state[4]

            # Convert to Frenet coordinates to get arc length s
            s, _, _ = self.track.cartesian_to_frenet(x, y, theta, use_raceline=False)

            # Render the lookahead curvature points
            self.track.render_lookahead_curvatures(
                e=e, vehicle_s=s, n_points=self.lookahead_n_points, ds=self.lookahead_ds
            )
        except Exception:
            # Silently skip on errors to avoid breaking rendering
            pass

    @classmethod
    def fullscale_vehicle_params(cls) -> dict:
        """
        This is copied as-is from commonroad-vehicle-models/PYTHON/vehiclemodels/parameters/parameters_vehicle1.yaml
        """
        params = {
            "mu": 1.0489,
            "C_Sf": 4.718,
            "C_Sr": 5.4562,
            "lf": 0.88392,
            "lr": 1.50876,
            "h": 0.074,
            "m": 1225.8878467253344,
            "I": 1538.8533713561394,
            "width": 1.674,
            "length": 4.298,
            # steering constraints
            "s_min": -0.91,
            "s_max": 0.91,
            "sv_min": -0.4,
            "sv_max": 0.4,
            # maximum curvature change
            "kappa_dot_max": 0.4,
            # maximum curvature rate rate
            "kappa_dot_dot_max": 20,
            # Longitudinal constraints
            "v_switch": 4.755,
            "a_max": 11.5,
            "v_min": -13.9,
            "v_max": 45.8,
            # maximum longitudinal jerk [m/s^3]
            "j_max": 10.0e3,
            # maximum longitudinal jerk change [m/s^4]
            "j_dot_max": 10.0e3,
            # Extra parameters (for future use in multibody simulation)
            # sprung mass [kg]  SMASS
            "m_s": 1094.542720290477,
            # unsprung mass front [kg]  UMASSF
            "m_uf": 65.67256321742863,
            # unsprung mass rear [kg]  UMASSR
            "m_ur": 65.67256321742863,
            # moments of inertia of sprung mass
            # moment of inertia for sprung mass in roll [kg m^2]  IXS
            "I_Phi_s": 244.04723069965206,
            # moment of inertia for sprung mass in pitch [kg m^2]  IYS
            "I_y_s": 1342.2597688480864,
            # moment of inertia for sprung mass in yaw [kg m^2]  IZZ
            "I_z": 1538.8533713561394,
            # moment of inertia cross product [kg m^2]  IXZ
            "I_xz_s": 0.0,
            # suspension parameters
            # suspension spring rate (front) [N/m]  KSF
            "K_sf": 21898.332429625985,
            # suspension damping rate (front) [N s/m]  KSDF
            "K_sdf": 1459.3902937206362,
            # suspension spring rate (rear) [N/m]  KSR
            "K_sr": 21898.332429625985,
            # suspension damping rate (rear) [N s/m]  KSDR
            "K_sdr": 1459.3902937206362,
            # geometric parameters
            # track width front [m]  TRWF
            "T_f": 1.389888,
            # track width rear [m]  TRWB
            "T_r": 1.423416,
            # lateral spring rate at compliant compliant pin joint between M_s and M_u [N/m]  KRAS
            "K_ras": 175186.65943700788,
            # auxiliary torsion roll stiffness per axle (normally negative) (front) [N m/rad]  KTSF
            "K_tsf": -12880.270509148304,
            # auxiliary torsion roll stiffness per axle (normally negative) (rear) [N m/rad]  KTSR
            "K_tsr": 0.0,
            # damping rate at compliant compliant pin joint between M_s and M_u [N s/m]  KRADP
            "K_rad": 10215.732056044453,
            # vertical spring rate of tire [N/m]  KZT
            "K_zt": 189785.5477234252,
            # center of gravity height of total mass [m]  HCG (mainly required for conversion to other vehicle models)
            "h_cg": 0.5577840000000001,
            # height of roll axis above ground (front) [m]  HRAF
            "h_raf": 0.0,
            # height of roll axis above ground (rear) [m]  HRAR
            "h_rar": 0.0,
            # M_s center of gravity above ground [m]  HS
            "h_s": 0.59436,
            # moment of inertia for unsprung mass about x-axis (front) [kg m^2]  IXUF
            "I_uf": 32.53963075995361,
            # moment of inertia for unsprung mass about x-axis (rear) [kg m^2]  IXUR
            "I_ur": 32.53963075995361,
            # wheel inertia, from internet forum for 235/65 R 17 [kg m^2]
            "I_y_w": 1.7,
            # lateral compliance rate of tire, wheel, and suspension, per tire [m/N]  KLT
            "K_lt": 1.0278264878518764e-05,
            # effective wheel/tire radius  chosen as tire rolling radius RR  taken from ADAMS documentation [m]
            "R_w": 0.344,
            # split of brake and engine torque
            "T_sb": 0.76,
            "T_se": 1,
            # suspension parameters
            # [rad/m]  DF
            "D_f": -0.6233595800524934,
            # [rad/m]  DR
            "D_r": -0.20997375328083986,
            # [needs conversion if nonzero]  EF
            "E_f": 0,
            # [needs conversion if nonzero]  ER
            "E_r": 0,
            # tire parameters from ADAMS handbook
            # longitudinal coefficients
            "tire_p_cx1": 1.6411,  # Shape factor Cfx for longitudinal force
            "tire_p_dx1": 1.1739,  # Longitudinal friction Mux at Fznom
            "tire_p_dx3": 0,  # Variation of friction Mux with camber
            "tire_p_ex1": 0.46403,  # Longitudinal curvature Efx at Fznom
            "tire_p_kx1": 22.303,  # Longitudinal slip stiffness Kfx/Fz at Fznom
            "tire_p_hx1": 0.0012297,  # Horizontal shift Shx at Fznom
            "tire_p_vx1": -8.8098e-006,  # Vertical shift Svx/Fz at Fznom
            "tire_r_bx1": 13.276,  # Slope factor for combined slip Fx reduction
            "tire_r_bx2": -13.778,  # Variation of slope Fx reduction with kappa
            "tire_r_cx1": 1.2568,  # Shape factor for combined slip Fx reduction
            "tire_r_ex1": 0.65225,  # Curvature factor of combined Fx
            "tire_r_hx1": 0.0050722,  # Shift factor for combined slip Fx reduction
            # lateral coefficients
            "tire_p_cy1": 1.3507,  # Shape factor Cfy for lateral forces
            "tire_p_dy1": 1.0489,  # Lateral friction Muy
            "tire_p_dy3": -2.8821,  # Variation of friction Muy with squared camber
            "tire_p_ey1": -0.0074722,  # Lateral curvature Efy at Fznom
            "tire_p_ky1": -21.92,  # Maximum value of stiffness Kfy/Fznom
            "tire_p_hy1": 0.0026747,  # Horizontal shift Shy at Fznom
            "tire_p_hy3": 0.031415,  # Variation of shift Shy with camber
            "tire_p_vy1": 0.037318,  # Vertical shift in Svy/Fz at Fznom
            "tire_p_vy3": -0.32931,  # Variation of shift Svy/Fz with camber
            "tire_r_by1": 7.1433,  # Slope factor for combined Fy reduction
            "tire_r_by2": 9.1916,  # Variation of slope Fy reduction with alpha
            "tire_r_by3": -0.027856,  # Shift term for alpha in slope Fy reduction
            "tire_r_cy1": 1.0719,  # Shape factor for combined Fy reduction
            "tire_r_ey1": -0.27572,  # Curvature factor of combined Fy
            "tire_r_hy1": 5.7448e-006,  # Shift factor for combined Fy reduction
            "tire_r_vy1": -0.027825,  # Kappa induced side force Svyk/Muy*Fz at Fznom
            "tire_r_vy3": -0.27568,  # Variation of Svyk/Muy*Fz with camber
            "tire_r_vy4": 12.12,  # Variation of Svyk/Muy*Fz with alpha
            "tire_r_vy5": 1.9,  # Variation of Svyk/Muy*Fz with kappa
            "tire_r_vy6": -10.704,  # Variation of Svyk/Muy*Fz with atan(kappa)
        }
        return params

    @classmethod
    def f1fifth_vehicle_params(cls) -> dict:
        params = {
            "mu": 1.1,
            "C_Sf": 5.3507,
            "C_Sr": 5.3507,
            "lf": 0.2725,
            "lr": 0.2585,
            "h": 0.1825,
            "m": 15.32,
            "I": 0.64332,
            "s_min": -0.4189,
            "s_max": 0.4189,
            "sv_min": -3.2,
            "sv_max": 3.2,
            "v_switch": 7.319,
            "a_max": 9.51,
            "v_min": -5.0,
            "v_max": 20.0,
            "width": 0.55,
            "length": 0.8,
        }
        return params

    @classmethod
    def f1tenth_vehicle_params(cls) -> dict:
        """
        Default parameters.
        """
        params = {
            "mu": 1.0489,
            "C_Sf": 4.718,
            "C_Sr": 5.4562,
            "lf": 0.15875,
            "lr": 0.17145,
            "h": 0.074,
            "m": 3.74,
            "I": 0.04712,
            "s_min": -0.4189,
            "s_max": 0.4189,
            "sv_min": -3.2,
            "sv_max": 3.2,
            "v_switch": 7.319,
            "a_max": 9.51,
            "v_min": -5.0,
            "v_max": 20.0,
            "width": 0.31,
            "length": 0.58,
        }
        return params

    @classmethod
    def f1tenth_std_drift_bias_params(cls) -> dict:
        """
        Returns params for Single Track Drift (STD) model with drift bias.

        Returns:
            dict: Parameter dictionary with drift bias for STD model
        """
        params = cls.f1tenth_std_vehicle_params().copy()

        # Overwrite specific params
        params["lf"] = 0.19812  # originally 0.15875 to promote rear weight bias
        params["lr"] = 0.13208  # originally 0.17145 to promote rear weight bias
        params["a_max"] = 7.0  # originally 9.51 (greater than 5.0 to allow more wheelspin)
        params["tire_p_dy1"] = 1.0  # adjusted from original 1.101 for more oversteer
        params["tire_p_ky1"] = -50.0  # adjusted from original -65.76 for more oversteer
        params["T_se"] = 0.0  # adjusted from original 0.5 to simulate RWD

        return params

    @classmethod
    def f1tenth_std_vehicle_params(cls) -> dict:
        """
        Returns default parameters for Single Track Drift (STD) model.
        Extends standard F1TENTH parameters with wheel dynamics and Pacejka tire model.

        Returns:
            dict: Complete parameter dictionary for STD model
        """
        params = {
            # =========================
            # ORIGINAL PARAMS FROM f1tenth_vehicle_params
            # =========================
            "mu": 1.0489,
            "C_Sf": 4.718,
            "C_Sr": 5.4562,
            "lf": 0.15875,
            "lr": 0.17145,
            "h": 0.074,
            "m": 3.74,
            "I_z": 0.04712,  # moment of inertia for sprung mass in yaw [kg m^2]
            "s_min": -0.5,  # originally -0.4189
            "s_max": 0.5,  # originally 0.4189
            "sv_min": -3.2,
            "sv_max": 3.2,
            "v_switch": 7.319,
            "a_max": 5.0,  # originally 9.51
            "v_min": -5.0,
            "v_max": 20.0,
            "width": 0.31,
            "length": 0.58,
            # =========================
            # NEW PARAMS
            # =========================
            # 1. New params copied from multi-body config
            # =========================
            # tire parameters from ADAMS handbook
            # longitudinal coefficients
            "tire_p_cx1": 1.6411,  # Shape factor Cfx for longitudinal force
            "tire_p_dx1": 1.23,  # Longitudinal friction Mux at Fznom
            "tire_p_dx3": 0,  # Variation of friction Mux with camber
            "tire_p_ex1": 0.46403,  # Longitudinal curvature Efx at Fznom
            "tire_p_kx1": 66.909,  # Longitudinal slip stiffness Kfx/Fz at Fznom
            "tire_p_hx1": 0.0012297,  # Horizontal shift Shx at Fznom
            "tire_p_vx1": -8.8098e-006,  # Vertical shift Svx/Fz at Fznom
            "tire_r_bx1": 13.276,  # Slope factor for combined slip Fx reduction
            "tire_r_bx2": -13.778,  # Variation of slope Fx reduction with kappa
            "tire_r_cx1": 1.2568,  # Shape factor for combined slip Fx reduction
            "tire_r_ex1": 0.65225,  # Curvature factor of combined Fx
            "tire_r_hx1": 0.0050722,  # Shift factor for combined slip Fx reduction
            # lateral coefficients
            "tire_p_cy1": 1.3507,  # Shape factor Cfy for lateral forces
            "tire_p_dy1": 1.101,  # Lateral friction Muy
            "tire_p_dy3": -2.8821,  # Variation of friction Muy with squared camber
            "tire_p_ey1": -0.0074722,  # Lateral curvature Efy at Fznom
            "tire_p_ky1": -65.76,  # Maximum value of stiffness Kfy/Fznom
            "tire_p_hy1": 0.0026747,  # Horizontal shift Shy at Fznom
            "tire_p_hy3": 0.031415,  # Variation of shift Shy with camber
            "tire_p_vy1": 0.037318,  # Vertical shift in Svy/Fz at Fznom
            "tire_p_vy3": -0.32931,  # Variation of shift Svy/Fz with camber
            "tire_r_by1": 7.1433,  # Slope factor for combined Fy reduction
            "tire_r_by2": 9.1916,  # Variation of slope Fy reduction with alpha
            "tire_r_by3": -0.027856,  # Shift term for alpha in slope Fy reduction
            "tire_r_cy1": 1.0719,  # Shape factor for combined Fy reduction
            "tire_r_ey1": -0.27572,  # Curvature factor of combined Fy
            "tire_r_hy1": 5.7448e-006,  # Shift factor for combined Fy reduction
            "tire_r_vy1": -0.027825,  # Kappa induced side force Svyk/Muy*Fz at Fznom
            "tire_r_vy3": -0.27568,  # Variation of Svyk/Muy*Fz with camber
            "tire_r_vy4": 12.12,  # Variation of Svyk/Muy*Fz with alpha
            "tire_r_vy5": 1.9,  # Variation of Svyk/Muy*Fz with kappa
            "tire_r_vy6": -10.704,  # Variation of Svyk/Muy*Fz with atan(kappa)
            # =========================
            # 2. New params modified for 1/10 scale
            # =========================
            "h_s": 0.074,  # height of center of gravity [m], copied from ETH simulator
            "R_w": 0.049,  # effective tire radius [m] estimated by me to be 49 mm which is 0.049 m
            "I_y_w": 0.00017,  # wheel inertia in [kg m^2], approximated by me using I = 1/2 * m * r^2, where m = 107.5g and r = 50 mm
            # split of brake and engine torque
            "T_sb": 0.5,  # torque split of brakes (percent of torque sent to front axle) [no units] - I'm assuming it's even front/back
            "T_se": 0.5,  # torque split of engine (percent of torque sent to front axle) [no units] - For AWD I'm assuming it's even front/back
        }
        return params

    @classmethod
    def default_config(cls) -> dict:
        """
        Default environment configuration.

        Can be overloaded in environment implementations, or by calling configure().

        Args:
            None

        Returns:
            a configuration dict
        """
        return {
            "seed": 12345,
            "map": "Spielberg",
            "scale": 1.0,
            "params": cls.f1tenth_vehicle_params(),
            "num_agents": 2,
            "timestep": 0.01,
            "ego_idx": 0,
            "integrator": "rk4",
            "model": "st",
            "control_input": ["speed", "steering_angle"],
            "observation_config": {"type": None},
            "reset_config": {"type": None},
            "scale": 1.0,
            "num_beams": 1080,
            "render_track_lines": False,
            "render_arc_length_annotations": False,
            "arc_length_annotation_interval": 2.0,
            "render_lookahead_curvatures": False,
            "lookahead_n_points": 10,
            "lookahead_ds": 0.3,
            "sparse_width_obs": False,
            "debug_frenet_projection": False,
            "normalize_obs": None,  # None = auto-set based on observation type
            "normalize_act": True,
            "predictive_collision": False,  # default Frenet-based boundary check
            "record_obs_min_max": False,
            "wall_deflection": False,  # default to no wall deflections
            "track_direction": "normal",  # "normal", "reverse", or "random"
            "training_mode": "race",  # "race" or "recover"
            "out_of_bounds_penalty": -50,
            "progress_gain": 5.0,
            "negative_vel_penalty": -1,
            "max_episode_steps": 4096,
            # Recovery mode parameters (safe defaults, no effect when training_mode="race")
            "recovery_map": "IMS",
            "recovery_s_init": 96,
            "recovery_s_max": 140,
            "recovery_v_range": [2, 20],
            "recovery_beta_range": [-1.047, 1.047],
            "recovery_r_range": [-1.571, 1.571],
            "recovery_yaw_range": [-1.047, 1.047],
            "recovery_euclid_gain": 1.0,
            "recovery_timestep_penalty": 1.0,
            "recovery_success_reward": 100,
            "recovery_collision_penalty": -50,
            "recovery_delta_thresh": 0.05,
            "recovery_beta_thresh": 0.05,
            "recovery_r_thresh": 0.1,
            "recovery_d_beta_thresh": 0.1,
            "recovery_d_r_thresh": 0.2,
            "recovery_frenet_u_thresh": 0.05,
        }

    def configure(self, config: dict) -> None:
        if config:
            self.config = deep_update(self.config, config)
            self.params = self.config["params"]

            if hasattr(self, "sim"):
                self.sim.update_params(self.config["params"])

            if hasattr(self, "action_space"):
                # if some parameters changed, recompute action space
                self.normalize_act = self.config["normalize_act"]
                self.action_type = CarAction(
                    self.config["control_input"], params=self.params, normalize=self.normalize_act
                )
                self.action_space = from_single_to_multi_action_space(self.action_type.space, self.num_agents)

    def _resolve_direction(self) -> None:
        """
        Resolve track direction based on configuration.

        Sets self.direction_reversed based on self.track_direction_config:
        - "normal": False (drive forward)
        - "reverse": True (drive backward)
        - "random": randomly choose 50/50
        """
        if self.track_direction_config == "normal":
            self.direction_reversed = False
        elif self.track_direction_config == "reverse":
            self.direction_reversed = True
        else:  # "random"
            self.direction_reversed = np.random.random() < 0.5

    def _check_done(self):
        """
        Check if the current episode should end. Distinguishes between
        natural termination (collision/boundary) and truncation (time limit).

        Reset behavior depends on collision detection mode:
        - Predictive (TTC): Reset when ego agent has TTC collision or all agents complete 2 laps
        - Frenet (Drift): Reset only when ego agent exceeds track boundaries

        Args:
            None

        Returns:
            terminated (bool): whether the episode ended due to a terminal state
            truncated (bool): whether the episode was truncated due to time limit
            toggle_list (list[int]): each agent's toggle list for crossing the finish zone
        """

        # this is assuming 2 agents
        # TODO: switch to maybe s-based
        left_t = 2
        right_t = 2

        poses_x = np.array(self.poses_x) - self.start_xs
        poses_y = np.array(self.poses_y) - self.start_ys
        delta_pt = np.dot(self.start_rot, np.stack((poses_x, poses_y), axis=0))
        temp_y = delta_pt[1, :]
        idx1 = temp_y > left_t
        idx2 = temp_y < -right_t
        temp_y[idx1] -= left_t
        temp_y[idx2] = -right_t - temp_y[idx2]
        temp_y[np.invert(np.logical_or(idx1, idx2))] = 0

        dist2 = delta_pt[0, :] ** 2 + temp_y**2
        closes = dist2 <= 0.1
        for i in range(self.num_agents):
            if closes[i] and not self.near_starts[i]:
                self.near_starts[i] = True
                self.toggle_list[i] += 1
            elif not closes[i] and self.near_starts[i]:
                self.near_starts[i] = False
                self.toggle_list[i] += 1
            self.lap_counts[i] = self.toggle_list[i] // 2
            if self.toggle_list[i] < 4:
                self.lap_times[i] = self.current_time

        # Determine natural termination based on collision detection mode
        if self.predictive_collision:
            terminated = (self.collisions[self.ego_idx]) or np.all(self.toggle_list >= 4)
        else:
            terminated = self.boundary_exceeded[self.ego_idx]

        # Truncate based on timestep limit
        truncated = self.current_step > self.max_episode_steps

        return bool(terminated), bool(truncated), self.toggle_list >= 4

    def _update_state(self):
        """
        Update the env's states according to observations.

        Note: When predictive_collision=False (Frenet mode), self.collisions
        reflects simulator TTC/GJK collision state, but is unused for reset/reward.
        Instead, self.boundary_exceeded tracks Frenet-based boundary violations.
        """
        self.poses_x = self.sim.agent_poses[:, 0]
        self.poses_y = self.sim.agent_poses[:, 1]
        self.poses_theta = self.sim.agent_poses[:, 2]
        self.collisions = self.sim.collisions

        # Check boundaries once per step in Frenet mode
        if not self.predictive_collision:
            for i in range(self.num_agents):
                self.boundary_exceeded[i] = self._check_boundary_frenet(i)

    def _update_obs_min_max(self):
        """
        Update observation min/max tracking with current observation values.
        Called after each step when tracking is enabled.
        """

        raw_features = getattr(self.observation_type, "_last_raw_features", None)
        if raw_features is None:
            return

        self.obs_tracker_step_count += 1

        for feature_name, feature_value in raw_features.items():
            # Handle array-valued vs scalar features
            if isinstance(feature_value, (list, np.ndarray)):
                curr_min = float(np.min(feature_value))
                curr_max = float(np.max(feature_value))
            else:
                curr_min = float(feature_value)
                curr_max = float(feature_value)

            # Skip invalid values (NaN or Inf)
            if np.isnan(curr_min) or np.isnan(curr_max) or np.isinf(curr_min) or np.isinf(curr_max):
                continue

            # Update tracker
            if curr_min < self.obs_min_max_tracker[feature_name]["min"]:
                self.obs_min_max_tracker[feature_name]["min"] = curr_min
            if curr_max > self.obs_min_max_tracker[feature_name]["max"]:
                self.obs_min_max_tracker[feature_name]["max"] = curr_max

    def _print_obs_min_max_stats(self):
        """
        Print observation min/max statistics at the end of training/evaluation.
        Shows recorded values compared to theoretical normalization bounds.
        """
        TABLE_WIDTH = 120
        FEATURE_WIDTH = 25
        OBS_WIDTH = 12

        # Print header
        print("\n" + "=" * TABLE_WIDTH)
        print("Observation Min/Max Statistics")
        print(f"Tracked over {self.obs_tracker_step_count:,} timesteps")
        print("=" * TABLE_WIDTH)
        print(
            f"{'Feature':<{FEATURE_WIDTH}} | {'Rec. Min':>{OBS_WIDTH}} | {'Rec. Max':>{OBS_WIDTH}} | {'Theor. Min':>{OBS_WIDTH}} | {'Theor. Max':>{OBS_WIDTH}} | {'Coverage':>{OBS_WIDTH}}"
        )
        print("-" * TABLE_WIDTH)

        bounds_violations = []

        for feature_name in self.observation_type.features:
            # Get recorded & theoretical min/max values
            rec_min = self.obs_min_max_tracker[feature_name]["min"]
            rec_max = self.obs_min_max_tracker[feature_name]["max"]
            theor_min, theor_max = self.observation_type.bounds[feature_name]

            # Handle case where no valid values were recorded
            if rec_min == float("inf") or rec_max == float("-inf"):
                print(
                    f"{feature_name:<{FEATURE_WIDTH}} | {'N/A':>{OBS_WIDTH}} | {'N/A':>{OBS_WIDTH}} | {'N/A':>{OBS_WIDTH}} | {'N/A':>{OBS_WIDTH}} | {'N/A':>{OBS_WIDTH}}"
                )
            else:
                # Calculate coverage
                theor_range = theor_max - theor_min
                rec_range = rec_max - rec_min
                coverage_str = f"{(rec_range / theor_range) * 100.0:6.1f}%" if theor_range > 0 else "N/A"

                # Check for violations
                exceeds = ""
                eps = 1e-4
                if rec_min < theor_min - eps or rec_max > theor_max + eps:
                    bounds_violations.append(feature_name)
                    exceeds = " ⚠️"

                print(
                    f"{feature_name:<{FEATURE_WIDTH}} | {rec_min:12.4f} | {rec_max:12.4f} | {theor_min:12.4f} | {theor_max:12.4f} | {coverage_str:>{OBS_WIDTH}}{exceeds}"
                )

        print("=" * TABLE_WIDTH)

        if bounds_violations:
            print("⚠️  WARNING: The following features exceeded theoretical bounds:")
            for feature_name in bounds_violations:
                print(f"    - {feature_name}")
            print("   Consider updating normalization bounds in calculate_norm_bounds()")
            print("=" * TABLE_WIDTH)

        print()

    def _check_boundary_frenet(self, agent_idx: int) -> bool:
        """
        Check if agent has exceeded track boundaries using Frenet coordinates.

        This method provides explicit boundary detection based on the agent's
        lateral deviation from the centerline. Used when predictive_collision=False
        (drift mode) to detect exact boundary crossings rather than predictive
        TTC-based collisions.

        Args:
            agent_idx (int): Index of the agent to check

        Returns:
            bool: True if agent has exceeded track boundaries, False otherwise

        Raises:
            RuntimeError: If Frenet conversion fails
            ValueError: If track boundary data is missing or invalid
        """
        # Get agent position
        x = self.poses_x[agent_idx]
        y = self.poses_y[agent_idx]
        theta = self.poses_theta[agent_idx]

        # Convert to Frenet coordinates
        try:
            s, ey, _ = self.track.cartesian_to_frenet(x, y, theta, use_raceline=False)
        except Exception as e:
            raise RuntimeError(
                f"Frenet coordinate conversion failed for agent {agent_idx} at position ({x:.2f}, {y:.2f}). "
                f"This is required for boundary checking with predictive_collision=False. Error: {e}"
            ) from e

        centerline = self.track.centerline

        # Check if boundary data is available
        if centerline.w_lefts is None or centerline.w_rights is None:
            raise ValueError(
                "Track boundary data (w_lefts, w_rights) is not available. "
                "Frenet-based boundary checking requires track width information. "
                "Ensure track is loaded with boundary data"
            )

        if not hasattr(centerline, "ss") or centerline.ss is None:
            raise ValueError(
                "Centerline arc length data (ss) is not available. "
                "Frenet-based boundary checking requires arc length information for interpolation. "
                "Ensure track is properly initialized"
            )

        # Find nearest waypoint index - np.argmin is efficient for this use case without numba
        idx = np.argmin(np.abs(centerline.ss - s))

        # Get track width at this position
        w_left = centerline.w_lefts[idx]
        w_right = centerline.w_rights[idx]
        half_width = (w_left + w_right) / 2

        # Check boundary violation
        if abs(ey) > half_width:
            return True  # Boundary exceeded

        return False  # Within boundaries

    def _get_reward(self):
        """
        Get the reward for the current step

        Reward structure depends on collision detection mode:
        - Predictive (TTC): progress - penalties (additive)
        - Frenet (Drift): -1 OR progress (exclusive)
        """

        reward = 0.0
        track_length = self.track.centerline.spline.s[-1]

        for i in range(self.num_agents):
            # current_s calculated as distance along track centerline from start, in meters
            current_s, _ = self.track.centerline.spline.calc_arclength_inaccurate(self.poses_x[i], self.poses_y[i])

            # progress is current - previous arc length
            prog = current_s - self.last_s[i]

            # correct forward/backward track wraparound
            prog = self._correct_wraparound_prog(prog=prog, track_length=track_length)

            # reward forward progress, multiplied by progress_gain
            prog_r = prog * self.progress_gain

            # penalize negative longitudinal velocity v_x, as agent should not drive backward
            agent = self.sim.agents[i]
            vel = agent.standard_state["v_x"]
            if vel < 0:
                reward += self.negative_vel_penalty

            # Apply reward based on collision detection strategy
            if self.predictive_collision:
                # Predictive TTC mode: additive reward structure
                # Reward = sum(progress) - sum(collision_penalties)
                reward += prog_r
                if self.collisions[i]:
                    reward += self.out_of_bounds_penalty
            else:
                # Frenet boundary mode (drift): exclusive reward structure
                # Reward = -1 if boundary exceeded, else progress
                if self.boundary_exceeded[i]:
                    reward += self.out_of_bounds_penalty  # Exclusive penalty for boundary violation
                else:
                    reward += prog_r  # Only get progress if within boundaries

            self.last_s[i] = current_s

        return reward

    def step(self, action, skip_integration=False):
        """
        Step function for the gym env

        Args:
            action (np.ndarray(num_agents, 2))
            skip_integration (bool): if True, skip dynamics integration (used in reset to generate obs)

        Returns:
            obs (dict): observation of the current step
            reward (float, default=self.timestep): step reward, currently is physics timestep
            done (bool): if the simulation is done
            info (dict): auxillary information dictionary
        """

        # call simulation step
        self.sim.step(action, skip_integration=skip_integration)

        # observation
        obs = self.observation_type.observe()

        # Track observation min/max if enabled
        if self.record_obs_min_max:
            self._update_obs_min_max()

        # increment time and step
        self.current_time = self.current_time + self.timestep
        self.current_step += 1

        # update data member
        self._update_state()

        # rendering observation
        self.render_obs = {
            "ego_idx": self.sim.ego_idx,
            "poses_x": self.sim.agent_poses[:, 0],
            "poses_y": self.sim.agent_poses[:, 1],
            "poses_theta": self.sim.agent_poses[:, 2],
            "steering_angles": self.sim.agent_steerings,
            "lap_times": self.lap_times,
            "lap_counts": self.lap_counts,
            "collisions": self.sim.collisions,
            "sim_time": self.current_time,
        }

        # check done
        terminated, truncated, toggle_list = self._check_done()
        info = {
            "checkpoint_done": toggle_list,
            "episode_length": self.current_step,
        }

        # calc reward
        reward = self._get_reward()

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """
        Reset the gym environment by given poses or full states.

        Args:
            seed: random seed for the reset
            options: dictionary of options for the reset:
                - "poses": np.ndarray (num_agents, 3) - [x, y, yaw] per agent
                - "states": np.ndarray (num_agents, 7) - full state per agent (STD model only)
                            [x, y, delta, v, yaw, yaw_rate, slip_angle]
                Note: Cannot specify both "poses" and "states".

        Returns:
            obs (dict): observation of the current step
            reward (float, default=self.timestep): step reward, currently is physics timestep
            done (bool): if the simulation is done
            info (dict): auxillary information dictionary
        """
        if seed is not None:
            np.random.seed(seed=seed)
        super().reset(seed=seed)

        # Re-randomize direction for random
        if self.track_direction_config == "random":
            self._resolve_direction()

        # Swap Track's active centerline/raceline references
        self.track.set_direction(self.direction_reversed)

        # reset counters and data members
        self.current_time = 0.0
        self.collisions = np.zeros((self.num_agents,))
        self.boundary_exceeded = np.zeros((self.num_agents,), dtype=bool)
        self.num_toggles = 0
        self.near_start = True
        self.near_starts = np.array([True] * self.num_agents)
        self.toggle_list = np.zeros((self.num_agents,))
        self.current_step = 0

        # Parse options for poses and states
        poses = None
        states = None

        if options is not None:
            has_poses = "poses" in options
            has_states = "states" in options

            # Mutual exclusion check
            if has_poses and has_states:
                raise ValueError("Cannot provide both 'poses' and 'states' in reset options.")

            if has_states:
                # Validate STD model requirement
                if self.model != DynamicModel.STD:
                    raise ValueError(
                        f"Full state initialization only supported for STD model. "
                        f"Current model: {self.model}. Optionally use 'poses' instead."
                    )

                states = options["states"]
                assert isinstance(states, np.ndarray) and states.shape == (
                    self.num_agents,
                    7,
                ), f"States must be a numpy array of shape (num_agents, 7), got {states.shape}"

                # Save poses for downstream use
                # State indices: [x, y, delta, v, yaw, yaw_rate, slip_angle]
                poses = np.column_stack([states[:, 0], states[:, 1], states[:, 4]])

            elif has_poses:
                poses = options["poses"]

        # If no poses derived yet, sample from reset function
        if poses is None:
            poses = self.reset_fn.sample()

        assert isinstance(poses, np.ndarray) and poses.shape == (
            self.num_agents,
            3,
        ), "Initial poses must be a numpy array of shape (num_agents, 3)"

        self.start_xs = poses[:, 0]
        self.start_ys = poses[:, 1]
        self.start_thetas = poses[:, 2]
        self.start_rot = np.array(
            [
                [
                    np.cos(-self.start_thetas[self.ego_idx]),
                    -np.sin(-self.start_thetas[self.ego_idx]),
                ],
                [
                    np.sin(-self.start_thetas[self.ego_idx]),
                    np.cos(-self.start_thetas[self.ego_idx]),
                ],
            ]
        )

        # call reset to simulator (pass states if provided)
        self.sim.reset(poses, states=states)

        # Initialize last_s to actual starting arc lengths for accurate first-step reward
        self.last_s = []
        for i in range(self.num_agents):
            s, _, _ = self.track.cartesian_to_frenet(poses[i, 0], poses[i, 1], poses[i, 2], use_raceline=False)
            self.last_s.append(s)

        # get no input observations without integrating dynamics, so that RaceCar states are not changed
        action = np.zeros((self.num_agents, 2))
        obs, _, _, _, info = self.step(action, skip_integration=True)

        return obs, info

    def _correct_wraparound_prog(self, prog: float, track_length: float, margin=10.0) -> float:
        """
        Validate that progress is within physically possible bounds.

        Args:
            prog (float): Progress in meters for current timestep
            agent_idx (int): Index of the agent
        """
        # max progress used for clipping
        max_progress = self.params["v_max"] * self.timestep * margin
        # half track length used for wraparound detection
        half_track = track_length / 2

        # forward wraparound
        if prog < -half_track:
            prog += track_length

        # backward wraparound
        elif prog > half_track:
            prog -= track_length

        # clip in case progress is still out of bounds
        return np.clip(prog, -max_progress, max_progress)

    def update_map(self, map_name: str):
        """
        Updates the map used by simulation

        Args:
            map_name (str): name of the map

        Returns:
            None
        """
        self.sim.set_map(map_name)
        self.track = Track.from_track_name(map_name)

    def update_params(self, params, index=-1):
        """
        Updates the parameters used by simulation for vehicles

        Args:
            params (dict): dictionary of parameters
            index (int, default=-1): if >= 0 then only update a specific agent's params

        Returns:
            None
        """
        self.sim.update_params(params, agent_idx=index)

    def add_render_callback(self, callback_func):
        """
        Add extra drawing function to call during rendering.

        Args:
            callback_func (function (EnvRenderer) -> None): custom function to called during render()
        """
        if self.renderer is not None:
            self.renderer.add_renderer_callback(callback_func)

    def render(self, mode="human"):
        """
        Renders the environment with pyglet. Use mouse scroll in the window to zoom in/out, use mouse click drag to pan. Shows the agents, the map, current fps (bottom left corner), and the race information near as text.

        Args:
            mode (str, default='human'): rendering mode, currently supports:
                'human': slowed down rendering such that the env is rendered in a way that sim time elapsed is close to real time elapsed
                'human_fast': render as fast as possible

        Returns:
            None
        """
        # NOTE: separate render (manage render-mode) from render_frame (actual rendering with pyglet)

        if self.render_mode not in self.metadata["render_modes"]:
            return

        self.renderer.update(state=self.render_obs)
        return self.renderer.render()

    def close(self):
        """
        Ensure renderer is closed upon deletion
        """
        # Print observation statistics if tracking was enabled
        if self.record_obs_min_max:
            self._print_obs_min_max_stats()

        if self.renderer is not None:
            self.renderer.close()
        super().close()
