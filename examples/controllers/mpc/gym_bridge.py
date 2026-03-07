from pathlib import Path

import numpy as np

from f1tenth_gym.envs.track import Track

from .config import CarConfig, KMPCConfig, PacejkaTireConfig, STMPCConfig
from .kmpc.kinematic_mpc import Kinematic_MPC_Controller
from .stmpc.single_track_mpc import Single_track_MPC_Controller


class KMPCGymBridge:
    """Bridge between F1TENTH gym environment and the Kinematic MPC controller.

    Uses the track centerline as reference path and the MPC's own FrenetConverter
    for coordinate conversion (not the gym's CubicSpline2D).
    """

    def __init__(self, track: Track, kmpc_config_path: str | Path, car_config_path: str | Path, ref_speed: float = 6.0):
        self.track = track
        kmpc_config = KMPCConfig.from_yaml(kmpc_config_path)
        car_config = CarConfig.from_yaml(car_config_path)

        cl = track.centerline
        xs = cl.xs.astype(np.float64)
        ys = cl.ys.astype(np.float64)
        ks = cl.ks.astype(np.float64)
        yaws = cl.yaws.astype(np.float64)
        w_lefts = cl.w_lefts.astype(np.float64)
        w_rights = cl.w_rights.astype(np.float64)

        n = len(xs)
        vx_ref = np.full(n, ref_speed, dtype=np.float64)

        # Recompute arc-length from (xs, ys) to match FrenetConverter/SplineTrack domains
        s_ref = np.zeros(n, dtype=np.float64)
        s_ref[1:] = np.cumsum(np.hypot(np.diff(xs), np.diff(ys)))

        # Build waypoint array (Nx8): [x, y, vx, 0, s_m, kappa, yaw, 0]
        # Only columns 2 (vx) and 4 (s_m) are used at runtime by main_loop
        self.waypoint_array = np.column_stack(
            [
                xs,
                ys,
                vx_ref,
                np.zeros(n),
                s_ref,
                ks,
                yaws,
                np.zeros(n),
            ]
        )

        self.controller = Kinematic_MPC_Controller(kmpc_config, car_config)
        self.controller.mpc_initialize_solver(xs, ys, vx_ref, ks, s_ref, w_lefts, w_rights)
        self.last_compute_time = 0.0

    def get_action(self, obs: dict) -> np.ndarray:
        agent_obs = obs["agent_0"]
        pose_x = float(agent_obs["pose_x"])
        pose_y = float(agent_obs["pose_y"])
        pose_theta = float(agent_obs["pose_theta"])

        # Frenet conversion via MPC's FrenetConverter
        s_d = self.controller.fren_conv.get_frenet(np.array([pose_x]), np.array([pose_y]))
        s = float(s_d[0])
        d = float(s_d[1])

        # Heading deviation via SplineTrack
        deriv = self.controller.spline.get_derivative(s)
        track_heading = np.arctan2(deriv[1], deriv[0])
        alpha = np.arctan2(np.sin(pose_theta - track_heading), np.cos(pose_theta - track_heading))

        # Feed actual vehicle state to MPC (instead of using MPC's own predictions)
        self.controller.speed = float(agent_obs["linear_vel_x"])
        self.controller.steering_angle_buf[:] = float(agent_obs["delta"])

        position_in_map = np.array([[pose_x, pose_y, pose_theta]])
        position_in_map_frenet = np.array([s, d, alpha])

        speed, steering = self.controller.main_loop(
            position_in_map, self.waypoint_array, position_in_map_frenet, self.last_compute_time
        )

        return np.array([[steering, speed]])

    def get_start_pose(self) -> tuple[float, float, float]:
        cl = self.track.centerline
        return float(cl.xs[0]), float(cl.ys[0]), float(cl.yaws[0])


class STMPCGymBridge:
    """Bridge between F1TENTH gym environment and the Single Track MPC controller.

    Uses the track centerline as reference path and the MPC's own FrenetConverter
    for coordinate conversion (not the gym's CubicSpline2D).
    """

    def __init__(
        self,
        track: Track,
        stmpc_config_path: str | Path,
        car_config_path: str | Path,
        tire_config_path: str | Path,
        ref_speed: float = 4.0,
    ):
        self.track = track
        stmpc_config = STMPCConfig.from_yaml(stmpc_config_path)
        car_config = CarConfig.from_yaml(car_config_path)
        tire_config = PacejkaTireConfig.from_yaml(tire_config_path)

        cl = track.centerline
        xs = cl.xs.astype(np.float64)
        ys = cl.ys.astype(np.float64)
        ks = cl.ks.astype(np.float64)
        yaws = cl.yaws.astype(np.float64)
        w_lefts = cl.w_lefts.astype(np.float64)
        w_rights = cl.w_rights.astype(np.float64)

        n = len(xs)
        vx_ref = np.full(n, ref_speed, dtype=np.float64)

        # Recompute arc-length from (xs, ys) to match FrenetConverter/SplineTrack domains
        s_ref = np.zeros(n, dtype=np.float64)
        s_ref[1:] = np.cumsum(np.hypot(np.diff(xs), np.diff(ys)))

        # Build waypoint array (Nx8): [x, y, vx, 0, s_m, kappa, yaw, 0]
        # Only columns 2 (vx) and 4 (s_m) are used at runtime by main_loop
        self.waypoint_array = np.column_stack(
            [
                xs,
                ys,
                vx_ref,
                np.zeros(n),
                s_ref,
                ks,
                yaws,
                np.zeros(n),
            ]
        )

        self.controller = Single_track_MPC_Controller(stmpc_config, car_config, tire_config)
        self.controller.mpc_initialize_solver(xs, ys, vx_ref, ks, s_ref, w_lefts, w_rights)
        self.last_compute_time = 0.0

    def get_action(self, obs: dict) -> np.ndarray:
        agent_obs = obs["agent_0"]
        pose_x = float(agent_obs["pose_x"])
        pose_y = float(agent_obs["pose_y"])
        pose_theta = float(agent_obs["pose_theta"])

        # Frenet conversion via MPC's FrenetConverter
        s_d = self.controller.fren_conv.get_frenet(np.array([pose_x]), np.array([pose_y]))
        s = float(s_d[0])
        d = float(s_d[1])

        # Heading deviation via SplineTrack
        deriv = self.controller.spline.get_derivative(s)
        track_heading = np.arctan2(deriv[1], deriv[0])
        alpha = np.arctan2(np.sin(pose_theta - track_heading), np.cos(pose_theta - track_heading))

        # Feed actual vehicle state to MPC (instead of using MPC's own predictions)
        vx = float(agent_obs["linear_vel_x"])
        self.controller.speed = vx
        delta = float(agent_obs["delta"])
        self.controller.steering_angle_buf[:] = delta

        # Open-loop startup
        v_min = self.controller.stmpc_config.v_min
        if vx < v_min:
            startup_speed = v_min + 3.0
            return np.array([[0.0, startup_speed]])

        position_in_map = np.array([[pose_x, pose_y, pose_theta]])
        position_in_map_frenet = np.array([s, d, alpha])

        # Additional dynamic states for single-track model
        linear_vel_y = float(agent_obs["linear_vel_y"])
        ang_vel_z = float(agent_obs["ang_vel_z"])
        # measured_acc = 0.0 since controller uses self.prev_acc internally (f110 car path)
        single_track_state = np.array([linear_vel_y, ang_vel_z, 0.0, delta])

        speed, steering, status = self.controller.main_loop(
            position_in_map, self.waypoint_array, position_in_map_frenet, single_track_state, self.last_compute_time
        )

        return np.array([[steering, speed]])

    def init_from_obs(self, obs: dict) -> None:
        """Sync the MPC controller's internal state with a gym observation (e.g. after reset)."""
        agent_obs = obs["agent_0"]

        # update x,y pose
        pose_x = float(agent_obs["pose_x"])
        pose_y = float(agent_obs["pose_y"])

        # update arclengths
        s_d = self.controller.fren_conv.get_frenet(np.array([pose_x]), np.array([pose_y]))
        s = float(s_d[0])

        self.controller.fre_s = s
        self.controller.previous_frenet_s = s
        self.controller.speed = float(agent_obs["linear_vel_x"])

        # update MPC target velocity to longitudinal velocity
        self.waypoint_array[:, 2] = float(agent_obs["linear_vel_x"])

    def get_start_pose(self) -> tuple[float, float, float]:
        cl = self.track.centerline
        return float(cl.xs[0]), float(cl.ys[0]), float(cl.yaws[0])
