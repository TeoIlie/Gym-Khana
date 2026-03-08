#! /usr/bin/env python3
import numpy as np

from ..config import CarConfig, KMPCConfig
from ..utils.frenet_converter import FrenetConverter
from ..utils.splinify import SplineTrack
from .acados_settings import acados_settings


class Kinematic_MPC_Controller:
    def __init__(self, kmpc_config: KMPCConfig, car_config: CarConfig) -> None:
        self.kmpc_config: KMPCConfig = kmpc_config
        self.car_config: CarConfig = car_config
        self.mpc_init_params()

    def mpc_init_params(self) -> None:
        self.t_MPC = 1 / self.kmpc_config.MPC_freq
        self.bound_inflation = self.kmpc_config.track_safety_margin
        self.t_delay = self.kmpc_config.t_delay + self.t_MPC

        # steering angle buffer
        buf_size = 2
        self.steering_angle_buf = np.zeros(buf_size)

        # Initial state
        self.mpc_sd = np.zeros((self.kmpc_config.N + 1, 2))
        self.u0 = np.zeros(2)
        self.fre_s = 0
        self.conp_time = 0
        self.speed = 0

    def mpc_initialize_solver(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        vx_ref: np.ndarray,
        kappa_ref: np.ndarray,
        s_ref: np.ndarray,
        d_left: np.ndarray,
        d_right: np.ndarray,
    ) -> None:
        """Initialises the MPC solver with raceline data.

        Args:
            xs: raceline x coordinates
            ys: raceline y coordinates
            vx_ref: reference velocities
            kappa_ref: reference curvatures
            s_ref: arc-length positions
            d_left: left track boundary distances from raceline
            d_right: right track boundary distances from raceline
        """
        self.fren_conv = FrenetConverter(xs, ys)

        n = len(xs)
        boundaries = np.zeros((n - 1, 2))
        waypoints = np.column_stack([xs, ys])[:-1]
        coords_path = np.array([boundaries, waypoints, boundaries])

        self.spline = SplineTrack(coords_direct=coords_path)
        self.nr_laps = 0

        self.s_ref = np.asarray(s_ref)
        self.constraint, self.model, self.acados_solver, self.model_params = acados_settings(
            self.s_ref, kappa_ref, vx_ref, d_left, d_right, self.kmpc_config, self.car_config
        )
        self.kappa = kappa_ref
        print("[MPC Controller] Solver initialized")

    def apply_warm_start(self, pose_frenet: np.ndarray):
        warm_start_trajectory = self.get_warm_start(pose_frenet=pose_frenet, const_v=1, const_steer_vel=0.0)
        for i in range(self.kmpc_config.N + 1):
            self.acados_solver.set(i, "x", warm_start_trajectory[i][: self.model.n_x])
            if i < self.kmpc_config.N:
                self.acados_solver.set(i, "u", warm_start_trajectory[i][self.model.n_x :])

    def main_loop(
        self,
        position_in_map: np.ndarray,
        waypoint_array_in_map: np.ndarray,
        position_in_map_frenet: np.ndarray,
        compute_time: float,
    ):
        """Run one MPC step.

        Args:
            position_in_map: shape (1, 3) — [x, y, theta]
            waypoint_array_in_map: shape (N, 8) — raceline data with columns [x, y, vx, d_m, s_m, kappa, psi, ax]
            position_in_map_frenet: shape (3,) — [s, d, alpha] in Frenet frame
            compute_time: solver computation time from previous step (for delay compensation)

        Returns:
            (speed, steering_angle) tuple
        """
        self.position_in_map = position_in_map
        self.waypoint_array_in_map = waypoint_array_in_map
        self.position_in_map_frenet = position_in_map_frenet
        self.conp_time = compute_time

        # update initial state from sensors
        track_length = self.spline.track_length - 0.1  # Needed as values are not exact
        center_car_s = self.position_in_map_frenet[0] + self.car_config.lr * np.cos(self.position_in_map_frenet[2])
        deriv_center = self.spline.get_derivative(center_car_s)
        alpha_center = self.position_in_map[0, 2] - np.arctan2(deriv_center[1], deriv_center[0])
        # make alpha_center between -pi and pi
        alpha_center = alpha_center % (2 * np.pi)
        if alpha_center > np.pi:
            alpha_center = alpha_center - 2 * np.pi

        if self.position_in_map_frenet[0] < 0.2 and self.fre_s // track_length != self.nr_laps:
            self.nr_laps = self.fre_s // track_length

        current_pos_s = self.position_in_map_frenet[0] + self.nr_laps * self.spline.track_length

        self.fre_s = current_pos_s
        self.fre_d = self.position_in_map_frenet[1]
        self.fre_alpha = alpha_center

        x0 = np.array(
            [self.fre_s, self.fre_d, self.fre_alpha, self.speed, self.steering_angle_buf[-1]], dtype=np.float64
        )

        # set the initial state for the mpc
        self.acados_solver.set(0, "lbx", x0)
        self.acados_solver.set(0, "ubx", x0)

        # dynamically change the target speed, weight parameters and constraints
        for i in range(self.kmpc_config.N + 1):
            idx = np.abs(self.waypoint_array_in_map[:, 4] - self.mpc_sd[i, 0] % self.spline.track_length).argmin()
            target_v_speed = self.waypoint_array_in_map[idx, 2]

            online_parameters = np.array(
                [
                    target_v_speed,
                    self.kmpc_config.qadv,
                    self.kmpc_config.qv,
                    self.kmpc_config.qn,
                    self.kmpc_config.qalpha,
                    self.kmpc_config.qac,
                    self.kmpc_config.qddelta,
                    self.kmpc_config.alat_max,
                    self.kmpc_config.track_safety_margin,
                    0.0,  # overtake_d — always 0 in time-trial mode
                ],
                dtype=np.float64,
            )

            self.acados_solver.set(i, "p", online_parameters)

            if i < self.kmpc_config.N:
                self.acados_solver.set(i, "lbu", np.array([self.kmpc_config.a_min, self.kmpc_config.ddelta_min]))
                self.acados_solver.set(i, "ubu", np.array([self.kmpc_config.a_max, self.kmpc_config.ddelta_max]))
                # do not change i+1 to i since mpc needs the initial state
                self.acados_solver.set(i + 1, "lbx", np.array([self.kmpc_config.v_min, self.kmpc_config.delta_min]))
                self.acados_solver.set(i + 1, "ubx", np.array([self.kmpc_config.v_max, self.kmpc_config.delta_max]))

        # Solve OCP
        status = self.acados_solver.solve()
        if status != 0:
            print("Solver failed, applying warm start")
            self.apply_warm_start(pose_frenet=[self.fre_s, self.fre_d, self.fre_alpha])

        # get solution with time delay compensation
        # TODO: make time_delay_step a config parameter
        time_delay_step = 3
        self.u0 = self.acados_solver.get(0, "u")
        self.pred_x = self.acados_solver.get(time_delay_step, "x")
        self.steering_angle = self.pred_x[4]
        self.speed = self.pred_x[3]

        # steering buffer — shift right and add predicted steering angle
        self.steering_angle_buf[1:] = self.steering_angle_buf[:-1]
        self.steering_angle_buf[0] = self.pred_x[-1]

        # update predicted trajectory for next step's reference speed lookup
        self.mpc_sd = np.array([self.acados_solver.get(j, "x")[:2] for j in range(self.kmpc_config.N + 1)])

        return self.speed, self.steering_angle

    #############
    # Utilities #
    #############
    def get_warm_start(self, pose_frenet: np.ndarray, const_v: float, const_steer_vel: float) -> np.array:
        warm_start = np.zeros((self.kmpc_config.N + 1, 7))
        warm_start[0] = np.array([pose_frenet[0], pose_frenet[1], pose_frenet[2], 1, const_steer_vel, 0, 0])
        for i in range(1, self.kmpc_config.N + 1):
            der_state = self._dynamics_of_car(0, warm_start[i - 1])
            warm_start[i] = warm_start[i - 1] + np.array(der_state) / self.kmpc_config.MPC_freq
        return warm_start

    def _dynamics_of_car(self, t, x0) -> list:
        s, n, alpha, v, delta, derv, derDelta = x0[0], x0[1], x0[2], x0[3], x0[4], x0[5], x0[6]
        xdot = self.model.f_expl_func(s, n, alpha, v, delta, derv, derDelta, self.model_params.p)
        return [float(xdot[0]), float(xdot[1]), float(xdot[2]), float(xdot[3]), float(xdot[4]), derv, derDelta]
