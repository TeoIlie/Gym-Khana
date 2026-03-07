#! /usr/bin/env python3
import numpy as np

from ..config import CarConfig, PacejkaTireConfig, STMPCConfig
from ..utils.frenet_converter import FrenetConverter
from ..utils.splinify import SplineTrack
from .acados_settings import acados_settings
from .indicies import StateIndex


class Single_track_MPC_Controller:
    def __init__(self, stmpc_config: STMPCConfig, car_config: CarConfig, tire_config: PacejkaTireConfig) -> None:
        self.stmpc_config: STMPCConfig = stmpc_config
        self.car_config: CarConfig = car_config
        self.tire_config: PacejkaTireConfig = tire_config
        self.mpc_init_params()

    def mpc_init_params(self) -> None:
        self.t_MPC = 1 / self.stmpc_config.MPC_freq

        print(
            f"[STMPC Controller] Steps Delay set to {self.stmpc_config.steps_delay}. "
            f"Equivalent to a delay of {1000 * self.stmpc_config.steps_delay * self.t_MPC:3.2f} milliseconds."
        )

        self.t_delay = self.stmpc_config.t_delay + self.t_MPC

        # steering angle buffer
        buf_size = 2
        self.steering_angle_buf = np.zeros(buf_size)

        # Initial state
        self.mpc_sd = np.zeros((self.stmpc_config.N + 1, 2))
        self.u0 = np.zeros(2)
        self.fre_s = 0
        self.previous_frenet_s = 0
        self.comp_time = 0
        self.prev_acc = 0
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
            vx_ref: reference velocities (unused by STMPC, kept for interface consistency)
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
            self.s_ref, kappa_ref, d_left, d_right, self.stmpc_config, self.car_config, self.tire_config
        )
        self.kappa = kappa_ref
        self.prev_acc = 0
        print("[STMPC Controller] Solver initialized")

    def apply_warm_start(self, pose_frenet: np.ndarray):
        warm_start_trajectory = self.get_warm_start(pose_frenet=pose_frenet, const_acc=1, const_steer_vel=0.0)
        for i in range(self.stmpc_config.N + 1):
            self.acados_solver.set(i, "x", warm_start_trajectory[i][: self.model.n_x])
            if i < self.stmpc_config.N:
                self.acados_solver.set(i, "u", warm_start_trajectory[i][self.model.n_x :])

    def main_loop(
        self,
        position_in_map: np.ndarray,
        waypoint_array_in_map: np.ndarray,
        position_in_map_frenet: np.ndarray,
        single_track_state: np.ndarray,
        compute_time: float,
    ):
        """Run one STMPC step.

        Args:
            position_in_map: shape (1, 3) — [x, y, theta]
            waypoint_array_in_map: shape (N, 8) — raceline data with columns [x, y, vx, d_m, s_m, kappa, psi, ax]
            position_in_map_frenet: shape (3,) — [s, d, alpha] in Frenet frame
            single_track_state: [v_y, yaw_rate, measured_acc, measured_steer]
            compute_time: solver computation time from previous step

        Returns:
            (speed, steering_angle, status) tuple
        """
        self.position_in_map = position_in_map
        self.waypoint_array_in_map = waypoint_array_in_map
        self.position_in_map_frenet = position_in_map_frenet
        self.vel_y = single_track_state[0]
        self.yaw_rate = single_track_state[1]
        self.measured_steer = single_track_state[3]
        self.measured_acc = self.prev_acc  # internal estimate (f110 car path, no accelerometer)
        self.comp_time = compute_time

        # update initial state from sensors
        center_car_s = self.position_in_map_frenet[0] + self.car_config.lr * np.cos(self.position_in_map_frenet[2])
        deriv_center = self.spline.get_derivative(center_car_s)
        alpha_center = self.position_in_map[0, 2] - np.arctan2(deriv_center[1], deriv_center[0])
        # make alpha_center between -pi and pi
        alpha_center = alpha_center % (2 * np.pi)
        if alpha_center > np.pi:
            alpha_center = alpha_center - 2 * np.pi

        # lap counting
        if self.position_in_map_frenet[0] < self.previous_frenet_s - 1:
            self.nr_laps += 1
        self.previous_frenet_s = self.position_in_map_frenet[0]

        current_pos_s = self.position_in_map_frenet[0] + self.nr_laps * self.spline.track_length

        self.fre_s = current_pos_s
        self.fre_d = self.position_in_map_frenet[1]
        self.fre_alpha = alpha_center

        x0 = np.array(
            [
                self.fre_s,
                self.fre_d,
                self.fre_alpha,
                self.speed,
                self.vel_y,
                self.measured_steer,
                self.yaw_rate,
                self.measured_acc,
            ],
            dtype=np.float64,
        )

        # set the initial state for the mpc
        self.acados_solver.set(0, "lbx", x0)
        self.acados_solver.set(0, "ubx", x0)

        # dynamically change the target speed, weight parameters and constraints
        for i in range(self.stmpc_config.N + 1):
            idx = np.abs(self.waypoint_array_in_map[:, 4] - self.mpc_sd[i, 0] % self.spline.track_length).argmin()
            target_v_speed = self.waypoint_array_in_map[idx, 2]

            if i < self.stmpc_config.steps_delay:
                # increase qddelta weight to effectively freeze steering during delay compensation
                multiplier = 1e6
                online_parameters = np.array(
                    [
                        target_v_speed,
                        self.stmpc_config.qadv,
                        self.stmpc_config.qv,
                        self.stmpc_config.qn,
                        self.stmpc_config.qalpha,
                        self.stmpc_config.qjerk,
                        multiplier * self.stmpc_config.qddelta,
                        self.stmpc_config.alat_max,
                        self.stmpc_config.a_min,
                        self.stmpc_config.a_max,
                        self.stmpc_config.track_safety_margin,
                        0.0,  # overtake_d — always 0 in time-trial mode
                    ],
                    dtype=np.float64,
                )
            else:
                online_parameters = np.array(
                    [
                        target_v_speed,
                        self.stmpc_config.qadv,
                        self.stmpc_config.qv,
                        self.stmpc_config.qn,
                        self.stmpc_config.qalpha,
                        self.stmpc_config.qjerk,
                        self.stmpc_config.qddelta,
                        self.stmpc_config.alat_max,
                        self.stmpc_config.a_min,
                        self.stmpc_config.a_max,
                        self.stmpc_config.track_safety_margin,
                        0.0,  # overtake_d — always 0 in time-trial mode
                    ],
                    dtype=np.float64,
                )

            self.acados_solver.set(i, "p", online_parameters)

            if i < self.stmpc_config.N:
                self.acados_solver.set(i, "lbu", np.array([self.stmpc_config.jerk_min, self.stmpc_config.ddelta_min]))
                self.acados_solver.set(i, "ubu", np.array([self.stmpc_config.jerk_max, self.stmpc_config.ddelta_max]))
                # do not change i+1 to i since mpc needs the initial state
                self.acados_solver.set(
                    i + 1,
                    "lbx",
                    np.array([self.stmpc_config.v_min, self.stmpc_config.delta_min, self.stmpc_config.a_min]),
                )
                self.acados_solver.set(
                    i + 1,
                    "ubx",
                    np.array([self.stmpc_config.v_max, self.stmpc_config.delta_max, self.stmpc_config.a_max]),
                )

        # Solve OCP
        status = self.acados_solver.solve()
        if status != 0:
            print("STMPC solver failed, applying warm start")
            self.apply_warm_start(pose_frenet=[self.fre_s, self.fre_d, self.fre_alpha])

        # get solution
        self.u0 = self.acados_solver.get(0, "u")
        self.prev_acc = self.acados_solver.get(0, "x")[StateIndex.ACCEL.value]
        delayed_index = self.stmpc_config.steps_delay + 1
        self.pred_x = self.acados_solver.get(delayed_index, "x")
        self.steering_angle = self.pred_x[StateIndex.STEERING_ANGLE_DELTA.value]
        self.speed = self.pred_x[StateIndex.VELOCITY_V_X.value]
        # Euler integration of jerk for acceleration estimate
        self.prev_acc = self.prev_acc + self.u0[0] / self.stmpc_config.MPC_freq

        # steering buffer — shift right and add predicted steering angle
        self.steering_angle_buf[1:] = self.steering_angle_buf[:-1]
        self.steering_angle_buf[0] = self.pred_x[StateIndex.STEERING_ANGLE_DELTA.value]

        # update predicted trajectory for next step's reference speed lookup
        self.mpc_sd = np.array([self.acados_solver.get(j, "x")[:2] for j in range(self.stmpc_config.N + 1)])

        if status == 0:
            return self.speed, self.steering_angle, status
        else:
            return 0, self.measured_steer, status

    #############
    # Utilities #
    #############
    def get_warm_start(self, pose_frenet: np.ndarray, const_acc: float, const_steer_vel: float) -> np.array:
        warm_start = np.zeros((self.stmpc_config.N + 1, 10))  # 8 states + 2 inputs
        warm_start[0] = np.array(
            [pose_frenet[0], pose_frenet[1], pose_frenet[2], 1, 0, 0, 0, const_acc, 0, const_steer_vel]
        )
        for i in range(1, self.stmpc_config.N + 1):
            xdot = self._dynamics_of_car(0, warm_start[i - 1])
            warm_start[i] = warm_start[i - 1] + np.array(xdot) / self.stmpc_config.MPC_freq
        return warm_start

    def _dynamics_of_car(self, t, x0) -> list:
        """Forward propagation dynamics. Uses the CasADi model from acados."""
        s = x0[0]
        n = x0[1]
        theta = x0[2]
        v_x = x0[3]
        v_y = x0[4]
        delta = x0[5]
        yaw_rate = x0[6]
        accel = x0[7]
        jerk = x0[8]
        derDelta = x0[9]
        xdot = self.model.f_expl_func(
            s, n, theta, v_x, v_y, delta, yaw_rate, accel, jerk, derDelta, self.model_params.p
        )
        return [
            float(xdot[0]),
            float(xdot[1]),
            float(xdot[2]),
            float(xdot[3]),
            float(xdot[4]),
            float(xdot[5]),
            float(xdot[6]),
            float(xdot[7]),
            jerk,
            derDelta,
        ]
