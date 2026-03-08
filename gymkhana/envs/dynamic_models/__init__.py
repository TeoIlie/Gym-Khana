"""
This module contains the dynamic models available in the F1Tenth Gym.
Each submodule contains a single model, and the equations or their source is documented alongside it. Many of the models are from the CommonRoad repository, available here: https://gitlab.lrz.de/tum-cps/commonroad-vehicle-models/
"""

import warnings
from enum import Enum
from typing import Optional

import numpy as np

from .kinematic import get_standardized_state_ks, vehicle_dynamics_ks
from .multi_body import get_standardized_state_mb, init_mb, vehicle_dynamics_mb
from .single_track import get_standardized_state_st, vehicle_dynamics_st
from .single_track_drift import get_standardized_state_std, init_std, vehicle_dynamics_std
from .utils import bang_bang_steer, p_accl


class DynamicModel(Enum):
    KS = 1  # Kinematic Single Track
    ST = 2  # Single Track
    MB = 3  # Multi-body Model
    STD = 4  # Single Track Drift

    @staticmethod
    def from_string(model: str):
        if model == "ks":
            warnings.warn("Chosen model is KS. This is different from previous versions of the gym.")
            return DynamicModel.KS
        elif model == "st":
            return DynamicModel.ST
        elif model == "mb":
            return DynamicModel.MB
        elif model == "std":
            return DynamicModel.STD
        else:
            raise ValueError(f"Unknown model type {model}")

    def get_initial_state(self, pose=None, state=None, params: Optional[dict] = None):
        """
        Get the initial state for the vehicle model.

        Args:
            pose: (3,) array [x, y, yaw] - legacy pose-only initialization
            state: (7,) array for STD model [x, y, delta, v, yaw, yaw_rate, slip_angle]
                   omega_f and omega_r are calculated automatically
            params: Vehicle parameters dictionary (required for MB and STD models)

        Returns:
            Full state vector for the model

        Note:
            Cannot specify both pose and state - use one or the other.
        """
        # Validation: pose and state are mutually exclusive
        if pose is not None and state is not None:
            raise ValueError("Cannot provide both 'pose' and 'state'. Use one or the other.")

        # Assert that if self is MB or STD, params is not None
        if (self == DynamicModel.MB or self == DynamicModel.STD) and params is None:
            raise ValueError("MultiBody and SingleTrackDrift models require parameters to be provided.")

        # initialize zero state
        if self == DynamicModel.KS:
            # state is [x, y, steer_angle, vel, yaw_angle]
            init_state = np.zeros(5)
        elif self == DynamicModel.ST:
            # state is [x, y, steer_angle, vel, yaw_angle, yaw_rate, slip_angle]
            init_state = np.zeros(7)
        elif self == DynamicModel.MB:
            # state is a 29D vector
            init_state = np.zeros(29)
        elif self == DynamicModel.STD:
            # state is [x, y, steer_angle, vel, yaw_angle, yaw_rate, slip_angle, omega_f, omega_r]
            init_state = np.zeros(9)
        else:
            raise ValueError(f"Unknown model type {self}")

        # If full state provided, copy first 7 elements (STD model only)
        if state is not None:
            if self != DynamicModel.STD:
                raise ValueError(f"Full state initialization only supported for STD model, not {self}")
            if len(state) != 7:
                raise ValueError(f"STD model requires 7-element state, got {len(state)}")
            init_state[:7] = state
        # set initial pose if provided
        elif pose is not None:
            init_state[0:2] = pose[0:2]
            init_state[4] = pose[2]

        # If state is MultiBody, we must inflate the state to 29D
        if self == DynamicModel.MB:
            init_state = init_mb(init_state, params)
        # If state is SingleTrackDrift, we must inflate to 9D (calculates omega_f, omega_r)
        elif self == DynamicModel.STD:
            init_state = init_std(init_state, params)
        return init_state

    @property
    def f_dynamics(self):
        if self == DynamicModel.KS:
            return vehicle_dynamics_ks
        elif self == DynamicModel.ST:
            return vehicle_dynamics_st
        elif self == DynamicModel.MB:
            return vehicle_dynamics_mb
        elif self == DynamicModel.STD:
            return vehicle_dynamics_std
        else:
            raise ValueError(f"Unknown model type {self}")

    def get_standardized_state_fn(self):
        """
        This function returns the standardized state information for the model.
        This needs to be a function, because the state information is different for each model.
        Slip is not directly available from the MB model.
        """
        if self == DynamicModel.KS:
            return get_standardized_state_ks
        elif self == DynamicModel.ST:
            return get_standardized_state_st
        elif self == DynamicModel.MB:
            return get_standardized_state_mb
        elif self == DynamicModel.STD:
            return get_standardized_state_std
        else:
            raise ValueError(f"Unknown model type {self}")
