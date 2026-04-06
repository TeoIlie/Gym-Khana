import warnings
from abc import abstractmethod
from enum import Enum
from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np

from .dynamic_models import bang_bang_steer, p_accl


class LongitudinalActionEnum(Enum):
    Accl = 1
    Speed = 2

    @staticmethod
    def from_string(action: str):
        if action == "accl":
            return AcclAction
        elif action == "speed":
            return SpeedAction
        else:
            raise ValueError(f"Unknown action type {action}")

    @staticmethod
    def is_valid(action: str) -> bool:
        return action in ("accl", "speed")


class LongitudinalAction:
    def __init__(self, normalize: bool) -> None:
        self._type = None
        self.normalize = normalize
        self.lower_limit = None
        self.upper_limit = None
        self.scale_factor = 1.0

    @abstractmethod
    def act(self, longitudinal_action: Any, **kwargs) -> float:
        raise NotImplementedError("longitudinal act method not implemented")

    @property
    def type(self) -> str:
        return self._type

    @property
    def space(self) -> gym.Space:
        return gym.spaces.Box(low=self.lower_limit, high=self.upper_limit, dtype=np.float32)


class AcclAction(LongitudinalAction):
    def __init__(self, params: Dict, normalize: bool) -> None:
        super().__init__(normalize=normalize)
        self._type = "accl"

        if normalize:
            # When normalized: action space is [-1, 1]
            self.lower_limit = -1.0
            self.upper_limit = 1.0
            # Scale factor: maximum acceleration magnitude
            self.scale_factor = params["a_max"]
        else:
            # Original behavior: action space is [-a_max, a_max]
            self.lower_limit = -params["a_max"]
            self.upper_limit = params["a_max"]
            self.scale_factor = 1.0

    def act(self, action: float, state, params) -> float:
        # Apply scaling when normalized
        # When normalize=True: maps [-1, 1] → [-a_max, a_max]
        # When normalize=False: pass-through (scale_factor=1.0)
        return action * self.scale_factor


class SpeedAction(LongitudinalAction):
    def __init__(self, params: Dict, normalize: bool) -> None:
        super().__init__(normalize=normalize)
        self._type = "speed"

        if normalize:
            # When normalized: action space is [-1, 1]
            self.lower_limit = -1.0
            self.upper_limit = 1.0
            # Scale factor: compute from v_min and v_max
            # Center point: (v_max + v_min) / 2
            # Range: (v_max - v_min) / 2
            # Mapping: normalized * range + center
            self.v_center = (params["v_max"] + params["v_min"]) / 2.0
            self.v_range = (params["v_max"] - params["v_min"]) / 2.0
        else:
            # Original behavior: action space is [v_min, v_max]
            self.lower_limit = params["v_min"]
            self.upper_limit = params["v_max"]

    def act(self, action: float, state: np.ndarray, params: Dict) -> float:
        # Scale normalized action to actual speed
        # When normalize=True: maps [-1, 1] → [v_min, v_max]
        # When normalize=False: pass-through
        if self.normalize:
            desired_speed = action * self.v_range + self.v_center
        else:
            desired_speed = action

        # Apply existing P controller logic
        accl = p_accl(
            desired_speed,
            state[3],
            params["a_max"],
            params["v_max"],
            params["v_min"],
        )

        return accl


class SteerAction:
    def __init__(self, normalize: bool) -> None:
        self._type = None
        self.normalize = normalize
        self.lower_limit = None
        self.upper_limit = None
        self.scale_factor = 1.0

    @abstractmethod
    def act(self, steer_action: Any, **kwargs) -> float:
        raise NotImplementedError("steer act method not implemented")

    @property
    def type(self) -> str:
        return self._type

    @property
    def space(self) -> gym.Space:
        return gym.spaces.Box(low=self.lower_limit, high=self.upper_limit, dtype=np.float32)


class SteeringAngleAction(SteerAction):
    def __init__(self, params: Dict, normalize: bool) -> None:
        super().__init__(normalize=normalize)
        self._type = "steering_angle"

        if normalize:
            # When normalized: action space is [-1, 1]
            self.lower_limit = -1.0
            self.upper_limit = 1.0
            # Scale factor: use maximum steering angle (guaranteed symmetric: s_min = -s_max)
            self.scale_factor = params["s_max"]
        else:
            # Original behavior: action space is [s_min, s_max]
            self.lower_limit = params["s_min"]
            self.upper_limit = params["s_max"]
            self.scale_factor = 1.0

    def act(self, action: float, state: np.ndarray, params: Dict) -> float:
        # Scale normalized action to actual steering angle
        # When normalize=True: maps [-1, 1] → [s_min, s_max]
        # When normalize=False: pass-through (scale_factor=1.0)
        desired_angle = action * self.scale_factor

        # Apply existing bang-bang controller to convert angle → velocity
        sv = bang_bang_steer(
            desired_angle,
            state[2],
            params["sv_max"],
        )
        return sv


class SteeringSpeedAction(SteerAction):
    def __init__(self, params: Dict, normalize: bool) -> None:
        super().__init__(normalize=normalize)
        self._type = "steering_speed"

        if normalize:
            # When normalized: action space is [-1, 1]
            self.lower_limit = -1.0
            self.upper_limit = 1.0
            # Scale factor: guaranteed symmetric (sv_min = -sv_max)
            self.scale_factor = params["sv_max"]
        else:
            # Original behavior: action space is [sv_min, sv_max]
            self.lower_limit = params["sv_min"]
            self.upper_limit = params["sv_max"]
            self.scale_factor = 1.0

    def act(self, action: float, state: np.ndarray, params: Dict) -> float:
        # Apply scaling when normalized
        # When normalize=True: maps [-1, 1] → [sv_min, sv_max]
        # When normalize=False: pass-through (scale_factor=1.0)
        return action * self.scale_factor


class SteerActionEnum(Enum):
    Steering_Angle = 1
    Steering_Speed = 2

    @staticmethod
    def from_string(action: str):
        if action == "steering_angle":
            return SteeringAngleAction
        elif action == "steering_speed":
            return SteeringSpeedAction
        else:
            raise ValueError(f"Unknown action type {action}")

    @staticmethod
    def is_valid(action: str) -> bool:
        return action in ("steering_angle", "steering_speed")


class CarAction:
    def __init__(self, control_mode: list[str, str], params: Dict, normalize: bool) -> None:
        long_act_type_fn = None
        steer_act_type_fn = None
        if type(control_mode) == str:  # only one control mode specified
            try:
                long_act_type_fn = LongitudinalActionEnum.from_string(control_mode)
            except ValueError:
                try:
                    steer_act_type_fn = SteerActionEnum.from_string(control_mode)
                except ValueError:
                    raise ValueError(f"Unknown control mode {control_mode}")
                if control_mode == "steering_speed":
                    warnings.warn(
                        f"Only one control mode specified, using {control_mode} for steering and defaulting to acceleration for longitudinal control"
                    )
                    long_act_type_fn = LongitudinalActionEnum.from_string("accl")
                else:
                    warnings.warn(
                        f"Only one control mode specified, using {control_mode} for steering and defaulting to speed for longitudinal control"
                    )
                    long_act_type_fn = LongitudinalActionEnum.from_string("speed")

            else:
                if control_mode == "accl":
                    warnings.warn(
                        f"Only one control mode specified, using {control_mode} for longitudinal control and defaulting to steering speed for steering"
                    )
                    steer_act_type_fn = SteerActionEnum.from_string("steering_speed")
                else:
                    warnings.warn(
                        f"Only one control mode specified, using {control_mode} for longitudinal control and defaulting to steering angle for steering"
                    )
                    steer_act_type_fn = SteerActionEnum.from_string("steering_angle")

        elif type(control_mode) == list:
            if len(control_mode) != 2:
                raise ValueError(f"control_input must have exactly 2 elements, got {len(control_mode)}")
            for mode in control_mode:
                if LongitudinalActionEnum.is_valid(mode):
                    if long_act_type_fn is not None:
                        raise ValueError(f"control_input has two longitudinal types: {control_mode}")
                    long_act_type_fn = LongitudinalActionEnum.from_string(mode)
                elif SteerActionEnum.is_valid(mode):
                    if steer_act_type_fn is not None:
                        raise ValueError(f"control_input has two steering types: {control_mode}")
                    steer_act_type_fn = SteerActionEnum.from_string(mode)
                else:
                    raise ValueError(
                        f"Unknown control mode '{mode}'. Valid: 'accl', 'speed', 'steering_angle', 'steering_speed'"
                    )
            if long_act_type_fn is None:
                raise ValueError("control_input must include a longitudinal type ('accl' or 'speed')")
            if steer_act_type_fn is None:
                raise ValueError("control_input must include a steering type ('steering_angle' or 'steering_speed')")
        else:
            raise ValueError(f"Unknown control mode {control_mode}")

        # Store normalize parameter for reference
        self.normalize = normalize

        # Pass normalize parameter to action instances
        self._longitudinal_action: LongitudinalAction = long_act_type_fn(params, normalize=normalize)
        self._steer_action: SteerAction = steer_act_type_fn(params, normalize=normalize)

    @abstractmethod
    def act(self, action: Any, **kwargs) -> Tuple[float, float]:
        steer_action = self._steer_action.act(action[0], **kwargs)
        longitudinal_action = self._longitudinal_action.act(action[1], **kwargs)
        return longitudinal_action, steer_action

    @property
    def type(self) -> Tuple[str, str]:
        return (self._steer_action.type, self._longitudinal_action.type)

    @property
    def space(self) -> gym.Space:
        low = np.array([self._steer_action.lower_limit, self._longitudinal_action.lower_limit]).astype(np.float32)
        high = np.array([self._steer_action.upper_limit, self._longitudinal_action.upper_limit]).astype(np.float32)

        return gym.spaces.Box(low=low, high=high, shape=(2,), dtype=np.float32)


def from_single_to_multi_action_space(single_agent_action_space: gym.spaces.Box, num_agents: int) -> gym.spaces.Box:
    return gym.spaces.Box(
        low=single_agent_action_space.low[None].repeat(num_agents, 0),
        high=single_agent_action_space.high[None].repeat(num_agents, 0),
        shape=(num_agents, single_agent_action_space.shape[0]),
        dtype=np.float32,
    )
