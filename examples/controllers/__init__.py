"""Controller abstraction layer for F1TENTH analysis scripts."""

from .base import Controller
from .learned_controller import LearnedController
from .mpc.stmpc_controller import STMPCController
from .steer_controller import (
    BETA_GAIN,
    FRENET_N_GAIN,
    FRENET_N_I,
    FRENET_U_GAIN,
    FRENET_U_I,
    K_HEADING_STANLEY,
    K_SOFT_STANLEY,
    K_STANLEY,
    R_GAIN,
    TARGET_SPEED,
    PDStabilityController,
    PDSteerController,
    StanleyController,
    get_config,
)


def create_controller(
    controller_type: str,
    target_speed: float = TARGET_SPEED,
    model_path: str | None = None,
    map: str = "IMS",
) -> Controller:
    """
    Factory function to create controllers.

    Args:
        controller_type: One of "learned", "stable", "steer", "stanley"
        target_speed: Target speed for all controllers
        model_path: Path to PPO model (required for "learned")
        map: Map name for environment configuration (default: "IMS")

    Returns:
        Controller instance implementing the Controller ABC

    Raises:
        ValueError: If controller_type is invalid or required params missing

    Example:
        >>> controller = create_controller("steer", target_speed=4.0, map="IMS")
        >>> config = controller.get_env_config()
        >>> env = gym.make("f1tenth_gym:f1tenth-v0", config=config)
        >>> action = controller.get_action(obs)
    """
    match controller_type:
        case "learned":
            if model_path is None:
                raise ValueError("model_path required for learned controller")
            return LearnedController(model_path=model_path, map=map)

        case "stable":
            return PDStabilityController(target_speed=target_speed, map=map)

        case "steer":
            return PDSteerController(target_speed=target_speed, map=map)

        case "stanley":
            return StanleyController(target_speed=target_speed, map=map)

        case "stmpc":
            return STMPCController(ref_speed=target_speed, map=map)

        case _:
            raise ValueError(f"Unknown controller_type: {controller_type}")


__all__ = [
    # Protocol
    "Controller",
    # Concrete implementations
    "PDSteerController",
    "PDStabilityController",
    "StanleyController",
    "LearnedController",
    "STMPCController",
    # Factory
    "create_controller",
    # Configuration utilities
    "get_config",
    # Constants
    "FRENET_U_I",
    "FRENET_N_I",
    "FRENET_N_GAIN",
    "FRENET_U_GAIN",
    "BETA_GAIN",
    "R_GAIN",
    "K_STANLEY",
    "K_SOFT_STANLEY",
    "K_HEADING_STANLEY",
    "TARGET_SPEED",
]
