"""
Test wall_deflection configuration for different observation types.
"""

import gymnasium as gym
import numpy as np
import pytest

from f1tenth_gym.envs.f110_env import F110Env


def test_wall_deflection_default_drift():
    """Test that wall_deflection=False by default for 'drift' observation type."""
    env = gym.make(
        "f1tenth_gym:f1tenth-v0",
        config={
            "map": "Spielberg",
            "num_agents": 1,
            "model": "std",
            "observation_config": {"type": "drift"},
            "params": F110Env.f1tenth_std_vehicle_params(),
        },
    )
    assert env.unwrapped.wall_deflection is False
    env.close()


def test_wall_deflection_default_other():
    """Test that wall_deflection=True by default for non-drift observation types."""
    env = gym.make(
        "f1tenth_gym:f1tenth-v0",
        config={
            "map": "Spielberg",
            "num_agents": 1,
            "observation_config": {"type": "original"},
            "params": F110Env.f1tenth_std_vehicle_params(),
        },
    )
    assert env.unwrapped.wall_deflection is True
    env.close()


def test_wall_deflection_default_kinematic_state():
    """Test that wall_deflection=True by default for 'kinematic_state' observation type."""
    env = gym.make(
        "f1tenth_gym:f1tenth-v0",
        config={
            "map": "Spielberg",
            "num_agents": 1,
            "observation_config": {"type": "kinematic_state"},
            "params": F110Env.f1tenth_std_vehicle_params(),
        },
    )
    assert env.unwrapped.wall_deflection is True
    env.close()


def test_wall_deflection_override_true_with_drift():
    """Test that wall_deflection=True with drift type sets to True with warning."""
    with pytest.warns(UserWarning, match="wall_deflection=False is recommended"):
        env = gym.make(
            "f1tenth_gym:f1tenth-v0",
            config={
                "map": "Spielberg",
                "num_agents": 1,
                "model": "std",
                "observation_config": {"type": "drift"},
                "params": F110Env.f1tenth_std_vehicle_params(),
                "wall_deflection": True,
            },
        )
        assert env.unwrapped.wall_deflection is True
        env.close()


def test_wall_deflection_override_false_with_non_drift():
    """Test that wall_deflection=False with non-drift type keeps False without warning."""
    # Should not raise a warning
    env = gym.make(
        "f1tenth_gym:f1tenth-v0",
        config={
            "map": "Spielberg",
            "num_agents": 1,
            "observation_config": {"type": "original"},
            "params": F110Env.f1tenth_std_vehicle_params(),
            "wall_deflection": False,
        },
    )
    assert env.unwrapped.wall_deflection is False
    env.close()


def test_wall_deflection_explicit_true_with_non_drift():
    """Test that wall_deflection=True with non-drift type works correctly."""
    env = gym.make(
        "f1tenth_gym:f1tenth-v0",
        config={
            "map": "Spielberg",
            "num_agents": 1,
            "observation_config": {"type": "kinematic_state"},
            "params": F110Env.f1tenth_std_vehicle_params(),
            "wall_deflection": True,
        },
    )
    assert env.unwrapped.wall_deflection is True
    env.close()


def test_wall_deflection_explicit_false_with_drift():
    """Test that wall_deflection=False with drift type works correctly."""
    env = gym.make(
        "f1tenth_gym:f1tenth-v0",
        config={
            "map": "Spielberg",
            "num_agents": 1,
            "model": "std",
            "observation_config": {"type": "drift"},
            "params": F110Env.f1tenth_std_vehicle_params(),
            "wall_deflection": False,
        },
    )
    assert env.unwrapped.wall_deflection is False
    env.close()


def test_wall_deflection_propagates_to_simulator():
    """Test that wall_deflection config propagates to Simulator."""
    env = gym.make(
        "f1tenth_gym:f1tenth-v0",
        config={
            "map": "Spielberg",
            "num_agents": 2,
            "observation_config": {"type": "original"},
            "params": F110Env.f1tenth_std_vehicle_params(),
            "wall_deflection": False,
        },
    )
    assert env.unwrapped.wall_deflection is False
    assert env.unwrapped.sim.wall_deflection is False
    env.close()


def test_wall_deflection_propagates_to_racecars():
    """Test that wall_deflection config propagates to all RaceCar instances."""
    env = gym.make(
        "f1tenth_gym:f1tenth-v0",
        config={
            "map": "Spielberg",
            "num_agents": 3,
            "observation_config": {"type": "original"},
            "params": F110Env.f1tenth_std_vehicle_params(),
            "wall_deflection": True,
        },
    )
    assert env.unwrapped.wall_deflection is True
    assert env.unwrapped.sim.wall_deflection is True

    # Check all agents have the correct wall_deflection setting
    for agent in env.unwrapped.sim.agents:
        assert agent.wall_deflection is True

    env.close()


def test_wall_deflection_multi_agent_consistency():
    """Test that all agents in multi-agent setup have consistent wall_deflection."""
    env = gym.make(
        "f1tenth_gym:f1tenth-v0",
        config={
            "map": "Spielberg",
            "num_agents": 5,
            "observation_config": {"type": "original"},
            "params": F110Env.f1tenth_std_vehicle_params(),
            "wall_deflection": False,
        },
    )

    # All agents should have wall_deflection=False
    for agent in env.unwrapped.sim.agents:
        assert agent.wall_deflection is False

    env.close()


def test_wall_deflection_with_none_obs_type():
    """Test that wall_deflection works with None observation type (defaults to True)."""
    env = gym.make(
        "f1tenth_gym:f1tenth-v0",
        config={
            "map": "Spielberg",
            "num_agents": 1,
            "observation_config": {"type": None},
            "params": F110Env.f1tenth_std_vehicle_params(),
        },
    )
    # When obs_type is None, default should be True (obs_type != "drift")
    assert env.unwrapped.wall_deflection is True
    env.close()
