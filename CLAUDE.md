# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## About F1TENTH

This is the F1TENTH Gym environment - a high-fidelity simulation platform for 1/10th scale autonomous racing cars. F1TENTH is an international autonomous racing competition and research platform that provides a standardized testbed for autonomous vehicle algorithms at reduced scale and cost.

### Autonomous Racing Context
The simulator models realistic vehicle dynamics, sensor data (LiDAR), and racing scenarios to enable development and testing of:
- **Path Planning**: Algorithms like Pure Pursuit (implemented in examples)
- **Control Systems**: Speed and steering control for high-speed racing
- **Perception**: LiDAR-based environment understanding
- **SLAM**: Simultaneous Localization and Mapping
- **Reinforcement Learning**: Training autonomous racing agents with PPO and other algorithms

### Research Applications
- **Multi-agent Racing**: Support for multiple competing vehicles
- **Safety Systems**: Collision detection and avoidance
- **Vehicle Dynamics**: Multiple models including kinematic, single-track, multi-body, and drift models
- **Sensor Simulation**: Accurate LiDAR ray-casting for perception research
- **Drifting**: STD (Single Track Drift) model with PAC2002 tire model for aggressive driving

## Development Commands

### Installation
```bash
# With pip (recommended for this project)
pip install -e .

# Or with poetry
poetry install
source $(poetry env info -p)/bin/activate
```

### Virtual Environment
This project uses a virtual environment at `rl_env/`. Always activate it before running commands:
```bash
source rl_env/bin/activate
```

### Testing
Run all tests:
```bash
python3 -m pytest
```

Run specific test file:
```bash
python3 -m pytest tests/test_f110_env.py
```

Run specific test function:
```bash
python3 -m pytest tests/test_f110_env.py::test_function_name -v
```

The CI runs pytest for Python versions 3.10-3.12.

### Running Examples
```bash
cd examples
python3 waypoint_follow.py  # Pure Pursuit path following
python3 p_steer_controller.py  # Simple P controller for centerline following
python3 drift_debug.py  # Debug drift behavior with visualization
```

### Training Models
Main training script with multiple modes:
```bash
# Train a new model
python train/ppo_race.py --m t

# Evaluate a local model (uses latest wandb run if --path not specified)
python train/ppo_race.py --m e
python train/ppo_race.py --m e --path /path/to/model.zip

# Download model from wandb and evaluate
python train/ppo_race.py --m d --run_id <wandb_run_id>

# Continue training from existing model
python train/ppo_race.py --m c --path /path/to/model.zip --additional_timesteps 10000000
```

### Formatting/Linting
Manual formatting:
```bash
black .
```

Linting runs automatically via VSCode settings (`.vscode/settings.json`) on save with:
- `black` for formatting
- `isort` for import sorting
- `flake8` for linting (max line length 120, ignoring E203 and E501)

Configuration:
- Line length: 120 characters
- Target: Python 3.10+
- Profile: black (for isort)

### Docker Support
```bash
docker build -t f1tenth_gym_container -f Dockerfile .
docker run --gpus all -it -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix f1tenth_gym_container
```

## Architecture Overview

### Core Autonomous Vehicle Components

**F110Env** (`f1tenth_gym/envs/f110_env.py`): Main gymnasium environment implementing F1/10th vehicle simulation with realistic dynamics. This is the entry point for creating environments via `gym.make('f1tenth_gym:f1tenth-v0', config={...})`.

**Dynamic Models** (`f1tenth_gym/envs/dynamic_models/`): Four vehicle dynamics models:
- **KS (Kinematic Single Track)**: `kinematic.py` - Simplest model, pure kinematics without tire forces
- **ST (Single Track)**: `single_track.py` - Single-track dynamics model without explicit tire model
- **MB (Multi-Body)**: `multi_body/` - Most detailed model with full tire modeling, may be overkill for RL
- **STD (Single Track Drift)**: `single_track_drift/` - Single-track dynamics + Pacejka PAC2002 tire model for drift simulation (recommended for RL drift training)

**Observation System** (`f1tenth_gym/envs/observation.py`): Factory pattern for different observation types including "rl" and "drift" observations. Supports lookahead curvature sampling and normalization.

**Action System** (`f1tenth_gym/envs/action.py`): Handles different control input types like `["speed", "steering_angle"]` or `["accl", "steering_angle"]`.

**Laser Models** (`f1tenth_gym/envs/laser_models.py`): LiDAR simulation with ray-casting for autonomous perception. Includes TTC (Time-To-Collision) calculations.

**Collision Models** (`f1tenth_gym/envs/collision_models.py`): Safety-critical collision detection systems with support for predictive TTC-based collision checking or Frenet-based collision checking.

**Base Classes** (`f1tenth_gym/envs/base_classes.py`): Core simulation infrastructure:
- `RaceCar`: Handles physics and laser scan of single vehicle
- `Simulator`: Multi-agent simulation orchestrator
- Step method at line 503 defines core simulation loop

**Track System** (`f1tenth_gym/envs/track/`): Racing circuit definitions with centerline and raceline support. Includes cubic spline interpolation for smooth trajectories.

**Reset System** (`f1tenth_gym/envs/reset/`): Different reset strategies for RL training including random static resets and map-based resets.

**Rendering** (`f1tenth_gym/envs/rendering/`): Visualization support using pygame and PyQt with debug visualization options.

### RL Training Infrastructure

**Configuration** (`train/config/`):
- `env_config.py`: Centralized Gym environment and RL parameter configuration
- `rl_config.yaml`: RL hyperparameters (n_envs, learning rate, batch size, etc.)
- `gym_config.yaml`: Gym environment parameters (map, model, timestep, etc.)
- Key functions:
  - `get_drift_test_config()`: Configuration for testing drift-trained models
  - `get_drift_train_config()`: Configuration for training drift models

**Training Scripts** (`train/`):
- `ppo_race.py`: Main training script with train/eval/download/continue modes (see Training Models commands above)
- `ppo_example.py`: Simpler example PPO training script
- `training_utils.py`: Shared utilities for output directory management, environment creation, and callbacks

### Package Structure
- **f1tenth_gym/**: Main package following gymnasium RL environment interface
- **examples/**: Reference implementations of autonomous racing algorithms
- **train/**: RL training scripts and configuration
- **tests/**: Comprehensive test suite including model validation
- **maps/**: Racing track definitions (real-world inspired circuits)
- **plan/**: Development planning documents (drift, reward, normalization plans)

### Key Dependencies for Autonomous Systems
- **gymnasium**: RL environment interface for agent training
- **stable-baselines3**: PPO and other RL algorithms
- **wandb**: Experiment tracking and model logging
- **numba**: JIT compilation for real-time performance in control loops
- **pygame/PyQt6**: Real-time visualization for algorithm development and debugging
- **numpy/scipy**: Mathematical foundations for control and estimation algorithms

## Important Configuration Options

### Vehicle Models and Control
```python
env = gym.make('f1tenth_gym:f1tenth-v0', config={
    'model': 'std',  # Use 'std' for drifting with PAC2002 tire model
    'control_input': ['accl', 'steering_angle'],  # Best for RL drift training
    'params': F110Env.f1tenth_std_vehicle_params(),  # Drift parameters for 1/10 scale
})
```

### Observation Configuration
```python
config = {
    'observation_config': {'type': 'drift'},  # or 'rl' for standard RL observations
    'lookahead_n_points': 10,  # Number of lookahead points (default: 10)
    'lookahead_ds': 0.3,  # Spacing between points in meters (default: 0.3m)
    'normalize_obs': True,  # Enable observation normalization (only for 'drift' obs)
    'normalize_act': True,  # Enable action normalization
}
```

### Collision and Safety
```python
config = {
    'predictive_collision': True,  # True for TTC collision checking, False for Frenet-based
    'wall_deflection': False,  # False treats edges as boundaries, True as walls causing collision
}
```

### Debugging and Visualization
```python
config = {
    'render_mode': 'human',  # Enable visualization
    'render_track_lines': True,  # Render centerline (green) and raceline (red)
    'render_lookahead_curvatures': True,  # Visualize lookahead curvature points (yellow)
    'debug_frenet_projection': True,  # Visualize Frenet coordinate accuracy
    'record_obs_min_max': True,  # Record min/max observations for normalization tuning
}
```

Debug with breakpoints by looping through environment steps (see `tests/drift_debug.py`).

## Branches and Fork History

- Original `f1tenth_gym` branch `main` → renamed to `f1tenth_main_original` in this fork
- Original `f1tenth_gym` branch `rl_example` → renamed to `main` in this fork (active development branch)
- This fork actively extends the RL capabilities of the original project

## Tire Parameters

Parameters for the 1/10 scale F1TENTH car with STD model are defined in `f1tenth_gym/envs/f110_env.py::f1tenth_std_vehicle_params()`. These mix existing F1TENTH params with tire parameters adjusted from fullscale car.

Test script `tests/model_validation/test_f1tenth_std_params.py` creates comparison figures and parameter YAML dumps in `figures/tire_params/` to maintain parameter history.

## Wandb Integration

Models and training runs are logged to: https://wandb.ai/teo-altum-quinque-queen-s-university/projects

## Known Platform Issues

- **Windows**: Must use Python 3.8 due to compilation dependencies
- **macOS Big Sur+**: May need `pip3 install pyglet==1.5.11` for OpenGL framework compatibility (ignore gym version warning)
