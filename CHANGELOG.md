# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- **Control debug panel**: real-time steering/throttle visualization (PyQt6 renderer)
- **Observation debug overlay**: live observation values overlay on the map (PyQt6 renderer)
- **ONNX export**: convert SB3 models to ONNX for sim-to-real transfer via `OnnxPolicyRunner`
- **Norm bounds extraction**: save normalization bounds to config for reuse in external packages
- **Custom neural network architecture**: configurable layer sizes via `net_arch` in RL config
- **Morris sensitivity analysis**: parameter sensitivity analysis script for STD model
- **Observation sensitivity analysis**: script for analyzing observation feature importance
- **Regression tests**: tests for action ordering and other previously encountered bugs
- **Trajectory comparison script**: compare KS, ST, STD model trajectories for sim2real validation

### Fixed
- Action ordering bug where `control_input` order affected action array mapping
- Rendering `rgb_array` mode bugfix
- Kinematic model yaw_rate bugfix
- Removed unused `abstractmethod` decorator on `CarAction.act` method

### Changed
- Refactored docstrings for consistency, correctness, and RTD readability
- Clarified supported Python versions in README
- Reorganized README for clearer table of contents
- Made unit test suite more efficient
- Moved commonroad submodule to analysis subfolder

### Removed
- Docker support (Poetry and PyPI are sufficient)
- Euclidean reward option for recovery learning

## [1.1.1] - 2026-03-15

### Fixed
- Version bump only (packaging fix)

## [1.1.0] - 2026-03-15

### Added
- **Recovery training**: dedicated training mode (`training_mode: "recover"`) with recovery-specific rewards, resets, and evaluation
- **Curriculum learning**: `CurriculumLearningCallback` that gradually expands recovery state initialization ranges as agent success rate improves
- **Transfer learning**: transfer pretrained model weights to new tasks with optional critic reset and `log_std` re-initialization
- **MPC controllers**: Kinematic MPC (KMPC) and Single-Track MPC (STMPC) via acados integration
- **Multi-map training**: train across multiple maps in parallel environments
- **Sparse width observations**: `sparse_width_obs` config to reduce observation dimensionality when track width varies little
- **P/PD controllers**: simple proportional controllers for benchmarking recovery performance
- **Performance metrics**: beta-r phase plane analysis, recovery trajectory plotting, controller comparison metrics
- PyPI publishing via `publish.yml` GitHub Actions workflow (triggered on tag push)
- ReadTheDocs documentation site with Sphinx RTD theme
- Logos, favicon, and branding assets

### Changed
- Refactored package for PyPI publishing (renamed `f1tenth_gym` to `gymkhana`)
- Migrated from Black to Ruff for linting, formatting, and import sorting with pre-commit hooks
- Refactored controllers with abstract base class
- Refactored training scripts with shared `train_common.py` and `train_utils.py`
- Updated map loading to use maintained source for better compatibility
- Reorganized project structure (analysis, figures, plans into dedicated directories)

### Fixed
- Mexico City track naming issue
- Eval env should use curriculum max ranges

## [1.0.0] - 2026-01-30

### Added
- **Drift dynamics**: Single Track Drift (STD) model with PAC2002 tire physics
- **Reverse/random driving direction**: `track_direction` config option (`normal`, `reverse`, `random`) for balanced cornering training
- **State reset**: `env.reset(options={"states": ...})` for full 7-d state initialization (STD model)
- **Arc-length visualization**: render arc-length annotations along the centerline for Frenet coordinate debugging
- **Observation types**: `FeaturesObservation` (drift) with slip angle, lookahead curvatures/widths; observation and action normalization
- **Predictive collision**: TTC-based collision checking as alternative to Frenet-based
- **Wall deflection mode**: configurable track boundary behavior (boundary vs. wall collision)
- **Progress reward gain**: configurable `progress_gain` multiplier for forward progress reward
- **Frenet projection debugging**: `debug_frenet_projection` visualization option
- **Lookahead curvature visualization**: render lookahead sampling points ahead of the vehicle
- **Wandb integration**: experiment tracking, model logging, download, and resume from wandb
- **Performance metrics**: beta-r phase plane analysis, recovery trajectory plotting
- PPO training infrastructure with SB3 and parallel environments

### Fixed
- Wraparound bug in progress tracking (robust fix with unit tests)
- Normalization logic when both actions and observations are normalized
- Reverse direction reset bug where random reset could point vehicle the wrong way due to cached reference line
- Reverse direction normalization fix where min/max curvature bounds need to be symmetric

### Changed
- Configuration logic simplified and centralized in `train/config/`
- Moved plotting files to `analysis/` folder

[Unreleased]: https://github.com/TeoIlie/Gym-Khana/compare/v1.1.1...HEAD
[1.1.1]: https://github.com/TeoIlie/Gym-Khana/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/TeoIlie/Gym-Khana/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/TeoIlie/Gym-Khana/releases/tag/v1.0.0
