# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Fixed
- Kinematic model yaw_rate bugfix
- Removed unused `abstractmethod` decorator on `CarAction.act` method

### Changed
- Refactored docstrings for consistency, correctness, and RTD readability
- Clarified supported Python versions in README
- Reorganized README for clearer table of contents
- Made unit test suite more efficient
- Renamed environment unit test file

## [1.1.1] - 2026-03-15

### Fixed
- Version bump only (packaging fix)

## [1.1.0] - 2026-03-15

### Added
- PyPI publishing via `publish.yml` GitHub Actions workflow (triggered on tag push)
- PyPI badge in README
- ReadTheDocs documentation site with Sphinx RTD theme
- Logos, favicon, and branding assets
- Inference testing script for DRL models
- VSCode debugging support for acados

### Changed
- Refactored package for PyPI publishing (renamed `f1tenth_gym` to `gymkhana`)
- Updated map loading to use maintained source for better compatibility

### Fixed
- Mexico City track naming issue

## [1.0.0] - 2026-01-30

### Added
- **Drift dynamics**: Single Track Drift (STD) model with PAC2002 tire physics
- **Recovery training**: dedicated training mode (`training_mode: "recover"`) with recovery-specific rewards, resets, and evaluation
- **Curriculum learning**: `CurriculumLearningCallback` that gradually expands recovery state initialization ranges as agent success rate improves
- **Transfer learning**: transfer pretrained model weights to new tasks with optional critic reset and `log_std` re-initialization
- **MPC controllers**: Kinematic MPC (KMPC) and Single-Track MPC (STMPC) via acados integration
- **Reverse/random driving direction**: `track_direction` config option (`normal`, `reverse`, `random`) for balanced cornering training
- **State reset**: `env.reset(options={"states": ...})` for full 7-d state initialization (STD model)
- **Arc-length visualization**: render arc-length annotations along the centerline for Frenet coordinate debugging
- **Multi-map training**: train across multiple maps in parallel environments
- **Observation types**: `FeaturesObservation` (drift) with slip angle, lookahead curvatures/widths; observation and action normalization
- **Sparse width observations**: `sparse_width_obs` config to reduce observation dimensionality when track width varies little
- **ONNX export**: convert SB3 models to ONNX for sim-to-real transfer via `OnnxPolicyRunner`
- **Norm bounds extraction**: save normalization bounds to config for reuse in external packages
- **Morris sensitivity analysis**: parameter sensitivity analysis script for STD model
- **Observation sensitivity analysis**: script for analyzing observation feature importance
- **Custom neural network architecture**: configurable layer sizes via `net_arch` in RL config
- **Wandb integration**: experiment tracking, model logging, download, and resume from wandb
- **Control debug panel**: real-time steering/throttle visualization (PyQt6 renderer)
- **Observation debug overlay**: live observation values overlay on the map (PyQt6 renderer)
- **Frenet projection debugging**: `debug_frenet_projection` visualization option
- **Lookahead curvature visualization**: render lookahead sampling points ahead of the vehicle
- **Performance metrics**: beta-r phase plane analysis, recovery trajectory plotting, controller comparison metrics
- **P/PD controllers**: simple proportional controllers for benchmarking recovery performance
- **Ruff formatting**: migrated from Black to Ruff for linting, formatting, and import sorting with pre-commit hooks
- **Regression tests**: tests for action ordering and other previously encountered bugs
- **Predictive collision**: TTC-based collision checking as alternative to Frenet-based
- **Wall deflection mode**: configurable track boundary behavior (boundary vs. wall collision)
- **Progress reward gain**: configurable `progress_gain` multiplier for forward progress reward
- **Docker removal**: removed Docker support in favor of Poetry and PyPI

### Fixed
- Action ordering bug where `control_input` order affected action array mapping
- Rendering `rgb_array` mode bugfix
- Wraparound bug in progress tracking (robust fix with unit tests)
- Normalization logic when both actions and observations are normalized
- Reverse direction reset bug where random reset could point vehicle the wrong way due to cached reference line
- Reverse direction normalization fix where min/max curvature bounds need to be symmetric

### Changed
- Refactored controllers with abstract base class
- Refactored training scripts with shared `train_common.py` and `train_utils.py`
- Reorganized project structure (analysis, figures, plans into dedicated directories)
- Moved plotting files to `analysis/` folder
- Configuration logic simplified and centralized in `train/config/`

[Unreleased]: https://github.com/TeoIlie/Gym-Khana/compare/v1.1.1...HEAD
[1.1.1]: https://github.com/TeoIlie/Gym-Khana/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/TeoIlie/Gym-Khana/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/TeoIlie/Gym-Khana/releases/tag/v1.0.0
