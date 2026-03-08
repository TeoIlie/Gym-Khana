# Plan: Rewrite Gym-Khana Documentation for Read the Docs

## Context
The `docs/` directory contains outdated Sphinx documentation inherited from the original f1tenth_gym project. All content references wrong package names, URLs, and APIs. The README.md is comprehensive and up-to-date. Goal: rewrite docs content from README, keep Sphinx infrastructure, host on Read the Docs.

## New Page Structure

```
docs/
  index.rst              -- New landing page with features, demo GIF, toctree
  installation.rst       -- Rewrite: pip, poetry, docker, acados (README §Quickstart + §Additional Dependencies)
  quickstart.rst         -- NEW: running examples + basic simulation loop code
  configuration.rst      -- Rewrite: all config options (README §Configuration, §Reset, §Debugging, §Maps)
  training.rst           -- NEW: ppo_race/recover, modes, CL, callbacks, wandb (README §Training + §CL + §Wandb)
  architecture.rst       -- NEW: package structure, models, important files (README §Important files + §Tire params)
  known_issues.rst       -- NEW: platform issues (README §Known issues)
  api/
    env.rst              -- GKEnv class, config dict, vehicle params
    base_classes.rst     -- RaceCar, Simulator, step method
    dynamic_models.rst   -- KS/ST/MB/STD model descriptions
    observation.rst      -- NEW (replaces obv.rst): observation factory, types, normalization
    action.rst           -- NEW: action types, normalization
    track.rst            -- NEW: Track class, Frenet conversion, map loading
```

**Delete**: `basic_usage.rst`, `customized_usage.rst`, `reproduce.rst`, `api/obv.rst`, `api/laser_models.rst`, `api/collision_models.rst`, `api/rendering.rst`

## Implementation Steps

### 1. Create `.readthedocs.yaml` (repo root)
```yaml
version: 2
build:
  os: ubuntu-22.04
  tools:
    python: "3.11"
sphinx:
  configuration: docs/conf.py
python:
  install:
    - requirements: docs/requirements.txt
```

### 2. Update `docs/requirements.txt`
Replace `breathe` with `sphinx>=7.0` and `sphinx_rtd_theme>=2.0`.

### 3. Update `docs/conf.py`
- `project` → `"Gym-Khana"`
- `author` → `"Teodor Ilie"`
- `copyright` → include Teodor Ilie, 2024-2026
- `github_user` → `"TeoIlie"`, `github_repo` → `"Gym-Khana"`, `github_version` → `"main"`
- Remove `breathe` from extensions and `breathe_projects` dict
- Update or remove `html_logo` (use `gym.svg` or comment out until a Gym-Khana logo exists)
- Keep `sphinx_rtd_theme`, `sphinx.ext.autosectionlabel`
- Add `"plan"` to `exclude_patterns` to suppress warnings about `docs/plan/` files

### 4. Update `docs/Doxyfile`
- `PROJECT_NAME` → `"Gym-Khana"`, `INPUT` → `../gymkhana`

### 5. Rewrite all .rst content files
Each page rewritten from README.md content (see page structure above). API .rst files become manual descriptions instead of broken Doxygen `doxygenfile` directives.

### 6. Handle assets
- Copy `figures/F1TENTH_PPO_Drift.gif` into `docs/assets/` for self-containment
- Stop referencing f1tenth-branded SVGs/PNGs from new pages
- Use `gym.svg` or no logo temporarily

### 7. Update `pyproject.toml` line 30
`documentation` URL → `"https://gym-khana.readthedocs.io/en/latest/"` (adjust slug after RTD registration)

### 8. Delete old files
Remove `basic_usage.rst`, `customized_usage.rst`, `reproduce.rst`, `api/obv.rst`, `api/laser_models.rst`, `api/collision_models.rst`, `api/rendering.rst`, `docs/html/` (stale built output)

## Verification
1. `cd docs && make html` — build locally, check for warnings
2. Open `docs/_build/html/index.html` in browser, verify all pages render
3. Verify no broken toctree references or missing files
4. After pushing, connect repo to readthedocs.org and trigger a build

## Files Modified
- **Create**: `.readthedocs.yaml`, `docs/quickstart.rst`, `docs/training.rst`, `docs/architecture.rst`, `docs/known_issues.rst`, `docs/configuration.rst`, `docs/api/observation.rst`, `docs/api/action.rst`, `docs/api/track.rst`
- **Rewrite**: `docs/index.rst`, `docs/installation.rst`, `docs/api/env.rst`, `docs/api/base_classes.rst`, `docs/api/dynamic_models.rst`
- **Update**: `docs/conf.py`, `docs/requirements.txt`, `docs/Doxyfile`, `pyproject.toml`
- **Delete**: `docs/basic_usage.rst`, `docs/customized_usage.rst`, `docs/reproduce.rst`, `docs/api/obv.rst`, `docs/api/laser_models.rst`, `docs/api/collision_models.rst`, `docs/api/rendering.rst`, `docs/html/` directory
