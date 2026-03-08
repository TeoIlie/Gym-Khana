# Plan: Rename Project to Gym-Khana

## Context
The project is a fork of f1tenth_gym with significant custom work (drift dynamics, recovery training, MPC controllers). The rename establishes it as an independent project, separate from the upstream f1tenth_gym. The name "Gym-Khana" is a play on words: "Gym" for RL Gymnasium and "Khana" from Gymkhana (drift/precision driving motorsport). This rename also prepares the project for future PyPI publishing.

## Naming Scheme

| Context | Current | New | Notes |
|---------|---------|-----|-------|
| Display name / branding | F1TENTH Gym | **Gym-Khana** | Stylized, used in README title, repo name, docs prose |
| GitHub repo | `TeoIlie/F1TENTH_Gym` | **`TeoIlie/Gym-Khana`** | Stylized |
| Python package / import | `f1tenth_gym` | **`gymkhana`** | Plain, no hyphen (Python identifier rules) |
| PyPI name | `f1tenth_gym` | **`gymkhana`** | Plain |
| Package directory | `f1tenth_gym/` | **`gymkhana/`** | Plain |
| Gym environment ID | `f1tenth-v0` | **`gymkhana-v0`** | Plain |
| `gym.make()` call | `gym.make("f1tenth_gym:f1tenth-v0")` | **`gym.make("gymkhana:gymkhana-v0")`** | Plain |
| Main env class | `F110Env` | **`GKEnv`** | |
| Main env file | `f110_env.py` | **`gymkhana_env.py`** | |
| Docker image | `f1tenth_gym_container` | **`gymkhana`** | |
| Cache directory | `~/.f1tenth_gym/maps/` | **`~/.gymkhana/maps/`** | |
| **Keep unchanged** | `f1tenth_vehicle_params()`, `f1tenth_std_vehicle_params()`, `f1tenth_std_drift_bias_params()` | Same | Describes physical 1/10th scale, not project |

## Steps

### Step 1: Rename GitHub repo (manual, before code changes)
- Rename `TeoIlie/F1TENTH_Gym` → `TeoIlie/Gym-Khana` via GitHub Settings > General > Repository name
- Update local remote: `git remote set-url origin https://github.com/TeoIlie/Gym-Khana.git`
- GitHub auto-redirects old URLs, so nothing breaks immediately

### Step 2: Rename package directory
- `git mv f1tenth_gym/ gymkhana/`
- This is the highest-impact change — all imports break until updated

### Step 3: Rename env file
- `git mv gymkhana/envs/f110_env.py gymkhana/envs/gymkhana_env.py`

### Step 4: Update class name F110Env → GKEnv
- **File:** `gymkhana/envs/gymkhana_env.py` — class definition (line 47)
- **File:** `gymkhana/envs/__init__.py` — export

### Step 5: Update gym registration
- **File:** `gymkhana/__init__.py`
  - Change ID from `"f1tenth-v0"` to `"gymkhana-v0"`
  - Change entry_point from `"f1tenth_gym.envs:F110Env"` to `"gymkhana.envs:GKEnv"`

### Step 6: Update pyproject.toml
- `name = "gymkhana"`
- `description` — update to reflect Gym-Khana identity
- `packages = [{include = "gymkhana"}]`
- `repository` — `https://github.com/TeoIlie/Gym-Khana`
- `documentation` — update or remove readthedocs link

### Step 7: Bulk find-and-replace across Python files
Mechanical replacements, applied in this order:
1. `from f1tenth_gym.` → `from gymkhana.` (all imports)
2. `import f1tenth_gym` → `import gymkhana`
3. `f1tenth_gym:f1tenth-v0` → `gymkhana:gymkhana-v0` (gym.make calls)
4. `"f1tenth-v0"` → `"gymkhana-v0"` (env ID strings)
5. `F110Env` → `GKEnv` (class references, excluding `f1tenth_vehicle_params` method names)
6. `f110_env` → `gymkhana_env` (module references in imports)

**Files affected (~50+):**
- `train/config/env_config.py` — env ID function, imports
- `train/train_utils.py` — imports
- `train/ppo_race.py`, `train/ppo_recover.py`, `train/ppo_example.py` — gym.make calls
- All `tests/test_*.py` files (~20 files)
- `examples/*.py` and `examples/controllers/*.py`
- `examples/analysis/**/*.py`

### Step 8: Update Dockerfile
- **File:** `Dockerfile`
  - Change `/f1tenth_gym` → `/gymkhana` (lines 46-52)
  - Change image name in any build instructions

### Step 9: Update cache directory reference
- **File:** `gymkhana/envs/track/track_utils.py`
  - Change `~/.f1tenth_gym/maps/` → `~/.gymkhana/maps/`

### Step 10: Update README.md
- Title: `# Gym-Khana`
- Description referencing the wordplay
- Badge URLs → `TeoIlie/Gym-Khana`
- Clone URL → `https://github.com/TeoIlie/Gym-Khana.git`
- Docker image name → `gymkhana`
- All code examples using new import/gym.make patterns

### Step 11: Update CLAUDE.md
- All references to `f1tenth_gym` → `gymkhana`, `F110Env` → `GKEnv`, `f110_env.py` → `gymkhana_env.py`
- Display name references → Gym-Khana
- Update architecture section with new names

### Step 12: Update other documentation
- `docs/plan/*.md` — update any code examples

### Step 13: Update CI workflows
- **Files:** `.github/workflows/*.yml`
  - Check for any hardcoded package name references

## Verification
1. `pip install -e .` — confirm package installs under new name
2. `python -c "import gymkhana"` — confirm import works
3. `python -c "import gymnasium as gym; env = gym.make('gymkhana:gymkhana-v0')"` — confirm env registration
4. `python3 -m pytest` — run full test suite
5. `ruff check .` — confirm no lint errors
6. Check examples still run: `cd examples && python3 waypoint_follow.py`
