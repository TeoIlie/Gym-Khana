"""
Centralized F1TENTH Gym Environment Configuration
"""

import multiprocessing
import os

import yaml

from f1tenth_gym.envs.f110_env import F110Env

# ====================================
# RL config
# ====================================
_rl_config_path = os.path.join(os.path.dirname(__file__), "rl_config.yaml")
with open(_rl_config_path, "r") as f:
    _rl_config = yaml.safe_load(f)

# RL training parameters
N_ENVS = _rl_config["core_mult"] * multiprocessing.cpu_count()  # CPU core count * multiplier
TOTAL_TIMESTEPS = _rl_config["total_timesteps"]
N_STEPS = _rl_config["n_steps"]
START_LEARNING_RATE = _rl_config["start_learning_rate"]
END_LEARNING_RATE = _rl_config["end_learning_rate"]
SEED = _rl_config["seed"]
EVAL_SEED = _rl_config["eval_seed"]
ACT_FUNC_NEG_SLOPE = _rl_config["act_func_neg_slope"]
ADDITIONAL_TIMESTEPS = _rl_config["additional_timesteps"]
TRANSFER_RESET_LOG_STD = _rl_config["transfer_reset_log_std"]
TRANSFER_RESET_CRITIC = _rl_config["transfer_reset_critic"]


# ====================================
# Gym config
# ====================================
_config_path = os.path.join(os.path.dirname(__file__), "gym_config.yaml")
with open(_config_path, "r") as f:
    _config = yaml.safe_load(f)

# Gym shared parameters
PROJECT_NAME = _config["project_name"]
RACE_TRAINING_MODE = _config["race_training_mode"]
MAP = _config["map"]
TRACK_POOL = _config["track_pool"]
MODEL = _config["model"]
TIMESTEP = _config["timestep"]
NUM_AGENTS = _config["num_agents"]
INTEGRATOR = _config["integrator"]
ACTION_INPUT = _config["action_input"]
OBS_TYPE = _config["obs_type"]
RESET_CONFIG = _config["reset_config"]
TRACK_DIRECTION = _config["track_direction"]
LOOKAHEAD_N_POINTS = _config["lookahead_n_points"]
LOOKAHEAD_DS = _config["lookahead_ds"]
SPARSE_WIDTH_OBS = _config["sparse_width_obs"]
NORM_OBS = _config["normalize_obs"]
RECORD_OBS_MIN_MAX = _config["record_obs_min_max"]
PREDICTIVE_COLLISION = _config["predictive_collision"]
NORM_ACT = _config["normalize_act"]
WALL_DEFLECTION = _config["wall_deflection"]

# Vehicle parameters
PARAMS = F110Env.f1tenth_std_drift_bias_params()

# Render/debug params
TEST_DEBUG_RENDER = _config["test_debug_render"]
TRAIN_DEBUG_RENDER = _config["train_debug_render"]

# Callback config
CKPT_SAVE_FREQ = _config["ckpt_save_freq"]
N_EVAL_EPISODES = _config["n_eval_episodes"]
BEST_MODEL = "best_model"

# LiDAR beams
NUM_BEAMS = _config["num_beams"]

# Recovery-specific configs
RECOVERY_PROJECT_NAME = _config["recovery_project_name"]
RECOVERY_TRAINING_MODE = _config["recovery_training_mode"]
RECOVERY_TRACK_POOL = _config["recovery_track_pool"]

# Curriculum learning config
CURRICULUM_CONFIG = _config.get("curriculum", {})


# ====================================
# Gym config functions
# ===================================
def get_env_id():
    return "f1tenth_gym:f1tenth-v0"


def _base_config(debug_render):
    """
    Returns environment params shared across all environments
    """
    return {
        "num_agents": NUM_AGENTS,
        "timestep": TIMESTEP,
        "integrator": INTEGRATOR,
        "model": MODEL,
        "num_beams": NUM_BEAMS,
        "control_input": ACTION_INPUT,
        "observation_config": {"type": OBS_TYPE},
        "reset_config": {"type": RESET_CONFIG},
        "render_lookahead_curvatures": debug_render,  # Enable lookahead curvature visualization
        "lookahead_n_points": LOOKAHEAD_N_POINTS,
        "lookahead_ds": LOOKAHEAD_DS,
        "sparse_width_obs": SPARSE_WIDTH_OBS,
        "debug_frenet_projection": debug_render,  # Enable Frenet projection debug visualization
        "params": PARAMS,
        "render_track_lines": debug_render,  # View track lines
        "render_arc_length_annotations": debug_render,
        "normalize_obs": NORM_OBS,
        "record_obs_min_max": RECORD_OBS_MIN_MAX,
        "predictive_collision": PREDICTIVE_COLLISION,
        "normalize_act": NORM_ACT,
        "wall_deflection": WALL_DEFLECTION,
    }


def _drift_overrides():
    """
    Drift/race-specific overrides
    """
    return {
        "training_mode": RACE_TRAINING_MODE,
        "map": MAP,
        "track_pool": TRACK_POOL,
        "track_direction": TRACK_DIRECTION,
    }


def _recovery_overrides():
    """
    Recovery-specific overrides. When curriculum learning is enabled, the recovery ranges
    are set to the curriculum's max_lo/max_hi values.
    """
    overrides = {
        "training_mode": RECOVERY_TRAINING_MODE,
        "track_pool": RECOVERY_TRACK_POOL,
    }

    if CURRICULUM_CONFIG.get("enabled", False):
        for gym_key, curriculum_key in [
            ("recovery_v_range", "v_range"),
            ("recovery_beta_range", "beta_range"),
            ("recovery_r_range", "r_range"),
            ("recovery_yaw_range", "yaw_range"),
        ]:
            vals = CURRICULUM_CONFIG[curriculum_key]
            overrides[gym_key] = [vals[2], vals[3]]  # [max_lo, max_hi]

    return overrides


def get_drift_test_config():
    """
    Returns gym drift TESTING environment config
    """
    return {**_base_config(TEST_DEBUG_RENDER), **_drift_overrides()}


def get_drift_train_config():
    """
    Returns gym drift TRAINING environment config
    """
    return {**_base_config(TRAIN_DEBUG_RENDER), **_drift_overrides()}


def get_recovery_test_config():
    """
    Returns gym recovery TESTING environment config
    """
    return {**_base_config(TEST_DEBUG_RENDER), **_recovery_overrides()}


def get_recovery_train_config():
    """
    Returns gym recovery TRAINING environment config
    """
    return {**_base_config(TRAIN_DEBUG_RENDER), **_recovery_overrides()}


def get_curriculum_config():
    """Returns curriculum learning config dict (empty dict if not configured)."""
    return CURRICULUM_CONFIG
