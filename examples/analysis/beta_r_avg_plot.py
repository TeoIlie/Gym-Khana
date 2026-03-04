"""
Beta-R Recovery Success Heatmap Analysis

This script systematically evaluates a controller's recovery capability across
a grid of initial (beta, r) conditions, averaging over variations in velocity
and yaw. The output is a heatmap showing recovery success rate at each (beta, r)
grid cell, plus printed aggregate metrics.

Usage — Steer controller:
    1. Set CONTROLLER_TYPE = "stanley" or "steer" and DESC to a description string
    2. Run: python examples/analysis/beta_r_avg_plot.py
    3. Output saved to figures/analysis/recover_heatmap/CONTROLLER_TYPE

Usage — Learned (PPO) controller:
    1. Set CONTROLLER_TYPE = "learned"
    2. Set LEARNED_TYPE to "drift" or "recover"
    3. Set RUN_ID to the wandb run ID (model downloaded to outputs/downloads/<RUN_ID>/)
    4. Set DESC to a short description of the model
    5. Run: python examples/analysis/beta_r_avg_plot.py
    6. Output saved to figures/analysis/recover_heatmap/<RUN_ID>/
"""

import argparse
import os

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from examples.controllers import create_controller
from train.config.env_config import get_env_id
from train.train_utils import get_output_dirs, print_header

CONTROLLER_TYPE = "stanley"
DESC = "stanley"
LEARNED_TYPE = ""
RUN_ID = ""

# CONTROLLER_TYPE = "learned"

# LEARNED_TYPE = "drift"
# RUN_ID = "178a1a5l"
# DESC = "drift model - CW & CCW on Drift_large, with `sparse_width_obs` = True"

# LEARNED_TYPE = "drift"
# RUN_ID = "iza03vyw"
# DESC = "drift model - CW & CCW on Drift_large, with `sparse_width_obs` = False"

# LEARNED_TYPE = "drift"
# RUN_ID = "bsoh5xyb"
# DESC = "drift model - CW & CCW on Drift_large, with `sparse_width_obs` = True"

# LEARNED_TYPE = "recover"
# RUN_ID = "p13d1mdz"
# DESC = "recovering model - original with Euclidean reward"

# LEARNED_TYPE = "recover"
# RUN_ID = "irdqwnhp"
# DESC = "recovering model - no Euclidean reward"

# LEARNED_TYPE = "recover"
# RUN_ID = "8m5f957h"
# DESC = "recovering model - Euclidean reward, larger beta, r ranges"

# LEARNED_TYPE = "recover"
# RUN_ID = "50x16c1d"
# DESC = "recovering model - Euclidean reward, curriculum learning, smaller beta, r ranges"

# LEARNED_TYPE = "recover"
# RUN_ID = "qhj88o3r"
# DESC = "recovering model - Euclidean reward, curriculum learning, larger beta, r ranges"

# LEARNED_TYPE = "recover"
# RUN_ID = "sysea5vx"
# DESC = "recovering model - no Euclidean reward, curriculum learning, 200 success reward, smaller beta, r ranges"

# LEARNED_TYPE = "recover"
# RUN_ID = "qh54psj2"
# DESC = "recovering model - original with Euclidean reward, no curriculum, small beta,r variations"

# LEARNED_TYPE = "recover"
# RUN_ID = "koa3rljd"
# DESC = "recovering model - Euclidean reward, curriculum learning, larger beta, r ranges"

# LEARNED_TYPE = "recover"
# RUN_ID = "w7bkr26u"
# DESC = "recovering model - no Euclidean reward, curriculum learning, 200 success reward, smaller beta, r ranges"

# LEARNED_TYPE = "recover"
# RUN_ID = "g3w88oqx"
# DESC = "recovering model - drift model 178a1a5l retrained with Fine-Tuning with Fresh Optimizer + LR Reset + log_std reset with --m f. No curriculum learning, small beta-r initial ranges, no Euclidean reward"

# LEARNED_TYPE = "recover"
# RUN_ID = "bwcm7l05"
# DESC = "recovering model - drift model 178a1a5l retrained by loading and continuing training with --m c. No curriculum learning, small beta-r initial ranges, no Euclidean reward"

# LEARNED_TYPE = "recover"
# RUN_ID = "pbmnxwcc"
# DESC = "recovering model - drift model 178a1a5l retrained with Fine-Tuning with Fresh Optimizer + LR Reset + log_std reset + Critic Reinitialization.\nNo curriculum learning, small beta-r initial ranges, no Euclidean reward"

S = 96  # Arc length on IMS straight section
SEED = 42

# Grid parameters (radians / rad/s / m/s)
BETA_VALUES = np.linspace(-1.39, 1.39, 7)  # 10 points, +/-60 deg
R_VALUES = np.linspace(-13, 13, 7)  # 10 points, +/-180 deg/s
V_VALUES = np.linspace(4, 5, 2)  # 3 points: [2, 7, 12]
YAW_VALUES = np.linspace(-0.17, 0.17, 3)  # 3 points: [-60, 0, +60] deg


def reset_at_state(eval_env, beta_rad, r_rad, v, yaw_offset_rad):
    """Reset environment with specified beta, r, velocity, and yaw offset.

    Args:
        eval_env: The gym environment
        beta_rad: Sideslip angle in radians
        r_rad: Yaw rate in rad/s
        v: Velocity in m/s
        yaw_offset_rad: Yaw offset from track heading in radians

    Returns:
        Initial observation
    """
    x, y, base_yaw = eval_env.unwrapped.track.frenet_to_cartesian(S, ey=0, ephi=0)
    yaw = base_yaw + yaw_offset_rad

    init_state = np.array(
        [
            [
                x,
                y,
                0.0,  # delta (steering angle)
                v,
                yaw,
                r_rad,
                beta_rad,
            ]
        ]
    )

    obs, _ = eval_env.reset(options={"states": init_state})
    return obs


def run_episode(eval_env, controller, beta, r, v, yaw):
    """Run one recovery episode from the given initial state.

    Returns:
        recovered: Whether the episode ended with successful recovery.
        time_seconds: Episode duration in seconds.
    """
    dt = eval_env.unwrapped.timestep
    obs = reset_at_state(eval_env, beta, r, v, yaw)

    done, trunc = False, False
    steps = 0
    info = {"recovered": False}

    while not done and not trunc:
        action = controller.get_action(obs)
        obs, _, done, trunc, info = eval_env.step(action)
        steps += 1

    recovered = done and info.get("recovered", False)
    return recovered, steps * dt


def run_grid_evaluation(eval_env, controller, stanley_states=None):
    """Run recovery evaluation across the full (beta, r, v, yaw) grid.

    Args:
        eval_env: The gym environment.
        controller: Controller providing get_action(obs).
        stanley_states: Optional set of (beta, r, v, yaw) tuples where Stanley succeeded.
            When provided, also tracks metrics on this subset.

    Returns:
        recovery_rates: array of recovery rate per (beta, r) cell
        mean_recovery_times: array of mean recovery time (seconds) per cell
        successful_states: list of (beta, r, v, yaw) tuples that recovered
    """
    n_beta = len(BETA_VALUES)
    n_r = len(R_VALUES)
    n_v = len(V_VALUES)
    n_yaw = len(YAW_VALUES)
    n_inner = n_v * n_yaw
    total_episodes = n_beta * n_r * n_inner

    # Per-cell accumulators
    recovery_counts = np.zeros((n_beta, n_r))
    recovery_time_sums = np.zeros((n_beta, n_r))

    successful_states = []

    # Stanley-subset accumulators
    stanley_total_count = 0
    stanley_recovery_times = []

    episode = 0
    for i, beta in enumerate(BETA_VALUES):
        for j, r in enumerate(R_VALUES):
            for v in V_VALUES:
                for yaw in YAW_VALUES:
                    recovered, time_s = run_episode(eval_env, controller, beta, r, v, yaw)

                    if recovered:
                        recovery_counts[i, j] += 1
                        recovery_time_sums[i, j] += time_s
                        successful_states.append((beta, r, v, yaw))

                    # Track Stanley-subset metrics
                    if stanley_states is not None and (beta, r, v, yaw) in stanley_states:
                        stanley_total_count += 1
                        if recovered:
                            stanley_recovery_times.append(time_s)

                    episode += 1
                    if episode % 100 == 0:
                        print(f"  Progress: {episode}/{total_episodes} episodes")

    recovery_rates = recovery_counts / n_inner

    # compute mean times, guarding against division by 0
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_recovery_times = np.where(recovery_counts > 0, recovery_time_sums / recovery_counts, np.nan)

    return recovery_rates, mean_recovery_times, successful_states, stanley_total_count, stanley_recovery_times


def plot_recovery_heatmap(beta_values, r_values, recovery_rates, output_path, controller_label=""):
    """Plot a heatmap of recovery success rate across the (beta, r) grid."""
    fig, ax = plt.subplots(figsize=(10, 8))

    beta_deg = np.rad2deg(beta_values)
    r_deg = np.rad2deg(r_values)
    pct = recovery_rates * 100

    im = ax.imshow(
        pct.T,
        origin="lower",
        aspect="auto",
        cmap="magma",
        vmin=0,
        vmax=100,
        interpolation="bilinear",
    )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Recovery Rate (%)", fontsize=12, fontweight="bold")

    # Annotate cells at integer positions (aligned with imshow pixels)
    for i in range(len(beta_deg)):
        for j in range(len(r_deg)):
            val = pct[i, j]
            color = "white" if val < 40 or val > 80 else "black"
            ax.text(
                i,
                j,
                f"{val:.0f}%",
                ha="center",
                va="center",
                fontsize=7,
                fontweight="bold",
                color=color,
            )

    # Use degree tick labels on integer positions
    ax.set_xticks(range(len(beta_deg)))
    ax.set_xticklabels([f"{v:.0f}" for v in beta_deg])
    ax.set_yticks(range(len(r_deg)))
    ax.set_yticklabels([f"{v:.0f}" for v in r_deg])

    ax.set_xlabel("Beta (sideslip angle) [deg]", fontsize=13, fontweight="bold")
    ax.set_ylabel("R (yaw rate) [deg/s]", fontsize=13, fontweight="bold")
    ax.set_title(
        f"Recovery Success Rate — {controller_label}\n"
        f"Averaged over v=[{V_VALUES[0]:.0f}..{V_VALUES[-1]:.0f}] m/s, "
        f"yaw=[{np.rad2deg(YAW_VALUES[0]):.0f}..{np.rad2deg(YAW_VALUES[-1]):.0f}] deg",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nHeatmap saved to: {output_path}")
    plt.show()


def save_metrics(title, summary_lines, recovery_times, output_path, desc=""):
    """Format, print, and save recovery metrics to a file.

    Args:
        title: Section header
        summary_lines: List of summary strings (grid info, rates, etc.).
        recovery_times: 1-D array of recovery times for successful episodes.
        output_path: File path to write the metrics to.
        desc: Optional description to append to the file.
    """
    lines = ["", "=" * 50, title, "=" * 50]
    lines += summary_lines
    if len(recovery_times) > 0:
        lines += [
            f"Mean recovery time:   {np.mean(recovery_times):.3f} s",
            f"Median recovery time: {np.median(recovery_times):.3f} s",
            f"Std recovery time:    {np.std(recovery_times):.3f} s",
        ]
    else:
        lines.append("No successful recoveries recorded.")
    lines.append("=" * 50)

    text = "\n".join(lines)
    print(text)

    with open(output_path, "w") as f:
        f.write(text + "\n")
        if desc:
            f.write(f"\nNote: {desc}\n")
    print(f"Metrics saved to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Beta-R Recovery Success Heatmap Analysis")
    parser.add_argument(
        "--controller_type", default=CONTROLLER_TYPE, help="Controller type: 'learned', 'stanley', 'steer'"
    )
    parser.add_argument("--learned_type", default=LEARNED_TYPE, help="Learned model type: 'drift' or 'recover'")
    parser.add_argument("--run_id", default=RUN_ID, help="Wandb run ID for the learned model")
    parser.add_argument("--desc", default=DESC, help="Short description of the model")
    return parser.parse_args()


def main():
    args = parse_args()
    controller_type = args.controller_type
    learned_type = args.learned_type
    run_id = args.run_id
    desc = args.desc

    print_header("Beta-R Recovery Success Heatmap Analysis")
    print(f"Controller: {controller_type}, Learned type: {learned_type}, Run ID: {run_id}")
    print(f"Description: {desc}")

    proj_root, _ = get_output_dirs()

    model_path = f"{proj_root}/outputs/downloads/{run_id}/model.zip" if controller_type == "learned" else None
    controller = create_controller(
        controller_type,
        model_path=model_path,
        map="IMS",
    )

    config = controller.get_env_config()
    config["training_mode"] = "recover"

    eval_env = gym.make(
        get_env_id(),
        config=config,
        render_mode=None,
    )

    np.random.seed(SEED)

    print(
        f"\nRunning grid evaluation: {len(BETA_VALUES)}x{len(R_VALUES)} cells, "
        f"{len(V_VALUES) * len(YAW_VALUES)} episodes per cell..."
    )

    # Load Stanley recovery states for learned controllers
    stanley_states_path = f"{proj_root}/figures/analysis/recover_heatmap/stanley_recovery_states.npz"

    # Load Stanley recovery states and validate they are a subset of current configured states
    stanley_states = None
    if controller_type == "learned" and os.path.exists(stanley_states_path):
        data = np.load(stanley_states_path)

        grids_match = (
            np.allclose(data["beta_values"], BETA_VALUES)
            and np.allclose(data["r_values"], R_VALUES)
            and np.allclose(data["v_values"], V_VALUES)
            and np.allclose(data["yaw_values"], YAW_VALUES)
        )
        if grids_match:
            stanley_states = set(map(tuple, data["states"]))
            print(f"Loaded {len(stanley_states)} Stanley recovery states for baseline comparison")
        else:
            print(
                "Warning: Stanley states were computed with a different grid — "
                "skipping baseline comparison. Re-run with stanley to regenerate."
            )
    elif controller_type == "learned":
        print(f"Warning: {stanley_states_path} not found — run Stanley evaluation first for baseline metrics")

    recovery_rates, mean_recovery_times, successful_states, stanley_total, stanley_times = run_grid_evaluation(
        eval_env, controller, stanley_states=stanley_states
    )

    # Save Stanley successful states with grid parameters
    if controller_type == "stanley" and successful_states:
        save_dir = os.path.dirname(stanley_states_path)
        os.makedirs(save_dir, exist_ok=True)
        np.savez(
            stanley_states_path,
            states=np.array(successful_states),
            beta_values=BETA_VALUES,
            r_values=R_VALUES,
            v_values=V_VALUES,
            yaw_values=YAW_VALUES,
        )
        print(f"Saved {len(successful_states)} Stanley recovery states to: {stanley_states_path}")

    subfolder = (
        f"{proj_root}/figures/analysis/recover_heatmap/{run_id}"
        if controller_type == "learned"
        else f"{proj_root}/figures/analysis/recover_heatmap/{controller_type}"
    )
    os.makedirs(subfolder, exist_ok=True)
    output_path = (
        f"{subfolder}/beta_r_recovery_heatmap_{controller_type}{'_' + learned_type if learned_type else ''}_policy.png"
    )
    controller_label = f"{controller_type}" + (f" ({learned_type})" if learned_type else "")
    plot_recovery_heatmap(BETA_VALUES, R_VALUES, recovery_rates, output_path, controller_label=controller_label)

    n_inner = len(V_VALUES) * len(YAW_VALUES)
    overall_rate = np.mean(recovery_rates) * 100
    overall_std = np.std(recovery_rates) * 100
    valid_times = mean_recovery_times[~np.isnan(mean_recovery_times)]
    save_metrics(
        "RECOVERY METRICS",
        [
            f"Grid: {len(BETA_VALUES)}x{len(R_VALUES)} (beta x r), {n_inner} (v, yaw) combos per cell",
            f"Total episodes: {len(BETA_VALUES) * len(R_VALUES) * n_inner}",
            f"Overall recovery rate: {overall_rate:.1f}% (std across cells: {overall_std:.1f}%)",
        ],
        valid_times,
        os.path.join(subfolder, "metrics.txt"),
        desc=desc,
    )

    # Stanley-baseline metrics for learned controllers
    if stanley_total > 0:
        n_recovered = len(stanley_times)
        rate = n_recovered / stanley_total * 100
        save_metrics(
            "STANLEY-BASELINE RECOVERY METRICS",
            [
                f"Stanley recovery states evaluated: {stanley_total}",
                f"Learned policy recovered: {n_recovered}/{stanley_total} ({rate:.1f}%)",
            ],
            np.array(stanley_times),
            os.path.join(subfolder, "stanley_recovery_states_metrics.txt"),
            desc=desc,
        )

    eval_env.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
