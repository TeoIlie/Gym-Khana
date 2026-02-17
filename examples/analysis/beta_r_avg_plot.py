"""
Beta-R Recovery Success Heatmap Analysis

This script systematically evaluates a controller's recovery capability across
a grid of initial (beta, r) conditions, averaging over variations in velocity
and yaw. The output is a heatmap showing recovery success rate at each (beta, r)
grid cell, plus printed aggregate metrics.

Usage:
    1. Edit MODEL_PATH, S, grid parameters as needed
    2. Run: python examples/analysis/beta_r_avg_plot.py
    3. View generated heatmap in tests/test_figures/
"""

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from examples.controllers import create_controller
from train.config.env_config import get_env_id
from train.train_utils import get_output_dirs, print_header

CONTROLLER_TYPE = "learned"
LEARNED_TYPE = "drift"
MODEL_PATH = "/outputs/downloads/178a1a5l/model.zip"

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


def run_grid_evaluation(eval_env, controller):
    """Run recovery evaluation across the full (beta, r, v, yaw) grid.

    Returns:
        recovery_rates: (10, 10) array of recovery rate per (beta, r) cell
        mean_recovery_times: (10, 10) array of mean recovery time (seconds) per cell
    """
    dt = eval_env.unwrapped.timestep
    n_beta = len(BETA_VALUES)
    n_r = len(R_VALUES)
    n_v = len(V_VALUES)
    n_yaw = len(YAW_VALUES)
    n_inner = n_v * n_yaw
    total_episodes = n_beta * n_r * n_inner

    # Per-cell accumulators
    recovery_counts = np.zeros((n_beta, n_r))
    recovery_time_sums = np.zeros((n_beta, n_r))
    recovery_time_counts = np.zeros((n_beta, n_r))

    episode = 0
    for i, beta in enumerate(BETA_VALUES):
        for j, r in enumerate(R_VALUES):
            for v in V_VALUES:
                for yaw in YAW_VALUES:
                    obs = reset_at_state(eval_env, beta, r, v, yaw)

                    done, trunc = False, False
                    steps = 0
                    info = {"recovered": False}

                    while not done and not trunc:
                        action = controller.get_action(obs)
                        obs, _, done, trunc, info = eval_env.step(action)
                        steps += 1

                    if done and info.get("recovered", False):
                        recovery_counts[i, j] += 1
                        recovery_time_sums[i, j] += steps * dt
                        recovery_time_counts[i, j] += 1

                    episode += 1
                    if episode % 100 == 0:
                        print(f"  Progress: {episode}/{total_episodes} episodes")

    recovery_rates = recovery_counts / n_inner
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_recovery_times = np.where(recovery_time_counts > 0, recovery_time_sums / recovery_time_counts, np.nan)

    return recovery_rates, mean_recovery_times


def plot_recovery_heatmap(beta_values, r_values, recovery_rates, output_path):
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
    controller_label = f"{CONTROLLER_TYPE}" + (f" ({LEARNED_TYPE})" if LEARNED_TYPE else "")
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


def print_metrics(recovery_rates, mean_recovery_times):
    """Print aggregate recovery metrics."""
    n_inner = len(V_VALUES) * len(YAW_VALUES)

    # Per-episode binary outcomes across all 900 episodes
    # recovery_rates is per-cell fraction; overall rate is mean of cells
    overall_rate = np.mean(recovery_rates) * 100
    overall_std = np.std(recovery_rates) * 100

    # Recovery times (only successful episodes)
    valid_times = mean_recovery_times[~np.isnan(mean_recovery_times)]

    print("\n" + "=" * 50)
    print("RECOVERY METRICS")
    print("=" * 50)
    print(f"Grid: {len(BETA_VALUES)}x{len(R_VALUES)} (beta x r), {n_inner} (v, yaw) combos per cell")
    print(f"Total episodes: {len(BETA_VALUES) * len(R_VALUES) * n_inner}")
    print(f"Overall recovery rate: {overall_rate:.1f}% (std across cells: {overall_std:.1f}%)")

    if len(valid_times) > 0:
        print(f"Mean recovery time:   {np.mean(valid_times):.3f} s")
        print(f"Median recovery time: {np.median(valid_times):.3f} s")
        print(f"Std recovery time:    {np.std(valid_times):.3f} s")
    else:
        print("No successful recoveries recorded.")
    print("=" * 50)


def main():
    print_header("Beta-R Recovery Success Heatmap Analysis")

    proj_root, _ = get_output_dirs()

    controller = create_controller(
        CONTROLLER_TYPE,
        model_path=proj_root + MODEL_PATH if CONTROLLER_TYPE == "learned" else None,
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

    recovery_rates, mean_recovery_times = run_grid_evaluation(eval_env, controller)

    output_path = (
        f"{proj_root}/tests/test_figures/"
        f"beta_r_recovery_heatmap_{CONTROLLER_TYPE}"
        f"{'_' + LEARNED_TYPE if CONTROLLER_TYPE == 'learned' else ''}_policy.png"
    )
    plot_recovery_heatmap(BETA_VALUES, R_VALUES, recovery_rates, output_path)

    print_metrics(recovery_rates, mean_recovery_times)

    eval_env.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
