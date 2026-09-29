"""
Beta-R Recovery Success Heatmap Analysis

Evaluates a controller's recovery capability across a grid of initial (beta, r)
conditions, averaging over velocity and yaw variations. Outputs:
  - Heatmap of recovery success rate per (beta, r) cell
  - Aggregate recovery metrics (rate, time stats)
  - For learned controllers compared against a Stanley baseline:
    - Recovery rate on Stanley-recoverable states and percentage improvement
    - Recovery time stats for states recovered by learned but NOT by Stanley

Usage — Stanley baseline (must run first to generate baseline .npz):
    python examples/analysis/beta_r_avg_plot.py --controller_type stanley

Usage — Learned (PPO) controller:
    python examples/analysis/beta_r_avg_plot.py --controller_type learned --learned_type recover --run_id <wandb_run_id>

Add --no-title to omit the figure title (useful when embedding in LaTeX, where the
caption carries the description).

Output saved to figures/analysis/recover_heatmap/<controller_type or run_id>/
"""

import argparse
import os

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from examples.analysis.fig_format import CMAP, latex_style
from examples.controllers import create_controller
from gymkhana.envs.gymkhana_env import GKEnv
from train.config.env_config import get_env_id
from train.train_utils import get_output_dirs, print_header

# ==========================
# CLASSIC CONTROLLERS
# ==========================

# CONTROLLER_TYPE = "stanley"
# DESC = "stanley"
# LEARNED_TYPE = ""
# RUN_ID = ""

# CONTROLLER_TYPE = "stmpc"
# DESC = "Single-track MPC controller with acados + CasAdi, ported from ForzaETH"
# LEARNED_TYPE = ""
# RUN_ID = ""

# ==========================
# LEARNED RL CONTROLLERS
# ==========================

CONTROLLER_TYPE = "learned"

# LEARNED_TYPE = "recover"
# RUN_ID = "f1mgktxe"
# DESC = "FINAL transfer model - drift model 8ncsx1rk retrained with Fine-Tuning with Fresh Optimizer + LR Reset + log_std reset + Critic Reinitialization. No curriculum learning, small beta-r initial ranges, no Euclidean reward"

# LEARNED_TYPE = "recover"
# RUN_ID = "irdqwnhp"
# DESC = "FINAL recovering model - no Euclidean reward"

# LEARNED_TYPE = "drift"
# RUN_ID = "8ncsx1rk"
# DESC = "FINAL drift model - CW & CCW on Drift_large, with `sparse_width_obs` = True, 3rd train with seed 123"

# LEARNED_TYPE = "recover"
# RUN_ID = "8m5f957h"
# DESC = "ABLATION recovering model - Euclidean reward, larger beta, r ranges"

LEARNED_TYPE = "recover"
RUN_ID = "qhj88o3r"
DESC = "ABLATION recovering model - Euclidean reward, curriculum learning, larger beta, r ranges"


# LEARNED_TYPE = "transfer"
# RUN_ID = "29hhclk1"
# DESC = "transfer model - tgkpbyrh retrained"

# LEARNED_TYPE = "drift"
# RUN_ID = "tgkpbyrh"
# DESC = "drift model - same changes as 6nyece2v, new params for new covers (f1tenth_std_new_tire_covers_params)"

# LEARNED_TYPE = "drift"
# RUN_ID = "vn7fzj2u"
# DESC = "drift model - same changes as 3g2kzk8l, but more DR pertubration (0.1 not 0.02) on all params"

# LEARNED_TYPE = "drift"
# RUN_ID = "gs9xfxhy"
# DESC = "drift model - new ID'd params, 'speed' not 'accl' control, drift_real obs space, log_std fix applied"

# LEARNED_TYPE = "drift"
# RUN_ID = "wx0w5eqr"
# DESC = "drift model - CW & CCW on track pool of [Drift_large, Drift_large_mirror, Drift2, Drift2mirror], with `sparse_width_obs` = True"

# LEARNED_TYPE = "recover"
# RUN_ID = "3y11m6mk"
# DESC = "drift model - CW & CCW on track pool of [Drift_large, Drift_large_mirror], with `sparse_width_obs` = True"

# LEARNED_TYPE = "recover"
# RUN_ID = "zv45r303"
# DESC = "transfer model - drift model bsoh5xyb retrained with Fine-Tuning with Fresh Optimizer + LR Reset + log_std reset with --m f. No curriculum learning, small beta-r initial ranges, no Euclidean reward"

# LEARNED_TYPE = "transfer"
# RUN_ID = "c76n9olh"
# DESC = "transfer model - drift model bsoh5xyb retrained by loading and continuing training with --m c. No curriculum learning, small beta-r initial ranges, no Euclidean reward "

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
# RUN_ID = "50x16c1d"
# DESC = "recovering model - Euclidean reward, curriculum learning, smaller beta, r ranges"

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

# LEARNED_TYPE = "recover"
# RUN_ID = "65daon9y"
# DESC = "recovering model - no Euclidean reward"

# LEARNED_TYPE = "recover"
# RUN_ID = "cn3rnv9v"
# DESC = "recovering model - Euclidean reward, larger beta, r ranges"

# LEARNED_TYPE = "recover"
# RUN_ID = "33t51j8g"
# DESC = "recovering model - Euclidean reward, curriculum learning, smaller beta, r ranges"

# LEARNED_TYPE = "recover"
# RUN_ID = "l2ww5lee"
# DESC = "transfer model - drift model bsoh5xyb retrained with Fine-Tuning with Fresh Optimizer + LR Reset + log_std reset + Critic Reinitialization. \nNo curriculum learning, small beta-r initial ranges, no Euclidean reward"

# LEARNED_TYPE = "recover"
# RUN_ID = "gpti2zb1"
# DESC = "recovering model - original with Euclidean reward, small beta, r ranges, and success reward 50,  collision penalty -100"

# Vehicle parameters, pinned here so edits to train/config/env_config.py or gymkhana/presets.py don't change them
EVAL_PARAMS = GKEnv.f1tenth_std_drift_bias_params()

# Learned-policy obs type and control input, pinned to what the evaluated model was trained with.
# Models trained before 2026-05-14 use "drift" + accl; later ones use "drift_real" + speed.
# Classic controllers set their own and ignore these.
EVAL_OBS_TYPE = "drift"
EVAL_CONTROL_INPUT = ["accl", "steering_angle"]

S = 96  # Arc length on IMS straight section
SEED = 42

# Grid parameters (radians / rad/s / m/s)
BETA_VALUES = np.linspace(-1.39, 1.39, 7)  # 7 points, +/-80 deg
R_VALUES = np.linspace(-13, 13, 7)  # 7 points, +/-745 deg/s
V_VALUES = np.linspace(4, 5, 2)  # 2 points: [4, 5]
YAW_VALUES = np.linspace(-0.17, 0.17, 3)  # 3 points: [-10, 0, +10] deg


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
    controller.on_reset(obs)

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
        recovery_times: list of recovery times (seconds), one per successful episode
        successful_states: list of (beta, r, v, yaw) tuples that recovered
        stanley_total: number of evaluated states in the Stanley-recoverable set
        stanley_times: recovery times on Stanley-recoverable states
        non_stanley_times: recovery times on states Stanley did not recover
    """
    n_beta = len(BETA_VALUES)
    n_r = len(R_VALUES)
    n_v = len(V_VALUES)
    n_yaw = len(YAW_VALUES)
    n_inner = n_v * n_yaw
    total_episodes = n_beta * n_r * n_inner

    recovery_counts = np.zeros((n_beta, n_r))
    recovery_times = []
    successful_states = []

    # Stanley-subset accumulators
    stanley_total_count = 0
    stanley_recovery_times = []
    non_stanley_recovery_times = []

    episode = 0
    for i, beta in enumerate(BETA_VALUES):
        for j, r in enumerate(R_VALUES):
            for v in V_VALUES:
                for yaw in YAW_VALUES:
                    recovered, time_s = run_episode(eval_env, controller, beta, r, v, yaw)

                    if recovered:
                        recovery_counts[i, j] += 1
                        recovery_times.append(time_s)
                        successful_states.append((beta, r, v, yaw))

                    # Track Stanley-subset and non-Stanley metrics
                    if stanley_states is not None:
                        if (beta, r, v, yaw) in stanley_states:
                            stanley_total_count += 1
                            if recovered:
                                stanley_recovery_times.append(time_s)
                        elif recovered:
                            non_stanley_recovery_times.append(time_s)

                    episode += 1
                    if episode % 10 == 0:
                        print(f"  Progress: {episode}/{total_episodes} episodes")

    return (
        recovery_counts / n_inner,
        recovery_times,
        successful_states,
        stanley_total_count,
        stanley_recovery_times,
        non_stanley_recovery_times,
    )


# Luminance below which white text out-contrasts black, from equating the two WCAG
# contrast ratios: 1.05 / (L + 0.05) = (L + 0.05) / 0.05  =>  L = sqrt(0.0525) - 0.05.
_WHITE_TEXT_MAX_LUMINANCE = 0.0525**0.5 - 0.05


def contrasting_text_color(rgba):
    """Return "white" or "black", whichever contrasts more against the given color.

    Keys off WCAG relative luminance rather than a raw data threshold, so the
    choice stays correct for any colormap.
    """

    def linearize(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (linearize(c) for c in rgba[:3])
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "white" if luminance <= _WHITE_TEXT_MAX_LUMINANCE else "black"


def plot_recovery_heatmap(beta_values, r_values, recovery_rates, output_path, controller_label="", show_title=True):
    """Plot a heatmap of recovery success rate across the (beta, r) grid.

    Args:
        beta_values: Sideslip angles in radians, one per grid column.
        r_values: Yaw rates in rad/s, one per grid row.
        recovery_rates: (len(beta_values), len(r_values)) array of rates in [0, 1].
        output_path: File path to write the figure to.
        controller_label: Controller name shown in the title.
        show_title: When False, the title is omitted entirely (for LaTeX figures
            where the caption carries the description).
    """
    with latex_style():
        _, ax = plt.subplots(figsize=(10, 8))

        beta_deg = np.rad2deg(beta_values)
        r_deg = np.rad2deg(r_values)
        pct = recovery_rates * 100

        im = ax.imshow(
            pct.T,
            origin="lower",
            aspect="auto",
            cmap=CMAP,
            vmin=0,
            vmax=100,
            interpolation="bilinear",
        )

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Recovery rate [%]")

        # Annotate cells at integer positions (aligned with imshow pixels). Bilinear
        # interpolation is exact at those positions, so the mapped cell value is the
        # background color actually under the label.
        for i in range(len(beta_deg)):
            for j in range(len(r_deg)):
                val = pct[i, j]
                color = contrasting_text_color(im.cmap(im.norm(val)))
                ax.text(
                    i,
                    j,
                    f"{val:.0f}%",
                    ha="center",
                    va="center",
                    family="sans-serif",  # set off from the serif body text
                    fontweight="bold",
                    fontsize=11,
                    color=color,
                )

        # Use degree tick labels on integer positions
        ax.set_xticks(range(len(beta_deg)))
        ax.set_xticklabels([f"{v:.0f}" for v in beta_deg])
        ax.set_yticks(range(len(r_deg)))
        ax.set_yticklabels([f"{v:.0f}" for v in r_deg])

        ax.set_xlabel(r"Sideslip angle $\beta$ [deg]")
        ax.set_ylabel(r"Yaw rate $r$ [deg/s]")
        if show_title:
            ax.set_title(
                f"Recovery Success Rate: {controller_label}\n"
                rf"Averaged over $v=[{V_VALUES[0]:.0f}, {V_VALUES[-1]:.0f}]$ m/s, "
                rf"$\psi=[{np.rad2deg(YAW_VALUES[0]):.0f}, {np.rad2deg(YAW_VALUES[-1]):.0f}]$ deg"
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
        recovery_times: Recovery times (seconds), one per successful episode.
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


def grid_matches(data):
    """Return True if the grid stored in a loaded .npz matches the current grid."""
    return (
        np.allclose(data["beta_values"], BETA_VALUES)
        and np.allclose(data["r_values"], R_VALUES)
        and np.allclose(data["v_values"], V_VALUES)
        and np.allclose(data["yaw_values"], YAW_VALUES)
    )


def save_grid_cache(
    path, recovery_rates, recovery_times, successful_states, stanley_total, stanley_times, non_stanley_times
):
    """Save run_grid_evaluation results to an .npz cache file."""
    np.savez(
        path,
        recovery_rates=recovery_rates,
        recovery_times=np.array(recovery_times),
        successful_states=np.array(successful_states) if successful_states else np.empty((0, 4)),
        stanley_total=np.array(stanley_total),
        stanley_times=np.array(stanley_times),
        non_stanley_times=np.array(non_stanley_times),
        beta_values=BETA_VALUES,
        r_values=R_VALUES,
        v_values=V_VALUES,
        yaw_values=YAW_VALUES,
    )
    print(f"Grid results cached to: {path}")


def load_grid_cache(path):
    """Load and validate a cached grid_results.npz file.

    Returns the six run_grid_evaluation outputs if the grid parameters match,
    or raises ValueError if the cache was built with different grid settings
    or predates per-episode recovery times.
    """
    data = np.load(path)
    if "recovery_times" not in data:
        raise ValueError(
            f"Cache {path} has no per-episode recovery times (old format). Re-run with --no-cache to regenerate."
        )
    if not grid_matches(data):
        raise ValueError(
            f"Cached grid parameters in {path} do not match the current BETA/R/V/YAW_VALUES. "
            "Delete the cache file or re-run with --no-cache to regenerate."
        )

    successful_states = [tuple(row) for row in data["successful_states"]]
    return (
        data["recovery_rates"],
        list(data["recovery_times"]),
        successful_states,
        int(data["stanley_total"]),
        list(data["stanley_times"]),
        list(data["non_stanley_times"]),
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Beta-R Recovery Success Heatmap Analysis")
    parser.add_argument(
        "--controller_type", default=CONTROLLER_TYPE, help="Controller type: 'learned', 'stanley', 'steer', 'stmpc'"
    )
    parser.add_argument("--learned_type", default=LEARNED_TYPE, help="Learned model type: 'drift' or 'recover'")
    parser.add_argument("--run_id", default=RUN_ID, help="Wandb run ID for the learned model")
    parser.add_argument("--desc", default=DESC, help="Short description of the model")
    parser.add_argument("--no-cache", action="store_true", help="Ignore existing cache and re-run grid evaluation")
    parser.add_argument(
        "--no-title",
        action="store_true",
        help="Omit the figure title (for LaTeX figures where the caption describes it)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    controller_type = args.controller_type
    learned_type = args.learned_type
    run_id = args.run_id
    desc = args.desc
    no_cache = args.no_cache

    print_header("Beta-R Recovery Success Heatmap Analysis")
    print(f"Controller: {controller_type}, Learned type: {learned_type}, Run ID: {run_id}")
    print(f"Description: {desc}")

    proj_root, _ = get_output_dirs()

    subfolder = (
        f"{proj_root}/figures/analysis/recover_heatmap/{run_id}"
        if controller_type == "learned"
        else f"{proj_root}/figures/analysis/recover_heatmap/{controller_type}"
    )
    os.makedirs(subfolder, exist_ok=True)
    cache_path = os.path.join(subfolder, "grid_results.npz")

    # Attempt to load from cache
    if not no_cache and os.path.exists(cache_path):
        print(f"\nFound cached grid results at: {cache_path}")
        recovery_rates, recovery_times, successful_states, stanley_total, stanley_times, non_stanley_times = (
            load_grid_cache(cache_path)
        )
        print("Cache loaded successfully — skipping grid evaluation.")
        eval_env = None
    else:
        if no_cache:
            print("\n--no-cache flag set; ignoring existing cache and re-running grid evaluation.")

        model_path = f"{proj_root}/outputs/downloads/{run_id}/model.zip" if controller_type == "learned" else None
        controller = create_controller(
            controller_type,
            model_path=model_path,
            map="IMS",
        )

        config = controller.get_env_config()
        config["training_mode"] = "recover"
        config["params"] = EVAL_PARAMS
        if controller_type == "learned":
            config["observation_config"] = {"type": EVAL_OBS_TYPE}
            config["control_input"] = EVAL_CONTROL_INPUT

        eval_env = gym.make(
            get_env_id(),
            config=config,
            render_mode=None,
        )

        controller.initialize(eval_env)

        np.random.seed(SEED)

        # Load Stanley recovery states for learned controllers
        stanley_states_path = f"{proj_root}/figures/analysis/recover_heatmap/stanley_recovery_states.npz"

        stanley_states = None
        if controller_type != "stanley" and os.path.exists(stanley_states_path):
            data = np.load(stanley_states_path)
            if grid_matches(data):
                stanley_states = set(map(tuple, data["states"]))
                print(f"Loaded {len(stanley_states)} Stanley recovery states for baseline comparison")
            else:
                raise ValueError(
                    "Stanley states were computed with a different grid — "
                    "re-run with --controller_type stanley to regenerate."
                )
        elif controller_type != "stanley":
            raise FileNotFoundError(
                f"{stanley_states_path} not found — run Stanley evaluation first for baseline metrics"
            )

        print(
            f"\nRunning grid evaluation: {len(BETA_VALUES)}x{len(R_VALUES)} cells, "
            f"{len(V_VALUES) * len(YAW_VALUES)} episodes per cell..."
        )

        recovery_rates, recovery_times, successful_states, stanley_total, stanley_times, non_stanley_times = (
            run_grid_evaluation(eval_env, controller, stanley_states=stanley_states)
        )

        # Cache results immediately after evaluation
        save_grid_cache(
            cache_path,
            recovery_rates,
            recovery_times,
            successful_states,
            stanley_total,
            stanley_times,
            non_stanley_times,
        )

        # Save Stanley successful states with grid parameters
        if controller_type == "stanley" and successful_states:
            np.savez(
                stanley_states_path,
                states=np.array(successful_states),
                beta_values=BETA_VALUES,
                r_values=R_VALUES,
                v_values=V_VALUES,
                yaw_values=YAW_VALUES,
            )
            print(f"Saved {len(successful_states)} Stanley recovery states to: {stanley_states_path}")

        eval_env.close()

    # Plotting and metrics (always runs, whether from cache or fresh)
    output_path = (
        f"{subfolder}/beta_r_recovery_heatmap_{controller_type}{'_' + learned_type if learned_type else ''}_policy.pdf"
    )
    controller_label = f"{controller_type}" + (f" ({learned_type})" if learned_type else "")
    plot_recovery_heatmap(
        BETA_VALUES,
        R_VALUES,
        recovery_rates,
        output_path,
        controller_label=controller_label,
        show_title=not args.no_title,
    )

    n_inner = len(V_VALUES) * len(YAW_VALUES)
    n_total = len(BETA_VALUES) * len(R_VALUES) * n_inner
    n_recovered_total = len(recovery_times)
    overall_rate = n_recovered_total / n_total * 100
    overall_std = np.std(recovery_rates) * 100
    save_metrics(
        "RECOVERY METRICS",
        [
            f"Grid: {len(BETA_VALUES)}x{len(R_VALUES)} (beta x r), {n_inner} (v, yaw) combos per cell",
            f"Total episodes: {n_total}",
            f"Overall recovery rate: {n_recovered_total}/{n_total} ({overall_rate:.1f}%, std across cells: {overall_std:.1f}%)",
        ],
        recovery_times,
        os.path.join(subfolder, "metrics.txt"),
        desc=desc,
    )

    # Stanley-baseline metrics for learned controllers
    if stanley_total > 0:
        n_recovered = len(stanley_times)
        rate = n_recovered / stanley_total * 100
        improvement = (n_recovered_total - stanley_total) / stanley_total * 100
        save_metrics(
            "STANLEY-BASELINE RECOVERY METRICS",
            [
                f"Stanley recovery states evaluated: {stanley_total}",
                f"Learned policy recovered: {n_recovered}/{stanley_total} ({rate:.1f}%)",
                f"Recovery rate improvement over Stanley: {improvement:+.1f}% ({n_recovered_total} vs {stanley_total} total recoveries)",
            ],
            stanley_times,
            os.path.join(subfolder, "stanley_recovery_states_metrics.txt"),
            desc=desc,
        )

        # Non-Stanley recovery time metrics (states learned recovered but Stanley didn't)
        save_metrics(
            "NON-STANLEY RECOVERY TIME METRICS",
            [
                f"States recovered by learned but NOT by Stanley: {len(non_stanley_times)}",
            ],
            non_stanley_times,
            os.path.join(subfolder, "non_stanley_recovery_states_metrics.txt"),
            desc=desc,
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
