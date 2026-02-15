"""
Beta-R Phase Plane Convergence Analysis

This script tests a controller's stability by initializing the vehicle
at extreme beta-r states in all 4 quadrants of the phase plane, then observing
convergence toward the stable equilibrium at (0, 0).

Usage:
    1. Edit MODEL_PATH, S, TARGET_SPEED, INIT_BETA_R as needed
    2. Run: python examples/analysis/beta_r_traj_IMS_plot.py
    3. View generated plot in beta_r_convergence.png
"""

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from examples.controllers import create_controller
from train.config.env_config import get_env_id
from train.train_utils import get_output_dirs, print_header

CONTROLLER_TYPE = "learned"  # "steer", "stable", or "learned"
MODEL_PATH = "/outputs/downloads/178a1a5l/model.zip"

# Initial vehicle position on straight section
S = 96  # Arc length [m]
TARGET_SPEED = 4  # Initial velocity [m/s]

# Initial beta, r values (permutation of +/- values is used for 4 initial states)
BETA = 35
R = 150

# Convergence criteria (in degrees and deg/s)
CONVERGENCE_BETA_THRESHOLD = 2.0  # [deg]
CONVERGENCE_R_THRESHOLD = 10.0  # [deg/s]
MAX_TIMESTEPS = 1000


def reset_at_beta_r(eval_env, beta_deg: float, r_deg: float):
    """Reset environment with specified beta (sideslip) and r (yaw rate).

    Args:
        eval_env: The gym environment
        beta_deg: Sideslip angle in degrees
        r_deg: Yaw rate in deg/s

    Returns:
        Initial observation
    """
    # Convert degrees to radians for internal state representation
    beta_rad = np.deg2rad(beta_deg)
    r_rad = np.deg2rad(r_deg)

    # Get Cartesian position from Frenet coordinates
    x, y, yaw = eval_env.unwrapped.track.frenet_to_cartesian(S, ey=0, ephi=0)

    init_state = np.array(
        [
            [
                x,  # x position (Cartesian coordinates)
                y,  # y position (Cartesian coordinates)
                0.0,  # delta (steering angle) [rad]
                TARGET_SPEED,  # v (velocity) [m/s]
                yaw,  # yaw angle (Cartesian orientation) [rad]
                r_rad,  # yaw_rate [rad/s]
                beta_rad,  # slip_angle [rad]
            ]
        ]
    )

    np.random.seed()
    obs, _ = eval_env.reset(options={"states": init_state})

    return obs


def plot_beta_r_convergence(trajectories, output_filename, target_speed):
    """Plot multiple beta-r trajectories showing convergence to equilibrium.

    Args:
        trajectories: List of dicts with 'beta', 'r', 'time', 'init' keys
        output_filename: Path to save the plot
        target_speed: Initial velocity in m/s
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot each trajectory with time-based coloring
    for i, traj in enumerate(trajectories):
        beta_deg = traj["beta"]
        r_deg = traj["r"]
        time = traj["time"]
        init_beta, init_r = traj["init"]

        # Create line segments colored by normalized time (0→1)
        points = np.array([beta_deg, r_deg]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Normalize time to [0, 1] for consistent color mapping
        norm_time = (time - time[0]) / (time[-1] - time[0]) if len(time) > 1 else time

        lc = LineCollection(segments, cmap="plasma", linewidths=2.0, alpha=0.8)
        lc.set_array(norm_time[:-1])
        ax.add_collection(lc)

        # Mark initial point
        ax.scatter(
            beta_deg[0],
            r_deg[0],
            s=150,
            c="green",
            marker="o",
            edgecolors="black",
            linewidths=2,
            zorder=10,
            label=f"Start {i + 1}: ({init_beta:.0f}°, {init_r:.0f}°/s)" if i < 4 else "",
        )

        # Mark final point
        ax.scatter(
            beta_deg[-1],
            r_deg[-1],
            s=150,
            c="red",
            marker="X",
            edgecolors="black",
            linewidths=2,
            zorder=10,
        )

    # Set axis limits with some padding
    all_beta = np.concatenate([t["beta"] for t in trajectories])
    all_r = np.concatenate([t["r"] for t in trajectories])
    beta_margin = (all_beta.max() - all_beta.min()) * 0.1
    r_margin = (all_r.max() - all_r.min()) * 0.1

    ax.set_xlim([all_beta.min() - beta_margin, all_beta.max() + beta_margin])
    ax.set_ylim([all_r.min() - r_margin, all_r.max() + r_margin])

    # Labels and styling
    ax.set_xlabel("Beta (sideslip angle) [deg]", fontsize=14, fontweight="bold")
    ax.set_ylabel("R (yaw rate) [deg/s]", fontsize=14, fontweight="bold")
    ax.set_title(
        f"Phase Plane Convergence: Beta vs R (v₀ = {target_speed} m/s)\nGreen circles = Start, Red X = End",
        fontsize=16,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.axhline(y=0, color="k", linestyle="-", linewidth=1.5, alpha=0.7)
    ax.axvline(x=0, color="k", linestyle="-", linewidth=1.5, alpha=0.7)

    # Colorbar for time progression
    sm = plt.cm.ScalarMappable(cmap="plasma", norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Normalized time progression (0 → 1)", fontsize=12, fontweight="bold")

    # Add metadata
    total_points = sum(len(t["beta"]) for t in trajectories)
    fig.suptitle(
        f"{len(trajectories)} trajectories, {total_points} total points",
        fontsize=11,
        y=0.995,
    )

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")

    print(f"\nPlot saved to: {output_filename}")
    for i, traj in enumerate(trajectories):
        print(
            f"  Trajectory {i + 1}: {len(traj['beta'])} points, "
            f"converged from ({traj['init'][0]:.0f}°, {traj['init'][1]:.0f}°/s) "
            f"to ({traj['beta'][-1]:.1f}°, {traj['r'][-1]:.1f}°/s)"
        )

    plt.show()


def main():
    print_header("Phase Plane Convergence Analysis for Trained PPO Drift Policy")

    proj_root, _ = get_output_dirs()

    # Create controller - single unified approach for all types!
    controller = create_controller(
        CONTROLLER_TYPE,
        target_speed=TARGET_SPEED,
        model_path=proj_root + MODEL_PATH if CONTROLLER_TYPE == "learned" else None,
        map="IMS",
    )

    # Get controller-specific config and create environment
    config = controller.get_env_config()
    eval_env = gym.make(
        get_env_id(),
        config=config,
        render_mode="human",
    )

    # Four initial states of beta-r in each of the 4 graph quadrants (degrees, deg/s)

    # Store trajectories separately
    trajectories = []

    init_beta_r = [(BETA, R), (-BETA, R), (BETA, -R), (-BETA, -R)]

    print(f"\nTesting {len(init_beta_r)} initial conditions...")
    print(f"Convergence criteria: |beta| < {CONVERGENCE_BETA_THRESHOLD}°, |r| < {CONVERGENCE_R_THRESHOLD}°/s")
    print(f"Max timesteps per trajectory: {MAX_TIMESTEPS}\n")

    for traj_idx, (init_beta, init_r) in enumerate(init_beta_r):
        print(f"\n--- Trajectory {traj_idx + 1}/{len(init_beta_r)} ---")
        print(f"Initial state: beta={init_beta}°, r={init_r}°/s")

        # Reset at specific beta-r state
        obs = reset_at_beta_r(eval_env, init_beta, init_r)

        # Data collection for this trajectory
        traj_beta = []
        traj_r = []
        traj_time = []

        # Record initial state (timestep 0) before any actions
        agent = eval_env.unwrapped.sim.agents[0]
        std_state = agent.standard_state
        beta_rad = std_state["slip"]
        r_rad = std_state["yaw_rate"]
        traj_beta.append(np.rad2deg(beta_rad))
        traj_r.append(np.rad2deg(r_rad))
        traj_time.append(0)

        converged = False
        timestep = 0
        for timestep in range(1, MAX_TIMESTEPS):
            # Get action from controller - unified interface!
            action = controller.get_action(obs)

            obs, reward, done, trunc, info = eval_env.step(action)
            eval_env.render()

            # Extract state
            agent = eval_env.unwrapped.sim.agents[0]
            std_state = agent.standard_state

            # Get beta and r (in radians, rad/s)
            beta_rad = std_state["slip"]
            r_rad = std_state["yaw_rate"]

            # Convert to degrees for storage and checking
            beta_deg = np.rad2deg(beta_rad)
            r_deg = np.rad2deg(r_rad)

            # Append data
            traj_beta.append(beta_deg)
            traj_r.append(r_deg)
            traj_time.append(timestep)

            print(f"\rStep {timestep}: Beta={beta_deg:6.2f}°, R={r_deg:6.1f}°/s", end="")

            # Check convergence
            if abs(beta_deg) < CONVERGENCE_BETA_THRESHOLD and abs(r_deg) < CONVERGENCE_R_THRESHOLD:
                converged = True
                print(f"\n✓ Converged at step {timestep}")
                break

        if not converged and timestep == MAX_TIMESTEPS - 1:
            print("\n⚠ Max timesteps reached without convergence")

        # Store trajectory
        trajectories.append(
            {
                "beta": np.array(traj_beta),
                "r": np.array(traj_r),
                "time": np.array(traj_time),
                "init": (init_beta, init_r),
                "converged": converged,
            }
        )

    eval_env.close()

    # Generate plot
    output_filename = proj_root + "/tests/test_figures/beta_r_convergence_" + CONTROLLER_TYPE + "_policy.png"
    print("\n\nGenerating convergence plot...")
    plot_beta_r_convergence(trajectories, output_filename, TARGET_SPEED)

    print("\nDone!")


if __name__ == "__main__":
    main()
