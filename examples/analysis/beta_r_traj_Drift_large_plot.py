"""
Beta (Sideslip) and R (Yaw Rate) Trajectory Plot

This scrip runs a learned policy in simulation for COLLECTION_LAPS laps,
following WARMUP_LAPS warm-up laps, and collects the vehicle beta, r values.
Optionally, the data can be filtered by arc length s-range (START_S to END_S).
The data is then plotted, coloured by arc-length, to visualize the drifting
behavior in the beta-r plane.

Usage:
    1. Edit MODEL_PATH, START_S, END_S, WARMUP_LAPS, COLLECTION_LAPS
    2. Run: python examples/analysis/beta_r_traj_plot.py
    3. View generated plots in beta_r_phase_plane.png
"""

import os

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from train.config.env_config import get_drift_test_config, get_env_id
from train.train_utils import get_output_dirs, print_header

MODEL_PATH = "/outputs/downloads/flt3rzle/model.zip"
CONFIG = get_drift_test_config()
CONFIG["render_arc_length_annotations"] = True
CONFIG["max_episode_steps"] = 20000
CONFIG["track_direction"] = "reverse"  # Set to "reverse" to see the graph in reverse

# Beta-R trajectory filtering parameters
START_S = 0.0  # Start arc length [meters]
END_S = None  # End arc length [meters] - None means full track

# Lap-based data collection parameters
WARMUP_LAPS = 2  # Number of warm-up laps before recording
COLLECTION_LAPS = 2  # Number of laps to record data
AGENT_IDX = 0  # Ego agent index (always 0 for single-agent)


def plot_beta_r_phase_plane(beta, r, s, output_filename, start_s, end_s):
    """Create beta vs r phase plane plot with connected trajectory lines."""
    from matplotlib.collections import LineCollection

    # Convert to degrees
    beta_deg = np.rad2deg(beta)
    r_deg = np.rad2deg(r)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create line segments colored by arc length
    points = np.array([beta_deg, r_deg]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = LineCollection(segments, cmap="turbo", linewidths=1.5, alpha=0.8)
    lc.set_array(s[:-1])  # Color by arc length
    line = ax.add_collection(lc)

    # Set axis limits based on data
    ax.set_xlim([beta_deg.min() - 1, beta_deg.max() + 1])
    ax.set_ylim([r_deg.min() - 5, r_deg.max() + 5])

    ax.set_xlabel("Beta (sideslip angle) [deg]", fontsize=14)
    ax.set_ylabel("R (yaw rate) [deg/s]", fontsize=14)
    ax.set_title("Phase Plane: Beta vs R", fontsize=16, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="k", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axvline(x=0, color="k", linestyle="--", linewidth=0.8, alpha=0.5)

    # Colorbar for arc length
    cbar = plt.colorbar(line, ax=ax)
    cbar.set_label("Arc length s [m]", fontsize=12)

    # Metadata
    fig.suptitle(f"s-range: [{start_s:.1f}, {end_s:.1f}]m, {len(beta)} points", fontsize=12, y=0.98)

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")

    print(f"\nPlot saved to: {output_filename}")
    print(f"  Beta range: [{np.min(beta_deg):.2f}, {np.max(beta_deg):.2f}] deg")
    print(f"  R range: [{np.min(r_deg):.2f}, {np.max(r_deg):.2f}] deg/s")

    plt.show()


def main():
    print_header("Beta-R Phase Plane Plot")

    proj_root, _ = get_output_dirs()

    # Load model
    model = PPO.load(proj_root + MODEL_PATH, print_system_info=True, device="cpu")

    # Create environment (keep render_mode="human")
    eval_env = gym.make(get_env_id(), config=CONFIG, render_mode="human")

    # Get track length for filtering
    track_length = eval_env.unwrapped.track.centerline.spline.s[-1]
    end_s = END_S if END_S is not None else track_length

    print(f"Track length: {track_length:.2f}m")
    print(f"Filtering s-range: [{START_S:.2f}, {end_s:.2f}]m")

    # Data collection lists
    beta_list = []
    r_list = []
    s_list = []

    np.random.seed()
    obs, info = eval_env.reset()

    # Run for MAX_STEPS and collect data
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, trunc, info = eval_env.step(action)
        eval_env.render()

        current_laps = eval_env.unwrapped.lap_counts[AGENT_IDX]
        print(f"\rLaps: {current_laps}", end="")

        if current_laps >= WARMUP_LAPS:
            # Extract state from environment
            agent = eval_env.unwrapped.sim.agents[0]
            std_state = agent.standard_state

            # Get beta and r
            beta = std_state["slip"]
            r = std_state["yaw_rate"]

            # Get Frenet s-coordinate
            x, y, theta = std_state["x"], std_state["y"], std_state["yaw"]
            s, ey, ephi = eval_env.unwrapped.track.cartesian_to_frenet(x, y, theta, use_raceline=False)

            # Collect all data (filter later)
            beta_list.append(beta)
            r_list.append(r)
            s_list.append(s)

        if current_laps >= WARMUP_LAPS + COLLECTION_LAPS:
            print("\nData collection complete.")
            break

        if done or trunc:
            obs, info = eval_env.reset()

    eval_env.close()

    # Filter data by s-range
    beta_arr = np.array(beta_list)
    r_arr = np.array(r_list)
    s_arr = np.array(s_list)

    mask = (s_arr >= START_S) & (s_arr <= end_s)
    beta_filtered = beta_arr[mask]
    r_filtered = r_arr[mask]
    s_filtered = s_arr[mask]

    print(f"\nData collected: {len(beta_list)} total points")
    print(f"Data after filtering: {len(beta_filtered)} points in s-range")

    if len(beta_filtered) == 0:
        print("ERROR: No data in specified s-range!")
        return

    # Create plot
    output_dir = proj_root + "/figures/analysis"
    os.makedirs(output_dir, exist_ok=True)
    output_filename = output_dir + "/beta_r_phase_plane.png"
    print("\nGenerating phase plane plot...")
    plot_beta_r_phase_plane(beta_filtered, r_filtered, s_filtered, output_filename, START_S, end_s)

    print("\nDone!")


if __name__ == "__main__":
    main()
