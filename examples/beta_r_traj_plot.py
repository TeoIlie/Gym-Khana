import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
import matplotlib.pyplot as plt

from train.config.env_config import get_drift_test_config, get_env_id
from train.training_utils import get_output_dirs, print_header

MODEL_PATH = "/outputs/downloads/ol035sw5/model.zip"
CONFIG = get_drift_test_config()
CONFIG["render_arc_length_annotations"] = True

# Beta-R trajectory filtering parameters
START_S = 0.0  # Start arc length [meters]
END_S = None  # End arc length [meters] - None means full track
MAX_STEPS = 5000  # Number of steps to run


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
    model = PPO.load(proj_root + MODEL_PATH, print_system_info=True, device="auto")

    # Create environment (keep render_mode="human")
    eval_env = gym.make(get_env_id(), config=CONFIG, render_mode="human")

    # Get track length for filtering
    track_length = eval_env.track.centerline.spline.s[-1]
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
    for step in range(MAX_STEPS):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, trunc, info = eval_env.step(action)
        eval_env.render()

        # Extract state from environment
        agent = eval_env.sim.agents[0]
        std_state = agent.standard_state

        # Get beta and r
        beta = std_state["slip"]
        r = std_state["yaw_rate"]

        # Get Frenet s-coordinate
        x, y, theta = std_state["x"], std_state["y"], std_state["yaw"]
        s, ey, ephi = eval_env.track.cartesian_to_frenet(x, y, theta, use_raceline=False)

        # Collect all data (filter later)
        beta_list.append(beta)
        r_list.append(r)
        s_list.append(s)

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
    output_filename = proj_root + "/tests/test_figures/beta_r_phase_plane.png"
    print("\nGenerating phase plane plot...")
    plot_beta_r_phase_plane(beta_filtered, r_filtered, s_filtered, output_filename, START_S, end_s)

    print("\nDone!")


if __name__ == "__main__":
    main()
