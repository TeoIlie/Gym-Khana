"""Sim2Real trajectory comparison.

Replays control commands from a real car recording (100Hz NPZ) through the
gymkhana KS simulator and produces comparison plots.

Usage:
    python examples/analysis/traj_compare.py --path /path/to/bag_100Hz.npz
"""

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Sim2Real trajectory comparison")
    parser.add_argument("--path", required=True, help="Path to 100Hz resampled NPZ file")
    args = parser.parse_args()

    # Load real data
    data = np.load(args.path)
    t = data["t"]
    cmd_speed = data["cmd_speed"]
    cmd_steer = data["cmd_steer"]
    vicon_x = data["vicon_x"]
    vicon_y = data["vicon_y"]
    vicon_yaw = data["vicon_yaw"]
    vicon_body_vx = data["vicon_body_vx"]

    # Output directory
    stem = Path(args.path).stem
    out_dir = os.path.join("figures", "analysis", "traj_compare", stem)
    os.makedirs(out_dir, exist_ok=True)

    # Create environment
    config = {
        "model": "ks",
        "num_agents": 1,
        "timestep": 0.01,
        "map": "Spielberg_blank",
        "control_input": ["speed", "steering_angle"],
        "observation_config": {"type": "kinematic_state"},
        "normalize_obs": False,
        "normalize_act": False,
    }
    env = gym.make("gymkhana:gymkhana-v0", config=config)
    obs, _ = env.reset(options={"poses": np.array([[vicon_x[0], vicon_y[0], vicon_yaw[0]]])})

    # Record initial state
    agent_obs = obs["agent_0"]
    sim_x = [float(agent_obs["pose_x"])]
    sim_y = [float(agent_obs["pose_y"])]
    sim_vx = [float(agent_obs["linear_vel_x"])]
    sim_delta = [float(agent_obs["delta"])]
    sim_cmd_speed = [cmd_speed[0]]
    sim_cmd_steer = [cmd_steer[0]]

    # Replay commands
    for i in range(len(cmd_speed)):
        action = np.array([[cmd_steer[i], cmd_speed[i]]])
        obs, _, _, _, _ = env.step(action)

        agent_obs = obs["agent_0"]
        sim_x.append(float(agent_obs["pose_x"]))
        sim_y.append(float(agent_obs["pose_y"]))
        sim_vx.append(float(agent_obs["linear_vel_x"]))
        sim_delta.append(float(agent_obs["delta"]))
        sim_cmd_speed.append(cmd_speed[i])
        sim_cmd_steer.append(cmd_steer[i])

    env.close()

    # Convert to arrays
    sim_x = np.array(sim_x)
    sim_y = np.array(sim_y)
    sim_vx = np.array(sim_vx)
    sim_delta = np.array(sim_delta)
    sim_cmd_speed = np.array(sim_cmd_speed)
    sim_cmd_steer = np.array(sim_cmd_steer)

    # Time axis for sim: initial state at t=0, then one point per step
    n = len(sim_x)
    if n <= len(t):
        sim_t = np.concatenate([[0.0], t[: n - 1] + 0.01]) if n > 1 else np.array([0.0])
    else:
        sim_t = t[:n]
    # Simpler: sim has n points starting from t=0 with 0.01 spacing
    sim_t = np.arange(n) * 0.01

    # Trim real data to match sim duration
    n_real = min(len(t), n)

    # Summary
    final_err = np.hypot(vicon_x[n_real - 1] - sim_x[n_real - 1], vicon_y[n_real - 1] - sim_y[n_real - 1])
    print(f"Steps simulated: {n - 1}")
    print(f"Duration: {sim_t[-1]:.2f}s")
    print(f"Final position error (real vs sim): {final_err:.4f} m")

    # --- Plot 1: XY Trajectory ---
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(vicon_x[:n_real], vicon_y[:n_real], label="Real (Vicon)", linewidth=1.5)
    ax.plot(sim_x[:n_real], sim_y[:n_real], label="Sim (KS model)", linewidth=1.5, linestyle="--")
    ax.plot(vicon_x[0], vicon_y[0], "ko", markersize=8, label="Start")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("XY Trajectory — Real vs Sim")
    ax.set_aspect("equal")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "xy_trajectory.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Plot 2: Velocity ---
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(t[:n_real], cmd_speed[:n_real], label="Commanded speed", linewidth=1, alpha=0.7)
    ax.plot(t[:n_real], vicon_body_vx[:n_real], label="Real velocity (Vicon)", linewidth=1)
    ax.plot(sim_t[:n_real], sim_cmd_speed[:n_real], label="Sim commanded speed", linewidth=1, linestyle="--")
    ax.plot(sim_t[:n_real], sim_vx[:n_real], label="Sim actual velocity", linewidth=1.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Velocity (m/s)")
    ax.set_title("Velocity — Command vs Actual")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "velocity.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Plot 3: Steering ---
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(t[:n_real], cmd_steer[:n_real], label="Commanded steering", linewidth=1, alpha=0.7)
    ax.plot(sim_t[:n_real], sim_cmd_steer[:n_real], label="Sim commanded steering", linewidth=1, linestyle="--")
    ax.plot(sim_t[:n_real], sim_delta[:n_real], label="Sim actual delta", linewidth=1.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Steering (rad)")
    ax.set_title("Steering — Command vs Actual")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "steering.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nPlots saved to {out_dir}/")


if __name__ == "__main__":
    main()
