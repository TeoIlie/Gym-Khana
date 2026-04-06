"""
Replay real-car commands in the Gym-Khana STD simulator and compare trajectories.

Loads a 100 Hz resampled .npz file (produced by bag_to_npz.py), replays the
recorded speed + steering commands open-loop through the STD dynamics model,
and compares the simulated trajectory against Vicon ground truth.

Outputs:
  - 4-panel comparison plot (XY path, heading, velocity, position error)
  - Printed RMSE metrics

Usage:
    python traj_compare.py <npz_file> [--map MAP] [--model MODEL]
    python traj_compare.py --bag-name <bag_name> [--map MAP]

Example:
    python traj_compare.py ~/data/run1_100Hz.npz
    python traj_compare.py ~/data/run1_100Hz.npz --map Spielberg
    python traj_compare.py ~/data/run1_100Hz.npz --model ks
"""

import argparse
import math
import sys
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from gymkhana.envs.gymkhana_env import GKEnv
from train.config.env_config import get_env_id


def load_data(npz_path: str) -> dict[str, np.ndarray]:
    d = dict(np.load(npz_path))
    required = [
        "t",
        "cmd_speed",
        "cmd_steer",
        "vicon_x",
        "vicon_y",
        "vicon_yaw",
        "vicon_body_vx",
        "vicon_body_vy",
        "vicon_r",
    ]
    missing = [k for k in required if k not in d]
    if missing:
        sys.exit(f"Missing arrays in {npz_path}: {missing}")
    return d


def build_initial_state(d: dict[str, np.ndarray], model: str) -> np.ndarray:
    """Build initial state from first Vicon sample, sized for the chosen model."""
    vx = d["vicon_body_vx"][0]
    vy = d["vicon_body_vy"][0]
    v = math.sqrt(vx**2 + vy**2)
    x0 = d["vicon_x"][0]
    y0 = d["vicon_y"][0]
    yaw0 = d["vicon_yaw"][0]

    if model == "ks":
        # KS state: [x, y, delta, v_x, yaw] — v_x is longitudinal velocity
        return np.array([[x0, y0, 0.0, vx, yaw0]])
    else:
        # ST and STD state: [x, y, delta, vel, yaw, yaw_rate, slip_angle]
        beta = math.atan2(vy, vx) if v > 0.5 else 0.0
        r = d["vicon_r"][0]
        return np.array([[x0, y0, 0.0, v, yaw0, r, beta]])


def run_replay(d: dict[str, np.ndarray], map_name: str, model: str):
    """Step the sim with recorded commands and return sim trajectory."""
    params = GKEnv.f1tenth_std_vehicle_params() if model == "std" else GKEnv.f1tenth_vehicle_params()

    env = gym.make(
        get_env_id(),
        config={
            "map": map_name,
            "num_agents": 1,
            "model": model,
            "control_input": ["speed", "steering_angle"],
            "normalize_act": False,
            "normalize_obs": False,
            "observation_config": {"type": "kinematic_state"},
            "timestep": 0.01,
            "params": params,
        },
        render_mode="human",
    )

    init_state = build_initial_state(d, model)
    if model == "std":
        env.reset(options={"states": init_state})
    else:
        # KS/ST don't support full state init; use pose (x, y, yaw) — starts from rest
        pose = np.array([[d["vicon_x"][0], d["vicon_y"][0], d["vicon_yaw"][0]]])
        env.reset(options={"poses": pose})

    n_steps = len(d["t"])
    sim_x = np.empty(n_steps)
    sim_y = np.empty(n_steps)
    sim_yaw = np.empty(n_steps)
    sim_vx = np.empty(n_steps)
    sim_vy = np.empty(n_steps)

    def extract_state(state):
        x, y, yaw, v = state[0], state[1], state[4], state[3]
        if model == "ks":
            return x, y, yaw, v, 0.0
        else:
            beta = state[6]
            return x, y, yaw, v * math.cos(beta), v * math.sin(beta)

    # Record initial state
    state = env.unwrapped.sim.agents[0].state
    sim_x[0], sim_y[0], sim_yaw[0], sim_vx[0], sim_vy[0] = extract_state(state)

    for i in range(1, n_steps):
        action = np.array([[d["cmd_speed"][i - 1], d["cmd_steer"][i - 1]]])
        env.step(action)

        state = env.unwrapped.sim.agents[0].state
        sim_x[i], sim_y[i], sim_yaw[i], sim_vx[i], sim_vy[i] = extract_state(state)

        env.render()

    env.close()
    return sim_x, sim_y, sim_yaw, sim_vx, sim_vy


def wrap_angle(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2 * np.pi) - np.pi


def compute_metrics(d, sim_x, sim_y, sim_yaw, sim_vx, sim_vy):
    pos_err = np.sqrt((sim_x - d["vicon_x"]) ** 2 + (sim_y - d["vicon_y"]) ** 2)
    heading_err = np.abs(wrap_angle(sim_yaw - d["vicon_yaw"]))
    real_speed = np.sqrt(d["vicon_body_vx"] ** 2 + d["vicon_body_vy"] ** 2)
    sim_speed = np.sqrt(sim_vx**2 + sim_vy**2)
    speed_err = np.abs(sim_speed - real_speed)

    metrics = {
        "pos_rmse_m": float(np.sqrt(np.mean(pos_err**2))),
        "heading_rmse_rad": float(np.sqrt(np.mean(heading_err**2))),
        "speed_rmse_ms": float(np.sqrt(np.mean(speed_err**2))),
        "max_pos_err_m": float(np.max(pos_err)),
        "max_pos_err_time_s": float(d["t"][np.argmax(pos_err)]),
        "final_pos_err_m": float(pos_err[-1]),
    }
    return metrics, pos_err, heading_err, speed_err


def plot_comparison(
    d, sim_x, sim_y, sim_yaw, sim_vx, sim_vy, pos_err, heading_err, speed_err, metrics, out_path, model: str
):
    t = d["t"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Sim-to-Real Trajectory Comparison (model={model.upper()})", fontsize=14)

    # 1. XY trajectory
    ax = axes[0, 0]
    ax.plot(d["vicon_x"], d["vicon_y"], label="Real (Vicon)", linewidth=1.5)
    ax.plot(sim_x, sim_y, label=f"Sim ({model.upper()})", linewidth=1.5, linestyle="--")
    ax.plot(d["vicon_x"][0], d["vicon_y"][0], "go", markersize=8, label="Start")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("XY Trajectory")
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # 2. Heading vs time
    ax = axes[0, 1]
    ax.plot(t, d["vicon_yaw"], label="Real", linewidth=1)
    ax.plot(t, sim_yaw, label="Sim", linewidth=1, linestyle="--")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Yaw (rad)")
    ax.set_title("Heading")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Speed vs time
    ax = axes[1, 0]
    real_speed = np.sqrt(d["vicon_body_vx"] ** 2 + d["vicon_body_vy"] ** 2)
    sim_speed = np.sqrt(sim_vx**2 + sim_vy**2)
    ax.plot(t, real_speed, label="Real", linewidth=1)
    ax.plot(t, sim_speed, label="Sim", linewidth=1, linestyle="--")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speed (m/s)")
    ax.set_title("Speed")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Position error vs time
    ax = axes[1, 1]
    ax.plot(t, pos_err, linewidth=1, color="red")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position Error (m)")
    ax.set_title(f"Position Error (RMSE={metrics['pos_rmse_m']:.3f} m)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    print(f"Plot saved to {out_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Replay real-car commands in sim and compare trajectories")
    parser.add_argument("npz_file", nargs="?", help="Path to 100 Hz resampled .npz file (fallback to --bag-name)")
    parser.add_argument(
        "--bag-name",
        help="Name of the bag that lives under examples/analysis/bags/<bag_name>/<bag_name>_100Hz.npz",
        default=None,
    )
    parser.add_argument("--map", default="Spielberg", help="Track map name (default: Spielberg)")
    parser.add_argument(
        "--model", default="std", choices=["ks", "st", "std"], help="Dynamics model: ks, st, or std (default: std)"
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    default_bags_root = script_dir / "bags"

    if args.npz_file:
        npz_path = Path(args.npz_file).resolve()
    elif args.bag_name:
        npz_path = (default_bags_root / args.bag_name / f"{args.bag_name}_100Hz.npz").resolve()
    else:
        parser.error("Specify either the npz_file path or --bag-name to build the default path")
    if not npz_path.exists():
        sys.exit(f"File not found: {npz_path}")

    print(f"Loading {npz_path} ...")
    d = load_data(str(npz_path))
    print(f"  {len(d['t'])} steps, {d['t'][-1]:.2f}s duration")

    print(f"Running sim replay (map={args.map}, model={args.model}) ...")
    sim_x, sim_y, sim_yaw, sim_vx, sim_vy = run_replay(d, args.map, args.model)

    print("\nComputing metrics ...")
    metrics, pos_err, heading_err, speed_err = compute_metrics(d, sim_x, sim_y, sim_yaw, sim_vx, sim_vy)

    print("\n=== Sim-to-Real Gap Metrics ===")
    print(f"  Position RMSE:    {metrics['pos_rmse_m']:.4f} m")
    print(
        f"  Heading RMSE:     {metrics['heading_rmse_rad']:.4f} rad "
        f"({math.degrees(metrics['heading_rmse_rad']):.2f} deg)"
    )
    print(f"  Speed RMSE:       {metrics['speed_rmse_ms']:.4f} m/s")
    print(f"  Max position err: {metrics['max_pos_err_m']:.4f} m at t={metrics['max_pos_err_time_s']:.2f}s")
    print(f"  Final position err: {metrics['final_pos_err_m']:.4f} m")

    repo_root = script_dir.parents[1]
    figures_dir = repo_root / "figures" / "analysis" / "traj_compare"
    figures_dir.mkdir(parents=True, exist_ok=True)
    bag_label = args.bag_name if args.bag_name else npz_path.stem
    out_path = figures_dir / f"{bag_label}_{args.model}.png"
    plot_comparison(
        d, sim_x, sim_y, sim_yaw, sim_vx, sim_vy, pos_err, heading_err, speed_err, metrics, out_path, args.model
    )


if __name__ == "__main__":
    main()
