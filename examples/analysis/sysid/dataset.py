"""STD system identification dataset loader.

Loads a 100 Hz Vicon-recorded NPZ bag, slices it into fixed-length windows
suitable for sim-vs-real loss evaluation, and (optionally) appends mirrored
copies to neutralize left/right asymmetry in the data.

Each Window holds:
  - the 7-element STD user state at t0 (used for env.reset),
  - the command sequence to replay (cmd_steer, cmd_speed),
  - the real body-frame signals to score against (v_x, v_y, yaw_rate, a_x).

See docs/plan/OPTUNA_SYS_ID_LOSS.md for the locked design decisions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import savgol_filter

CHANNELS = ("yaw_rate", "v_y", "a_x", "v_x")

# init_state layout: [x, y, delta, v, yaw, yaw_rate, beta]
# Under left/right mirror (reflection across the longitudinal body axis):
#   x, v stay; y, delta, yaw, yaw_rate, beta flip sign.
_MIRROR_INIT_SIGNS = np.array([1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0])


@dataclass(frozen=True)
class Window:
    t0_idx: int
    init_state: np.ndarray  # shape (7,)
    cmd_steer: np.ndarray  # shape (N,)
    cmd_speed: np.ndarray  # shape (N,)
    real_v_x: np.ndarray  # shape (N+1,)
    real_v_y: np.ndarray
    real_yaw_rate: np.ndarray
    real_a_x: np.ndarray
    is_mirrored: bool


@dataclass(frozen=True)
class Dataset:
    windows: list[Window]
    variances: dict[str, float]
    dt: float


def mirror_window(w: Window) -> Window:
    return Window(
        t0_idx=w.t0_idx,
        init_state=w.init_state * _MIRROR_INIT_SIGNS,
        cmd_steer=-w.cmd_steer,
        cmd_speed=w.cmd_speed.copy(),
        real_v_x=w.real_v_x.copy(),
        real_v_y=-w.real_v_y,
        real_yaw_rate=-w.real_yaw_rate,
        real_a_x=w.real_a_x.copy(),
        is_mirrored=not w.is_mirrored,
    )


def load_dataset(
    npz_path: str,
    window_length_s: float = 1.5,
    stride_s: float = 0.5,
    min_speed: float = 0.3,
    mirror: bool = True,
    sg_window: int = 21,
    sg_polyorder: int = 2,
    dt: float = 0.01,
) -> Dataset:
    data = np.load(npz_path)
    t = data["t"]
    cmd_speed = data["cmd_speed"]
    cmd_steer = data["cmd_steer"]
    vicon_x = data["vicon_x"]
    vicon_y = data["vicon_y"]
    vicon_yaw = data["vicon_yaw"]
    vicon_body_vx = data["vicon_body_vx"]
    vicon_body_vy = data["vicon_body_vy"]
    vicon_r = data["vicon_r"]

    n_total = len(t)
    n_steps = int(round(window_length_s / dt))
    stride = int(round(stride_s / dt))
    if n_total < n_steps + 1:
        raise ValueError(f"Bag too short for window_length_s={window_length_s} (n_total={n_total})")

    # Smoothed real longitudinal accel: computed once on the full bag, then sliced.
    real_a_x_full = np.gradient(savgol_filter(vicon_body_vx, sg_window, sg_polyorder), dt)

    speed_full = np.hypot(vicon_body_vx, vicon_body_vy)
    beta_full = np.where(speed_full > min_speed, np.arctan2(vicon_body_vy, vicon_body_vx), 0.0)

    windows: list[Window] = []
    t0_max = n_total - n_steps - 1
    for t0 in range(0, t0_max + 1, stride):
        end = t0 + n_steps  # exclusive for cmds; cmds drive N steps, real signals span N+1 samples
        cmd_s = cmd_steer[t0:end]
        cmd_v = cmd_speed[t0:end]
        rvx = vicon_body_vx[t0 : end + 1]
        rvy = vicon_body_vy[t0 : end + 1]
        rr = vicon_r[t0 : end + 1]
        rax = real_a_x_full[t0 : end + 1]

        if np.mean(speed_full[t0 : end + 1]) < min_speed:
            continue
        if not all(np.all(np.isfinite(a)) for a in (rvx, rvy, rr, rax, cmd_s, cmd_v)):
            continue

        init_state = np.array(
            [
                vicon_x[t0],
                vicon_y[t0],
                cmd_steer[t0],
                speed_full[t0],
                vicon_yaw[t0],
                vicon_r[t0],
                beta_full[t0],
            ],
            dtype=float,
        )

        windows.append(
            Window(
                t0_idx=int(t0),
                init_state=init_state,
                cmd_steer=cmd_s.astype(float),
                cmd_speed=cmd_v.astype(float),
                real_v_x=rvx.astype(float),
                real_v_y=rvy.astype(float),
                real_yaw_rate=rr.astype(float),
                real_a_x=rax.astype(float),
                is_mirrored=False,
            )
        )

    if not windows:
        raise ValueError(f"No windows survived filtering for {npz_path}")

    # Variances computed from the physical (non-mirrored) signals so the NMSE scale
    # does not depend on whether mirroring is enabled. Mirrored windows are scored
    # against the same per-channel scale.
    variances = {
        "yaw_rate": float(np.var(np.concatenate([w.real_yaw_rate for w in windows]))),
        "v_y": float(np.var(np.concatenate([w.real_v_y for w in windows]))),
        "a_x": float(np.var(np.concatenate([w.real_a_x for w in windows]))),
        "v_x": float(np.var(np.concatenate([w.real_v_x for w in windows]))),
    }

    if mirror:
        windows = windows + [mirror_window(w) for w in windows]

    return Dataset(windows=windows, variances=variances, dt=dt)


def _plot_dataset_overview(dataset: Dataset, npz_path: str, out_path: str, mirror: bool = False) -> None:
    import os

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = np.load(npz_path)
    t = data["t"]
    vicon_x = data["vicon_x"]
    vicon_y = data["vicon_y"]
    vicon_body_vx = data["vicon_body_vx"]
    vicon_body_vy = data["vicon_body_vy"]
    vicon_r = data["vicon_r"]
    cmd_speed = data["cmd_speed"]
    cmd_steer = data["cmd_steer"]

    if mirror:
        # Reflect raw bag signals so they match the mirrored windows we're about to overlay.
        vicon_y = -vicon_y
        vicon_body_vy = -vicon_body_vy
        vicon_r = -vicon_r
        cmd_steer = -cmd_steer

    real_a_x = np.gradient(savgol_filter(vicon_body_vx, 21, 2), dataset.dt)
    real_speed = np.hypot(vicon_body_vx, vicon_body_vy)
    real_beta = np.where(real_speed > 0.2, np.arctan2(vicon_body_vy, vicon_body_vx), np.nan)

    selected = [w for w in dataset.windows if w.is_mirrored == mirror]
    n_steps = len(selected[0].cmd_steer) if selected else 0

    suffix = " [MIRRORED]" if mirror else ""
    fig, grid = plt.subplots(2, 4, figsize=(32, 14))
    fig.suptitle(
        f"Dataset overview{suffix} — {os.path.basename(npz_path)} ({len(selected)} windows)",
        fontsize=14,
    )
    grid[0, 3].axis("off")
    axes = [grid[0, 0], grid[0, 1], grid[0, 2], grid[1, 0], grid[1, 1], grid[1, 2], grid[1, 3]]

    def shade_windows(ax):
        for w in selected:
            ax.axvspan(t[w.t0_idx], t[w.t0_idx + n_steps], color="orange", alpha=0.15, linewidth=0)

    ax = axes[0]
    ax.plot(vicon_x, vicon_y, label="Real (Vicon)", linewidth=1.0)
    for w in selected:
        ax.plot(w.init_state[0], w.init_state[1], "o", color="orange", markersize=4)
    ax.plot(vicon_x[0], vicon_y[0], "ko", markersize=8, label="Bag start")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("XY Trajectory (orange dots = window init_state)")
    ax.set_aspect("equal")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    shade_windows(ax)
    ax.plot(t, cmd_speed, label="cmd_speed", linewidth=1, alpha=0.6)
    ax.plot(t, vicon_body_vx, label="Real vx", linewidth=1)
    for w in selected:
        ax.plot(t[w.t0_idx], w.init_state[3], "o", color="orange", markersize=4)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Velocity (m/s)")
    ax.set_title("vx (orange dots = init speed)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    shade_windows(ax)
    ax.plot(t, vicon_body_vy, label="Real vy", linewidth=1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Velocity (m/s)")
    ax.set_title("vy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[3]
    shade_windows(ax)
    ax.plot(t, cmd_steer, label="cmd_steer", linewidth=1)
    for w in selected:
        ax.plot(t[w.t0_idx], w.init_state[2], "o", color="orange", markersize=4)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Steering (rad)")
    ax.set_title("Steering cmd (orange dots = init delta)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[4]
    shade_windows(ax)
    ax.plot(t, vicon_r, label="Real yaw rate", linewidth=1)
    for w in selected:
        ax.plot(t[w.t0_idx], w.init_state[5], "o", color="orange", markersize=4)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Yaw rate (rad/s)")
    ax.set_title("Yaw rate (orange dots = init yaw_rate)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[5]
    shade_windows(ax)
    ax.plot(t, real_a_x, label="Real a_x (smoothed)", linewidth=1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Acceleration (m/s²)")
    ax.set_title("Longitudinal acceleration")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[6]
    shade_windows(ax)
    ax.plot(t, real_beta, label="Real β", linewidth=1)
    for w in selected:
        ax.plot(t[w.t0_idx], w.init_state[6], "o", color="orange", markersize=4)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Slip angle β (rad)")
    ax.set_title("Slip angle (orange dots = init beta)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    # optional dataset validation plots can be generated
    import argparse
    import os
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Validate sysid dataset loading by plotting bag overview")
    parser.add_argument("--path", required=True, help="Path to 100Hz NPZ bag")
    parser.add_argument("--window-length-s", type=float, default=1.5)
    parser.add_argument("--stride-s", type=float, default=0.5)
    parser.add_argument("--min-speed", type=float, default=0.3)
    parser.add_argument("--no-mirror", action="store_true")
    args = parser.parse_args()

    ds = load_dataset(
        args.path,
        window_length_s=args.window_length_s,
        stride_s=args.stride_s,
        min_speed=args.min_speed,
        mirror=not args.no_mirror,
    )

    n_orig = sum(not w.is_mirrored for w in ds.windows)
    print(f"windows: {len(ds.windows)} total ({n_orig} originals + {len(ds.windows) - n_orig} mirrors)")
    print(f"variances: {ds.variances}")

    stem = Path(args.path).stem
    out_dir = os.path.join("figures", "analysis", "sysid", stem)
    out_path = os.path.join(out_dir, "dataset_overview.png")
    _plot_dataset_overview(ds, args.path, out_path, mirror=False)
    print(f"plot saved to {out_path}")

    if not args.no_mirror:
        out_path_mirror = os.path.join(out_dir, "dataset_overview_mirror.png")
        _plot_dataset_overview(ds, args.path, out_path_mirror, mirror=True)
        print(f"plot saved to {out_path_mirror}")
