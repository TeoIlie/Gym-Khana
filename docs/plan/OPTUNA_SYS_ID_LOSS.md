# Plan: Loss Function Module for STD System Identification

## Context

We are building Optuna-based system identification for the STD (Single Track Drift) vehicle dynamics model — single-track dynamics with the full PAC2002 tire model (lateral + longitudinal coefficients). The optimization works by replaying real-car command sequences (recorded in a Vicon motion-capture space) through the gymkhana simulator with trial parameters and scoring how well the simulated trajectory matches reality.

STD's user-facing state is 7 elements (`[x, y, delta, v, yaw, yaw_rate, slip_angle]`) — identical to STP. The two extra internal states (`omega_front`, `omega_rear`) are initialized from `v` at reset (`single_track_drift.py:42-43, 244-245`), so the loss/rollout interface is the same dimensionality regardless of model choice.

This plan covers **only the loss function and the dataset preparation** that feeds it — not the Optuna study, parameter search space, or CLI. The loss is the foundation: every later piece (objective function, Optuna study, plots) calls into it. Getting it right means getting per-channel weighting, normalization, windowing, warmup, and mirroring correct so Optuna minimizes a quantity that genuinely reflects sim2real fidelity for downstream policy transfer — not artifacts of integration drift or unit imbalance.

## Design decisions (locked)

| Decision | Choice | Rationale |
|---|---|---|
| Channels in loss | `yaw_rate`, `v_y`, `a_x`, `v_x` | Direct dynamics outputs; XY excluded (integrates yaw drift); `delta`/`β` redundant. |
| Channel weights | `w_r=3.0, w_vy=2.0, w_ax=1.0, w_vx=0.5` | Bias toward lateral dynamics — primary tire-param signal. |
| Per-channel normalization | NMSE: divide MSE by `var(real)` over full dataset | Makes channels commensurable across units (rad/s vs m/s² vs m/s). |
| Window length | Configurable, default **1.5 s** (150 samples @ 100 Hz) | Long enough to excite saturation, short enough to limit chaotic divergence. |
| Window stride | Configurable, default 0.5 s | Overlap is fine; more samples helps TPE. |
| Warmup discard | First **0.2 s** of each window dropped from loss | Eats steering-servo transient from `delta_init = cmd_steer[t0]` (we don't have real `delta`). |
| Reset state | Full 7-element STD user state from Vicon at `t0`, with `delta = cmd_steer[t0]` | `env.reset(options={"states": ...})` accepts full state (confirmed); `omega_front`/`omega_rear` derived internally from `v`. |
| Low-speed mask | Drop windows where `mean(speed) < 0.3 m/s` | Kinematic blend dominates — doesn't constrain Pacejka. |
| Mirroring | Default **on**, configurable off | Removes left/right bias from imbalanced data; STP/STD are structurally symmetric. |
| Sim signal smoothing | None — finite-difference raw `sim_v_x` for `sim_a_x` | Sim is noise-free; smoothing just adds lag. Real `a_x` smoothed once up front. |
| Real `a_x` smoothing | Savitzky–Golay, window=21, polyorder=2 (matches `traj_compare.py:136`) | Done once at dataset load, not per trial. |

## Files to create

1. **`examples/analysis/sysid/dataset.py`** — load NPZ, build windows, compute per-channel variances, apply mirroring.
2. **`examples/analysis/sysid/loss.py`** — pure functions: per-channel NMSE, weighted total, rollout-window scoring.
3. **`examples/analysis/sysid/rollout.py`** — single-window rollout helper that wraps `gym.make` + reset with full state + step loop. Factored from `traj_compare.py:64-107`.
4. **`tests/sysid/test_loss.py`** — unit tests on synthetic data: identity rollout → loss = 0; perturbed params → loss > 0; mirroring symmetry → identical loss on mirrored input under symmetric params.

## Module designs

### `dataset.py`

```python
@dataclass(frozen=True)
class Window:
    t0_idx: int                          # start index in source NPZ
    init_state: np.ndarray               # shape (7,) — STD user state vector
    cmd_steer: np.ndarray                # shape (N,)
    cmd_speed: np.ndarray                # shape (N,)
    real_v_x: np.ndarray                 # shape (N+1,) — body-frame
    real_v_y: np.ndarray                 # shape (N+1,)
    real_yaw_rate: np.ndarray            # shape (N+1,)
    real_a_x: np.ndarray                 # shape (N+1,) — pre-smoothed, pre-differentiated
    is_mirrored: bool

@dataclass(frozen=True)
class Dataset:
    windows: list[Window]
    variances: dict[str, float]          # {"yaw_rate": ..., "v_y": ..., "a_x": ..., "v_x": ...}
    dt: float                            # 0.01

def load_dataset(
    npz_path: str,
    window_length_s: float = 1.5,
    stride_s: float = 0.5,
    warmup_s: float = 0.2,               # passed through; consumed in loss, not here
    min_speed: float = 0.3,
    mirror: bool = True,
    sg_window: int = 21,
    sg_polyorder: int = 2,
) -> Dataset
```

Construction order:
1. Load NPZ keys per `traj_compare.py:39-48`.
2. Pre-compute `real_a_x = gradient(savgol_filter(vicon_body_vx, ...), t)`.
3. Slide window: at each candidate `t0`, build init state `[vicon_x, vicon_y, cmd_steer, speed, vicon_yaw, vicon_r, beta]` where `speed = hypot(vx,vy)` and `beta = atan2(vy,vx)` (masked at low speed → 0).
4. Skip windows failing the speed mask or containing NaNs.
5. Compute per-channel variances over all retained windows' real signals.
6. If `mirror`, append a flipped copy per `mirror(window)` (negate `y, yaw, yaw_rate, v_y, beta, cmd_steer`).

### `loss.py`

```python
def channel_nmse(sim: np.ndarray, real: np.ndarray, variance: float) -> float:
    return float(np.mean((sim - real) ** 2) / (variance + 1e-9))

def window_loss(
    sim_states: SimStates,               # bag of (v_x, v_y, yaw_rate, a_x) arrays of shape (N+1,)
    window: Window,
    variances: dict[str, float],
    weights: dict[str, float],
    warmup_steps: int,
) -> tuple[float, dict[str, float]]:
    """Returns (weighted_total, per_channel_dict). Slices off warmup before scoring."""

def dataset_loss(
    rollout_fn: Callable[[Window], SimStates],
    dataset: Dataset,
    weights: dict[str, float] = DEFAULT_WEIGHTS,
    warmup_s: float = 0.2,
) -> tuple[float, dict[str, float]]:
    """Rolls out every window via rollout_fn, averages weighted total + per-channel breakdown."""

DEFAULT_WEIGHTS = {"yaw_rate": 3.0, "v_y": 2.0, "a_x": 1.0, "v_x": 0.5}
```

`dataset_loss` is what the eventual Optuna objective will call: it accepts a `rollout_fn` closure that captures the trial's params dict, so the loss module stays independent of Optuna and the env.

### `rollout.py`

```python
def make_rollout_fn(params: dict, dt: float = 0.01) -> Callable[[Window], SimStates]:
    """Build a closure that constructs an env once and rolls out windows.
    Returns SimStates with v_x, v_y, yaw_rate, a_x (finite-diffed from v_x, no smoothing)."""
```

Reuses the env-construction config from `traj_compare.py:65-76` but with **`model="std"`** and `params=GKEnv.f1tenth_std_vehicle_params()`; control_input remains `["speed","steering_angle"]`. Step loop from lines 92-105. Reset uses `options={"states": init_state.reshape(1, 7)}` per the explored interface in `gymkhana_env.py:1128-1144`.

## Reused existing code

- `gymkhana.envs.gymkhana_env.GKEnv` — env construction; `reset(options={"states": ...})` at `gymkhana_env.py:1075-1192`.
- `gymkhana.envs.params.load_params` — YAML loader at `gymkhana/envs/params/__init__.py:11-38`.
- `GKEnv.f1tenth_std_vehicle_params()` — base params loaded from `gymkhana/envs/params/f1tenth_std.yaml`.
- Savitzky–Golay smoothing config — copied from `traj_compare.py:136`.

## Verification

1. **Identity sanity check**: build a `Dataset` from an existing NPZ, build a `rollout_fn` with the YAML defaults, compute `dataset_loss`. Record per-channel NMSE values — these are the **baseline** any later Optuna run must beat.
2. **Self-consistency**: feed real signals as if they were sim signals (`sim = real`) → loss must be 0 across all channels.
3. **Mirror symmetry test** (`tests/sysid/test_loss.py`): for symmetric STD params, loss on mirrored window must equal loss on original window (within 1e-6). Catches sign-flip bugs in the mirror transform.
4. **Smoke test integration**: run `dataset_loss` end-to-end on one real NPZ; confirm runtime is acceptable (target < 5 s on 30 s log → enables 2000 trials in ~3 hours).
5. **Channel breakdown sanity**: print per-channel NMSE for the baseline; confirm `yaw_rate` and `v_y` are the dominant residuals (if `v_x` dominates, weights are off and need re-tuning before launching Optuna).

## Out of scope (future plans)

- Optuna study setup, sampler choice, parameter search-space bounds.
- Multi-NPZ aggregation across multiple recordings.
- Adding `omega_front`/`omega_rear` channels to the loss — wheel speeds are not measured in current bags; `a_x` is the only longitudinal-tire constraint available.
- CLI / wandb logging.
- Per-window weighting by maneuver type.
