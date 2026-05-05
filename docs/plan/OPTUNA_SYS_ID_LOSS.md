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
| Reset state | Full 9-element STD user state from Vicon at `t0`, with `delta = cmd_steer[t0]` and `omega_front = omega_rear = rs_core_speed[t0]/R_w` | `env.reset(options={"states": ...})` accepts the 9-wide branch (`user_state_lens()`); `omega` seeded from VESC under the AWD assumption (single scalar to both wheels). |
| Low-speed mask | Drop windows where `mean(speed) < 0.3 m/s` | Kinematic blend dominates — doesn't constrain Pacejka. |
| Mirroring | Default **on**, configurable off | Removes left/right bias from imbalanced data; STP/STD are structurally symmetric. |
| Sim signal smoothing | None — finite-difference raw `sim_v_x` for `sim_a_x` | Sim is noise-free; smoothing just adds lag. Real `a_x` smoothed once up front. |
| Real `a_x` smoothing | Savitzky–Golay, window=21, polyorder=2 (matches `traj_compare.py:136`) | Done once at dataset load, not per trial. |
| Variance domain | Computed on non-mirrored windows only | Decouples the NMSE scale from the mirror flag — antisymmetric channels (`v_y`, `yaw_rate`) would otherwise show inflated variance when mirroring is enabled. |

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
    init_state: np.ndarray               # shape (9,) — STD user state vector + omega_f, omega_r
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
    min_speed: float = 0.3,
    mirror: bool = True,
    sg_window: int = 21,
    sg_polyorder: int = 2,
    dt: float = 0.01,
) -> Dataset
```

Construction order:
1. Load NPZ keys per `traj_compare.py:39-48`.
2. Pre-compute `real_a_x = gradient(savgol_filter(vicon_body_vx, ...), t)`.
3. Slide window: at each candidate `t0`, build init state `[vicon_x, vicon_y, cmd_steer, speed, vicon_yaw, vicon_r, beta]` where `speed = hypot(vx,vy)` and `beta = atan2(vy,vx)` (masked at low speed → 0).
4. Skip windows failing the speed mask or containing NaNs.
5. Compute per-channel variances over the non-mirrored retained windows' real signals (so the NMSE scale is independent of the mirror flag).
6. If `mirror`, append a flipped copy per `mirror_window(w)` (negate `y, delta, yaw, yaw_rate, v_y, beta, cmd_steer`).

### `loss.py`

```python
def channel_nmse(sim: np.ndarray, real: np.ndarray, variance: float) -> float:
    return float(np.mean((sim - real) ** 2) / (variance + 1e-9))

def window_loss(
    sim: dict[str, np.ndarray],          # keys = CHANNELS, each shape (N+1,)
    window: Window,
    variances: dict[str, float],
    weights: dict[str, float],
    warmup_steps: int,
) -> tuple[float, dict[str, float]]:
    """Returns (weighted_total, per_channel_dict). Slices off warmup before scoring."""

def dataset_loss(
    rollout_fn: Callable[[Window], dict[str, np.ndarray]],
    dataset: Dataset,
    weights: dict[str, float] = DEFAULT_WEIGHTS,
    warmup_s: float = 0.2,
) -> tuple[float, dict[str, float]]:
    """Rolls out every window via rollout_fn, averages weighted total + per-channel breakdown."""

DEFAULT_WEIGHTS = {"yaw_rate": 3.0, "v_y": 2.0, "a_x": 1.0, "v_x": 0.5}
```

Sim signals are passed as a plain `dict[str, np.ndarray]` keyed by `CHANNELS` (imported from `dataset.py`), not a dedicated dataclass. This keeps the interface symmetric with `variances` / `weights` / per-channel return dicts, removes a cross-module type-placement question, and makes identity tests trivial (pass `{"v_x": w.real_v_x, ...}` directly).

`dataset_loss` is what the eventual Optuna objective will call: it accepts a `rollout_fn` closure that captures the trial's params dict, so the loss module stays independent of Optuna and the env.

### `rollout.py`

```python
class Rollout:
    """Owns a single GKEnv instance and replays Windows through it.

    Construct once per worker process; call `set_params(params)` between trials
    to hot-swap PAC2002 coefficients without rebuilding the env. Each call to
    `run(window)` resets the env to the window's init_state and steps through
    the recorded command sequence.
    """
    def __init__(self, params: dict | None = None): ...
    def set_params(self, params: dict) -> None:
        """Hot-swap params via env.configure({'params': params})."""
    def run(self, window: Window) -> dict[str, np.ndarray]:
        """Reset to window.init_state, step N commands, return CHANNELS dict
        of shape (N+1,) each."""

def make_rollout_fn(params: dict) -> Callable[[Window], dict[str, np.ndarray]]:
    """Convenience: build a Rollout and return its `run` bound method.
    Useful for Phase 1 tests and the loss-module identity sanity check.
    Phase 3 study code should construct `Rollout` directly and reuse it
    across trials within a worker (call set_params each trial)."""
```

**Env construction.** Start from `train.config.env_config.get_drift_train_config()` and override:

| Key | Override | Rationale |
|---|---|---|
| `num_agents` | `1` | `reset(options={"states": ...})` shape is `(1, 7)`. |
| `model` | `"std"` | Sysid targets the STD model. |
| `control_input` | `["speed", "steering_angle"]` | Match bag command channels (`cmd_speed`, `cmd_steer`). |
| `normalize_obs` | `False` | Rollout reads raw `agent.state`, not the obs vector. |
| `normalize_act` | `False` | Bag commands are physical units (rad, m/s); avoid double-scaling. |
| `prevent_instability` | `False` | Otherwise RaceCar may revert mid-rollout and contaminate the loss signal. |
| `track_direction` | `"normal"` | `"random"` would call RNG on every reset, breaking sampler-determinism invariant. |
| `render_mode` / render flags / `record_obs_min_max` | off | No-op overhead during sysid. |
| `params` | from constructor / `set_params` | The thing we're identifying. |

Map / `training_mode="race"` are left at the drift-train defaults. Vicon coordinates do not lie on the gym track, so `boundary_exceeded` will fire on step 1 — harmless because `Rollout.run` ignores `terminated`/`truncated` and steps the full command sequence regardless.

**Reset.** `env.reset(options={"states": window.init_state.reshape(1, 9)})`. The 9-element STD user state is `[x, y, delta, v, yaw, yaw_rate, beta, omega_front, omega_rear]` exactly as `Window.init_state` is constructed. Wheel angular velocities are seeded from VESC `rs_core_speed[t0]/R_w` under the AWD assumption (single scalar to both wheels); `init_std`'s 9-wide branch (`single_track_drift/__init__.py:80-82`) consumes them verbatim via `compute_wheel_speeds=False`.

**Step loop.** For `k in range(N)`:
```python
action = np.array([[window.cmd_steer[k], window.cmd_speed[k]]], dtype=np.float64)
env.step(action)
```
Action ordering is `[steering_angle, speed]` regardless of `control_input` order (CarAction normalizes internally).

**Sim signal extraction.** Use `observation_config={"type": "dynamic_state"}` and read body-frame signals straight from the per-step obs dict:
- `v_x = obs[agent_id]["linear_vel_x"]`
- `v_y = obs[agent_id]["linear_vel_y"]`
- `yaw_rate = obs[agent_id]["ang_vel_z"]`

These are sourced from `agent.standard_state` inside `FeaturesObservation.observe`, which for STD is exactly body-frame `v_x`/`v_y` (matches `vicon_body_vx`/`vicon_body_vy` in the bag and how `traj_compare.py:99-102` reads them). Obs values arrive as `float32`; cast to `float64` on assignment to keep loss arithmetic in double precision.

After collecting all `N+1` samples, compute `a_x = np.gradient(v_x, dt)` — no smoothing (sim is noise-free; matches the no-sim-smoothing decision in the locked design table).

## Reused existing code

- `gymkhana.envs.gymkhana_env.GKEnv` — env construction; `reset(options={"states": ...})` at `gymkhana_env.py:1075-1192`.
- `gymkhana.envs.params.load_params` — YAML loader at `gymkhana/envs/params/__init__.py:11-38`.
- `GKEnv.f1tenth_std_vehicle_params()` — base params loaded from `gymkhana/envs/params/f1tenth_std.yaml`.
- Savitzky–Golay smoothing config — copied from `traj_compare.py:136`.

## Implementation notes (post-build)

- `loss.py` imports `CHANNELS` from `dataset.py` as the single source of truth for channel keys; `getattr(window, f"real_{ch}")` is used to fetch the real-signal array (channel names match `Window.real_*` field suffixes).
- `window_loss` asserts `sim[ch].shape == real[ch].shape` per channel, catching rollout shape bugs immediately rather than letting numpy broadcast silently. **Contract for `rollout.py`:** the returned dict must have all four `CHANNELS` keys, each a numpy array of shape `(N+1,)` matching `Window.real_v_x`.
- `dataset_loss` raises `ValueError` if `warmup_steps >= signal_len` instead of silently returning NaN from `np.mean([])`.
- Per-channel aggregation across windows is the **arithmetic mean** of per-window NMSEs (equal weight per window), not a globally pooled MSE / total variance.

## Verification

Status: ✅ **all checks pass.** Phase 1 module set is complete.

1. ✅ **Self-consistency / identity invariant**: feeding real signals as sim → 0 across all channels (`test_window_loss_identity`, `test_dataset_loss_identity`).
2. ✅ **Mirror symmetry (loss)**: under sign-symmetric sim, mirrored windows score identically to originals (`test_dataset_loss_mirror_invariance_under_symmetric_sim`).
3. ✅ **Warmup discard / weighting / aggregation**: per-channel NMSE responds only to post-warmup error, `weighted_total = Σ wᵢ·nmseᵢ`, `dataset_loss = arithmetic mean across windows` (six dedicated tests).
4. ✅ **Mirror symmetry (rollout)**: under default symmetric STD params, mirrored windows produce sign-flipped sim signals on antisymmetric channels and identical signals on symmetric ones (`test_mirror_invariant_under_default_params`).
5. ✅ **Sampler determinism (rollout)**: same window replayed twice produces bit-identical sim signals (`test_run_is_deterministic`).
6. ✅ **Identity sanity check on real bag**: `dataset_loss` runs end-to-end on `examples/analysis/bags/circle_Apr6_100Hz.npz` with YAML defaults. Recorded baseline (no mirror, 11 windows): **total = 5.0224**, per-channel NMSE = `{v_y: 1.85, a_x: 0.33, yaw_rate: 0.30, v_x: 0.15}`. _Updated 2026-05-05: windows now seeded with VESC wheel speed (`omega_f = omega_r = rs_core_speed/R_w`, AWD assumption). On this steady-circle bag the change is a slight uptick (was 4.7960) — no-slip seeding happens to be accurate at constant ~1 m/s. The benefit is expected on aggressive launch/drift bags where no-slip is wrong._
7. ✅ **Smoke test integration**: full pipeline runs in ~2.6 s on 11 windows (~160 ms/window, ~1.6 ms/step) — comfortably inside the OVERVIEW's ~1 ms/step trial-budget assumption.
8. ✅ **Channel breakdown sanity**: `v_y` dominates the residual (1.83), then `a_x` (0.30), `yaw_rate` (0.25), `v_x` (0.15) — confirms weights are sensible (lateral-heavy, exactly the regime we want Optuna to attack).
9. ✅ **Visual validation**: `python -m examples.analysis.sysid.rollout --path <bag>` overlays sim windows onto Vicon signals across all loss channels + steering + slip; eyeball-confirmed init_state reset and command replay are correct on `circle_Apr6_100Hz.npz`.

## Out of scope (future plans)

- Optuna study setup, sampler choice, parameter search-space bounds.
- Multi-NPZ aggregation across multiple recordings.
- Adding `omega_front`/`omega_rear` channels to the loss. (VESC wheel-speed feedback is now read into the dataset for the *reset*, but not yet scored. Reset alone removes the cold-start transient; adding a loss channel would *constrain* longitudinal-tire identifiability — defer until Phase 4 `a_x` plateaus.)
- CLI / wandb logging.
- Per-window weighting by maneuver type.
