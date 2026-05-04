# MVP Optuna System Identification Pipeline (STP target)

## Context

The STP (Single Track Pacejka) dynamics model in `gymkhana/envs/dynamic_models/single_track_pacejka/single_track_pacejka.py` is a dynamic single-track bicycle with a **lateral-only** Pacejka Magic Formula tire model — no longitudinal slip, no wheel-spin states, 7-element state vector. Its tire params are 8 coefficients: `B_f, C_f, D_f, E_f, B_r, C_r, D_r, E_r`, plus `mu`, `I_z`. These are seeded from a different vehicle's on-track sysid (`f1tenth_stp.yaml` line 5–6), so the sim-to-real gap on the actual lab car is unknown.

We have 16 NPZ rosbags (100 Hz) under `examples/analysis/bags/` containing commanded speed/steering, Vicon ground truth `(x, y, yaw, body_vx, body_vy, yaw_rate)`, and VESC wheel-speed feedback. The goal is to use Optuna to tune the hard-to-measure subset of STP params so the simulator matches the real car.

**Staging strategy**: target STP first because it has fewer params (~10 tunable), no longitudinal-slip / wheel-spin states (so no observability questions about `ω_f, ω_r`), and a simpler 7-state vector. Once the pipeline is validated on STP, the user will augment it for the more complex STD model in a follow-up step (adds longitudinal Pacejka coefficients, wheel-spin dynamics, and the `ω` residual channel that VESC enables).

The output is a tuned YAML in `gymkhana/envs/params/` that drop-in replaces `f1tenth_stp.yaml` for downstream training.

## Decisions locked in

- **Target model**: STP (`vehicle_dynamics_stp`, 7-state). STD comes later.
- **Location**: `examples/sysid/`
- **Loss**: multi-step rollout error with configurable window size `W`. Each window starts by resetting the sim state to the real state, then applies the bag's control sequence for `W` steps, comparing the rolled-out sim trajectory to the real trajectory across the window. Windows tile the bag (non-overlapping) so every sample contributes once.
- **Actuator dynamics**: already represented in the simulator as rate limits (`sv_min`, `sv_max` on steering velocity; `a_max` on acceleration via `accl_constraints`). These are exposed as tunable params in the MVP rather than treated as a separate "lag" model — fitting them lets Optuna match the real car's effective steering and throttle responsiveness without adding a new dynamics term.
- **Data scope**: single representative bag with intra-bag train/val split

## Directory layout

```
examples/sysid/
  README.md                  # how to run; STP-first staging note
  param_space.py             # tunable params + bounds + base param loading
  data_loader.py             # NPZ → time-aligned (state, control) arrays for STP
  loss.py                    # multi-step rollout loss
  objective.py               # Optuna objective fn
  run_optuna.py              # entry point: study, trials, save best YAML
  configs/
    mvp_stp.yaml             # bag path, n_trials, window size, channel weights
  results/                   # gitignored: study .db, tuned YAMLs, plots
```

Add `results/` to `.gitignore`. Designed so that adding STD support later is a matter of adding a `mvp_std.yaml` config + a small `model: "stp" | "std"` switch in `run_optuna.py` — the loss, data loader, and objective are written generically against a `(dynamics_fn, state_layout)` pair.

## Reusable code (do not reimplement)

- `vehicle_dynamics_stp(x, u_init, params)` — `gymkhana/envs/dynamic_models/single_track_pacejka/single_track_pacejka.py`. Call directly; no env needed. 7-state, returns derivatives.
- `RK4Integrator.integrate(f, x, u, dt, params)` — `gymkhana/envs/integrator.py:32`. Same RK4 the env uses, so fit-time and runtime integration match.
- `p_accl(desired_speed, V, ...)` — converts bag's `cmd_speed` to `accl`. Used by `SpeedAction` in `gymkhana/envs/action.py:157`.
- `bang_bang_steer(target_delta, current_delta, ...)` — converts bag's `cmd_steer` to `steering_velocity`. Used by `SteerAction`.
- `load_params(name)` — `gymkhana/envs/params/__init__.py`. Supports YAML inheritance via `_base`/`_overrides`.
- `get_standardized_state_st(x)` — STP shares ST's 7-state layout; the existing dispatcher in `dynamic_models/__init__.py` reuses ST's standardizer for STP. We do the same in the loss to extract `v_x, v_y, yaw_rate, β`.

## Component specifications

### `param_space.py`

- Loads `f1tenth_stp.yaml` via `load_params("f1tenth_stp")` as the base (frozen) params.
- Defines `TUNABLE_STP` dict, param name → `(low, high, log_scale)`. MVP subset (~13 params):
  - `mu` (0.5, 1.5)
  - `I_z` (0.02, 0.08, log)
  - Front Pacejka: `B_f` (1, 20), `C_f` (0.5, 3), `D_f` (0.3, 1.5), `E_f` (-2, 1)
  - Rear Pacejka: `B_r` (1, 20), `C_r` (0.5, 4), `D_r` (0.3, 1.5), `E_r` (-2, 1)
  - Actuator rate limits: `sv_max` (1.0, 8.0), `a_max` (3.0, 10.0). `sv_min` is constrained to `-sv_max` (symmetric) inside `suggest_params` to halve the search dimension; same for any sign-symmetric pair.
  - Bounds chosen to bracket the seeded values in `f1tenth_stp.yaml` by ~3× while keeping shape factors `C` and curvature `E` in physically sane ranges. Actuator bounds bracket the seeded `sv_max=4.95` and `a_max=6.0`.
- `suggest_params(trial, base_params, tunable) -> dict`: returns a fresh params dict with tunables overridden via `trial.suggest_float`. Generic over `tunable` so STD can pass its own dict later.

### `data_loader.py`

- `load_bag(path) -> dict` of arrays.
- `build_state_stp(bag) -> np.ndarray` of shape `(N, 7)` matching the STP state layout `[x, y, delta, V, psi, psi_dot, beta]`. Map:
  - `[0,1,4]` ← `vicon_x, vicon_y, vicon_yaw`
  - `[2]` (delta) ← approximated from `cmd_steer` for MVP. **Wheel angle is unobserved by Vicon.** Document as a known limitation.
  - `[3]` (V) ← `hypot(vicon_body_vx, vicon_body_vy)`
  - `[5]` (yaw_rate) ← `vicon_r`
  - `[6]` (β) ← `arctan2(vicon_body_vy, vicon_body_vx)`, masked to 0 where `V < 0.2`
- VESC wheel-speed feedback is **not used as a state residual in STP** (the model has no wheel-spin states), but the loader optionally records it for cross-validation: a sanity check that the bag's `V` agrees with `R_w · ω_VESC` — if it doesn't, the wheel is slipping and that segment is a poor fit for STP (which can't represent longitudinal slip). VESC becomes a first-class signal in the STD extension.
- `build_controls_stp(bag, x_traj, params) -> np.ndarray` of shape `(N, 2)`: convert `cmd_speed → accl` via `p_accl`, `cmd_steer → steering_velocity` via `bang_bang_steer`, using the corresponding step's state for the converters.
- `train_val_split(N, window_size, val_frac=0.2)` returns two arrays of **window start indices**. Use a **temporal block split**: last `val_frac` of the bag (rounded to whole windows) becomes val. Avoids leaking samples across the boundary.

### `loss.py`

- `rollout_loss(x_traj, u_traj, dynamics_fn, params, dt, integrator, channel_weights, window_indices, window_size, state_layout) -> float`:
  - For each window start index `i0` in `window_indices`:
    1. Initialize `x_sim = x_traj[i0]` (reset sim state to real state at window start).
    2. For `k` in `0..W-1`: apply control `u_traj[i0+k]`, step `x_sim = integrator.integrate(dynamics_fn, x_sim, u_traj[i0+k], dt, params)`. Accumulate residuals against `x_traj[i0+k+1]`.
    3. Sum per-channel squared errors across the window.
  - Window tiling: `window_indices = range(0, N - W, W)` → non-overlapping, every step contributes once. `W=1` recovers one-step error; large `W` approaches full-trajectory rollout.
  - `state_layout` ("stp" or "std") tells the loss which indices to read for each channel. STP uses ST's standardizer pattern.
  - Channels and default weights for STP (computed once on the full bag, reused across trials):
    - `v_x` (V·cos(β)): w=1.0, normalized by std across bag
    - `v_y` (V·sin(β)): w=1.5, normalized by std
    - `yaw_rate`: w=1.0, normalized by std
    - `β`: w=2.0, normalized by std (drift-relevant; explicit term so error is angular not Cartesian — see prior conversation)
  - Note: STP has no `ω` channel. The STD extension adds `omega_driven` (w≈0.5) which is what makes longitudinal-slip params identifiable; for STP, only lateral params are tunable so `(v_y, β, yaw_rate)` are the load-bearing residuals.
  - Return mean squared error across all windows × steps × channels.
  - **Numerical guard**: if any state becomes non-finite mid-rollout, abort the window early and add a large finite penalty rather than `nan` — keeps Optuna's TPE prior from being poisoned.

### `objective.py`

- `make_objective(x_traj, u_traj, dynamics_fn, base_params, tunable, dt, channel_weights, train_window_indices, window_size, state_layout)`:
  - Returns `objective(trial)` closure.
  - Inside: `params = suggest_params(trial, base_params, tunable)`, then `return rollout_loss(...)`.
  - Catch `np.linalg.LinAlgError` / `ValueError` from numerical blowups → return `float("inf")`.

### `run_optuna.py`

- Loads `configs/mvp_stp.yaml`: `model`, `bag_path`, `n_trials`, `study_name`, `channel_weights`, `window_size`, `val_frac`.
- Resolves model dispatch: `model: stp` → `(dynamics_fn=vehicle_dynamics_stp, base_yaml="f1tenth_stp", tunable=TUNABLE_STP, state_layout="stp")`. STD will plug in the same way later.
- Builds `state` and `control` arrays once (outside the trial loop — critical for speed).
- Creates Optuna study with TPE sampler, SQLite storage at `results/<study_name>.db`.
- Runs `study.optimize(objective, n_trials=...)`.
- After completion:
  - Compute val loss on the held-out 20% with `study.best_params`.
  - Print summary: train loss, val loss, per-channel breakdown.
  - Write `results/<study_name>_tuned.yaml` with `_base: f1tenth_stp` + `_overrides:` block (uses existing inheritance machinery).
  - Manual visual check via `examples/analysis/traj_compare.py --model stp` (documented in README).

### `configs/mvp_stp.yaml`

```yaml
model: stp
bag_path: examples/analysis/bags/rosbag2_2026_04_27-16_24_49_100Hz.npz
study_name: mvp_stp
n_trials: 500
dt: 0.01
window_size: 50  # 0.5s at 100Hz; tune per dataset. W=1 = one-step; W=N = full rollout
val_frac: 0.2
channel_weights:
  v_x: 1.0
  v_y: 1.5
  yaw_rate: 1.0
  beta: 2.0
```

### `pyproject.toml`

Add a new poetry group:

```toml
[tool.poetry.group.sysid.dependencies]
optuna = "^4.0.0"
```

`numpy`, `scipy`, `pyyaml`, `matplotlib` already covered by base/dev. Install with `poetry install --with sysid`.

## Files to modify / create

**Create:**
- `examples/sysid/README.md`
- `examples/sysid/param_space.py`
- `examples/sysid/data_loader.py`
- `examples/sysid/loss.py`
- `examples/sysid/objective.py`
- `examples/sysid/run_optuna.py`
- `examples/sysid/configs/mvp_stp.yaml`

**Modify:**
- `pyproject.toml` — add `[tool.poetry.group.sysid.dependencies]`
- `.gitignore` — add `examples/sysid/results/`

**Do not modify:**
- The dynamics model, integrator, or any `gymkhana/` code. The pipeline is read-only against the simulator.

## Forward-compatibility notes (for the STD follow-up step)

To keep the STP MVP a clean foundation:

- Pass `dynamics_fn` and `state_layout` as arguments throughout `loss.py` / `objective.py` rather than hard-coding `vehicle_dynamics_stp` — STD only needs to add a new dispatch entry.
- `param_space.py`: keep `TUNABLE_STP` as one dict and add `TUNABLE_STD` later. Don't merge them.
- `build_state_*` and `build_controls_*` are model-specific (different state shapes) — write them as separate functions, not a polymorphic one. STD will add `build_state_std` that includes `ω_f, ω_r` from VESC.
- The `omega_driven` channel and `vesc_axle` config flag are deliberately not introduced in STP — they get added when STD support lands.

## Known limitations to document in README

1. `delta` is not directly observed — approximated from `cmd_steer`. STP uses Pacejka only for lateral forces, so `delta` error directly biases lateral-tire fits; consider this when interpreting `B_f, D_f` results.
2. Actuator dynamics are modeled only as rate limits (`sv_max`, `a_max`) — there is no first-order lag or pure dead-time term. If real-car responsiveness is dominated by command-to-actuation latency (servo delay, motor controller filter) rather than rate saturation, the fit will distort tire params to compensate. Adding an explicit lag/delay is a follow-up.
3. Single-bag fit risks overfitting to a specific maneuver. Validate qualitatively on other bags via `traj_compare.py` before trusting the result.
4. STP cannot represent longitudinal slip — segments with hard accel/brake (especially where VESC `R_w · ω` diverges from Vicon `V`) are out-of-model. The loader should optionally exclude or down-weight these segments; for MVP, just flag them in the README and recommend picking a bag dominated by steady cornering.
5. `window_size` is a key knob: too small (W~1) under-exercises integration coupling and can let actuator-rate-limit effects hide in tire params; too large (W~hundreds) lets `delta` approximation error and any model bias compound and dominate the loss. Start with W≈50 (0.5 s) and sweep in `{10, 50, 100, 200}` once the pipeline runs end-to-end.

## Implementation steps

Each step is independently testable: before moving to the next, run the test described and confirm it passes / produces sane output. Do not bundle steps.

### Step 1 — Scaffolding & dependency

**Do:**
- Create `examples/sysid/` directory tree (empty files for `param_space.py`, `data_loader.py`, `loss.py`, `objective.py`, `run_optuna.py`, `configs/mvp_stp.yaml`, `README.md`).
- Add `[tool.poetry.group.sysid.dependencies]` with `optuna` to `pyproject.toml`.
- Add `examples/sysid/results/` to `.gitignore`.

**Test:**
- `poetry install --with sysid` succeeds.
- `python -c "import optuna; print(optuna.__version__)"` works.
- `python -c "from gymkhana.envs.dynamic_models.single_track_pacejka import vehicle_dynamics_stp"` works.

### Step 2 — Data loader

**Do:**
- Implement `load_bag(path)`, `build_state_stp(bag)`, `build_controls_stp(bag, x_traj, params)`, `train_val_split(N, W, val_frac)` in `data_loader.py`.

**Test:**
- Write a small `__main__` block (or a throwaway notebook cell) that loads one bag, prints state/control shapes, and asserts:
  - `state.shape == (N, 7)`, all `np.isfinite`.
  - `V[0]` matches `hypot(vicon_body_vx[0], vicon_body_vy[0])` to 1e-6.
  - β masked to 0 below the speed threshold; finite elsewhere.
  - controls shape `(N, 2)`, accl bounded by `±a_max`, steering velocity bounded by `±sv_max`.
  - `train_val_split` returns disjoint window-start arrays whose union covers all whole windows.

### Step 3 — Standalone rollout sanity (no loss yet)

**Do:**
- In `loss.py`, implement a helper `rollout(x0, u_window, dynamics_fn, params, dt, integrator) -> np.ndarray` of shape `(W+1, state_dim)`.

**Test:**
- Pick window `i0=0`, `W=50`. Run rollout with seeded `f1tenth_stp.yaml` params.
- Plot `rollout` vs `x_traj[i0:i0+W+1]` for `(x, y)` and `V`. Eyeball that they're in the same neighborhood (not identical — that's the whole point of sysid).
- Confirm no NaNs, no blowup.

### Step 4 — Loss function

**Do:**
- Implement `rollout_loss(...)` per spec, including channel weights, std normalization (computed once on the full bag and passed in), numerical guard.

**Test:**
- Compute loss with seeded params on the full bag — record the number; this is the **baseline** to beat.
- Inject a deliberately bad `D_f = 0.05` (peak lateral force × 13 → tiny) and confirm loss is **strictly higher**. If not, the loss is broken.
- Inject `D_f = D_f_seed` exactly — confirm loss returns to baseline (deterministic).
- Confirm loss is finite when params force a numerical blowup (e.g., `mu=0.01`); should hit the penalty path, not return NaN.

### Step 5 — Param space & objective

**Do:**
- Implement `TUNABLE_STP` and `suggest_params(trial, base, tunable)` in `param_space.py`.
- Implement `make_objective(...)` in `objective.py`.

**Test:**
- Build an Optuna study with `n_trials=10`, no storage, single-thread.
- Run it. Confirm:
  - All 10 trials complete (no exceptions).
  - At least one trial has loss < baseline from Step 4.
  - `study.best_params` keys match `TUNABLE_STP` keys exactly.
  - `sv_min == -sv_max` in best params (symmetry enforced).

### Step 6 — End-to-end runner

**Do:**
- Implement `run_optuna.py`: load config, build state/controls once, set up SQLite-backed study, run optimization, print per-channel loss breakdown, write tuned YAML with `_base` / `_overrides`.

**Test:**
- `python examples/sysid/run_optuna.py --config examples/sysid/configs/mvp_stp.yaml --n-trials 50`.
- Confirm:
  - `results/mvp_stp.db` exists.
  - `results/mvp_stp_tuned.yaml` exists, has correct `_base: f1tenth_stp` and `_overrides:` with all 13 tunables.
  - Val loss printed and is finite.
  - Per-channel breakdown printed (4 channels for STP).

### Step 7 — YAML inheritance round-trip

**Do:**
- Copy `results/mvp_stp_tuned.yaml` into `gymkhana/envs/params/mvp_stp_tuned.yaml`.

**Test:**
- `python -c "from gymkhana.envs.params import load_params; p = load_params('mvp_stp_tuned'); assert 'D_f' in p; print(p['D_f'])"` works and reflects the tuned (not seed) value.
- All STP-required keys present (`B_f, C_f, D_f, E_f, B_r, C_r, D_r, E_r, mu, m, lf, lr, h_s, I_z, s_min, s_max, sv_min, sv_max, v_switch, a_max, v_min, v_max`).

### Step 8 — Visual sim2real validation

**Do:**
- Run `examples/analysis/traj_compare.py` twice on the **same bag used for fitting**: once with seed params, once with tuned params (passed via the env's `params` config).

**Test:**
- Save both plots side-by-side. Confirm tuned plot shows visibly tighter `v_y`, `β`, and `yaw_rate` panels on the held-out tail of the bag.
- If no improvement: investigate (likely loss weights or window size, not pipeline correctness).

### Step 9 — Cross-bag generalization spot-check

**Do:**
- Run `traj_compare.py --model stp` with tuned params on **a different bag** not used for fitting.

**Test:**
- Confirm tuned params still beat seed params qualitatively. If they overfit (tuned worse than seed on held-out bag), document and flag — multi-bag fitting becomes the natural next iteration.

### Step 10 — Window-size sweep (optional but recommended)

**Do:**
- Re-run Step 6 with `window_size ∈ {10, 50, 100, 200}` (4 separate studies, separate `study_name`s).

**Test:**
- Compare val losses and tuned-param distributions across W. Document the chosen W in the README and pick the one with best val loss + parameter stability across runs.

---

## Verification

End-to-end smoke test (manual, no automated test in MVP):

1. `poetry install --with sysid`
2. `python examples/sysid/run_optuna.py --config examples/sysid/configs/mvp_stp.yaml --n-trials 20` — verifies the loop runs, study DB writes, val loss computes.
3. Inspect `results/mvp_stp_tuned.yaml` — confirm overrides written, base reference correct, file loads via `load_params("mvp_stp_tuned")` (after copying into `gymkhana/envs/params/`).
4. Run `python examples/analysis/traj_compare.py --path <bag> --model stp` once with default params and once with tuned params — confirm visual improvement on the held-out portion of the bag, especially in `v_y`, `β`, and `yaw_rate` panels.
5. Full run: `--n-trials 500`, expect ~5-15 minutes on CPU depending on bag length. Per-channel loss breakdown printed at end should show `v_y` and `β` improvements vs default params.

## Out of scope (deliberate, for follow-up work)

- **STD support** (longitudinal Pacejka, wheel-spin states, `ω_driven` residual using VESC) — explicit next step after STP MVP is validated.
- Multi-bag fitting with cross-validation
- Explicit first-order actuator lag and pure dead-time (only rate limits `sv_max`/`a_max` are tuned in MVP)
- CMA-ES or other samplers beyond TPE
- Automated regression tests in `tests/`
- Wandb logging of trials
