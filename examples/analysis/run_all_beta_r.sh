#!/usr/bin/env bash
# Run beta_r_avg_plot.py for the baselines and the final/ablation learned models sequentially.
# All learned models below were trained with "drift" obs + accl control, matching EVAL_OBS_TYPE /
# EVAL_CONTROL_INPUT in beta_r_avg_plot.py.
set -e

cd "$(dirname "$0")/../.."
# Non-interactive backend so plt.show() doesn't block between runs
export MPLBACKEND=Agg
# --no-cache regenerates every grid; remove it to reuse existing grid_results.npz caches
SCRIPT="python examples/analysis/beta_r_avg_plot.py --no-cache"

# 1. Stanley baseline (must run first: generates the recovery states the others compare against)
$SCRIPT --controller_type stanley --learned_type "" --run_id "" --desc "stanley"

# 2. STMPC baseline
$SCRIPT --controller_type stmpc --learned_type "" --run_id "" --desc "Single-track MPC controller with acados + CasAdi, ported from ForzaETH"

# 3-5. Final models
$SCRIPT --controller_type learned --learned_type recover --run_id f1mgktxe --desc "FINAL transfer model - drift model 8ncsx1rk retrained with Fine-Tuning with Fresh Optimizer + LR Reset + log_std reset + Critic Reinitialization. No curriculum learning, small beta-r initial ranges, no Euclidean reward"
$SCRIPT --controller_type learned --learned_type recover --run_id irdqwnhp --desc "FINAL recovering model - no Euclidean reward"
$SCRIPT --controller_type learned --learned_type drift --run_id 8ncsx1rk --desc "FINAL drift model - CW & CCW on Drift_large, with sparse_width_obs = True, 3rd train with seed 123"

# 6-7. Ablation models
$SCRIPT --controller_type learned --learned_type recover --run_id 8m5f957h --desc "ABLATION recovering model - Euclidean reward, larger beta, r ranges"
$SCRIPT --controller_type learned --learned_type recover --run_id qhj88o3r --desc "ABLATION recovering model - Euclidean reward, curriculum learning, larger beta, r ranges"

echo "All evaluations complete!"
