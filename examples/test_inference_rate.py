"""Benchmark inference rate of a trained PPO model.

Measures pure model.predict() latency without any environment overhead,
to determine achievable control loop frequency on real hardware.

Usage:
    python examples/test_inference_rate.py --path outputs/models/<run_id>/best_model/best_model.zip
    python examples/test_inference_rate.py --path outputs/models/<run_id>/best_model/best_model.zip --n 50000
"""

import argparse
import time

import numpy as np
import torch
from stable_baselines3 import PPO


def benchmark_model(model, n_iters: int, warmup: int) -> np.ndarray:
    """Run inference in a tight loop and return per-call times in seconds."""
    obs_space = model.observation_space
    # Warmup
    for _ in range(warmup):
        obs = obs_space.sample()
        model.predict(obs, deterministic=True)

    # Timed runs
    times = np.empty(n_iters)
    for i in range(n_iters):
        obs = obs_space.sample()
        t0 = time.perf_counter()
        model.predict(obs, deterministic=True)
        times[i] = time.perf_counter() - t0

    return times


def print_report(times: np.ndarray, device: str):
    """Print timing statistics."""
    ms = times * 1000
    mean_ms = np.mean(ms)
    hz = 1000.0 / mean_ms

    print(f"\n{'=' * 50}")
    print(f"  Device: {device}")
    print(f"  Iterations: {len(ms)}")
    print(f"{'=' * 50}")
    print(f"  Mean:   {mean_ms:.4f} ms  ({hz:.0f} Hz)")
    print(f"  Median: {np.median(ms):.4f} ms")
    print(f"  Std:    {np.std(ms):.4f} ms")
    print(f"  Min:    {np.min(ms):.4f} ms")
    print(f"  Max:    {np.max(ms):.4f} ms")
    print(f"  P95:    {np.percentile(ms, 95):.4f} ms")
    print(f"  P99:    {np.percentile(ms, 99):.4f} ms")
    print()

    targets = [100, 200, 500, 1000]
    print("  Control loop feasibility:")
    for rate in targets:
        budget_ms = 1000.0 / rate
        ok = "YES" if mean_ms < budget_ms else "NO"
        print(f"    {rate:>4d} Hz ({budget_ms:.1f} ms budget): {ok}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Benchmark PPO inference rate")
    parser.add_argument("--path", required=True, help="Path to model .zip file")
    parser.add_argument("--n", type=int, default=10000, help="Number of inference iterations (default: 10000)")
    parser.add_argument("--warmup", type=int, default=100, help="Warmup iterations (default: 100)")
    args = parser.parse_args()

    # CPU benchmark
    print(f"Loading model from {args.path} (device=cpu)")
    model_cpu = PPO.load(args.path, device="cpu")
    print(f"Observation space: {model_cpu.observation_space}")
    print(f"Action space: {model_cpu.action_space}")

    times_cpu = benchmark_model(model_cpu, args.n, args.warmup)
    print_report(times_cpu, "cpu")

    # CUDA benchmark if available
    if torch.cuda.is_available():
        print(f"Loading model from {args.path} (device=cuda)")
        model_cuda = PPO.load(args.path, device="cuda")
        times_cuda = benchmark_model(model_cuda, args.n, args.warmup)
        print_report(times_cuda, "cuda")


if __name__ == "__main__":
    main()
