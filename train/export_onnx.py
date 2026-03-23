"""Export an SB3 PPO policy to ONNX for deployment on systems with incompatible numpy versions.

Extracts only the deterministic actor network (no value head, no log_std), producing a
lightweight ONNX model that depends only on onnxruntime + numpy at inference time.

Usage:
    python train/export_onnx.py --path <model path> [--output policy.onnx]
"""

import argparse
import os
import warnings

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from stable_baselines3 import PPO


class DeterministicOnnxPolicy(nn.Module):
    """Minimal wrapper around the SB3 actor for ONNX export.

    Forward pass: obs -> features_extractor -> mlp_extractor.forward_actor -> action_net
    This matches model.predict(obs, deterministic=True) without the value head.
    """

    def __init__(self, policy):
        super().__init__()
        self.features_extractor = policy.pi_features_extractor
        self.mlp_extractor = policy.mlp_extractor
        self.action_net = policy.action_net

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Used once at export time to get the computation graph with a dummy input"""
        features = self.features_extractor(obs)
        latent_pi = self.mlp_extractor.forward_actor(features)
        return self.action_net(latent_pi)


def export_to_onnx(model_or_path, output_path: str):
    """Export an SB3 PPO model's deterministic policy to ONNX.

    Args:
        model_or_path: Either a loaded PPO model or a path to a .zip file.
        output_path: Where to save the .onnx file.
    """
    if isinstance(model_or_path, (str, os.PathLike)):
        model = PPO.load(str(model_or_path), device="cpu")
    else:
        model = model_or_path

    policy = model.policy
    policy.eval()

    obs_dim = model.observation_space.shape[0]
    act_dim = model.action_space.shape[-1]

    det_policy = DeterministicOnnxPolicy(policy)
    det_policy.eval()

    dummy = torch.randn(1, obs_dim, dtype=torch.float32)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*legacy TorchScript-based.*")
        torch.onnx.export(
            det_policy,
            dummy,
            output_path,
            input_names=["obs"],
            output_names=["action"],
            dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
            opset_version=14,
            dynamo=False,
        )

    # Validate: compare SB3 vs ONNX on random inputs
    session = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    test_obs = np.random.randn(5, obs_dim).astype(np.float32)

    sb3_actions, _ = model.predict(test_obs, deterministic=True)
    onnx_actions = session.run(["action"], {"obs": test_obs})[0]
    onnx_actions = np.clip(onnx_actions, -1.0, 1.0)

    # SB3 may return (batch, 1, act_dim); flatten both to (batch, act_dim)
    sb3_flat = sb3_actions.reshape(len(test_obs), -1)
    onnx_flat = onnx_actions.reshape(len(test_obs), -1)
    max_diff = np.max(np.abs(sb3_flat - onnx_flat))
    if not np.allclose(sb3_flat, onnx_flat, atol=1e-5):
        raise RuntimeError(f"ONNX output differs from SB3 (max diff: {max_diff:.2e}). Export may be incorrect.")

    # Clean up the external data file that onnx creates unnecessarily for small models
    external_data = output_path + ".data"
    if os.path.exists(external_data):
        os.remove(external_data)

    print(f"Exported {output_path}  (obs_dim={obs_dim}, act_dim={act_dim}, max_diff={max_diff:.1e})")


def main():
    parser = argparse.ArgumentParser(description="Export SB3 PPO policy to ONNX")
    parser.add_argument("--path", required=True, help="Path to SB3 model .zip file")
    parser.add_argument("--output", default="", help="Output .onnx path (default: alongside .zip)")
    args = parser.parse_args()

    if args.output:
        output_path = args.output
    else:
        base = os.path.splitext(args.path)[0]
        output_path = base + ".onnx"

    export_to_onnx(args.path, output_path)


if __name__ == "__main__":
    main()
