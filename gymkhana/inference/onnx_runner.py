"""Lightweight ONNX inference runner for deployed PPO policies.
No dependency on stable-baselines3 or torch. Requires only numpy and onnxruntime.
"""

import numpy as np
import onnxruntime as ort


class OnnxPolicyRunner:
    """Run a deterministic PPO policy exported to ONNX.

    Usage:
        runner = OnnxPolicyRunner("policy.onnx")
        action = runner.predict(obs)
    """

    def __init__(self, onnx_path: str, clip_actions: tuple[float, float] = (-1.0, 1.0)):
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(onnx_path, providers=providers)

        input_shape = self.session.get_inputs()[0].shape
        self.obs_dim = input_shape[1]
        self.clip_low, self.clip_high = clip_actions

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Run deterministic inference.

        Args:
            obs: Observation, shape (obs_dim,) or (batch, obs_dim).

        Returns:
            Action array, shape (act_dim,) or (batch, act_dim).
        """
        squeeze = obs.ndim == 1
        if squeeze:
            obs = obs[np.newaxis, :]

        obs = obs.astype(np.float32)
        action = self.session.run(["action"], {"obs": obs})[0]
        action = np.clip(action, self.clip_low, self.clip_high)

        if squeeze:
            action = action[0]
        return action
