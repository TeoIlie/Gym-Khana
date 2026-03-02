import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import gymnasium as gym
import torch
from stable_baselines3 import PPO

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from train.config.env_config import (
    END_LEARNING_RATE,
    START_LEARNING_RATE,
)
from train.train_utils import linear_schedule


def apply_transfer_resets(model, reset_log_std=-0.5):
    """Replicate the exact reset operations from train_common.py:305-326."""
    # Fresh LR schedule
    model.learning_rate = linear_schedule(START_LEARNING_RATE, END_LEARNING_RATE)
    model.lr_schedule = model.learning_rate

    # Fresh Adam optimizer
    model.policy.optimizer = model.policy.optimizer_class(
        model.policy.parameters(),
        lr=model.learning_rate(1.0),
        **model.policy.optimizer_kwargs,
    )

    # Reset update counter
    model._n_updates = 0

    # Reset log_std
    if reset_log_std is not None:
        if not hasattr(model.policy, "log_std"):
            raise AttributeError("Policy has no log_std parameter (not a continuous action distribution)")
        model.policy.log_std.data.fill_(reset_log_std)


class TestTransferTrainingResets(unittest.TestCase):
    """Test that transfer training resets work correctly."""

    @classmethod
    def setUpClass(cls):
        cls._tmp_dir = tempfile.mkdtemp()
        cls._env = gym.make("Pendulum-v1")
        model = PPO("MlpPolicy", cls._env, n_steps=64, device="cpu")
        model.learn(total_timesteps=128)
        cls._model_path = os.path.join(cls._tmp_dir, "test_model.zip")
        model.save(cls._model_path)
        # Save source log_std for comparison
        cls._source_log_std = model.policy.log_std.data.clone()

    @classmethod
    def tearDownClass(cls):
        cls._env.close()
        import shutil

        shutil.rmtree(cls._tmp_dir, ignore_errors=True)

    def _load_source(self):
        env = gym.make("Pendulum-v1")
        return PPO.load(self._model_path, env=env, device="cpu")

    def _get_named_weight_dict(self, model, exclude_log_std=True):
        result = {}
        for name, param in model.policy.named_parameters():
            if exclude_log_std and name == "log_std":
                continue
            result[name] = param.data.clone()
        return result

    def test_source_optimizer_has_state(self):
        model = self._load_source()
        state = model.policy.optimizer.state
        self.assertGreater(len(state), 0, "Source optimizer should have non-empty state after training")
        model.policy.optimizer = None  # help GC
        model.get_env().close()

    def test_network_weights_preserved(self):
        source = self._load_source()
        source_weights = self._get_named_weight_dict(source)

        target = self._load_source()
        apply_transfer_resets(target)
        target_weights = self._get_named_weight_dict(target)

        for name in source_weights:
            self.assertTrue(
                torch.equal(source_weights[name], target_weights[name]),
                f"Weight {name} changed after reset",
            )
        source.get_env().close()
        target.get_env().close()

    def test_optimizer_state_fresh(self):
        model = self._load_source()
        apply_transfer_resets(model)
        state = model.policy.optimizer.state
        self.assertEqual(len(state), 0, "New optimizer should have empty state")
        model.get_env().close()

    def test_optimizer_lr_is_start_lr(self):
        model = self._load_source()
        apply_transfer_resets(model)
        lr = model.policy.optimizer.param_groups[0]["lr"]
        self.assertAlmostEqual(lr, START_LEARNING_RATE, places=10)
        model.get_env().close()

    def test_optimizer_eps_preserved(self):
        model = self._load_source()
        apply_transfer_resets(model)
        eps = model.policy.optimizer.param_groups[0]["eps"]
        self.assertEqual(eps, 1e-5, "Adam eps should be SB3 default 1e-5")
        model.get_env().close()

    def test_optimizer_tracks_all_policy_params(self):
        model = self._load_source()
        apply_transfer_resets(model)

        opt_ptrs = set()
        for group in model.policy.optimizer.param_groups:
            for p in group["params"]:
                opt_ptrs.add(p.data_ptr())

        policy_ptrs = {p.data_ptr() for p in model.policy.parameters()}
        self.assertEqual(opt_ptrs, policy_ptrs, "Optimizer should track all policy parameters")
        model.get_env().close()

    def test_lr_schedule_endpoints(self):
        model = self._load_source()
        apply_transfer_resets(model)

        self.assertAlmostEqual(model.lr_schedule(1.0), START_LEARNING_RATE, places=10)
        self.assertAlmostEqual(model.lr_schedule(0.0), END_LEARNING_RATE, places=10)
        self.assertIs(model.lr_schedule, model.learning_rate, "lr_schedule and learning_rate should be the same object")
        model.get_env().close()

    def test_log_std_reset_to_value(self):
        model = self._load_source()
        apply_transfer_resets(model, reset_log_std=-0.5)

        expected = torch.full_like(model.policy.log_std.data, -0.5)
        self.assertTrue(torch.equal(model.policy.log_std.data, expected))
        model.get_env().close()

    def test_log_std_skip_when_none(self):
        model = self._load_source()
        original_log_std = model.policy.log_std.data.clone()
        apply_transfer_resets(model, reset_log_std=None)

        self.assertTrue(
            torch.equal(model.policy.log_std.data, original_log_std),
            "log_std should be unchanged when reset_log_std=None",
        )
        model.get_env().close()

    def test_log_std_in_optimizer_after_reset(self):
        model = self._load_source()
        apply_transfer_resets(model, reset_log_std=-0.5)

        opt_ptrs = set()
        for group in model.policy.optimizer.param_groups:
            for p in group["params"]:
                opt_ptrs.add(p.data_ptr())

        self.assertIn(
            model.policy.log_std.data_ptr(),
            opt_ptrs,
            "log_std should be tracked by the new optimizer",
        )
        model.get_env().close()

    def test_n_updates_reset_to_zero(self):
        model = self._load_source()
        # Precondition: source model should have _n_updates > 0 after training
        self.assertGreater(model._n_updates, 0, "Source model should have _n_updates > 0")

        apply_transfer_resets(model)
        self.assertEqual(model._n_updates, 0)
        model.get_env().close()


class TestTransferTrainArgParsing(unittest.TestCase):
    """Test CLI argument parsing for transfer training mode."""

    def _run_main_with_args(self, args_list):
        """Run main() with mocked sys.argv and a mock profile, return the mock for transfer_train."""
        from dataclasses import dataclass

        @dataclass
        class FakeProfile:
            display_name: str = "test"
            train_config: dict = None
            track_pool: list = None

        profile = FakeProfile()

        with (
            patch("sys.argv", ["prog"] + args_list),
            patch("train.train_common.transfer_train") as mock_transfer,
        ):
            from train.train_common import main

            try:
                main(profile)
            except SystemExit as e:
                return mock_transfer, e
            return mock_transfer, None

    def test_mode_f_requires_path(self):
        mock_transfer, exit_exc = self._run_main_with_args(["--m", "f"])
        self.assertIsNotNone(exit_exc, "--m f without --path should raise SystemExit")
        mock_transfer.assert_not_called()

    def test_reset_log_std_not_passed_uses_default(self):
        mock_transfer, exit_exc = self._run_main_with_args(["--m", "f", "--path", "/tmp/model.zip"])
        self.assertIsNone(exit_exc)
        mock_transfer.assert_called_once_with(profile=unittest.mock.ANY, model_path="/tmp/model.zip")
        self.assertNotIn(
            "reset_log_std",
            mock_transfer.call_args.kwargs,
            "reset_log_std should not be passed; default lives on function signature",
        )


if __name__ == "__main__":
    unittest.main()
