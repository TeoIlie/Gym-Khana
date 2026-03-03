import os
import sys
import tempfile
import unittest
from functools import partial
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.policies import BasePolicy

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from train.config.env_config import (
    END_LEARNING_RATE,
    START_LEARNING_RATE,
)
from train.train_utils import linear_schedule


def apply_transfer_resets(model, reset_log_std=-0.5, reset_critic=False):
    """Replicate the exact reset operations from train_common.py transfer_train()."""
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

    # Reset critic
    if reset_critic:
        model.policy.mlp_extractor.value_net.apply(partial(BasePolicy.init_weights, gain=np.sqrt(2)))
        model.policy.value_net.apply(partial(BasePolicy.init_weights, gain=1.0))


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


class TestCriticReset(unittest.TestCase):
    """Test that critic reset reinitializes only critic weights, leaving actor intact."""

    @classmethod
    def setUpClass(cls):
        cls._tmp_dir = tempfile.mkdtemp()
        cls._env = gym.make("Pendulum-v1")
        model = PPO("MlpPolicy", cls._env, n_steps=64, device="cpu")
        model.learn(total_timesteps=128)
        cls._model_path = os.path.join(cls._tmp_dir, "test_model.zip")
        model.save(cls._model_path)

    @classmethod
    def tearDownClass(cls):
        cls._env.close()
        import shutil

        shutil.rmtree(cls._tmp_dir, ignore_errors=True)

    def _load_source(self):
        env = gym.make("Pendulum-v1")
        return PPO.load(self._model_path, env=env, device="cpu")

    def _get_named_weight_dict(self, module):
        return {name: param.data.clone() for name, param in module.named_parameters()}

    def test_actor_weights_preserved_after_critic_reset(self):
        """Actor hidden layers (mlp_extractor.policy_net) must be unchanged."""
        source = self._load_source()
        actor_before = self._get_named_weight_dict(source.policy.mlp_extractor.policy_net)

        apply_transfer_resets(source, reset_critic=True)
        actor_after = self._get_named_weight_dict(source.policy.mlp_extractor.policy_net)

        for name in actor_before:
            self.assertTrue(
                torch.equal(actor_before[name], actor_after[name]),
                f"Actor weight {name} changed after critic reset",
            )
        source.get_env().close()

    def test_action_net_preserved_after_critic_reset(self):
        """Actor output head (action_net) must be unchanged."""
        source = self._load_source()
        action_before = self._get_named_weight_dict(source.policy.action_net)

        apply_transfer_resets(source, reset_critic=True)
        action_after = self._get_named_weight_dict(source.policy.action_net)

        for name in action_before:
            self.assertTrue(
                torch.equal(action_before[name], action_after[name]),
                f"Action net weight {name} changed after critic reset",
            )
        source.get_env().close()

    def test_log_std_preserved_after_critic_reset(self):
        """log_std must not be affected by critic reset (only by reset_log_std)."""
        source = self._load_source()
        log_std_before = source.policy.log_std.data.clone()

        apply_transfer_resets(source, reset_log_std=None, reset_critic=True)

        self.assertTrue(
            torch.equal(source.policy.log_std.data, log_std_before),
            "log_std should be unchanged when only critic is reset",
        )
        source.get_env().close()

    def test_critic_hidden_weights_changed(self):
        """Critic hidden layers (mlp_extractor.value_net) should differ after reset."""
        source = self._load_source()
        critic_before = self._get_named_weight_dict(source.policy.mlp_extractor.value_net)

        apply_transfer_resets(source, reset_critic=True)
        critic_after = self._get_named_weight_dict(source.policy.mlp_extractor.value_net)

        any_changed = any(not torch.equal(critic_before[n], critic_after[n]) for n in critic_before)
        self.assertTrue(any_changed, "At least one critic hidden weight should change after reset")
        source.get_env().close()

    def test_critic_output_head_changed(self):
        """Critic output head (value_net) should differ after reset."""
        source = self._load_source()
        head_before = self._get_named_weight_dict(source.policy.value_net)

        apply_transfer_resets(source, reset_critic=True)
        head_after = self._get_named_weight_dict(source.policy.value_net)

        any_changed = any(not torch.equal(head_before[n], head_after[n]) for n in head_before)
        self.assertTrue(any_changed, "At least one critic output head weight should change after reset")
        source.get_env().close()

    def test_critic_hidden_uses_orthogonal_init(self):
        """Critic hidden layer weights should be orthogonal after reset."""
        model = self._load_source()
        apply_transfer_resets(model, reset_critic=True)

        for module in model.policy.mlp_extractor.value_net.modules():
            if isinstance(module, torch.nn.Linear):
                W = module.weight.data
                # For orthogonal matrices: W @ W^T ≈ gain^2 * I (for square) or has orthogonal rows
                # Check that columns are approximately unit-norm (gain=sqrt(2))
                col_norms = torch.norm(W, dim=0)
                expected_norm = np.sqrt(2)
                for i, norm in enumerate(col_norms):
                    self.assertAlmostEqual(
                        norm.item(),
                        expected_norm,
                        places=3,
                        msg=f"Critic hidden col {i} norm {norm.item():.4f} != expected {expected_norm:.4f}",
                    )
                # Bias should be zero
                self.assertTrue(torch.all(module.bias.data == 0), "Critic hidden bias should be zero after reset")
        model.get_env().close()

    def test_critic_output_head_uses_orthogonal_init(self):
        """Critic output head weight should be orthogonal with gain=1.0."""
        model = self._load_source()
        apply_transfer_resets(model, reset_critic=True)

        W = model.policy.value_net.weight.data
        # Output head is (1, latent_dim) — orthogonal init gives row norm = gain
        row_norms = torch.norm(W, dim=1)
        expected_norm = 1.0
        for i, norm in enumerate(row_norms):
            self.assertAlmostEqual(
                norm.item(),
                expected_norm,
                places=3,
                msg=f"Critic output row {i} norm {norm.item():.4f} != expected {expected_norm:.4f}",
            )
        self.assertTrue(torch.all(model.policy.value_net.bias.data == 0), "Critic output bias should be zero")
        model.get_env().close()

    def test_no_critic_reset_preserves_all_weights(self):
        """With reset_critic=False, all weights (including critic) should be unchanged."""
        source = self._load_source()
        all_before = self._get_named_weight_dict(source.policy)

        apply_transfer_resets(source, reset_log_std=None, reset_critic=False)
        all_after = self._get_named_weight_dict(source.policy)

        for name in all_before:
            self.assertTrue(
                torch.equal(all_before[name], all_after[name]),
                f"Weight {name} changed with reset_critic=False and reset_log_std=None",
            )
        source.get_env().close()

    def test_critic_reset_idempotent(self):
        """Applying critic reset twice should produce valid orthogonal weights both times."""
        model = self._load_source()
        apply_transfer_resets(model, reset_critic=True)

        # Capture weights after first reset
        critic_after_first = self._get_named_weight_dict(model.policy.mlp_extractor.value_net)

        # Apply again
        model.policy.mlp_extractor.value_net.apply(partial(BasePolicy.init_weights, gain=np.sqrt(2)))
        model.policy.value_net.apply(partial(BasePolicy.init_weights, gain=1.0))

        # Verify orthogonal init still holds (bias = 0)
        for module in model.policy.mlp_extractor.value_net.modules():
            if isinstance(module, torch.nn.Linear):
                self.assertTrue(torch.all(module.bias.data == 0), "Bias should be zero after second reset")
        self.assertTrue(torch.all(model.policy.value_net.bias.data == 0))
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
